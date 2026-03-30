from collections import OrderedDict
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from visual_memory.config import Settings
from visual_memory.engine.embedding import make_combined_embedding
from visual_memory.engine.model_registry import registry
from visual_memory.engine.text_recognition import TextRecognizer
from visual_memory.learning import ProjectionHead
from visual_memory.utils import crop_object, find_match, deduplicate_matches, get_logger, mean_luminance, estimate_text_likelihood, collect_system_metrics
from visual_memory.utils.logger import LogTag
from visual_memory.database import DatabaseStore

_settings = Settings()
_log = get_logger(__name__)

_SCAN_CACHE_MAX = 50
_CONFIDENCE_HIGH = 0.6   # mirrors estimator.CONFIDENCE_HIGH


def _direction_from_box(bbox: list, img_w: int) -> str:
    cx = (bbox[0] + bbox[2]) / 2
    nx = (cx / img_w) * 2 - 1  # -1=far left, 1=far right
    if nx < -0.5:  return "to your left"
    if nx < -0.15: return "slightly left"
    if nx < 0.15:  return "ahead"
    if nx < 0.5:   return "slightly right"
    return                "to your right"


def _confidence_label(similarity: float) -> str:
    if similarity >= 0.35: return "high"
    if similarity >= 0.25: return "medium"
    return                        "low"


class ScanPipeline:
    def __init__(self, focal_length_px: float, db_path: str | Path = None):
        self.img_embedder    = registry.img_embedder
        self.text_embedder   = registry.text_embedder   if _settings.enable_ocr   else None
        self.detector        = registry.yoloe_detector
        self.ocr_client      = TextRecognizer() if _settings.enable_ocr else None
        self.estimator       = registry.depth_estimator if _settings.enable_depth else None
        self.focal_length_px = focal_length_px

        # learning state - readable/settable at runtime via set_enable_learning / set_head_weight
        self._enable_learning: bool = _settings.enable_learning
        self._head_weight: float    = _settings.projection_head_weight
        self._head_ramp_at: int     = _settings.projection_head_ramp_at
        # updated by reload_head() after each successful /retrain; drives auto-scaling blend
        self._triplet_count: int    = 0

        self.db = DatabaseStore(db_path if db_path is not None else Path(_settings.db_path))

        self._head = ProjectionHead(dim=_settings.projection_head_dim)
        self._head_trained = self._load_head()
        self._head.eval()
        items = self.db.get_all_items()
        self.database_embeddings = [(item["label"], item["combined_embedding"]) for item in items]

        # scan_id -> {label -> (anchor_emb, query_emb)}; capped at _SCAN_CACHE_MAX entries
        self._emb_cache: OrderedDict = OrderedDict()
        # scan_id -> [PIL Image, ...] in left-to-right match order; for /crop by index
        self._crop_cache: OrderedDict = OrderedDict()

    def _load_head(self) -> bool:
        """Load projection head weights: DB first, file fallback.

        Returns True if weights were loaded from either source.
        """
        state_dict = self.db.load_projection_head()
        if state_dict is not None:
            self._head.load_state_dict(state_dict)
            return True
        return self._head.load(Path(_settings.projection_head_path))

    def reload_database(self):
        """Refresh in-memory database embeddings from the DB store."""
        items = self.db.get_all_items()
        self.database_embeddings = [(item["label"], item["combined_embedding"]) for item in items]

    def get_cached_crop(self, scan_id: str, index: int):
        """Return the PIL Image crop at the given match index, or None."""
        crops = self._crop_cache.get(scan_id)
        if crops is None or index < 0 or index >= len(crops):
            return None
        return crops[index]

    def _store_crops(self, scan_id: str, crops: list) -> None:
        if scan_id not in self._crop_cache:
            if len(self._crop_cache) >= _SCAN_CACHE_MAX:
                self._crop_cache.popitem(last=False)
        self._crop_cache[scan_id] = crops

    def get_cached_embeddings(self, scan_id: str, label: str):
        """Return (anchor_emb, query_emb) for a previous scan result, or None."""
        entry = self._emb_cache.get(scan_id)
        if entry is None:
            return None
        return entry.get(label)

    def reload_head(self) -> bool:
        """Re-read projection head weights (DB first, file fallback) after a retrain.

        Returns True if weights were loaded from either source.
        """
        self._head_trained = self._load_head()
        self._head.eval()
        return self._head_trained

    def set_enable_learning(self, enabled: bool) -> None:
        """Toggle whether the projection head is applied during scan at runtime."""
        self._enable_learning = enabled

    def set_head_weight(self, weight: float, ramp_at: int) -> None:
        """Update the head blend weight ceiling and ramp target at runtime.

        weight  - max blend fraction (0.0 = never use head, 1.0 = full projection)
        ramp_at - triplet count at which weight is fully reached
        """
        self._head_weight = float(weight)
        self._head_ramp_at = int(ramp_at)

    def _apply_head(self, emb: torch.Tensor) -> torch.Tensor:
        """Blend raw embedding with projected embedding according to current weight.

        Alpha ramps linearly from 0 to _head_weight as _triplet_count reaches
        _head_ramp_at, then stays at _head_weight.  This gives the head more
        influence automatically as the user accumulates feedback data.

        Returns emb unchanged when learning is off or no weights are loaded.
        """
        if not self._head_trained or not self._enable_learning:
            return emb
        alpha = min(
            self._head_weight,
            self._head_weight * self._triplet_count / max(self._head_ramp_at, 1),
        )
        if alpha <= 0.0:
            return emb
        with torch.no_grad():
            projected = self._head.project(emb)
        if alpha >= 1.0:
            return projected
        blended = (1.0 - alpha) * emb + alpha * projected
        return F.normalize(blended, dim=1)

    def run(self, query_image, scan_id: str | None = None, focal_length_px: float | None = None):
        """
        query_image: PIL Image
        scan_id: optional UUID string; if provided, caches embeddings for /feedback
        focal_length_px: overrides self.focal_length_px for this call only
        returns structured JSON dict
        """
        t0 = time.monotonic()
        stage_start = t0
        timing = {
            "prepare_ms": 0,
            "detect_ms": 0,
            "embed_ms": 0,
            "ocr_ms": 0,
            "match_ms": 0,
            "dedup_ms": 0,
            "depth_ms": 0,
            "ocr_batch_requests": 0,
            "ocr_batch_items": 0,
            "ocr_batch_non_empty": 0,
        }
        registry.prepare_for_scan()
        timing["prepare_ms"] = round((time.monotonic() - stage_start) * 1000)
        stage_start = time.monotonic()
        _focal = focal_length_px if focal_length_px is not None else self.focal_length_px

        lum = mean_luminance(query_image)
        if lum < _settings.darkness_threshold:
            _log.info({
                "event": "scan_complete",
                "tag": LogTag.PERF,
                "match_count": 0,
                "boxes_detected": 0,
                "is_dark": True,
                "duration_ms": round((time.monotonic() - t0) * 1000),
                **timing,
                **collect_system_metrics(),
            })
            return {
                "matches": [],
                "count": 0,
                "is_dark": True,
                "darkness_level": round(lum, 2),
                "message": "Image is too dark for detection. Enable the flashlight and retry.",
            }

        boxes, scores = self.detector.detect_all(query_image)
        timing["detect_ms"] = round((time.monotonic() - stage_start) * 1000)

        if not boxes:
            _log.info({
                "event": "scan_complete",
                "tag": LogTag.PERF,
                "match_count": 0,
                "boxes_detected": 0,
                "duration_ms": round((time.monotonic() - t0) * 1000),
                **timing,
                **collect_system_metrics(),
            })
            return {"matches": [], "count": 0}

        # PASS 1: combined similarity matching
        # batch: crop all first, embed in one forward pass (was: embed() inside per-box loop)
        stage_start = time.monotonic()
        crops = [crop_object(query_image, box) for box in boxes]
        img_embs = self.img_embedder.batch_embed(crops)  # (N, 1024); one forward pass

        # project DB embeddings once for all boxes this scan call
        projected_db = [(p, self._apply_head(e)) for p, e in self.database_embeddings]
        projected_db_map = {p: e for p, e in projected_db}
        timing["embed_ms"] = round((time.monotonic() - stage_start) * 1000)

        matches = []
        ocr_ms = 0.0
        match_ms = 0.0
        text_likelihoods = [estimate_text_likelihood(crop) for crop in crops]
        ocr_text_by_index: dict[int, str] = {}
        text_emb_by_index: dict[int, torch.Tensor] = {}

        if self.ocr_client is not None:
            ocr_indices = [
                i for i, likelihood in enumerate(text_likelihoods)
                if likelihood >= _settings.ocr_text_likelihood_threshold
            ]
            if ocr_indices:
                ocr_t0 = time.monotonic()
                ocr_results = self.ocr_client.recognize_batch([crops[i] for i in ocr_indices])
                ocr_ms += (time.monotonic() - ocr_t0) * 1000

                for idx, ocr_result in zip(ocr_indices, ocr_results):
                    text = str((ocr_result or {}).get("text", "") or "")
                    if text:
                        ocr_text_by_index[idx] = text

                if self.text_embedder is not None and ocr_text_by_index:
                    ordered = sorted(ocr_text_by_index.items(), key=lambda kv: kv[0])
                    text_batch = self.text_embedder.batch_embed_text([text for _, text in ordered])
                    for row, (idx, _) in enumerate(ordered):
                        text_emb_by_index[idx] = text_batch[row:row + 1]

        for i, (box, score) in enumerate(zip(boxes, scores)):
            img_emb = img_embs[i:i+1]  # (1, 1024)
            text_likelihood = text_likelihoods[i]
            ocr_text = ocr_text_by_index.get(i, "")
            text_emb = text_emb_by_index.get(i)
            combined = make_combined_embedding(img_emb, text_emb)

            projected_query = self._apply_head(combined)

            match_t0 = time.monotonic()
            match_path, similarity = find_match(
                projected_query,
                projected_db,
                _settings.similarity_threshold,
            )
            match_ms += (time.monotonic() - match_t0) * 1000

            if match_path:
                matched_label = Path(match_path).stem
                _log.info({
                    "event": "scan_text_match",
                    "label": matched_label,
                    "similarity": round(similarity, 4),
                    "text_likelihood": round(text_likelihood, 3),
                    "ocr_text_length": len(ocr_text),
                })
                if scan_id is not None:
                    anchor_emb = projected_db_map.get(match_path)
                    if anchor_emb is not None:
                        if scan_id not in self._emb_cache:
                            if len(self._emb_cache) >= _SCAN_CACHE_MAX:
                                self._emb_cache.popitem(last=False)
                            self._emb_cache[scan_id] = {}
                        self._emb_cache[scan_id][matched_label] = (anchor_emb, projected_query)
                direction_auto = _direction_from_box(box, query_image.width)
                self.db.add_sighting(
                    label=matched_label,
                    direction=direction_auto,
                    similarity=similarity,
                )
                entry = {
                    "box": box,
                    "label": matched_label,
                    "similarity": similarity,
                    "text_likelihood": round(text_likelihood, 3),
                    "_crop": crops[i],
                }
                if ocr_text:
                    entry["ocr_text"] = ocr_text
                matches.append(entry)
        timing["ocr_ms"] = round(ocr_ms)
        timing["match_ms"] = round(match_ms)

        if not matches:
            return {"matches": [], "count": 0}

        # Deduplicate
        dedup_t0 = time.monotonic()
        if _settings.enable_dedup:
            matches = deduplicate_matches(matches, iou_threshold=_settings.dedup_iou_threshold)
        timing["dedup_ms"] = round((time.monotonic() - dedup_t0) * 1000)

        if not matches:
            return {"matches": [], "count": 0}

        # Sort left to right by box center x (explicit spatial order for narration)
        matches.sort(key=lambda m: (m["box"][0] + m["box"][2]) / 2)

        # PASS 2: Depth (only once, only if enabled)
        if not _settings.enable_depth:
            output_matches = []
            output_crops = []
            for m in matches:
                direction = _direction_from_box(m["box"], query_image.width)
                sim = float(m["similarity"])
                if sim >= _CONFIDENCE_HIGH:
                    narration = f"{m['label'].capitalize()} {direction}."
                else:
                    narration = f"May be a {m['label']} {direction}, focus to verify."
                out = {
                    "label": m["label"],
                    "similarity": sim,
                    "confidence": _confidence_label(sim),
                    "direction": direction,
                    "narration": narration,
                    "text_likelihood": m["text_likelihood"],
                    "box": m["box"],
                }
                if "ocr_text" in m:
                    out["ocr_text"] = m["ocr_text"]
                output_matches.append(out)
                output_crops.append(m["_crop"])
            if scan_id is not None:
                self._store_crops(scan_id, output_crops)
            _log.info({
                "event": "scan_complete",
                "tag": LogTag.PERF,
                "match_count": len(output_matches),
                "boxes_detected": len(boxes),
                "depth": False,
                "duration_ms": round((time.monotonic() - t0) * 1000),
                **timing,
                **collect_system_metrics(),
            })
            return {"matches": output_matches, "count": len(output_matches)}

        depth_t0 = time.monotonic()
        depth_map = self.estimator.estimate(query_image, focal_length_px=_focal)

        output_matches = []
        output_crops = []

        for m in matches:
            distance_ft = self.estimator.get_depth_at_bbox(depth_map, m["box"])
            direction   = self.estimator.get_direction(m["box"], query_image.width)

            narration = self.estimator.build_narration(
                m["label"],
                direction,
                distance_ft,
                m["similarity"]
            )

            if narration:
                out = {
                    "label": m["label"],
                    "similarity": float(m["similarity"]),
                    "confidence": _confidence_label(float(m["similarity"])),
                    "distance_ft": float(distance_ft),
                    "direction": direction,
                    "narration": narration,
                    "text_likelihood": m["text_likelihood"],
                    "box": m["box"],
                }
                if "ocr_text" in m:
                    out["ocr_text"] = m["ocr_text"]
                output_matches.append(out)
                output_crops.append(m["_crop"])

        if scan_id is not None:
            self._store_crops(scan_id, output_crops)
        timing["depth_ms"] = round((time.monotonic() - depth_t0) * 1000)

        _log.info({
            "event": "scan_complete",
            "tag": LogTag.PERF,
            "match_count": len(output_matches),
            "boxes_detected": len(boxes),
            "depth": True,
            "duration_ms": round((time.monotonic() - t0) * 1000),
            **timing,
            **collect_system_metrics(),
        })
        return {
            "matches": output_matches,
            "count": len(output_matches)
        }
