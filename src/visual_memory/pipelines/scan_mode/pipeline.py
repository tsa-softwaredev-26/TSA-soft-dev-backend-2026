from pathlib import Path
from visual_memory.config import Settings
from visual_memory.engine.embedding import make_combined_embedding
from visual_memory.engine.model_registry import registry
from visual_memory.learning import ProjectionHead
from visual_memory.utils import crop_object, find_match, load_folder_images, deduplicate_matches, get_logger

_settings = Settings()
_log = get_logger(__name__)


class ScanPipeline:
    def __init__(self, database_dir: Path, focal_length_px: float):
        self.img_embedder    = registry.img_embedder
        self.text_embedder   = registry.text_embedder   if _settings.enable_ocr   else None
        self.detector        = registry.yoloe_detector
        self.text_recognizer = registry.text_recognizer if _settings.enable_ocr   else None
        self.estimator       = registry.depth_estimator if _settings.enable_depth else None
        self.focal_length_px = focal_length_px

        self._head = ProjectionHead(dim=_settings.projection_head_dim)
        _head_path = Path(_settings.projection_head_path)
        self._head_trained = self._head.load(_head_path)
        self._head.eval()

        self.database_images = load_folder_images(database_dir)
        self.database_embeddings = self._embed_database()

    def _embed_database(self):
        """Embed each database image as a combined (image+text) embedding."""
        if not self.database_images:
            return []

        # batch: one model forward pass for all DB images (was: embed() per image in a loop)
        paths, imgs = zip(*self.database_images)
        img_embs = self.img_embedder.batch_embed(list(imgs))  # (N, 1024) — one forward pass

        embeddings = []
        for i, (file_path, img) in enumerate(self.database_images):
            img_emb = img_embs[i:i+1]  # (1, 1024) — same shape as embed() output
            text_emb = None
            if self.text_recognizer is not None:
                ocr_result = self.text_recognizer.recognize(img)
                text_emb = self.text_embedder.embed_text(ocr_result["text"]) if ocr_result["text"] else None
            combined = make_combined_embedding(img_emb, text_emb)
            embeddings.append((file_path, combined))
        return embeddings

    def run(self, query_image):
        """
        query_image: PIL Image
        returns structured JSON dict
        """

        boxes, scores = self.detector.detect_all(query_image)

        if not boxes:
            return {"matches": [], "count": 0}

        # ---- PASS 1: combined similarity matching ----
        # batch: crop all first, embed in one forward pass (was: embed() inside per-box loop)
        crops = [crop_object(query_image, box) for box in boxes]
        img_embs = self.img_embedder.batch_embed(crops)  # (N, 1024) — one forward pass

        matches = []

        for i, (box, score) in enumerate(zip(boxes, scores)):
            img_emb = img_embs[i:i+1]  # (1, 1024)
            ocr_text = ""
            text_emb = None
            if self.text_recognizer is not None:
                ocr_result = self.text_recognizer.recognize(crops[i])
                ocr_text = ocr_result["text"]
                text_emb = self.text_embedder.embed_text(ocr_text) if ocr_text else None
            combined = make_combined_embedding(img_emb, text_emb)

            if self._head_trained:
                projected_query = self._head.project(combined)
                projected_db = [(p, self._head.project(e)) for p, e in self.database_embeddings]
            else:
                projected_query = combined
                projected_db = self.database_embeddings

            match_path, similarity = find_match(
                projected_query,
                projected_db,
                _settings.similarity_threshold,
            )

            if match_path:
                _log.info({
                    "event": "scan_text_match",
                    "label": Path(match_path).stem,
                    "similarity": round(similarity, 4),
                    "ocr_text_length": len(ocr_text),
                })
                entry = {
                    "box": box,
                    "label": Path(match_path).stem,
                    "similarity": similarity,
                }
                if ocr_text:
                    entry["ocr_text"] = ocr_text
                matches.append(entry)

        if not matches:
            return {"matches": [], "count": 0}

        # Deduplicate
        if _settings.enable_dedup:
            matches = deduplicate_matches(matches, iou_threshold=_settings.dedup_iou_threshold)

        if not matches:
            return {"matches": [], "count": 0}

        # ---- PASS 2: Depth (only once, only if enabled) ----
        if not _settings.enable_depth:
            output_matches = []
            for m in matches:
                out = {
                    "label": m["label"],
                    "similarity": float(m["similarity"]),
                }
                if "ocr_text" in m:
                    out["ocr_text"] = m["ocr_text"]
                output_matches.append(out)
            return {"matches": output_matches, "count": len(output_matches)}

        depth_map = self.estimator.estimate(query_image, focal_length_px=self.focal_length_px)

        output_matches = []

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
                    "distance_ft": float(distance_ft),
                    "direction": direction,
                    "narration": narration,
                }
                if "ocr_text" in m:
                    out["ocr_text"] = m["ocr_text"]
                output_matches.append(out)

        return {
            "matches": output_matches,
            "count": len(output_matches)
        }
