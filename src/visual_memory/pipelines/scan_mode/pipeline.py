from pathlib import Path
from visual_memory.config import Settings
from visual_memory.engine.embedding import CLIPEmbedder, make_combined_embedding
from visual_memory.engine.object_detection import YoloeDetector
from visual_memory.engine.depth import DepthEstimator
from visual_memory.engine.text_recognition import TextRecognizer
from visual_memory.utils import crop_object, find_match, load_folder_images, deduplicate_matches, get_logger

_settings = Settings()
_log = get_logger(__name__)


class ScanPipeline:
    def __init__(self, database_dir: Path, focal_length_px: float):
        self.embedder = CLIPEmbedder()
        self.text_recognizer = TextRecognizer()
        self.detector = YoloeDetector()
        self.estimator = DepthEstimator(focal_length_px=focal_length_px)

        self.database_images = load_folder_images(database_dir)
        self.database_embeddings = self._embed_database()

    def _embed_database(self):
        """Embed each database image as a combined (image+text) embedding."""
        embeddings = []
        for file_path, img in self.database_images:
            img_emb = self.embedder.embed(img)
            ocr_result = self.text_recognizer.recognize(img)
            text_emb = self.embedder.embed_text(ocr_result["text"]) if ocr_result["text"] else None
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
        matches = []

        for box, score in zip(boxes, scores):
            cropped = crop_object(query_image, box)

            img_emb = self.embedder.embed(cropped)
            ocr_result = self.text_recognizer.recognize(cropped)
            text_emb = self.embedder.embed_text(ocr_result["text"]) if ocr_result["text"] else None
            combined = make_combined_embedding(img_emb, text_emb)

            match_path, similarity = find_match(
                combined,
                self.database_embeddings,
                _settings.similarity_threshold,
            )

            if match_path:
                _log.info({
                    "event": "scan_text_match",
                    "label": Path(match_path).stem,
                    "similarity": round(similarity, 4),
                    "ocr_text_length": len(ocr_result["text"]),
                })
                entry = {
                    "box": box,
                    "label": Path(match_path).stem,
                    "similarity": similarity,
                }
                if ocr_result["text"]:
                    entry["ocr_text"] = ocr_result["text"]
                matches.append(entry)

        if not matches:
            return {"matches": [], "count": 0}

        # Deduplicate
        matches = deduplicate_matches(matches, iou_threshold=_settings.dedup_iou_threshold)

        if not matches:
            return {"matches": [], "count": 0}

        # ---- PASS 2: Depth (only once) ----
        depth_map = self.estimator.estimate(query_image)

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
