from pathlib import Path
from visual_memory.config import Settings
from visual_memory.engine.embedding import ImageEmbedder, TextEmbedder
from visual_memory.engine.object_detection import YoloeDetector
from visual_memory.engine.depth import DepthEstimator
<<<<<<< Updated upstream
from visual_memory.utils import crop_object, find_match, load_folder_images, deduplicate_matches
=======
from visual_memory.engine.text_recognition import TextRecognizer
from visual_memory.utils import crop_object, find_match, load_folder_images, deduplicate_matches, get_logger, cosine_similarity
>>>>>>> Stashed changes

_settings = Settings()


class ScanPipeline:
    def __init__(self, database_dir: Path, focal_length_px: float):
        self.embedder = ImageEmbedder()
        self.text_recognizer = TextRecognizer()
        self.text_embedder = TextEmbedder()
        self.detector = YoloeDetector()
        self.estimator = DepthEstimator(focal_length_px=focal_length_px)

        self.database_images = load_folder_images(database_dir)
        self.database_embeddings = self._embed_database()

    def _embed_database(self):
        """Embed each database image (visual + optional text) and return list of tuples."""
        embeddings = []
        for file_path, img in self.database_images:
            image_embedding = self.embedder.embed(img)
            ocr_result = self.text_recognizer.recognize(img)
            text_embedding = self.text_embedder.embed(ocr_result["text"]) if ocr_result["text"] else None
            embeddings.append((file_path, image_embedding, text_embedding))
        return embeddings

    def run(self, query_image):
        """
        query_image: PIL Image
        returns structured JSON dict
        """

        boxes, scores = self.detector.detect_all(query_image)

        if not boxes:
            return {"matches": [], "count": 0}

        # ---- PASS 1: similarity matching ----
        # Build image-only database embeddings list for find_match (expects (path, embedding) pairs)
        image_db = [(p, img_emb) for p, img_emb, _ in self.database_embeddings]

        matches = []

        for box, score in zip(boxes, scores):
            cropped = crop_object(query_image, box)
            image_embedding = self.embedder.embed(cropped)

            match_path, image_similarity = find_match(
                image_embedding,
                image_db,
                _settings.similarity_threshold
            )

            # Text-based matching: OCR the crop and compare against database text embeddings
            ocr_result = self.text_recognizer.recognize(cropped)
            crop_text_embedding = self.text_embedder.embed(ocr_result["text"]) if ocr_result["text"] else None

            best_text_path = None
            best_text_similarity = 0.0
            if crop_text_embedding is not None:
                for db_path, _, db_text_emb in self.database_embeddings:
                    if db_text_emb is None:
                        continue
                    sim = float(cosine_similarity(crop_text_embedding, db_text_emb))
                    if sim > best_text_similarity:
                        best_text_similarity = sim
                        best_text_path = db_path

            # Promote text match if it clears the threshold and beats image match
            text_match_path = best_text_path if best_text_similarity >= _settings.text_similarity_threshold else None

            if text_match_path and best_text_similarity > image_similarity:
                final_path = text_match_path
                final_similarity = best_text_similarity
                match_via = "text"
            elif match_path:
                final_path = match_path
                final_similarity = image_similarity
                match_via = "image"
            else:
                final_path = None
                final_similarity = 0.0
                match_via = None

            if final_path:
                _log.info({
                    "event": "scan_text_match",
                    "label": Path(final_path).stem,
                    "match_via": match_via,
                    "image_similarity": round(image_similarity, 4),
                    "text_similarity": round(best_text_similarity, 4),
                    "ocr_text_length": len(ocr_result["text"]),
                })
                entry = {
                    "box": box,
                    "label": Path(final_path).stem,
                    "similarity": final_similarity,
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