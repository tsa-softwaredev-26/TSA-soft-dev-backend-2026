from pathlib import Path
from visual_memory.config import Settings
from visual_memory.engine.embedding import ImageEmbedder
from visual_memory.engine.object_detection import YoloeDetector
from visual_memory.engine.depth import DepthEstimator
from visual_memory.utils import crop_object, find_match, load_folder_images, deduplicate_matches

_settings = Settings()


class ScanPipeline:
    def __init__(self, database_dir: Path, focal_length_px: float):
        self.embedder = ImageEmbedder()
        self.detector = YoloeDetector()
        self.estimator = DepthEstimator(focal_length_px=focal_length_px)

        self.database_images = load_folder_images(database_dir)
        self.database_embeddings = self._embed_database()

    def _embed_database(self):
        embeddings = []
        for file_path, img in self.database_images:
            embedding = self.embedder.embed(img)
            embeddings.append((file_path, embedding))
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
        matches = []

        for box, score in zip(boxes, scores):
            cropped = crop_object(query_image, box)
            embedding = self.embedder.embed(cropped)

            match_path, similarity = find_match(
                embedding,
                self.database_embeddings,
                _settings.similarity_threshold
            )

            if match_path:
                matches.append({
                    "box": box,
                    "label": Path(match_path).stem,
                    "similarity": similarity
                })

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
                output_matches.append({
                    "label": m["label"],
                    "similarity": float(m["similarity"]),
                    "distance_ft": float(distance_ft),
                    "direction": direction,
                    "narration": narration
                })

        return {
            "matches": output_matches,
            "count": len(output_matches)
        }