from ultralytics import YOLOE
from PIL import Image

class YoloeDetector:
    """
    YOLOE detector with fixed semantics:
    - prompt-free model: ALL detections with set confidence & iou (intersection) thresholds, not very accurate class labels; 
        breadth>accuracy of detections for similarity algorithm to handle in depth since its quiet fast
    
    - prompt-based model: (tried to do with "household object" prompt but unused right now) BEST detection only with prompt
        grounding dino works better for prompt-based, able to handle more abstract concepts
    """

    def __init__(
        self,
        # prompt_model_path="yoloe-26l-seg.pt",
        prompt_free_model_path="ObjectDetection/yoloe-26l-seg-pf.pt",
        # classes=None,
        confidence_threshold=.35, 
        intersection_threshold=.45
    ):

        # Prompt-based (text-conditioned, BEST only)
        # self.prompt_model = YOLOE(prompt_model_path)
        # if classes is not None:
        #     self.prompt_model.set_classes(
        #         classes, self.prompt_model.get_text_pe(classes)
        #     )

        # Prompt-free (objectness, ALL detections)
        self.prompt_free_model = YOLOE(prompt_free_model_path)
        self.confidence_threshold = confidence_threshold
        self.intersection_threshold = intersection_threshold

    def detect_all(self, image_path):
        """
        Prompt-free detection: return ALL boxes.

        Returns:
            tuple: (boxes, scores, class_ids) or (None, None, None) if no detection
        """

        results = self.prompt_free_model.predict(image_path, verbose=True, conf=self.confidence_threshold, iou=self.intersection_threshold)

        if len(results[0].boxes) == 0:
            return None, None

        boxes = results[0].boxes.xyxy.tolist()
        scores = results[0].boxes.conf.tolist()


        return boxes, scores



    # def detect_best(self, image_path):
    #     """
    #     Prompt-based detection: return BEST box only.

    #     Returns:
    #         tuple: (box, score, class_id) or (None, None, None) if no detection
    #     """
    #     results = self.prompt_model.predict(image_path, verbose=False)

    #     if len(results[0].boxes) == 0:
    #         return None, None, None


    #     box = results[0].boxes.xyxy[0].tolist()
    #     score = results[0].boxes.conf[0].item()
    #     class_id = results[0].boxes.cls[0].item()

    #     return box, score, class_id

