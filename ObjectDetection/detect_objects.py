from ultralytics import YOLOE
from PIL import Image


class YoloeDetector:
    """
    YOLOE object detector wrapper class.
    
    Handles model loading and object detection using YOLOE.
    """
    
    def __init__(self, model_path="yoloe-26l-seg.pt", classes=None):
        """
        Initialize YOLOE detector.
        
        Args:
            model_path: Path to YOLOE model file
            classes: Optional list of classes to filter detections
        """
        self.model = YOLOE(model_path)
        if classes:
            self.model.set_classes(classes, self.model.get_text_pe(classes))
    
    def detect(self, image_path):
        """
        Detect objects in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (boxes, scores, class_ids):
            - boxes: List of [x1, y1, x2, y2] bounding boxes
            - scores: List of confidence scores
            - class_ids: List of class IDs
        """
        results = self.model.predict(image_path)
        boxes = results[0].boxes.xyxy.tolist()  # List of [x1, y1, x2, y2]
        scores = results[0].boxes.conf.tolist()  # List of confidence scores
        class_ids = results[0].boxes.cls.tolist()  # List of class IDs 
        return boxes, scores, class_ids


def crop_object(image, x1, y1, x2, y2):
    """
    Crop an object from an image using bounding box coordinates.
    
    Args:
        image: PIL Image object
        x1: Left coordinate
        y1: Top coordinate
        x2: Right coordinate
        y2: Bottom coordinate
        
    Returns:
        Cropped PIL Image
    """
    cropped_image = image.crop((x1, y1, x2, y2))
    return cropped_image