from ultralytics import YOLO
import cv2
import numpy as np

class BBoxDetector:
    def __init__(self, model_path="yolov8n.pt"):
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)

    def detect_car(self, image_rgb):
        """
        Detects the largest car in the image.
        Returns: bbox [x1, y1, x2, y2] or None
        """
        results = self.model(image_rgb, verbose=False)
        
        # Filter for 'car' class (COCO class id 2) or 'truck' (7) or 'bus' (5)
        # Usually car is 2.
        car_boxes = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in [2, 5, 7]: # car, bus, truck
                    xyxy = box.xyxy[0].cpu().numpy()
                    car_boxes.append(xyxy)
        
        if not car_boxes:
            return None
        
        # Return the largest box by area
        best_box = max(car_boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        return best_box

    def detect_license_plate(self, image_rgb):
        """
        Placeholder for license plate detection. 
        In a real scenario, you'd use a fine-tuned model for plates.
        Here we might return None or a dummy box if we don't have a specific model.
        """
        # For this task, we assume we might want to detect plates to blur them or replace them.
        # If we don't have a plate model, we can't reliably return a box.
        return None
