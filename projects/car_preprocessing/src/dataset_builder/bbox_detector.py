from ultralytics import YOLO
import numpy as np

class BBoxDetector:
    def __init__(self, model_path="yolov8n.pt"):
        print("\n[BBoxDetector] Initialisation du modèle YOLO...")
        self.model = YOLO(model_path)
        print("[BBoxDetector] YOLO chargé avec succès.")

    def detect_car(self, image_rgb):
        """
        Détecte la voiture / camion / bus le plus grand dans l'image.
        Sortie : bbox [x1, y1, x2, y2] ou None
        """
        print("[BBoxDetector] Détection des véhicules...")
        h, w = image_rgb.shape[:2]

        results = self.model(image_rgb, verbose=False)
        car_boxes = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id in [2, 5, 7]:   # car, bus, truck
                    xyxy = box.xyxy[0].cpu().numpy()
                    car_boxes.append(xyxy)

        if not car_boxes:
            print("[BBoxDetector] Aucune voiture détectée.")
            return None

        best_box = max(car_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        print(f"[BBoxDetector] BBOX détectée (brute): {best_box}")

        # Clamp dans les limites de l'image
        x1, y1, x2, y2 = best_box
        x1, x2 = max(0, x1), min(w - 1, x2)
        y1, y2 = max(0, y1), min(h - 1, y2)

        if x2 <= x1 or y2 <= y1:
            print("[BBoxDetector] BBOX invalide après clamp.")
            return None

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        print(f"[BBoxDetector] BBOX corrigée : {bbox}")
        return bbox

    def detect_license_plate(self, image_rgb):
        return None
