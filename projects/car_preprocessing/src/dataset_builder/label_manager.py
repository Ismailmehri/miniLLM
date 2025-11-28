import json
import os

class LabelManager:
    def __init__(self, annotations_dir):
        self.annotations_dir = annotations_dir
        os.makedirs(annotations_dir, exist_ok=True)

    def save_annotation(self, image_name, bbox, mask_path, labels=None):
        """
        Saves annotations to a JSON file.
        """
        data = {
            "image_name": image_name,
            "bbox": bbox.tolist() if bbox is not None else None,
            "mask_path": mask_path,
            "labels": labels or {}
        }
        
        json_path = os.path.join(self.annotations_dir, f"{os.path.splitext(image_name)[0]}.json")
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
            
    def load_annotation(self, image_name):
        json_path = os.path.join(self.annotations_dir, f"{os.path.splitext(image_name)[0]}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)
        return None
