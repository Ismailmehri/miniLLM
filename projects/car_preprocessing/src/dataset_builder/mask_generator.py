import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import cv2

class MaskGenerator:
    def __init__(self, model_type="vit_h", checkpoint_path="models/sam_vit_h_4b8939.pth", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading SAM model ({model_type}) on {self.device}...")
        # Note: In a real scenario, ensure the checkpoint exists. 
        # For now, we assume the user will place the checkpoint in the correct path.
        # If not found, this might raise an error.
        try:
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            self.sam.to(device=self.device)
            self.predictor = SamPredictor(self.sam)
        except Exception as e:
            print(f"Warning: Could not load SAM model from {checkpoint_path}. Error: {e}")
            self.predictor = None

    def generate_mask(self, image_rgb, bbox=None):
        """
        Generates a binary mask for the car in the image.
        If bbox is provided [x1, y1, x2, y2], it uses it as a prompt.
        Otherwise, it could use automatic mask generation (not implemented here for simplicity, 
        assuming we use YOLO bbox as prompt).
        """
        if self.predictor is None:
            # Return a dummy mask if model failed to load (for testing without weights)
            return np.ones((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8) * 255

        self.predictor.set_image(image_rgb)

        if bbox is not None:
            box = np.array(bbox)
            masks, _, _ = self.predictor.predict(
                box=box,
                multimask_output=False
            )
            mask = masks[0].astype(np.uint8) * 255
            return mask
        else:
            # Fallback or auto-segmentation logic could go here
            return np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
