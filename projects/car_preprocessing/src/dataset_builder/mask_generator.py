from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import numpy as np

class MaskGenerator:
    def __init__(self):
        print("[MaskGenerator] Chargement SAM2 officiel (CPU only)...")

        self.device = "cpu"

        try:
            self.predictor = SAM2ImagePredictor.from_pretrained(
                "facebook/sam2-hiera-tiny",
                device=self.device,
                dtype=torch.float32
            )

            print("[MaskGenerator] SAM2 chargé avec succès (CPU).")

        except Exception as e:
            print("[MaskGenerator] ERREUR : SAM2 ne peut pas être chargé → fallback rectangle.")
            print(e)
            self.predictor = None

    def generate_mask(self, image_rgb, bbox):
        h, w = image_rgb.shape[:2]

        if bbox is None or self.predictor is None:
            return np.zeros((h, w), dtype=np.uint8)

        try:
            # SAM2 predictor attend une vraie image RGB numpy
            self.predictor.set_image(image_rgb)

            # Format SAM2 pour une bounding box
            prompts = {
                "boxes": [[[
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3])
                ]]]
            }

            with torch.no_grad():
                masks, _, _ = self.predictor.predict(
                    input_prompts=prompts,
                    multimask_output=False
                )

            mask = (masks[0] * 255).astype(np.uint8)
            return mask

        except Exception as e:
            print("[MaskGenerator] ERREUR SAM2 → fallback rectangle.")
            print(e)
            mask = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = map(int, bbox)
            mask[y1:y2, x1:x2] = 255
            return mask
