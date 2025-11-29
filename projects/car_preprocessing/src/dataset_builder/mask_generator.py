from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import numpy as np
import cv2

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
            # Format SAM2 pour une bounding box
            # SAM expects box as np.array([x1, y1, x2, y2])
            box = np.array(bbox)

            with torch.no_grad():
                masks, _, _ = self.predictor.predict(
                    box=box,
                    multimask_output=False
                )

            mask = (masks[0] * 255).astype(np.uint8)
            return mask

        except Exception as e:
            print(f"[MaskGenerator] ERREUR SAM2: {e}")
            
            # Fallback : Tentative avec rembg
            try:
                print("[MaskGenerator] Tentative de détourage avec rembg...")
                from rembg import remove
                from PIL import Image
                
                # Convertir numpy array en PIL Image
                img_pil = Image.fromarray(image_rgb)
                
                # Détourage
                output = remove(img_pil)
                
                # Récupérer le canal Alpha comme masque
                mask = np.array(output)[:, :, 3]
                
                # Binariser le masque (0 ou 255)
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                
                print("[MaskGenerator] Détourage rembg réussi.")
                return mask
                
            except ImportError:
                print("[MaskGenerator] 'rembg' n'est pas installé. Installez-le avec 'pip install rembg'.")
            except Exception as e_rembg:
                print(f"[MaskGenerator] Erreur rembg: {e_rembg}")

            # Fallback ultime : Rectangle de la BBox
            print("[MaskGenerator] Fallback : Rectangle simple.")
            mask = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = map(int, bbox)
            # Clip coordinates to image dimensions
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            mask[y1:y2, x1:x2] = 255
            return mask
