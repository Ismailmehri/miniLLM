import os
import cv2
import glob
from tqdm import tqdm
from .mask_generator import MaskGenerator
from .bbox_detector import BBoxDetector
from .label_manager import LabelManager
from .utils_image import apply_white_background, center_and_resize_car
from .studio_processor import apply_professional_studio_look
from .license_plate_replacer import LicensePlateReplacer
from PIL import Image
import numpy as np

class DatasetBuilder:
    def __init__(self, raw_data_dir, output_dir):
        print("\n[DatasetBuilder] Initialisation du pipeline...")

        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir

        self.processed_dir = os.path.join(output_dir, "processed")
        self.annotations_dir = os.path.join(output_dir, "annotations")

        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)

        print("[DatasetBuilder] Chargement des modules...")
        self.mask_generator = MaskGenerator()
        self.bbox_detector = BBoxDetector()
        self.label_manager = LabelManager(self.annotations_dir)

        print("[DatasetBuilder] Prêt.")

    def run(self):
        print("\n[DatasetBuilder] Lecture des images...")
        image_paths = glob.glob(os.path.join(self.raw_data_dir, "*"))
        image_paths = [p for p in image_paths if p.lower().endswith((".jpg", ".jpeg", ".png"))]

        print(f"[DatasetBuilder] {len(image_paths)} images détectées.")

        for image_path in tqdm(image_paths):
            self.process_image(image_path)

    def process_image(self, image_path):
        filename = os.path.basename(image_path)
        print(f"\n[Process] Traitement de {filename}")

        image = cv2.imread(image_path)
        if image is None:
            print("[Process] ERREUR : impossible de lire l'image.")
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 1. DETECTION BBOX
        print("[Process] Étape 1 : Détection de la voiture...")
        bbox = self.bbox_detector.detect_car(image_rgb)
        if bbox is None:
            print("[Process] Aucun véhicule détecté. Ignoré.")
            return

        # 2. GENERATION MASQUE
        print("[Process] Étape 2 : Génération du masque...")
        mask = self.mask_generator.generate_mask(image_rgb, bbox)

        # 3. CRÉATION IMAGE RGBA POUR STUDIO PROCESSOR
        print("[Process] Étape 3 : Création de l'image RGBA...")
        # Convertir en PIL pour manipulation RGBA
        img_pil = Image.fromarray(image_rgb)
        mask_pil = Image.fromarray(mask)
        
        # Créer l'image RGBA (RGB + Alpha)
        rgba_image = img_pil.convert('RGBA')
        rgba_image.putalpha(mask_pil)
        
        # 4. APPLICATION DU RENDU STUDIO PROFESSIONNEL
        print("[Process] Étape 4 : Application du rendu studio professionnel...")
        processed_pil = apply_professional_studio_look(
            rgba_image,
            output_size=(600, 575),
            padding=20,
            shadow_opacity=0.35,
            shadow_blur_radius=30,
            shadow_offset_y=15,
            shadow_flatten_factor=0.5,
            sharpness_factor=1.3,
            contrast_factor=1.15,
            saturation_factor=1.2,
            replace_plate=True
        )
        
        # Convertir PIL → numpy pour sauvegarde avec cv2
        processed = np.array(processed_pil)

        # 5. SAUVEGARDE IMAGES
        save_path = os.path.join(self.processed_dir, filename)
        cv2.imwrite(save_path, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
        print(f"[Process] Image modifiée sauvegardée → {save_path}")

        # 6. SAUVEGARDE MASQUE
        mask_filename = f"{os.path.splitext(filename)[0]}_mask.png"
        cv2.imwrite(os.path.join(self.annotations_dir, mask_filename), mask)
        print(f"[Process] Masque sauvegardé → {mask_filename}")

        # 7. SAUVEGARDE ANNOTATION JSON
        self.label_manager.save_annotation(
            image_name=filename,
            bbox=bbox,
            mask_path=mask_filename,
            labels={"processed": True}
        )

        print("[Process] Image traitée avec succès.")
