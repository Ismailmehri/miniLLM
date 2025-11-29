import os
import cv2
import glob
from tqdm import tqdm
from .mask_generator import MaskGenerator
from .bbox_detector import BBoxDetector
from .label_manager import LabelManager
from .utils_image import apply_white_background, center_and_resize_car

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

        # 3. APPLICATION FOND BLANC ET REDIMENSIONNEMENT
        print("[Process] Étape 3 : Application du fond blanc et redimensionnement...")
        # processed = apply_white_background(image_rgb, mask)
        processed = center_and_resize_car(image_rgb, mask, bbox)

        # 4. SAUVEGARDE IMAGES
        save_path = os.path.join(self.processed_dir, filename)
        cv2.imwrite(save_path, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
        print(f"[Process] Image modifiée sauvegardée → {save_path}")

        # 5. SAUVEGARDE MASQUE
        mask_filename = f"{os.path.splitext(filename)[0]}_mask.png"
        cv2.imwrite(os.path.join(self.annotations_dir, mask_filename), mask)
        print(f"[Process] Masque sauvegardé → {mask_filename}")

        # 6. SAUVEGARDE ANNOTATION JSON
        self.label_manager.save_annotation(
            image_name=filename,
            bbox=bbox,
            mask_path=mask_filename,
            labels={"processed": True}
        )

        print("[Process] Image traitée avec succès.")
