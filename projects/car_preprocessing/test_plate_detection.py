"""
Script de test pour vérifier la détection de plaques.
"""
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append("src")

from dataset_builder.license_plate_replacer import LicensePlateReplacer

# Trouver une image de test
image_path = Path("data/raw")
images = list(image_path.glob("*.jpg")) + list(image_path.glob("*.jpeg")) + list(image_path.glob("*.png"))

if not images:
    print("Aucune image trouvée dans data/raw")
    sys.exit(1)

print(f"\n=== TEST DE DÉTECTION DE PLAQUES ===\n")
print(f"Test sur: {images[0].name}\n")

# Charger l'image
image = cv2.imread(str(images[0]))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Créer le replacer
replacer = LicensePlateReplacer()

# Tester la détection
bbox = replacer.detect_plate_opencv(image_rgb)

if bbox is not None:
    print(f"\n✅ PLAQUE DÉTECTÉE!")
    print(f"Position: {bbox}")
    
    # Dessiner la bbox sur l'image pour visualiser
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.imwrite("test_plate_detection.jpg", image)
    print(f"\nImage avec bbox sauvegardée: test_plate_detection.jpg")
else:
    print("\n❌ AUCUNE PLAQUE DÉTECTÉE")
    print("La détection a échoué.")

print("\n" + "="*50 + "\n")
