"""
Script pour vérifier visuellement le replacement de plaques.
"""
from pathlib import Path
from PIL import Image

processed_dir = Path("data/processed")
images = list(processed_dir.glob("*.jpg"))

print(f"\n=== VÉRIFICATION DES IMAGES TRAITÉES ===\n")
print(f"Nombre d'images: {len(images)}\n")

for img_path in images:
    print(f"Image: {img_path.name}")
    img = Image.open(img_path)
    print(f"Taille: {img.size}")
    print(f"Mode: {img.mode}")
    print()

print("Les images ont été traitées. Vérifiez visuellement si les plaques ont été remplacées par PLANY.TN")
print("dans le dossier: data/processed/")
