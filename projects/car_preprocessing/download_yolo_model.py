"""
Script pour télécharger un modèle YOLO pré-entraîné pour la détection de plaques.
"""
import os
from pathlib import Path
import urllib.request

# Créer le dossier models s'il n'existe pas
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

print("=== TÉLÉCHARGEMENT DE MODÈLE YOLO POUR PLAQUES ===\n")

# Option 1: Modèle léger et rapide (recommandé pour commencer)
model_url = "https://github.com/niconielsen32/LicensePlateDetector/raw/main/license_plate_detector.pt"
model_path = models_dir / "license_plate_detector.pt"

print(f"Téléchargement du modèle depuis:")
print(f"{model_url}\n")
print(f"Destination: {model_path}\n")

try:
    print("Téléchargement en cours...")
    urllib.request.urlretrieve(model_url, model_path)
    print(f"\n✅ Modèle téléchargé avec succès!")
    print(f"📁 Emplacement: {model_path.absolute()}")
    print(f"📊 Taille: {model_path.stat().st_size / (1024*1024):.2f} MB")
    
    print("\n" + "="*60)
    print("UTILISATION:")
    print("="*60)
    print("\nLe modèle sera automatiquement utilisé si vous le placez dans:")
    print(f"  {model_path.absolute()}")
    print("\nOu vous pouvez le spécifier manuellement dans le code:")
    print("  replacer = LicensePlateReplacer(model_path='models/license_plate_detector.pt')")
    
except Exception as e:
    print(f"\n❌ Erreur lors du téléchargement: {e}")
    print("\nSi le téléchargement automatique échoue, vous pouvez:")
    print("1. Télécharger manuellement depuis:")
    print("   https://github.com/niconielsen32/LicensePlateDetector")
    print("2. Placer le fichier .pt dans le dossier 'models/'")
    print("3. Le renommer en 'license_plate_detector.pt'")
