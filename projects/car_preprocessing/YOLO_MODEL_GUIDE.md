# Guide : Détection de Plaques d'Immatriculation avec YOLO

## 📋 Résumé

Le système de remplacement de plaques fonctionne avec **deux méthodes** :

1. **OpenCV (Fallback)** : Détection basique, toujours disponible ✅
2. **YOLO (Optionnel)** : Détection précise avec deep learning 🎯

## 🚀 Utilisation Actuelle

**Actuellement, le système utilise OpenCV** et fonctionne correctement. Vous n'avez **rien à faire** si vous êtes satisfait des résultats.

## 🎯 Améliorer la Précision avec YOLO (Optionnel)

Si vous voulez une détection plus précise des plaques, vous pouvez ajouter un modèle YOLO.

### Option 1 : Téléchargement Automatique (Recommandé)

```bash
python download_yolo_model.py
```

Ce script va :
- Télécharger un modèle YOLOv8 pré-entraîné pour les plaques
- Le placer dans `models/license_plate_detector.pt`
- Le système l'utilisera automatiquement

### Option 2 : Téléchargement Manuel

1. **Téléchargez un modèle** depuis l'une de ces sources :
   - [niconielsen32/LicensePlateDetector](https://github.com/niconielsen32/LicensePlateDetector) (Recommandé)
   - [Ultralytics License Plate Models](https://github.com/ultralytics/ultralytics)

2. **Placez le fichier** dans le dossier `models/` :
   ```
   projects/car_preprocessing/
   └── models/
       └── license_plate_detector.pt  ← ICI
   ```

3. **C'est tout !** Le système détectera automatiquement le modèle.

### Option 3 : Entraîner Votre Propre Modèle

Si vous avez un dataset de plaques tunisiennes/africaines :

```python
from ultralytics import YOLO

# Charger un modèle de base
model = YOLO('yolov8n.pt')

# Entraîner sur votre dataset
model.train(
    data='license_plates.yaml',  # Votre configuration
    epochs=100,
    imgsz=640
)

# Sauvegarder
model.save('models/license_plate_detector.pt')
```

## 📁 Structure des Fichiers

```
projects/car_preprocessing/
├── models/                          # Dossier pour les modèles
│   └── license_plate_detector.pt   # Modèle YOLO (optionnel)
├── src/
│   └── dataset_builder/
│       ├── license_plate_replacer.py  # Module de remplacement
│       └── studio_processor.py        # Utilise le replacer
└── download_yolo_model.py          # Script de téléchargement
```

## 🔍 Comment Vérifier Quel Modèle est Utilisé

Regardez les logs lors de l'exécution :

```
[PlateReplacer] Modèle YOLO chargé: models/license_plate_detector.pt  ← YOLO
```

ou

```
[PlateReplacer] Pas de modèle YOLO fourni. Utilisation du fallback OpenCV.  ← OpenCV
```

## ⚙️ Configuration Avancée

Si vous voulez spécifier manuellement le chemin du modèle :

```python
from dataset_builder.license_plate_replacer import LicensePlateReplacer

# Avec YOLO
replacer = LicensePlateReplacer(model_path="models/license_plate_detector.pt")

# Sans YOLO (OpenCV uniquement)
replacer = LicensePlateReplacer(model_path=None)
```

## 📊 Comparaison

| Méthode | Précision | Vitesse | Installation |
|---------|-----------|---------|--------------|
| **OpenCV** | Bonne | Rapide | ✅ Aucune |
| **YOLO** | Excellente | Moyenne | Télécharger modèle |

## ❓ FAQ

**Q: Dois-je absolument télécharger le modèle YOLO ?**  
R: Non ! Le système fonctionne très bien avec OpenCV seul.

**Q: Quelle est la différence de précision ?**  
R: YOLO détecte mieux les plaques dans des angles difficiles ou avec peu de contraste.

**Q: Quel est le poids du modèle YOLO ?**  
R: Environ 6-25 MB selon le modèle (YOLOv8n = ~6MB, YOLOv8s = ~25MB).

**Q: Le modèle fonctionne-t-il hors ligne ?**  
R: Oui, une fois téléchargé, tout fonctionne en local.

## 🎓 Ressources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [License Plate Detection Tutorial](https://github.com/niconielsen32/LicensePlateDetector)
- [Ultralytics Models](https://github.com/ultralytics/ultralytics)
