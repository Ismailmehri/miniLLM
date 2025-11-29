"""
Module de détection et remplacement automatique de plaques d'immatriculation.
Auteur: Expert Senior Python & Computer Vision
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path


class LicensePlateReplacer:
    """
    Classe pour détecter et remplacer les plaques d'immatriculation
    avec une plaque virtuelle brandée "PLANY.TN".
    """
    
    def __init__(self, model_path=None):
        """
        Initialise le détecteur de plaques.
        
        Args:
            model_path: Chemin vers le modèle YOLO .pt entraîné sur les plaques
                       Si None, utilise la méthode de fallback OpenCV
        """
        self.model = None
        self.use_yolo = False
        
        if model_path and Path(model_path).exists():
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                self.use_yolo = True
                print(f"[PlateReplacer] Modèle YOLO chargé: {model_path}")
            except Exception as e:
                print(f"[PlateReplacer] Erreur chargement YOLO: {e}")
                print("[PlateReplacer] Utilisation du fallback OpenCV.")
        else:
            print("[PlateReplacer] Pas de modèle YOLO fourni. Utilisation du fallback OpenCV.")
    
    def detect_plate_yolo(self, image_rgb):
        """
        Détecte la plaque d'immatriculation avec YOLO.
        
        Args:
            image_rgb: Image numpy array RGB
        
        Returns:
            bbox: [x1, y1, x2, y2] ou None si non détectée
        """
        if not self.use_yolo or self.model is None:
            return None
        
        try:
            results = self.model(image_rgb, verbose=False)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Prendre la détection avec la meilleure confiance
                boxes = results[0].boxes
                best_idx = boxes.conf.argmax()
                bbox = boxes.xyxy[best_idx].cpu().numpy()
                
                print(f"[PlateReplacer] Plaque détectée (YOLO): {bbox}")
                return bbox
        except Exception as e:
            print(f"[PlateReplacer] Erreur détection YOLO: {e}")
        
        return None
    
    def detect_plate_opencv(self, image_rgb):
        """
        Détecte la plaque d'immatriculation avec OpenCV - Version simplifiée et robuste.
        
        Nouvelle stratégie ultra-simple:
        1. Chercher UNIQUEMENT dans le bas de l'image (zone pare-chocs)
        2. Utiliser des méthodes basiques mais efficaces
        3. Retourner TOUJOURS une position par défaut si rien n'est trouvé
        
        Args:
            image_rgb: Image numpy array RGB
        
        Returns:
            bbox: [x1, y1, x2, y2] - TOUJOURS une bbox (estimation si pas trouvé)
        """
        print("[PlateReplacer] Détection simplifiée de plaque...")
        
        h, w = image_rgb.shape[:2]
        
        # STRATÉGIE: Deviner une position par défaut si on ne trouve rien
        # Une plaque est TOUJOURS dans le bas de l'image, centrée horizontalement
        # Dimensions typiques: ~15-25% de la largeur, ~3-5% de la hauteur
        
        # Position par défaut (estimation intelligente)
        default_plate_w = int(w * 0.20)  # 20% de la largeur
        default_plate_h = int(default_plate_w / 4)  # Ratio 4:1
        default_x1 = (w - default_plate_w) // 2  # Centré
        default_y1 = int(h * 0.85)  # 85% vers le bas
        default_bbox = [default_x1, default_y1, default_x1 + default_plate_w, default_y1 + default_plate_h]
        
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Zone de recherche: 20% inférieur de l'image
        roi_y_start = int(h * 0.75)
        roi = gray[roi_y_start:, :]
        roi_h, roi_w = roi.shape
        
        if roi_h < 20 or roi_w < 50:
            print("[PlateReplacer] Image trop petite, utilisation position par défaut.")
            print(f"[PlateReplacer] Position estimée: {default_bbox}")
            return np.array(default_bbox)
        
        # Appliquer un seuillage adaptatif
        binary = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphologie pour nettoyer
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Trouver les contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            print("[PlateReplacer] Aucun contour, utilisation position par défaut.")
            print(f"[PlateReplacer] Position estimée: {default_bbox}")
            return np.array(default_bbox)
        
        # Filtrer les contours par taille et ratio
        candidates = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Critères minimaux
            if cw < 20 or ch < 5:
                continue
            
            ratio = cw / float(ch) if ch > 0 else 0
            
            # Ratio de plaque: entre 2:1 et 8:1 (très permissif)
            if not (1.5 <= ratio <= 8.0):
                continue
            
            area = cw * ch
            
            # Taille raisonnable (pas trop petit, pas toute l'image)
            min_area = 100
            max_area = roi_w * roi_h * 0.3
            
            if not (min_area <= area <= max_area):
                continue
            
            # Calcul simple: préférer les régions plus larges et plus basses
            score = area * (y / roi_h)  # Plus bas = mieux
            
            candidates.append({
                'bbox': [x, y + roi_y_start, x + cw, y + roi_y_start + ch],
                'score': score,
                'ratio': ratio,
                'area': area
            })
        
        if candidates:
            # Prendre le meilleur candidat
            best = max(candidates, key=lambda c: c['score'])
            bbox = best['bbox']
            print(f"[PlateReplacer] Plaque détectée: {bbox}")
            print(f"[PlateReplacer] Ratio: {best['ratio']:.2f}, Aire: {best['area']}")
            return np.array(bbox)
        
        # Si rien n'est trouvé, utiliser la position par défaut
        print("[PlateReplacer] Aucune plaque trouvée, utilisation position par défaut.")
        print(f"[PlateReplacer] Position estimée: {default_bbox}")
        return np.array(default_bbox)
    
    def generate_plany_plate(self, width, height):
        """
        Génère une plaque d'immatriculation virtuelle "PLANY.TN".
        
        Style: Plaque européenne/tunisienne
        - Fond blanc
        - Bande bleue à gauche (optionnel)
        - Texte noir "PLANY.TN" centré
        
        Args:
            width: Largeur de la plaque en pixels
            height: Hauteur de la plaque en pixels
        
        Returns:
            PIL.Image: Image de la plaque générée
        """
        # Créer une image blanche
        plate = Image.new('RGB', (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(plate)
        
        # Ajouter une bande bleue à gauche (style européen)
        blue_width = int(width * 0.1)  # 10% de la largeur
        draw.rectangle([0, 0, blue_width, height], fill=(0, 51, 153))  # Bleu EU
        
        # Ajouter le texte "PLANY.TN"
        text = "PLANY.TN"
        
        # Essayer de charger une police appropriée
        try:
            # Taille de police adaptative
            font_size = int(height * 0.6)
            # Essayer Arial Bold (disponible sur Windows)
            font = ImageFont.truetype("arialbd.ttf", font_size)
        except:
            try:
                # Fallback: police par défaut
                font = ImageFont.load_default()
            except:
                font = None
        
        # Calculer la position centrée du texte
        if font:
            # Utiliser textbbox pour obtenir les dimensions du texte
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            # Estimation approximative si pas de font
            text_width = len(text) * (height // 2)
            text_height = height // 2
        
        # Position centrée (en tenant compte de la bande bleue)
        text_x = blue_width + (width - blue_width - text_width) // 2
        text_y = (height - text_height) // 2
        
        # Dessiner le texte en noir
        draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
        
        # Ajouter une bordure noire fine
        draw.rectangle([0, 0, width-1, height-1], outline=(0, 0, 0), width=2)
        
        return plate
    
    def apply_perspective_transform(self, plate_image, dst_points, src_image_shape):
        """
        Applique une transformation de perspective à la plaque pour qu'elle
        suive l'angle du pare-chocs.
        
        Args:
            plate_image: PIL.Image de la plaque générée
            dst_points: 4 points de destination [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            src_image_shape: Tuple (height, width) de l'image source
        
        Returns:
            numpy array: Plaque transformée
        """
        # Convertir PIL → numpy
        plate_np = np.array(plate_image)
        h, w = plate_np.shape[:2]
        
        # Points source (coins de la plaque rectangulaire)
        src_points = np.float32([
            [0, 0],           # Top-left
            [w, 0],           # Top-right
            [w, h],           # Bottom-right
            [0, h]            # Bottom-left
        ])
        
        # Calculer la matrice d'homographie
        dst_points = np.float32(dst_points)
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Appliquer la transformation
        warped = cv2.warpPerspective(
            plate_np,
            matrix,
            (src_image_shape[1], src_image_shape[0])
        )
        
        return warped
    
    def replace_license_plate(self, image, use_perspective=False):
        """
        Fonction principale: détecte et remplace la plaque d'immatriculation.
        
        Args:
            image: PIL.Image ou numpy array RGB
            use_perspective: Si True, applique une transformation de perspective
        
        Returns:
            PIL.Image: Image avec plaque remplacée
        """
        # Convertir en numpy si nécessaire
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            is_pil = True
        else:
            image_np = image.copy()
            is_pil = False
        
        print("[PlateReplacer] Recherche de plaque d'immatriculation...")
        
        # Étape 1: Détection de la plaque
        bbox = None
        
        if self.use_yolo:
            bbox = self.detect_plate_yolo(image_np)
        
        if bbox is None:
            bbox = self.detect_plate_opencv(image_np)
        
        # detect_plate_opencv retourne toujours une bbox (estimation si non trouvée)
        
        # Étape 2: Générer la nouvelle plaque "PLANY.TN"
        x1, y1, x2, y2 = map(int, bbox)
        plate_width = x2 - x1
        plate_height = y2 - y1
        
        print(f"[PlateReplacer] Génération de la plaque PLANY.TN ({plate_width}x{plate_height})...")
        new_plate = self.generate_plany_plate(plate_width, plate_height)
        
        # Étape 3: Intégration de la plaque
        if use_perspective:
            # TODO: Détecter les 4 coins de la plaque pour perspective transform
            # Pour l'instant, on utilise la bounding box rectangulaire
            print("[PlateReplacer] Transformation de perspective non implémentée. Utilisation du collage simple.")
        
        # Méthode simple: coller la plaque redimensionnée
        new_plate_np = np.array(new_plate)
        
        # Remplacer la zone de la plaque
        image_np[y1:y2, x1:x2] = new_plate_np
        
        print("[PlateReplacer] Plaque remplacée avec succès!")
        
        return Image.fromarray(image_np)


# ============================================================================
# FONCTION STANDALONE POUR INTÉGRATION FACILE
# ============================================================================

def replace_license_plate(image, model_path=None):
    """
    Fonction standalone pour remplacer la plaque d'immatriculation.
    
    Args:
        image: PIL.Image ou numpy array RGB
        model_path: Chemin optionnel vers le modèle YOLO de détection de plaques
    
    Returns:
        PIL.Image: Image avec plaque remplacée par "PLANY.TN"
    """
    replacer = LicensePlateReplacer(model_path=model_path)
    return replacer.replace_license_plate(image)


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    """
    Exemple d'utilisation.
    """
    # Exemple 1: Avec une image PIL
    # img = Image.open("voiture.jpg")
    # result = replace_license_plate(img)
    # result.save("voiture_plany.jpg")
    
    # Exemple 2: Avec un modèle YOLO personnalisé
    # replacer = LicensePlateReplacer(model_path="license_plate_detector.pt")
    # img = Image.open("voiture.jpg")
    # result = replacer.replace_license_plate(img)
    # result.save("voiture_plany.jpg")
    
    print("Module license_plate_replacer chargé.")
