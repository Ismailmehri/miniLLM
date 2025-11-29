"""
Module de post-traitement pour simuler un rendu studio professionnel.
Auteur: Expert Senior Python & Computer Vision
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2


def correct_orientation(rgba_image, auto_rotate=True, rotation_angle=None):
    """
    Corrige l'orientation d'une image RGBA détourée pour qu'elle soit bien droite.
    
    Cette fonction analyse le canal alpha pour détecter l'angle d'inclinaison
    de l'objet et le redresse automatiquement.
    
    Méthodes de détection utilisées:
    1. Rectangle minimal englobant (Minimum Area Rectangle) via OpenCV
    2. Analyse des contours de l'objet
    3. Détection de l'axe principal via moments d'image
    
    Args:
        rgba_image: PIL.Image en mode RGBA (objet sur fond transparent)
        auto_rotate: Si True, détecte et corrige automatiquement l'angle
        rotation_angle: Angle de rotation manuel (degrés, sens horaire)
    
    Returns:
        PIL.Image: Image RGBA redressée
    """
    
    if not isinstance(rgba_image, Image.Image):
        raise TypeError("rgba_image doit être un objet PIL.Image")
    
    if rgba_image.mode != 'RGBA':
        raise ValueError("L'image doit être en mode RGBA")
    
    print("[Orientation] Analyse de l'orientation de l'objet...")
    
    # ========================================================================
    # ÉTAPE 1: EXTRACTION DU CANAL ALPHA ET CONVERSION EN FORMAT OPENCV
    # ========================================================================
    
    # Convertir PIL → numpy array
    rgba_array = np.array(rgba_image)
    
    # Extraire le canal alpha (masque binaire de l'objet)
    alpha_channel = rgba_array[:, :, 3]
    
    # Vérifier si l'image n'est pas vide
    if np.sum(alpha_channel) == 0:
        print("[Orientation] Image vide, aucune correction nécessaire.")
        return rgba_image
    
    # ========================================================================
    # ÉTAPE 2: DÉTECTION AUTOMATIQUE DE L'ANGLE D'INCLINAISON
    # ========================================================================
    
    if auto_rotate and rotation_angle is None:
        # Méthode 1: Rectangle minimal englobant (Minimum Area Rectangle)
        # Cette méthode trouve le plus petit rectangle qui englobe l'objet
        # et retourne son angle d'orientation
        
        # Binariser le masque (seuillage)
        _, binary_mask = cv2.threshold(alpha_channel, 127, 255, cv2.THRESH_BINARY)
        
        # Trouver les contours de l'objet
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            print("[Orientation] Aucun contour détecté, aucune correction nécessaire.")
            return rgba_image
        
        # Prendre le plus grand contour (l'objet principal)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Calculer le rectangle minimal englobant
        # minAreaRect retourne: ((center_x, center_y), (width, height), angle)
        rect = cv2.minAreaRect(main_contour)
        (center_x, center_y), (width, height), angle = rect
        
        print(f"[Orientation] Rectangle minimal détecté: angle = {angle:.2f}°")
        
        # ========================================================================
        # LOGIQUE DE CORRECTION D'ANGLE
        # ========================================================================
        # OpenCV retourne un angle entre -90° et 0°
        # On veut que l'objet soit horizontal (angle proche de 0°)
        
        # Si le rectangle est plus large que haut, l'objet est déjà horizontal
        # Si le rectangle est plus haut que large, il faut le faire pivoter de 90°
        
        if width < height:
            # L'objet est vertical, on le tourne de 90°
            rotation_angle = angle + 90
        else:
            # L'objet est horizontal, on corrige juste le petit angle
            rotation_angle = angle
        
        # Normaliser l'angle pour éviter les grandes rotations
        # On veut corriger uniquement les petites inclinaisons (-45° à +45°)
        if abs(rotation_angle) > 45:
            rotation_angle = 0
        
        print(f"[Orientation] Angle de correction calculé: {rotation_angle:.2f}°")
    
    elif rotation_angle is not None:
        print(f"[Orientation] Utilisation de l'angle manuel: {rotation_angle:.2f}°")
    else:
        print("[Orientation] Aucune rotation demandée.")
        return rgba_image
    
    # ========================================================================
    # ÉTAPE 3: APPLICATION DE LA ROTATION
    # ========================================================================
    
    if abs(rotation_angle) < 0.5:
        print("[Orientation] Angle négligeable, aucune rotation appliquée.")
        return rgba_image
    
    print(f"[Orientation] Application de la rotation de {rotation_angle:.2f}°...")
    
    # Convertir l'angle en radians pour les calculs
    # Note: PIL.Image.rotate() prend l'angle en degrés, sens anti-horaire
    # On inverse le signe pour corriger dans le bon sens
    rotation_angle_pil = -rotation_angle
    
    # Appliquer la rotation avec PIL (gestion automatique de la transparence)
    # expand=True agrandit le canvas pour contenir toute l'image tournée
    rotated_image = rgba_image.rotate(
        rotation_angle_pil,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(0, 0, 0, 0)  # Fond transparent
    )
    
    print("[Orientation] Rotation appliquée avec succès.")
    
    # ========================================================================
    # ÉTAPE 4: RECADRAGE POUR ENLEVER LES BORDS TRANSPARENTS EXCESSIFS
    # ========================================================================
    
    # Après rotation, il peut y avoir beaucoup d'espace transparent
    # On recadre pour garder uniquement la zone utile
    
    alpha_rotated = rotated_image.split()[3]
    bbox = alpha_rotated.getbbox()
    
    if bbox:
        # Recadrer l'image selon la bounding box
        rotated_image = rotated_image.crop(bbox)
        print("[Orientation] Image recadrée après rotation.")
    
    return rotated_image


def apply_professional_studio_look(
    rgba_image,
    output_size=(1024, 1024),
    padding=100,
    shadow_opacity=0.3,
    shadow_blur_radius=50,
    shadow_offset_y=30,
    shadow_flatten_factor=0.6,
    sharpness_factor=1.3,
    contrast_factor=1.15,
    saturation_factor=1.2,
    background_color=(255, 255, 255),
    auto_correct_orientation=True,
    replace_plate=False
):
    """
    Applique un post-traitement professionnel à une image RGBA détourée.
    
    Cette fonction simule une photo prise en studio professionnel avec :
    - Correction automatique de l'orientation (redressement)
    - Ombre portée réaliste (drop shadow avec perspective)
    - Color grading (netteté, contraste, saturation)
    - Composition centrée sur fond blanc
    
    Args:
        rgba_image: PIL.Image en mode RGBA (objet sur fond transparent)
        output_size: Tuple (width, height) pour la taille finale (carré recommandé)
        padding: Marge autour de l'objet en pixels
        shadow_opacity: Opacité de l'ombre (0.0 à 1.0)
        shadow_blur_radius: Rayon du flou gaussien pour l'ombre
        shadow_offset_y: Décalage vertical de l'ombre (pixels)
        shadow_flatten_factor: Facteur d'aplatissement de l'ombre (0.0 à 1.0)
        sharpness_factor: Facteur de netteté (1.0 = original, >1.0 = plus net)
        contrast_factor: Facteur de contraste (1.0 = original, >1.0 = plus de contraste)
        saturation_factor: Facteur de saturation (1.0 = original, >1.0 = plus saturé)
        background_color: Couleur de fond RGB (tuple)
        auto_correct_orientation: Si True, corrige automatiquement l'orientation
        replace_plate: Si True, remplace la plaque par PLANY.TN
    
    Returns:
        PIL.Image: Image finale en mode RGB avec rendu studio professionnel
    """
    
    if not isinstance(rgba_image, Image.Image):
        raise TypeError("rgba_image doit être un objet PIL.Image")
    
    if rgba_image.mode != 'RGBA':
        raise ValueError("L'image doit être en mode RGBA")
    
    # ========================================================================
    # ÉTAPE 0: CORRECTION DE L'ORIENTATION (NOUVEAU)
    # ========================================================================
    
    if auto_correct_orientation:
        rgba_image = correct_orientation(rgba_image, auto_rotate=True)
    
    # ========================================================================
    # ÉTAPE 0.5: REMPLACEMENT DE LA PLAQUE (NOUVEAU)
    # ========================================================================
    
    if replace_plate:
        print("[Studio] Remplacement de la plaque d'immatriculation...")
        try:
            from .license_plate_replacer import LicensePlateReplacer
            
            # Convertir RGBA → RGB pour la détection
            rgb_for_plate = np.array(rgba_image.convert('RGB'))
            
            # Chercher automatiquement un modèle YOLO
            from pathlib import Path
            yolo_model_path = None
            possible_paths = [
                Path("models/license_plate_detector.pt"),
                Path("../models/license_plate_detector.pt"),
                Path("../../models/license_plate_detector.pt")
            ]
            
            for path in possible_paths:
                if path.exists():
                    yolo_model_path = str(path)
                    break
            
            # Remplacer la plaque (avec YOLO si disponible, sinon OpenCV)
            replacer = LicensePlateReplacer(model_path=yolo_model_path)
            plate_replaced = replacer.replace_license_plate(rgb_for_plate)
            
            # Reconvertir en RGBA en gardant le canal alpha
            alpha_backup = rgba_image.split()[3]
            rgba_image = plate_replaced.convert('RGBA')
            rgba_image.putalpha(alpha_backup)
            
            print("[Studio] Plaque remplacée.")
        except Exception as e:
            print(f"[Studio] Erreur remplacement plaque: {e}")
            print("[Studio] Continuation sans remplacement de plaque.")
    
    # ========================================================================
    # ÉTAPE 1: EXTRACTION DU CANAL ALPHA ET CALCUL DE LA BOUNDING BOX
    # ========================================================================
    
    # Extraire le canal alpha (masque de l'objet)
    alpha_channel = rgba_image.split()[3]
    
    # Trouver la bounding box de l'objet (zone non-transparente)
    bbox = alpha_channel.getbbox()
    
    if bbox is None:
        # Image complètement transparente, retourner un fond blanc
        return Image.new('RGB', output_size, background_color)
    
    # Extraire l'objet selon sa bounding box
    cropped_rgba = rgba_image.crop(bbox)
    obj_width, obj_height = cropped_rgba.size
    
    # ========================================================================
    # ÉTAPE 2: CALCUL DES DIMENSIONS FINALES AVEC PADDING
    # ========================================================================
    
    # Calculer l'échelle pour que l'objet rentre dans output_size avec padding
    max_obj_width = output_size[0] - (2 * padding)
    max_obj_height = output_size[1] - (2 * padding)
    
    scale = min(max_obj_width / obj_width, max_obj_height / obj_height)
    
    # Nouvelles dimensions de l'objet
    new_width = int(obj_width * scale)
    new_height = int(obj_height * scale)
    
    # Redimensionner l'objet (haute qualité)
    resized_rgba = cropped_rgba.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )
    
    # ========================================================================
    # ÉTAPE 3: GÉNÉRATION DE L'OMBRE PORTÉE RÉALISTE (DROP SHADOW)
    # ========================================================================
    
    print("[Studio] Génération de l'ombre portée...")
    
    # 3.1: Extraire le canal alpha de l'objet redimensionné
    alpha_resized = resized_rgba.split()[3]
    
    # 3.2: Créer le masque d'ombre à partir de l'alpha
    # L'ombre est une copie du canal alpha
    shadow_mask = alpha_resized.copy()
    
    # 3.3: Aplatir l'ombre (perspective transformation simplifiée)
    # On réduit la hauteur de l'ombre pour simuler une projection au sol
    shadow_width = new_width
    shadow_height = int(new_height * shadow_flatten_factor)
    
    # Redimensionner l'ombre (aplatissement)
    shadow_flattened = shadow_mask.resize(
        (shadow_width, shadow_height),
        Image.Resampling.LANCZOS
    )
    
    # 3.4: Appliquer un flou gaussien pour adoucir l'ombre
    shadow_blurred = shadow_flattened.filter(
        ImageFilter.GaussianBlur(radius=shadow_blur_radius)
    )
    
    # 3.5: Ajuster l'opacité de l'ombre
    # Convertir en numpy pour manipulation rapide
    shadow_array = np.array(shadow_blurred).astype(np.float32)
    shadow_array = shadow_array * shadow_opacity
    shadow_final = Image.fromarray(shadow_array.astype(np.uint8), mode='L')
    
    # ========================================================================
    # ÉTAPE 4: COMPOSITION SUR FOND BLANC
    # ========================================================================
    
    print("[Studio] Composition sur fond blanc...")
    
    # 4.1: Créer le canvas blanc final
    final_canvas = Image.new('RGB', output_size, background_color)
    
    # 4.2: Calculer la position centrée de l'objet
    obj_x = (output_size[0] - new_width) // 2
    obj_y = (output_size[1] - new_height) // 2
    
    # 4.3: Calculer la position de l'ombre (décalée vers le bas)
    shadow_x = (output_size[0] - shadow_width) // 2
    shadow_y = obj_y + new_height - shadow_height + shadow_offset_y
    
    # 4.4: Créer une image RGBA pour l'ombre (noir avec alpha)
    shadow_rgba = Image.new('RGBA', output_size, (0, 0, 0, 0))
    
    # Créer l'ombre noire avec le masque d'opacité
    shadow_layer = Image.new('RGBA', (shadow_width, shadow_height), (0, 0, 0, 255))
    shadow_layer.putalpha(shadow_final)
    
    # Coller l'ombre sur le canvas RGBA
    shadow_rgba.paste(shadow_layer, (shadow_x, shadow_y), shadow_layer)
    
    # 4.5: Convertir le canvas blanc en RGBA pour la composition
    canvas_rgba = final_canvas.convert('RGBA')
    
    # 4.6: Composer l'ombre sur le canvas
    canvas_with_shadow = Image.alpha_composite(canvas_rgba, shadow_rgba)
    
    # 4.7: Composer l'objet par-dessus l'ombre
    canvas_with_shadow.paste(resized_rgba, (obj_x, obj_y), resized_rgba)
    
    # Convertir en RGB
    result = canvas_with_shadow.convert('RGB')
    
    # ========================================================================
    # ÉTAPE 5: COLOR GRADING (AMÉLIORATION DE L'IMAGE)
    # ========================================================================
    
    print("[Studio] Application du color grading...")
    
    # 5.1: Augmenter la netteté (Sharpness)
    enhancer_sharpness = ImageEnhance.Sharpness(result)
    result = enhancer_sharpness.enhance(sharpness_factor)
    
    # 5.2: Augmenter le contraste
    enhancer_contrast = ImageEnhance.Contrast(result)
    result = enhancer_contrast.enhance(contrast_factor)
    
    # 5.3: Augmenter la saturation
    enhancer_color = ImageEnhance.Color(result)
    result = enhancer_color.enhance(saturation_factor)
    
    print("[Studio] Rendu professionnel terminé.")
    
    return result


def process_folder(input_folder, output_folder, **kwargs):
    """
    Traite un dossier d'images RGBA en boucle.
    
    Args:
        input_folder: Chemin du dossier contenant les images RGBA
        output_folder: Chemin du dossier de sortie
        **kwargs: Arguments à passer à apply_professional_studio_look
    """
    import os
    from pathlib import Path
    
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extensions supportées
    extensions = ['.png', '.PNG']
    
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
    
    print(f"[Studio] {len(image_files)} images trouvées.")
    
    for img_file in image_files:
        print(f"\n[Studio] Traitement de {img_file.name}...")
        
        try:
            # Charger l'image RGBA
            rgba_img = Image.open(img_file)
            
            if rgba_img.mode != 'RGBA':
                print(f"[Studio] ATTENTION: {img_file.name} n'est pas en RGBA. Conversion...")
                rgba_img = rgba_img.convert('RGBA')
            
            # Appliquer le traitement studio
            result = apply_professional_studio_look(rgba_img, **kwargs)
            
            # Sauvegarder
            output_file = output_path / f"{img_file.stem}_studio.jpg"
            result.save(output_file, 'JPEG', quality=95)
            
            print(f"[Studio] ✓ Sauvegardé: {output_file.name}")
            
        except Exception as e:
            print(f"[Studio] ✗ ERREUR sur {img_file.name}: {e}")
            continue
    
    print(f"\n[Studio] Traitement terminé. {len(image_files)} images traitées.")


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    """
    Exemple d'utilisation en ligne de commande.
    """
    
    # Exemple 1: Traiter une seule image
    # rgba_img = Image.open("voiture_detouree.png")
    # result = apply_professional_studio_look(rgba_img)
    # result.save("voiture_studio.jpg", quality=95)
    
    # Exemple 2: Traiter un dossier complet
    process_folder(
        input_folder="./images_detourees",
        output_folder="./images_studio",
        output_size=(1024, 1024),
        padding=100,
        shadow_opacity=0.3,
        shadow_blur_radius=50,
        shadow_offset_y=30,
        sharpness_factor=1.3,
        contrast_factor=1.15,
        saturation_factor=1.2
    )
