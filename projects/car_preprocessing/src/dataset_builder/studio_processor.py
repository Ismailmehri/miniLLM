"""
Module de post-traitement pour simuler un rendu studio professionnel.
Auteur: Expert Senior Python & Computer Vision
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2


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
    background_color=(255, 255, 255)
):
    """
    Applique un post-traitement professionnel à une image RGBA détourée.
    
    Cette fonction simule une photo prise en studio professionnel avec :
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
    
    Returns:
        PIL.Image: Image finale en mode RGB avec rendu studio professionnel
    """
    
    if not isinstance(rgba_image, Image.Image):
        raise TypeError("rgba_image doit être un objet PIL.Image")
    
    if rgba_image.mode != 'RGBA':
        raise ValueError("L'image doit être en mode RGBA")
    
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
