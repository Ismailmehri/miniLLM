import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

def apply_white_background(image_rgb, mask):
    print("[Utils] Application du fond blanc...")

    white = np.ones_like(image_rgb) * 255
    mask_norm = (mask / 255.0).astype(np.float32)
    mask_norm = np.expand_dims(mask_norm, axis=2)

    out = (image_rgb * mask_norm + white * (1 - mask_norm)).astype(np.uint8)
    print("[Utils] Fond blanc appliqué.")
    return out

def add_shadow(image, bbox, shadow_color=(50, 50, 50), opacity=0.4):
    """
    Ajoute une ombre elliptique sous la voiture.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    
    # Centre et dimensions de l'ombre
    cx = (x1 + x2) // 2
    cy = y2  # Juste en bas de la voiture
    
    # Largeur de l'ombre = largeur de la voiture
    shadow_w = x2 - x1
    # Hauteur de l'ombre = une fraction de la largeur (ellipse aplatie)
    shadow_h = int(shadow_w * 0.2)
    
    # Créer un calque pour l'ombre
    shadow_layer = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Dessiner l'ellipse
    cv2.ellipse(shadow_layer, (cx, cy), (shadow_w // 2, shadow_h // 2), 0, 0, 360, (*shadow_color, 255), -1)
    
    # Flouter l'ombre (Gaussian Blur)
    ksize = int(shadow_w * 0.5) | 1 # Impair
    shadow_layer = cv2.GaussianBlur(shadow_layer, (ksize, ksize), 0)
    
    # Appliquer l'opacité
    shadow_layer[:, :, 3] = (shadow_layer[:, :, 3] * opacity).astype(np.uint8)
    
    # Convertir l'image de base en RGBA si besoin
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        
    # Superposition (Alpha Blending)
    # On utilise PIL pour faciliter le blending RGBA
    pil_img = Image.fromarray(image)
    pil_shadow = Image.fromarray(shadow_layer)
    
    pil_img.alpha_composite(pil_shadow)
    
    return np.array(pil_img)

def center_and_resize_car(image_rgb, mask, bbox, target_size=(240, 230), padding=20):
    """
    Centre la voiture dans une image de taille target_size avec un fond blanc,
    avec ombre et lissage des bords.
    """
    print(f"[Utils] Redimensionnement, centrage et amélioration vers {target_size}...")
    
    # 0. Lissage du masque (Feathering)
    # Flou léger sur le masque pour adoucir les bords
    mask_blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # 1. Appliquer le fond blanc sur l'image originale (avec masque flouté)
    masked_image = apply_white_background(image_rgb, mask_blurred)
    
    # 2. Découper la voiture (Crop) selon la BBox
    x1, y1, x2, y2 = map(int, bbox)
    h, w = masked_image.shape[:2]
    
    # Sécurité
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    car_crop = masked_image[y1:y2, x1:x2]
    
    # 3. Calculer la nouvelle taille
    car_h, car_w = car_crop.shape[:2]
    target_w, target_h = target_size
    
    avail_w = target_w - (padding * 2)
    avail_h = target_h - (padding * 2)
    
    scale = min(avail_w / car_w, avail_h / car_h)
    new_w = int(car_w * scale)
    new_h = int(car_h * scale)
    
    # Redimensionnement de la voiture
    car_resized = cv2.resize(car_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 4. Créer le canvas blanc final
    final_image = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    
    # 5. Ajouter l'ombre sur le canvas blanc AVANT de coller la voiture
    # On simule une bbox centrée pour l'ombre
    pos_x = (target_w - new_w) // 2
    pos_y = (target_h - new_h) // 2
    
    fake_bbox = [pos_x, pos_y, pos_x + new_w, pos_y + new_h]
    final_image = add_shadow(final_image, fake_bbox)
    
    # Convertir en RGB (add_shadow retourne RGBA)
    final_image = cv2.cvtColor(final_image, cv2.COLOR_RGBA2RGB)
    
    # 6. Coller la voiture au centre
    # Attention : car_resized a un fond blanc, il faut le "mixer" ou juste le coller
    # Comme le fond est blanc et le canvas aussi, on peut coller direct, 
    # MAIS pour garder l'ombre visible, il faut que le fond de car_resized soit transparent ou bien géré.
    # PROBLÈME : apply_white_background met du blanc en dur.
    # SOLUTION : On doit refaire le détourage ICI pour avoir de la transparence.
    
    # Recupérer le masque cropé et redimensionné
    mask_crop = mask_blurred[y1:y2, x1:x2]
    mask_resized = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Recupérer l'image brute cropée et redimensionnée
    raw_crop = image_rgb[y1:y2, x1:x2]
    raw_resized = cv2.resize(raw_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Coller avec Alpha Blending
    for c in range(3):
        alpha = mask_resized / 255.0
        final_image[pos_y:pos_y+new_h, pos_x:pos_x+new_w, c] = \
            raw_resized[:, :, c] * alpha + \
            final_image[pos_y:pos_y+new_h, pos_x:pos_x+new_w, c] * (1 - alpha)
            
    print("[Utils] Rendu réaliste terminé.")
    return final_image
