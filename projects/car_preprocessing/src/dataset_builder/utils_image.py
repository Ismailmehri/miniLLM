import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def apply_white_background(image_rgb, mask):
    """
    Applies a white background to the image using the mask.
    mask should be 0 for background, 255 for foreground.
    """
    # Create white background
    white_bg = np.ones_like(image_rgb) * 255
    
    # Normalize mask to 0-1
    mask_norm = mask / 255.0
    mask_norm = np.expand_dims(mask_norm, axis=2)
    
    # Combine
    result = image_rgb * mask_norm + white_bg * (1 - mask_norm)
    return result.astype(np.uint8)

def replace_license_plate(image_rgb, plate_bbox, text="PLANY.TN"):
    """
    Replaces the area defined by plate_bbox with a white rectangle and text.
    plate_bbox: [x1, y1, x2, y2]
    """
    if plate_bbox is None:
        return image_rgb
        
    x1, y1, x2, y2 = map(int, plate_bbox)
    
    # Draw white rectangle
    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 255, 255), -1)
    
    # Add text
    # Convert to PIL for better text rendering
    pil_img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    # Load font (default if not found)
    try:
        # Try to load a standard font, or use default
        font = ImageFont.truetype("arial.ttf", size=int((y2-y1)*0.8))
    except IOError:
        font = ImageFont.load_default()
        
    # Calculate text size to center it
    # This is a bit simplified, might need adjustment for perfect centering
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    text_x = center_x - text_width // 2
    text_y = center_y - text_height // 2
    
    draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
    
    return np.array(pil_img)
