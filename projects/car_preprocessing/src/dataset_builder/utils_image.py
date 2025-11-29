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
