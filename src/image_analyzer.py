from PIL import Image
import os
import cv2
import numpy as np

def analyze_image(image_path):
    """
    Handles initial loading and metadata extraction of the input image
    without loading a massive tensor into PyTorch memory.
    Also calculates Laplacian Variance to measure image sharpness/detail.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    img_info = {}
    with Image.open(image_path) as img:
        img_info['width'] = img.width
        img_info['height'] = img.height
        img_info['mode'] = img.mode
        img_info['format'] = img.format
        img_info['channels'] = len(img.getbands())
        
    img_info['file_size_bytes'] = os.path.getsize(image_path)
    img_info['pixels'] = img_info['width'] * img_info['height']
    
    # Calculate Laplacian variance to estimate detail density (sharpness)
    # We only take a center crop to avoid massive RAM/CPU usage on huge files
    cv_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if cv_img is not None:
        h, w = cv_img.shape
        crop_size = min(2000, h, w) # 2000x2000 is plenty to judge sharpness
        cy, cx = h // 2, w // 2
        crop = cv_img[cy - crop_size//2 : cy + crop_size//2, cx - crop_size//2 : cx + crop_size//2]
        img_info['laplacian_variance'] = cv2.Laplacian(crop, cv2.CV_64F).var()
    else:
        img_info['laplacian_variance'] = 0.0
        
    return img_info
