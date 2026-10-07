import cv2
import os
from src.progress import report_progress

def restore_resolution(image, original_width, original_height, target_scale=1):
    """
    Phase 11: Restores the image to the requested dimensions based on scale.
    scale=1 restores to exact original.
    scale=2 restores to 2x original.
    scale=4 restores to 4x original.
    """
    h, w = image.shape[:2]
    
    target_w = original_width * target_scale
    target_h = original_height * target_scale
    
    if w == target_w and h == target_h:
        return image
        
    report_progress(f"Adjusting output resolution: {w}x{h} -> {target_w}x{target_h} ({target_scale}x scale)")
    
    if w > target_w:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC
        
    restored = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
    return restored

def save_image(image, output_path):
    """
    Saves the final processed numpy array to disk.
    """
    report_progress(f"Saving final image to {output_path}...")
    success = cv2.imwrite(output_path, image)
    if not success:
        raise IOError(f"OpenCV failed to write image to {output_path}. Check file permissions and path.")

def patch_basicsr_compat():
    """
    Fixes a known compatibility issue where basicsr crashes on newer versions of PyTorch/Torchvision
    because functional_tensor was deprecated. This safely intercepts the import and maps it.
    """
    import sys
    try:
        import torchvision.transforms.functional_tensor
    except ImportError:
        try:
            import torchvision.transforms.functional as TF
            sys.modules['torchvision.transforms.functional_tensor'] = TF
        except ImportError:
            pass
