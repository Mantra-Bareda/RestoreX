import cv2
import numpy as np

def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 8, eps: float = 1e-4) -> np.ndarray:
    """
    Fast Guided Filter.
    guide: High-res RGB image, normalized to [0, 1]
    src: Upscaled coarse alpha mask, single channel [0, 1]
    """
    try:
        # If opencv-contrib-python is installed
        guide_uint8 = np.uint8(guide * 255) if guide.dtype != np.uint8 else guide
        src_uint8 = np.uint8(src * 255) if src.dtype != np.uint8 else src
        
        filtered = cv2.ximgproc.guidedFilter(
            guide=guide_uint8,
            src=src_uint8,
            radius=radius,
            eps=eps * 255 * 255
        )
        if len(filtered.shape) > 2:
            filtered = filtered[:, :, 0] if filtered.shape[2] > 1 else filtered[:, :]
        return filtered.astype(np.float32) / 255.0
    except AttributeError:
        # Fallback to pure NumPy/OpenCV box-filter implementation
        if len(guide.shape) == 3:
            guide = cv2.cvtColor(np.uint8(guide * 255), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            
        return _box_guided_filter(guide, src, radius, eps)

def _box_guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """
    Grayscale Guided Filter using boxFilter
    """
    mean_I = cv2.boxFilter(I, cv2.CV_32F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (r, r))
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (r, r))

    var_I = mean_II - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))

    q = mean_a * I + mean_b
    return q
