import cv2
import numpy as np

class TextPreserver:
    def __init__(self, color_threshold: int = 30):
        self.color_threshold = color_threshold
        
    def _estimate_background_color(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        corners = [
            img[0:10, 0:10],
            img[0:10, w-10:w],
            img[h-10:h, 0:10],
            img[h-10:h, w-10:w]
        ]
        medians = [np.median(c, axis=(0, 1)) for c in corners]
        return np.median(medians, axis=0)

    def generate_text_mask(self, img: np.ndarray) -> np.ndarray:
        bg_color = self._estimate_background_color(img)
        
        img_float = img.astype(np.float32)
        bg_float = bg_color.astype(np.float32)
        
        delta_e = np.sqrt(np.sum((img_float - bg_float)**2, axis=2))
        chroma_mask = (delta_e > self.color_threshold).astype(np.uint8) * 255
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use chroma mask directly, smoothed slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        base_mask = cv2.morphologyEx(chroma_mask, cv2.MORPH_CLOSE, kernel)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(base_mask, connectivity=8)
        
        text_mask = np.zeros_like(gray)
        candidate_count = 0
        
        # 0 is background
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            if area < 10 or area > (img.shape[0] * img.shape[1] * 0.05):
                continue
                
            aspect_ratio = w / float(h)
            if aspect_ratio > 15 or aspect_ratio < 0.1:
                continue
                
            density = area / (w * h)
            if density < 0.15:
                continue
                
            # For confirmed candidates, allow the dense chroma mask in that bounding box
            glyph_mask = chroma_mask[y:y+h, x:x+w]
            text_mask[y:y+h, x:x+w] = cv2.bitwise_or(text_mask[y:y+h, x:x+w], glyph_mask)
            candidate_count += 1
            
        print(f"[INFO] Text preservation active: Detected {candidate_count} glyph candidate components.", flush=True)
        
        # Slight dilation and blur for anti-aliasing letterforms
        text_mask = cv2.dilate(text_mask, kernel, iterations=1)
        text_mask = cv2.GaussianBlur(text_mask, (3, 3), 0)
        
        return text_mask

    def blend_masks(self, neural_mask: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
        if neural_mask.dtype != np.float32:
            neural_mask = neural_mask.astype(np.float32) / 255.0
        if text_mask.dtype != np.float32:
            text_mask = text_mask.astype(np.float32) / 255.0
            
        merged = np.clip(neural_mask + text_mask, 0.0, 1.0)
        return merged
