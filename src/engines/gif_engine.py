import os
import cv2
import numpy as np
import gc
from PIL import Image

from src.engines.restore_engine import RestoreXEngine
from src.engines.bg_remover import BackgroundRemoverEngine

class GifEngine:
    def __init__(self, action: str, device: str = "cpu", cpu_cores: int = 4):
        self.action = action
        self.device = device
        self.cpu_cores = cpu_cores
        
        self.restore_engine = None
        self.bg_engine = None
        
        if self.action == "upscale":
            self.restore_engine = RestoreXEngine("realesr-animevideov3.pth", self.device)
        elif self.action == "remove-bg":
            self.bg_engine = BackgroundRemoverEngine("u2netp.onnx", "CPU" if self.device == "cpu" else "CUDA", self.cpu_cores)
            
    def _get_bounding_box(self, diff_mask: np.ndarray) -> tuple:
        y_indices, x_indices = np.where(diff_mask > 0)
        if len(y_indices) == 0:
            return None
        return (np.min(x_indices), np.min(y_indices), np.max(x_indices), np.max(y_indices))

    def _process_frame(self, img: np.ndarray, scale: int) -> np.ndarray:
        if self.action == "upscale":
            # For animevideo, scale determines output size but model is internally 4x
            res = self.restore_engine.run_restoration(img, tile_size=256, tile_overlap=32, scale=scale)
            # Ensure it is RGBA if input was RGBA
            if img.shape[2] == 4 and res.shape[2] == 3:
                alpha = cv2.resize(img[:, :, 3], (res.shape[1], res.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                res = cv2.cvtColor(res, cv2.COLOR_RGB2RGBA)
                res[:, :, 3] = alpha
            return res
        elif self.action == "remove-bg":
            return self.bg_engine.run_removal(img, keep_text=False)
        return img

    def process_gif(self, input_path: str, output_path: str, scale: int = 2, out_format: str = "gif"):
        print(f"[INFO] Loading GIF: {input_path}", flush=True)
        img = Image.open(input_path)
        
        frames = []
        delays = []
        
        try:
            while True:
                delays.append(img.info.get('duration', 100))
                # Convert to RGBA
                frame = img.convert('RGBA')
                frames.append(np.array(frame))
                img.seek(img.tell() + 1)
        except EOFError:
            pass
            
        num_frames = len(frames)
        print(f"[INFO] Extracted {num_frames} frames.", flush=True)
        
        processed_frames = []
        prev_orig_frame = None
        prev_proc_frame = None
        
        diff_threshold = 12
        skip_count = 0
        
        for i, frame_np in enumerate(frames):
            if prev_orig_frame is not None:
                diff = cv2.absdiff(frame_np, prev_orig_frame)
                gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGBA2GRAY)
                _, diff_mask = cv2.threshold(gray_diff, diff_threshold, 255, cv2.THRESH_BINARY)
                
                changed_pixels = np.count_nonzero(diff_mask)
                total_pixels = frame_np.shape[0] * frame_np.shape[1]
                
                if changed_pixels == 0:
                    processed_frames.append(prev_proc_frame.copy())
                    skip_count += 1
                    print(f"[INFO] Frame {i+1}/{num_frames} ({(i+1)/num_frames*100:.1f}%) processed [Cached: 100% skipped]", flush=True)
                    continue
                    
                elif changed_pixels / total_pixels < 0.2:
                    bbox = self._get_bounding_box(diff_mask)
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        pad = 16
                        x1 = max(0, x1 - pad)
                        y1 = max(0, y1 - pad)
                        x2 = min(frame_np.shape[1], x2 + pad)
                        y2 = min(frame_np.shape[0], y2 + pad)
                        
                        crop = frame_np[y1:y2, x1:x2]
                        proc_crop = self._process_frame(crop, scale)
                        
                        proc_frame = prev_proc_frame.copy()
                        
                        if self.action == "upscale":
                            out_x1 = x1 * scale
                            out_y1 = y1 * scale
                            out_x2 = x2 * scale
                            out_y2 = y2 * scale
                        else:
                            out_x1, out_y1, out_x2, out_y2 = x1, y1, x2, y2
                            
                        # Ensure crop fits bounding box shape
                        proc_crop_resized = cv2.resize(proc_crop, (out_x2 - out_x1, out_y2 - out_y1))
                        proc_frame[out_y1:out_y2, out_x1:out_x2] = proc_crop_resized
                        
                        processed_frames.append(proc_frame)
                        prev_proc_frame = proc_frame
                        prev_orig_frame = frame_np
                        
                        skipped_area = 1.0 - ((x2-x1)*(y2-y1)/total_pixels)
                        print(f"[INFO] Frame {i+1}/{num_frames} ({(i+1)/num_frames*100:.1f}%) processed [Cached: {skipped_area*100:.1f}% skipped]", flush=True)
                        continue

            proc_frame = self._process_frame(frame_np, scale)
            processed_frames.append(proc_frame)
            
            prev_orig_frame = frame_np
            prev_proc_frame = proc_frame
            
            print(f"[INFO] Frame {i+1}/{num_frames} ({(i+1)/num_frames*100:.1f}%) processed [Cached: 0.0% skipped]", flush=True)
            gc.collect()

        print(f"[INFO] Total frames perfectly cached: {skip_count}/{num_frames}", flush=True)
        print(f"[INFO] Encoding sequence to {out_format.upper()}...", flush=True)
        
        pil_frames = [Image.fromarray(f) for f in processed_frames]
        
        if out_format.lower() == "gif":
            # Fast simple quantization for Pillow that looks good
            pil_frames[0].save(
                output_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=delays,
                loop=0,
                disposal=2,
                optimize=True
            )
        else:
            pil_frames[0].save(
                output_path,
                format='WebP',
                save_all=True,
                append_images=pil_frames[1:],
                duration=delays,
                loop=0,
                lossless=True,
                method=4
            )
            
        print(f"[INFO] Animation saved to {output_path}", flush=True)
