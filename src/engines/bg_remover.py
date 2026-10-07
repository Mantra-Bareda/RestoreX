import os
import cv2
import numpy as np
import gc

from src.models.registry import verify_and_download_model
from src.engines.guided_filter import guided_filter

class BackgroundRemoverEngine:
    def __init__(self, model_name: str, provider: str, cpu_cores: int = 4):
        self.model_name = model_name
        self.model_path = verify_and_download_model(model_name)
        
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = cpu_cores
        
        providers = ['CPUExecutionProvider']
        if provider == "CUDA":
            providers = ['CUDAExecutionProvider'] + providers
        elif provider == "DirectML":
            providers = ['DmlExecutionProvider'] + providers
        elif provider == "OpenVINO":
            providers = ['OpenVINOExecutionProvider'] + providers
            
        self.session = ort.InferenceSession(self.model_path, sess_options=sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        
        if "u2netp" in model_name or "silueta" in model_name:
            self.target_size = (320, 320)
        else:
            self.target_size = (1024, 1024)

    def run_removal(self, img: np.ndarray, keep_text: bool = False) -> np.ndarray:
        h, w = img.shape[:2]
        
        print(f"[INFO] Downsampling image to {self.target_size[0]}x{self.target_size[1]}...", flush=True)
        img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Standard ImageNet Normalization
        img_float = img_resized.astype(np.float32) / 255.0
        
        # Convert BGR (cv2 default) to RGB for the model
        img_float = cv2.cvtColor(img_float, cv2.COLOR_BGR2RGB)
        
        img_float -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
        img_float /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # HWC -> NCHW
        img_tensor = np.transpose(img_float, (2, 0, 1))
        img_tensor = np.expand_dims(img_tensor, axis=0)
        
        print("[INFO] Mask Inference...", flush=True)
        outputs = self.session.run(None, {self.input_name: img_tensor})
        mask = outputs[0][0, 0] # 1-channel mask
        
        # Release tensor and garbage collect
        del outputs, img_tensor, img_float, img_resized
        gc.collect()
        
        # Min-max normalization
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        
        print(f"[INFO] Upscaling mask back to {w}x{h}...", flush=True)
        mask_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_CUBIC)
        
        if keep_text:
            from src.engines.text_preserver import TextPreserver
            preserver = TextPreserver()
            text_mask = preserver.generate_text_mask(img)
            mask_full = preserver.blend_masks(mask_full, text_mask)
            
        print("[INFO] Guided Filter Snap...", flush=True)
        guide = img.astype(np.float32) / 255.0
        refined_mask = guided_filter(guide, mask_full, radius=8, eps=1e-4)
        
        refined_mask_uint8 = np.clip(refined_mask * 255, 0, 255).astype(np.uint8)
        
        print("[INFO] RGBA Export...", flush=True)
        if img.shape[2] == 4:
            b, g, r, _ = cv2.split(img)
        else:
            b, g, r = cv2.split(img)
            
        rgba = cv2.merge((b, g, r, refined_mask_uint8))
        
        return rgba
