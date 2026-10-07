import os
import cv2
import numpy as np
import torch
import gc
from typing import Dict, Any

from src.models.registry import verify_and_download_model
from src.engines.tiler import generate_tile_coordinates, extract_tile, get_feathering_weight_matrix

# Need to patch basicsr compat here since we rely on it deeply
try:
    import torchvision
    from torchvision.transforms import functional_tensor
except ImportError:
    import torchvision
    import torchvision.transforms.functional as functional
    import sys
    sys.modules['torchvision.transforms.functional_tensor'] = functional

class RestoreXEngine:
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self.model_path = verify_and_download_model(model_name)
        self.upsampler = self._load_realesrgan()

    def _load_realesrgan(self):
        from realesrgan import RealESRGANer
        if self.model_name == "realesr-general-x4v3.pth":
            from basicsr.archs.srvgg_arch import SRVGGNetCompact
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            
        upsampler = RealESRGANer(
            scale=4,
            model_path=self.model_path,
            dni_weight=None,
            model=model,
            tile=0, # We handle primary tiling externally, fallback to internal if OOM
            tile_pad=10,
            pre_pad=10,
            half=(self.device == "cuda"),
            device=self.device
        )
        return upsampler

    def run_restoration(self, img: np.ndarray, tile_size: int, tile_overlap: int, scale: int) -> np.ndarray:
        # Preserve original dimensions and channels
        h, w = img.shape[:2]
        c = img.shape[2] if len(img.shape) == 3 else 1
        
        # Internal scale for the network is always 4x
        out_w = w * 4
        out_h = h * 4
        out_overlap = tile_overlap * 4
        
        print(f"[INFO] Creating output composite canvas {out_w}x{out_h}", flush=True)
        # Handle memmap for massive arrays could be implemented here; keeping in RAM for general use
        output_img = np.zeros((out_h, out_w, c), dtype=np.float32)
        weight_mat = np.zeros((out_h, out_w, 1), dtype=np.float32)
        
        coords = generate_tile_coordinates(w, h, tile_size, tile_overlap)
        total_tiles = len(coords)
        print(f"[INFO] Total primary tiles to process: {total_tiles}", flush=True)
        
        current_sub_tile_size = tile_size
        
        for i, coord in enumerate(coords):
            tile_crop = extract_tile(img, coord)
            
            while True:
                try:
                    if current_sub_tile_size < tile_size:
                        self.upsampler.tile_size = current_sub_tile_size
                    else:
                        self.upsampler.tile_size = 0
                        
                    output_crop, _ = self.upsampler.enhance(tile_crop, outscale=4)
                    break
                except (RuntimeError, MemoryError) as e:
                    error_msg = str(e).lower()
                    if "out of memory" in error_msg or "oom" in error_msg or isinstance(e, MemoryError):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        
                        if current_sub_tile_size == tile_size:
                            current_sub_tile_size = tile_size // 2
                        else:
                            current_sub_tile_size = current_sub_tile_size // 2
                            
                        if current_sub_tile_size < 32:
                            raise RuntimeError("Unrecoverable Out Of Memory error during tile processing.")
                            
                        print(f"\n[OOM RECOVERY] Tile {i+1} OOM. Retrying with internal sub-tiling at {current_sub_tile_size}...", flush=True)
                    else:
                        raise e
                        
            out_coord = {
                'x': coord['x'] * 4,
                'y': coord['y'] * 4,
                'x_end': coord['x_end'] * 4,
                'y_end': coord['y_end'] * 4
            }
            
            # Ensure crop fits perfectly (RealESRGAN might return slightly padded if outscale is wonky, though unlikely)
            crop_h = out_coord['y_end'] - out_coord['y']
            crop_w = out_coord['x_end'] - out_coord['x']
            output_crop = output_crop[:crop_h, :crop_w]
            if len(output_crop.shape) == 2:
                output_crop = np.expand_dims(output_crop, axis=-1)
                
            feather_weight = get_feathering_weight_matrix(crop_h, crop_w, out_overlap, coord, w, h)
            
            output_img[out_coord['y']:out_coord['y_end'], out_coord['x']:out_coord['x_end']] += output_crop * feather_weight
            weight_mat[out_coord['y']:out_coord['y_end'], out_coord['x']:out_coord['x_end']] += feather_weight
            
            del output_crop, feather_weight, tile_crop
            gc.collect()
            
            percentage = ((i + 1) / total_tiles) * 100
            print(f"[INFO] Tile {i+1}/{total_tiles} ({percentage:.1f}%) processed.", flush=True)
            
        output_img = output_img / (weight_mat + 1e-6)
        output_img = output_img.clip(0, 255).astype(np.uint8)
        
        if scale != 4:
            target_w = w * scale
            target_h = h * scale
            print(f"[INFO] Adjusting output resolution via Lanczos to {target_w}x{target_h} ({scale}x scale)", flush=True)
            output_img = cv2.resize(output_img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            
        return output_img
