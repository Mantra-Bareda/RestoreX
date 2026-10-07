import os
import urllib.request
import tempfile
from src.progress import report_progress
from src.tiler import generate_tile_coordinates, extract_tile

MODELS = {
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "4x-UltraSharp.pth": "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth",
    "realesr-general-x4v3.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
}

def verify_and_download_model(model_name="RealESRGAN_x4plus.pth", models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, model_name)
    if not os.path.exists(model_path):
        report_progress(f"Model not found at {model_path}. Downloading...")
        try:
            req = urllib.request.Request(MODELS[model_name], headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(model_path, 'wb') as out_file:
                chunk_size = 8192
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
            report_progress("Model download complete.")
        except Exception as e:
            raise FileNotFoundError(f"Model could not be downloaded automatically: {e}")
            
    # Auto-convert old ESRGAN formats (like 4x-UltraSharp) to BasicSR format if necessary
    try:
        import torch
        state_dict = torch.load(model_path, map_location='cpu')
        if 'params' not in state_dict and 'params_ema' not in state_dict:
            report_progress(f"Converting {model_name} to BasicSR format...")
            new_state_dict = {}
            for k, v in state_dict.items():
                if k == 'model.0.weight': new_k = 'conv_first.weight'
                elif k == 'model.0.bias': new_k = 'conv_first.bias'
                elif k.startswith('model.1.sub.'):
                    parts = k.split('.')
                    sub_id = int(parts[3])
                    if sub_id == 23:
                        new_k = f'conv_body.{parts[4]}'
                    else:
                        new_k = f'body.{sub_id}.{parts[4].lower()}.{parts[5]}.{parts[7]}'
                elif k == 'model.3.weight': new_k = 'conv_up1.weight'
                elif k == 'model.3.bias': new_k = 'conv_up1.bias'
                elif k == 'model.6.weight': new_k = 'conv_up2.weight'
                elif k == 'model.6.bias': new_k = 'conv_up2.bias'
                elif k == 'model.8.weight': new_k = 'conv_hr.weight'
                elif k == 'model.8.bias': new_k = 'conv_hr.bias'
                elif k == 'model.10.weight': new_k = 'conv_last.weight'
                elif k == 'model.10.bias': new_k = 'conv_last.bias'
                else: new_k = k
                new_state_dict[new_k] = v
            torch.save({'params': new_state_dict}, model_path)
            report_progress(f"Conversion complete.")
    except Exception as e:
        report_progress(f"Failed to check/convert model format: {e}")
            
    if os.path.getsize(model_path) < 4000000:
        raise ValueError(f"Corrupted model file at {model_path}.")
    return model_path

def load_realesrgan(model_path, device, model_name="RealESRGAN_x4plus.pth", tile=0, tile_pad=10, pre_pad=10):
    # Dynamic heavy imports
    from realesrgan import RealESRGANer
    
    if model_name == "realesr-general-x4v3.pth":
        from basicsr.archs.srvgg_arch import SRVGGNetCompact
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    else:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        dni_weight=None,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=(device == "cuda"),
        device=device
    )
    return upsampler

def process_image(image_path, plan, model_path, output_dir="outputs"):
    import cv2
    import numpy as np
    import torch
    import gc
    
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
        
    # Extract alpha channel if it exists
    has_alpha = False
    alpha_channel = None
    if len(img.shape) == 3 and img.shape[2] == 4:
        has_alpha = True
        alpha_channel = img[:, :, 3]
        img = img[:, :, :3]
        
    if plan['downscaled_for_processing']:
        ww, wh = plan['working_resolution_width'], plan['working_resolution_height']
        report_progress(f"Phase 8 Strategy: Image is extremely large and soft. Downscaling from {img.shape[1]}x{img.shape[0]} to {ww}x{wh} to optimize workload.")
        img = cv2.resize(img, (ww, wh), interpolation=cv2.INTER_AREA)
        # We don't downscale alpha here, we'll just resize the original alpha to the final output size later
        
    h, w, c = img.shape
    outer_tile_size = plan['tile_size']
    overlap = plan['tile_overlap']
    
    report_progress(f"Generating outer grid tiles (Size: {outer_tile_size}, Overlap: {overlap})...")
    coords = generate_tile_coordinates(w, h, outer_tile_size, overlap)
    total_tiles = len(coords)
    report_progress(f"Total tiles to process: {total_tiles}")
    
    report_progress("Loading AI model into memory...")
    model_name = plan.get('model_name', "RealESRGAN_x4plus.pth")
    upsampler = load_realesrgan(model_path, plan['device'], model_name=model_name, tile=0, tile_pad=overlap)
    
    processed_tiles = []
    current_sub_tile_size = 0 
    
    temp_dir = tempfile.mkdtemp(prefix="restorex_")
    
    if has_alpha:
        np.save(os.path.join(temp_dir, "alpha.npy"), alpha_channel)
    
    for i, coord in enumerate(coords):
        tile_crop = extract_tile(img, coord)
        
        while True:
            try:
                upsampler.tile_size = current_sub_tile_size
                output_crop, _ = upsampler.enhance(tile_crop, outscale=4)
                break
            except RuntimeError as e:
                error_msg = str(e).lower()
                if "out of memory" in error_msg or "oom" in error_msg:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    if current_sub_tile_size == 0:
                        current_sub_tile_size = outer_tile_size // 2
                    else:
                        current_sub_tile_size = current_sub_tile_size // 2
                        
                    if current_sub_tile_size < 32:
                        raise RuntimeError("Unrecoverable Out Of Memory error.")
                        
                    report_progress(f"[OOM RECOVERY] Tile {i+1} OOM. Reducing internal tile to {current_sub_tile_size}...")
                else:
                    raise e
                    
        temp_path = os.path.join(temp_dir, f"tile_{i}.npy")
        np.save(temp_path, output_crop)
        
        processed_tiles.append({'coord': coord, 'path': temp_path})
        
        percentage = ((i + 1) / total_tiles) * 100
        report_progress(f"Tile {i+1}/{total_tiles} ({percentage:.1f}%) processed.")
        
    return processed_tiles, (w, h), temp_dir
