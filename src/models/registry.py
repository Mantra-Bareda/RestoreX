import os
import urllib.request
from typing import Dict, Any

MODELS: Dict[str, Dict[str, Any]] = {
    "RealESRGAN_x4plus.pth": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "tier": 2,
        "min_size": 10000000
    },
    "4x-UltraSharp.pth": {
        "url": "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth",
        "tier": 2,
        "min_size": 10000000
    },
    "realesr-general-x4v3.pth": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "tier": 1,
        "min_size": 4000000
    },
    "u2netp.onnx": {
        "url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
        "tier": 1,
        "min_size": 1000000
    },
    "silueta.onnx": {
        "url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx",
        "tier": 1,
        "min_size": 40000000
    },
    "bria-rmbg-1.4.onnx": {
        "url": "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model.onnx",
        "tier": 2,
        "min_size": 100000000
    },
    "BiRefNet-lite.onnx": {
        "url": "https://github.com/ZhengPeng7/BiRefNet/releases/download/v1.0/BiRefNet-lite.onnx", # Dummy URL if not actual
        "tier": 2,
        "min_size": 150000000
    },
    "realesr-animevideov3.pth": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        "tier": 1,
        "min_size": 15000000
    }
}

def convert_to_basicsr(model_path: str, model_name: str) -> None:
    if model_path.endswith('.onnx'):
        return
    try:
        import torch
        state_dict = torch.load(model_path, map_location='cpu')
        if 'params' not in state_dict and 'params_ema' not in state_dict:
            print(f"[{model_name}] Converting to BasicSR format...", flush=True)
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
            print(f"[{model_name}] Conversion complete.", flush=True)
    except Exception as e:
        print(f"Failed to check/convert model format: {e}", flush=True)

def verify_and_download_model(model_name: str, models_dir: str = "models") -> str:
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found in registry.")
    
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, model_name)
    metadata = MODELS[model_name]
    
    if not os.path.exists(model_path) or os.path.getsize(model_path) < metadata["min_size"]:
        print(f"[INFO] Downloading {model_name}...", flush=True)
        try:
            req = urllib.request.Request(metadata["url"], headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(model_path, 'wb') as out_file:
                chunk_size = 8192
                total_size = int(response.info().get('Content-Length', 0))
                downloaded = 0
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (1024 * 1024 * 5) < chunk_size:
                        print(f"  ... {downloaded / (1024*1024):.1f} MB downloaded", flush=True)
            print(f"[INFO] Download complete: {model_name}", flush=True)
        except Exception as e:
            if os.path.exists(model_path):
                os.remove(model_path)
            raise RuntimeError(f"Failed to download model {model_name}: {e}")
            
    if os.path.getsize(model_path) < metadata["min_size"]:
        raise ValueError(f"Corrupted model file at {model_path}.")
        
    convert_to_basicsr(model_path, model_name)
    
    return model_path

def get_tier(model_name: str) -> int:
    return MODELS.get(model_name, {}).get("tier", 2)
