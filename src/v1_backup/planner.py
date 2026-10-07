import math

def create_plan(sys_info, img_info, args):
    """
    Calculates the optimal processing configuration based on system info and image properties.
    Prioritizes conservative, safe defaults to prevent Out-Of-Memory errors.
    """
    plan = {}
    
    # 1. Device & Precision
    from src.system_info import determine_execution_device
    device = determine_execution_device(sys_info, args.device)
    plan['device'] = device
    plan['precision'] = "fp16" if device == "cuda" else "fp32"
    plan['batch_size'] = 1
    plan['tile_overlap'] = 48 
    
    # 2. Tile Size Calculation
    tile_size = 256 # Safe default
    ram_gb = sys_info.get('ram_available_bytes', 0) / (1024**3)
    
    if device == "cuda":
        vram_bytes = 0
        if sys_info['gpus']:
            gpu = sys_info['gpus'][0]
            vram_bytes = gpu.get('vram_free_bytes')
            if vram_bytes is None:
                vram_bytes = gpu.get('vram_total_bytes', 0)
        vram_gb = vram_bytes / (1024**3)
        
        if vram_gb < 2.0:
            tile_size = 128
        elif vram_gb < 4.0:
            tile_size = 256
        elif vram_gb < 8.0:
            tile_size = 384
        else:
            tile_size = 512
    else:
        if ram_gb < 4.0:
            tile_size = 128
        elif ram_gb < 8.0:
            tile_size = 256
        else:
            tile_size = 384
            
    if args.tile and args.tile.lower() != "auto":
        try:
            user_tile = int(args.tile)
            if user_tile > 0:
                tile_size = user_tile
        except ValueError:
            pass
            
    # Lightweight model is much smaller and faster, we can process larger chunks at once
    if hasattr(args, 'model') and args.model == 'realesr-general-x4v3.pth':
        # Safely double the tile size but cap it to avoid breaking memory bounds entirely
        tile_size = min(tile_size * 2, 800)
            
    plan['tile_size'] = tile_size
    plan['model_name'] = getattr(args, 'model', 'RealESRGAN_x4plus.pth')
    
    # 3. Phase 8: High-Resolution Image Strategy
    pixels = img_info['pixels']
    w, h = img_info['width'], img_info['height']
    
    working_w, working_h = w, h
    downscaled = False
    
    # If the image is insanely large (>25MP), processing it at 1:1 is a waste of time.
    # If it's quite large (>12MP), we only process it 1:1 if it actually has sharp, high-frequency detail.
    MASSIVE_THRESHOLD = 25_000_000
    LARGE_THRESHOLD = 12_000_000
    TARGET_PIXELS = 10_000_000 # Downscale to ~10MP working resolution
    
    laplacian_var = img_info.get('laplacian_variance', 0)
    
    if pixels > MASSIVE_THRESHOLD:
        downscaled = True
    elif pixels > LARGE_THRESHOLD:
        # A variance < 150 typically means soft, blurry, or heavily processed/denoised by a phone.
        if laplacian_var < 150: 
            downscaled = True
            
    if downscaled:
        scale_factor = math.sqrt(TARGET_PIXELS / float(pixels))
        working_w = int(w * scale_factor)
        working_h = int(h * scale_factor)
        
    plan['working_resolution_width'] = working_w
    plan['working_resolution_height'] = working_h
    plan['original_width'] = w
    plan['original_height'] = h
    plan['downscaled_for_processing'] = downscaled
    
    return plan
