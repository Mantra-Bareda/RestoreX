import numpy as np
import cv2
import os
from src.progress import report_progress

def create_tile_mask(x, y, w, h, total_W, total_H, overlap):
    """
    Creates a 2D weight mask for linear blending of overlapping tiles.
    Multiplies X and Y gradients so overlapping corners sum perfectly to 1.0,
    eliminating brightness discontinuities.
    """
    mask_x = np.ones((1, w, 1), dtype=np.float32)
    mask_y = np.ones((h, 1, 1), dtype=np.float32)
    
    if overlap <= 0:
        return np.ones((h, w, 1), dtype=np.float32)
        
    # Left overlap fade
    if x > 0:
        mask_x[:, :overlap, :] = np.linspace(0, 1, overlap).reshape(1, -1, 1)
    # Right overlap fade
    if x + w < total_W:
        mask_x[:, -overlap:, :] = np.linspace(1, 0, overlap).reshape(1, -1, 1)
        
    # Top overlap fade
    if y > 0:
        mask_y[:overlap, :, :] = np.linspace(0, 1, overlap).reshape(-1, 1, 1)
    # Bottom overlap fade
    if y + h < total_H:
        mask_y[-overlap:, :, :] = np.linspace(1, 0, overlap).reshape(-1, 1, 1)
        
    return mask_y * mask_x

def reconstruct_image(processed_tiles, orig_size, plan, sys_info, temp_dir):
    """
    Phase 7: Reconstructs the final high-resolution image from processed tiles on disk.
    Uses memory-mapped arrays if the output resolution is too large for available RAM.
    """
    orig_w, orig_h = orig_size
    scale = 4 # RealESRGAN_x4plus is a 4x model
    
    total_W = orig_w * scale
    total_H = orig_h * scale
    overlap = plan['tile_overlap'] * scale # Upscale the overlap region
    
    required_bytes = total_H * total_W * 4 * 4 # 3 float32 channels (canvas) + 1 float32 channel (weight)
    avail_ram = sys_info.get('ram_available_bytes', 0)
    
    # Memory-conscious processing: use disk-backed memmap if array size > 50% free RAM
    use_memmap = required_bytes > (avail_ram * 0.5)
    
    report_progress(f"Reconstructing {total_W}x{total_H} image...")
    
    if use_memmap:
        report_progress(f"Canvas size ({required_bytes / 1024**3:.2f} GB) exceeds safe RAM limits. Using disk-backed memory mapping...")
        canvas = np.memmap(os.path.join(temp_dir, 'canvas.dat'), dtype='float32', mode='w+', shape=(total_H, total_W, 3))
        weight_sum = np.memmap(os.path.join(temp_dir, 'weight.dat'), dtype='float32', mode='w+', shape=(total_H, total_W, 1))
    else:
        canvas = np.zeros((total_H, total_W, 3), dtype=np.float32)
        weight_sum = np.zeros((total_H, total_W, 1), dtype=np.float32)
        
    total = len(processed_tiles)
    for i, tile_info in enumerate(processed_tiles):
        coord = tile_info['coord']
        path = tile_info['path']
        
        # Scale coordinates
        x, y = coord['x'] * scale, coord['y'] * scale
        w, h = coord['w'] * scale, coord['h'] * scale
        
        # Load upscaled tile from disk
        tile_img = np.load(path).astype(np.float32)
        
        # Generate blending mask
        mask = create_tile_mask(x, y, w, h, total_W, total_H, overlap)
        
        # Accumulate
        canvas[y:y+h, x:x+w] += tile_img * mask
        weight_sum[y:y+h, x:x+w] += mask
        
        # Remove temp tile to save disk space
        try:
            os.remove(path)
        except OSError:
            pass
            
        if (i + 1) % max(1, total // 10) == 0 or i == total - 1:
            percentage = ((i + 1) / total) * 100
            report_progress(f"Reconstructed {i+1}/{total} tiles ({percentage:.1f}%).")
            
    report_progress("Normalizing reconstructed image...")
    # Prevent division by zero on extreme edges
    weight_sum = np.clip(weight_sum, 1e-6, None)
    
    # We apply division block by block to avoid large RAM spikes if using memmap
    final_image = np.zeros((total_H, total_W, 3), dtype=np.uint8)
    
    block_size = 2000
    for y in range(0, total_H, block_size):
        for x in range(0, total_W, block_size):
            y_end = min(y + block_size, total_H)
            x_end = min(x + block_size, total_W)
            
            chunk = canvas[y:y_end, x:x_end] / weight_sum[y:y_end, x:x_end]
            final_image[y:y_end, x:x_end] = np.clip(chunk, 0, 255).astype(np.uint8)
            
    # Cleanup memory mapped files
    if use_memmap:
        del canvas
        del weight_sum
        try:
            os.remove(os.path.join(temp_dir, 'canvas.dat'))
            os.remove(os.path.join(temp_dir, 'weight.dat'))
        except OSError:
            pass
            
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass
        
    return final_image
