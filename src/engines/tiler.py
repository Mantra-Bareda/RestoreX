import numpy as np
import math

def generate_tile_coordinates(width: int, height: int, tile_size: int, tile_overlap: int):
    coords = []
    y = 0
    while y < height:
        x = 0
        while x < width:
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)
            coords.append({
                'x': x, 'y': y,
                'x_end': x_end, 'y_end': y_end
            })
            if x_end == width:
                break
            x += (tile_size - tile_overlap)
        if y_end == height:
            break
        y += (tile_size - tile_overlap)
    return coords

def extract_tile(img: np.ndarray, coord: dict) -> np.ndarray:
    return img[coord['y']:coord['y_end'], coord['x']:coord['x_end']]

def get_feathering_weight_matrix(h: int, w: int, overlap: int, coord: dict, img_width: int, img_height: int) -> np.ndarray:
    # Linear feathering weight matrix for seamless stitching
    weight = np.ones((h, w, 1), dtype=np.float32)
    
    # Left edge
    if coord['x'] != 0 and w >= overlap:
        fade = np.linspace(0, 1, overlap).reshape(1, overlap, 1)
        weight[:, :overlap] *= fade
        
    # Right edge
    if coord['x_end'] != img_width and w >= overlap:
        fade = np.linspace(1, 0, overlap).reshape(1, overlap, 1)
        weight[:, -overlap:] *= fade
        
    # Top edge
    if coord['y'] != 0 and h >= overlap:
        fade = np.linspace(0, 1, overlap).reshape(overlap, 1, 1)
        weight[:overlap, :] *= fade
        
    # Bottom edge
    if coord['y_end'] != img_height and h >= overlap:
        fade = np.linspace(1, 0, overlap).reshape(overlap, 1, 1)
        weight[-overlap:, :] *= fade
        
    return weight
