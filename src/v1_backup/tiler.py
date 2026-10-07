def generate_tile_coordinates(width, height, tile_size, overlap):
    """
    Generates a list of bounding boxes for tiles covering the image.
    Returns a list of dicts: {'x': x, 'y': y, 'w': w, 'h': h}
    """
    tiles = []
    
    step_x = tile_size - overlap
    step_y = tile_size - overlap
    
    if step_x <= 0 or step_y <= 0:
        raise ValueError("Tile overlap is too large for the chosen tile size.")
        
    # We collect top-left coordinates of each tile
    y = 0
    while y < height:
        x = 0
        while x < width:
            box_x = x
            box_y = y
            box_w = min(tile_size, width - x)
            box_h = min(tile_size, height - y)
            
            # To avoid tiny edge tiles that might lack context, we shift the tile back 
            # so it has size 'tile_size' if we are at the edge, provided the image is larger than tile_size.
            if box_x + tile_size > width and width >= tile_size:
                box_x = width - tile_size
                box_w = tile_size
                
            if box_y + tile_size > height and height >= tile_size:
                box_y = height - tile_size
                box_h = tile_size
                
            coord = {'x': box_x, 'y': box_y, 'w': box_w, 'h': box_h}
            if coord not in tiles:
                tiles.append(coord)
                
            x += step_x
        y += step_y
        
    return tiles

def extract_tile(image_array, coord):
    """
    Extracts the tile crop from the numpy image array.
    """
    x, y, w, h = coord['x'], coord['y'], coord['w'], coord['h']
    return image_array[y:y+h, x:x+w]
