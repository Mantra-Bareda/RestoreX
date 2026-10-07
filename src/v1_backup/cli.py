import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="RestoreX: Recover visually useful detail from poor-quality photographs."
    )
    
    # In interactive mode, arguments are optional. So we use nargs='?' for positional
    parser.add_argument(
        "input",
        type=str,
        nargs='?',
        help="Path to the input image file."
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Path to the output image file."
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="RealESRGAN_x4plus.pth",
        choices=["RealESRGAN_x4plus.pth", "4x-UltraSharp.pth", "realesr-general-x4v3.pth"],
        help="Upscaling model to use."
    )
    
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        choices=[1, 2, 4],
        help="Target output scale (1 for original size, 2 for 2x, 4 for 4x). Default is 1."
    )
    
    parser.add_argument(
        "--tile",
        type=str,
        default="auto",
        help="Tile size for processing (e.g., 'auto', '256', '512'). Default is 'auto'."
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device ('auto', 'cpu', 'cuda'). Default is 'auto'."
    )
    
    parser.add_argument(
        "--face-enhance",
        action="store_true",
        help="Enable GFPGAN face restoration (optional)."
    )
    
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Apply subtle post-processing (local contrast, slight sharpening, artifact reduction)."
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging."
    )
    
    return parser.parse_args()
