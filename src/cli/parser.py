import argparse

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RestoreX v2 CLI - Image/GIF Restoration and Matting")
    
    parser.add_argument('input', type=str, nargs='?', help='Path to input image, GIF, or folder')
    parser.add_argument('-o', '--output', type=str, help='Required output file or destination directory')
    
    parser.add_argument('--mode', type=str, choices=['restore', 'remove-bg', 'all'], default='restore', help='Workflow mode')
    
    parser.add_argument('--tier', type=int, choices=[1, 2], help='1 (lightweight) or 2 (heavy). Auto if omitted')
    parser.add_argument('--model', type=str, help='Manual model override')
    
    parser.add_argument('--scale', type=int, default=1, choices=[1, 2, 4], help='Output scaling factor')
    parser.add_argument('--tile', type=int, help='Manual tile size')
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'], default='auto', help='Execution device')
    
    parser.add_argument('--keep-text', action='store_true', help='Enable typography protection during BG removal')
    parser.add_argument('--format', type=str, choices=['png', 'jpg', 'webp', 'gif'], help='Output format override')
    parser.add_argument('--face-enhance', action='store_true', help='Toggle face enhancement')
    
    parser.add_argument('--verbose', action='store_true', help='Detailed execution trace')
    parser.add_argument('--interactive', action='store_true', help='Force interactive terminal wizard')
    
    return parser
