import sys
import os
import cv2

from src.core.cleanup import init_cleanup, cleanup_resources, register_temp_file
from src.core.system_probe import SystemProbe
from src.core.preflight import assess_task
from src.cli.parser import get_parser
from src.cli.interactive import run_interactive_wizard

def main():
    init_cleanup()
    
    probe = SystemProbe()
    probe_data = probe.probe()
    
    parser = get_parser()
    
    # If no arguments provided or --interactive is set, run wizard
    if len(sys.argv) == 1 or '--interactive' in sys.argv:
        args_dict = run_interactive_wizard(probe_data)
        class Args:
            pass
        args = Args()
        for k, v in args_dict.items():
            setattr(args, k, v)
        setattr(args, 'tile', None)
        setattr(args, 'format', None)
    else:
        args = parser.parse_args()
        if not args.input or not args.output:
            parser.print_help()
            sys.exit(1)
            
    # Output Format Rules
    out_ext = os.path.splitext(args.output)[1].lower()
    is_gif = args.input.lower().endswith('.gif')
    
    if args.mode in ['remove-bg', 'all'] and not is_gif and out_ext not in ['.png', '.webp']:
        print(f"[WARNING] Background removal requires transparency. Changing output from {out_ext} to .png", flush=True)
        args.output = os.path.splitext(args.output)[0] + ".png"
        
    try:
        if is_gif:
            print("[INFO] Initializing GIF Engine...", flush=True)
            from src.engines.gif_engine import GifEngine
            
            preflight = assess_task("gif-process", args.tier or 1, probe_data)
            if preflight.warning_message:
                print(preflight.warning_message, flush=True)
                
            out_format = args.format or out_ext.strip('.')
            if out_format not in ['gif', 'webp']:
                out_format = 'gif'
                
            # If "all" mode on GIF, fallback to upscale since gif engine doesn't chain yet
            action = 'upscale' if args.mode not in ['upscale', 'remove-bg'] else args.mode
            if args.mode == 'remove-bg':
                action = 'remove-bg'
                
            engine = GifEngine(action, preflight.execution_device, probe_data['cpu_cores'])
            engine.process_gif(args.input, args.output, args.scale, out_format)
            
        else:
            img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Could not read input image: {args.input}")
                
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4 and args.mode == 'restore':
                # Convert RGBA to RGB for standard models
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                
            result_img = img
            
            if args.mode in ['restore', 'all']:
                preflight = assess_task("restore", args.tier or 2, probe_data)
                if preflight.warning_message:
                    print(preflight.warning_message, flush=True)
                if not preflight.is_safe:
                    print("[INFO] Safety override active. Using suggested tier.", flush=True)
                    tier = preflight.suggested_tier
                else:
                    tier = args.tier or 2
                    
                model_name = args.model
                if not model_name:
                    model_name = "RealESRGAN_x4plus.pth" if tier == 2 else "realesr-general-x4v3.pth"
                    
                print(f"[INFO] Initializing Restore Engine with {model_name}...", flush=True)
                from src.engines.restore_engine import RestoreXEngine
                engine = RestoreXEngine(model_name, preflight.execution_device)
                
                from src.config.settings import TILES
                
                # Calculate dynamic tile size using system probe
                calc_tile_sz = probe.calculate_dynamic_tile_size(
                    probe_data['available_ram_gb'],
                    probe_data['cpu_cores'],
                    probe_data['cpu_freq_ghz'],
                    probe_data['has_gpu'],
                    probe_data['vram_gb'],
                    tier
                )
                
                tile_sz = getattr(args, 'tile', None) or calc_tile_sz
                print(f"[INFO] Using chunk/tile size: {tile_sz}", flush=True)
                
                result_img = engine.run_restoration(result_img, tile_size=tile_sz, tile_overlap=TILES.TILE_OVERLAP, scale=args.scale)
                if getattr(args, "face_enhance", False):
                    print("[WARNING] Face Enhancement (GFPGAN) is not yet implemented in V2.", flush=True)
                
            if args.mode in ['remove-bg', 'all']:
                preflight = assess_task("remove-bg", args.tier or 2, probe_data)
                if preflight.warning_message:
                    print(preflight.warning_message, flush=True)
                    
                tier = args.tier or preflight.suggested_tier
                model_name = args.model
                if not model_name:
                    model_name = "bria-rmbg-1.4.onnx" if tier == 2 else "u2netp.onnx"
                    
                print(f"[INFO] Initializing BG Remover with {model_name}...", flush=True)
                from src.engines.bg_remover import BackgroundRemoverEngine
                engine = BackgroundRemoverEngine(model_name, preflight.execution_device, probe_data['cpu_cores'])
                
                result_img = engine.run_removal(result_img, keep_text=args.keep_text)
                
            print(f"[INFO] Saving output to {args.output}...", flush=True)
            cv2.imwrite(args.output, result_img)
            
    except Exception as e:
        print(f"\n[ERROR] Processing Failed: {str(e)}", flush=True)
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            print("Remediation: CUDA Out of Memory: Try re-running with --tier 1 or --tile 128", flush=True)
        sys.exit(1)
        
    cleanup_resources()
    print("[INFO] Done.", flush=True)

if __name__ == "__main__":
    main()
