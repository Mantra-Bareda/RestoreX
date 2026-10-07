import os
import sys
import time
import platform
import subprocess
import shutil

from src.system_info import get_system_info
from src.image_analyzer import analyze_image
from src.planner import create_plan
from src.inference import verify_and_download_model, process_image
from src.reconstruction import reconstruct_image
from src.postprocess import apply_postprocessing
from src.utils import restore_resolution, save_image, patch_basicsr_compat
from src.progress import report_progress

# Global state for cached system info so we don't re-analyze every menu jump
SYS_INFO = None

class DummyArgs:
    """A dummy class to simulate argparse namespace in interactive mode"""
    pass

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("╔══════════════════════════════════════════════╗")
    print("║              RESTOREX                   ║")
    print("║       AI Image Detail Restoration            ║")
    print("╚══════════════════════════════════════════════╝\n")

def get_input(prompt):
    try:
        return input(prompt).strip()
    except (KeyboardInterrupt, EOFError):
        print("\n[WARNING] Process interrupted by user.")
        sys.exit(0)

def auto_output_name(input_path):
    directory = os.path.dirname(input_path)
    base = os.path.basename(input_path)
    name, ext = os.path.splitext(base)
    
    count = 0
    while True:
        suffix = f"_{count}" if count > 0 else ""
        out_name = f"{name}_restored{suffix}{ext}"
        out_path = os.path.join(directory, out_name)
        if not os.path.exists(out_path):
            return out_path
        count += 1

def open_file_dialog():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        # Bring to front
        root.attributes('-topmost', True)
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[
                ("Image files", "*.jpg;*.jpeg;*.png;*.webp"),
                ("All files", "*.*")
            ]
        )
        root.destroy()
        return file_path
    except Exception as e:
        print(f"\n[ERROR] File dialog is not supported on this environment: {e}")
        time.sleep(2)
        return ""

def run_interactive():
    global SYS_INFO
    clear_screen()
    print_header()
    
    print("[INFO] Initializing...")
    if SYS_INFO is None:
        SYS_INFO = get_system_info()
        
    cpu_cores = SYS_INFO.get('cpu_cores_physical', 'Unknown')
    ram_gb = SYS_INFO.get('ram_total_bytes', 0) / (1024**3)
    gpus = SYS_INFO.get('gpus', [])
    gpu_name = gpus[0]['name'] if gpus else "None"
    backend = "CUDA" if SYS_INFO.get('cuda_available') else "CPU"
    
    print("\nSystem:")
    print(f"  CPU: {cpu_cores} Cores")
    print(f"  RAM: {ram_gb:.1f} GB")
    print(f"  GPU: {gpu_name}")
    print(f"  Backend: {backend}\n")
    
    main_menu()

def main_menu():
    while True:
        print("What would you like to do?\n")
        print("  1. Upscale / Restore Image")
        print("  2. Exit\n")
        
        choice = get_input("Select an option:\n> ")
        if choice == '1':
            image_selection_menu()
        elif choice == '2':
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid selection.\n")

def image_selection_menu():
    while True:
        clear_screen()
        print_header()
        print("How would you like to provide the image?\n")
        print("1. Enter file path manually")
        print("2. Select image using file dialog")
        print("3. Back\n")
        
        choice = get_input("Select:\n> ")
        
        if choice == '1':
            path = get_input("Enter image file path:\n> ")
            # Remove surrounding quotes if dragged into terminal
            path = path.strip("'\"")
            if validate_and_confirm_image(path):
                return
        elif choice == '2':
            path = open_file_dialog()
            if path and validate_and_confirm_image(path):
                return
        elif choice == '3':
            clear_screen()
            print_header()
            return
            
def validate_and_confirm_image(path):
    if not os.path.exists(path):
        print(f"\n[ERROR] File not found: {path}")
        time.sleep(2)
        return False
        
    valid_exts = ['.jpg', '.jpeg', '.png', '.webp']
    _, ext = os.path.splitext(path)
    if ext.lower() not in valid_exts:
        print(f"\n[ERROR] Unsupported image format: {ext}")
        print("Supported formats: JPG / JPEG, PNG, WEBP")
        time.sleep(3)
        return False
        
    try:
        img_info = analyze_image(path)
    except Exception as e:
        print(f"\n[ERROR] Failed to read image: {e}")
        time.sleep(2)
        return False
        
    while True:
        clear_screen()
        print_header()
        print("✓ Image selected\n")
        
        # Display shortened path if too long
        display_path = path if len(path) < 60 else "..." + path[-57:]
        
        print("Path:")
        print(f" {display_path}\n")
        print("Name:")
        print(f" {os.path.basename(path)}\n")
        print("Size:")
        print(f" {img_info['width']} × {img_info['height']}\n")
        print("Format:")
        print(f" {img_info['format']}\n")
        
        print("What would you like to do?\n")
        print("1. Continue")
        print("2. Select another image")
        print("3. Back\n")
        
        choice = get_input("Select:\n> ")
        
        if choice == '1':
            settings_menu(path, img_info)
            return True
        elif choice == '2':
            return False # Return to image selection
        elif choice == '3':
            return False

def settings_menu(img_path, img_info):
    while True:
        clear_screen()
        print_header()
        print("Image:")
        print(os.path.basename(img_path))
        print(f"Resolution: {img_info['width']} × {img_info['height']}\n")
        
        print("Select AI Model:\n")
        print("1. Standard (RealESRGAN_x4plus) - Good balance of speed and quality")
        print("2. Heavy (4x-UltraSharp) - High fidelity, very sharp, slower")
        print("3. Lightweight (realesr-general-x4v3) - Very fast, uses larger chunks, lower detail")
        print("4. Back\n")
        
        model_choice = get_input("Select:\n> ")
        model_name = "RealESRGAN_x4plus.pth"
        
        if model_choice == '1': model_name = "RealESRGAN_x4plus.pth"
        elif model_choice == '2': model_name = "4x-UltraSharp.pth"
        elif model_choice == '3': model_name = "realesr-general-x4v3.pth"
        elif model_choice == '4': return
        else: continue
        
        while True:
            clear_screen()
            print_header()
            print("Select restoration scale:\n")
            print("1. 1× — Restore detail, keep original resolution (Recommended)")
            print("2. 2× — Restore + 2× internal/output scaling")
            
            # Show 4x only if it's not the lightweight model
            if model_name != "realesr-general-x4v3.pth":
                print("3. 4× — Restore + 4× internal/output scaling")
                
            print("4. Back\n")
            
            choice = get_input("Select:\n> ")
            
            scale = 1
            if choice == '1': scale = 1
            elif choice == '2': scale = 2
            elif choice == '3' and model_name != "realesr-general-x4v3.pth": scale = 4
            elif choice == '4': break # go back to model selection
            else: continue
                
            while True:
                clear_screen()
                print_header()
                print("Select optional enhancements:\n")
                print("1. Standard (No extras)")
                print("2. Add Face Enhancement (GFPGAN)")
                print("3. Add Post-Processing Refinement (Contrast/Artifact cleanup)")
                print("4. Add Both")
                print("5. Back\n")
                
                opt_choice = get_input("Select:\n> ")
                face_enhance = False
                refine = False
                
                if opt_choice == '1': pass
                elif opt_choice == '2': face_enhance = True
                elif opt_choice == '3': refine = True
                elif opt_choice == '4': face_enhance = True; refine = True
                elif opt_choice == '5': break
                else: continue
                
                # Setup arguments and auto-plan
                args = DummyArgs()
                args.input = img_path
                args.output = auto_output_name(img_path)
                args.scale = scale
                args.device = "auto"
                args.tile = "auto"
                args.model = model_name
                args.face_enhance = face_enhance
                args.refine = refine
                
                plan = create_plan(SYS_INFO, img_info, args)
                
                confirm_processing(args, img_info, plan)
                return # Once completed or cancelled from confirmation, unwind to main menu

def confirm_processing(args, img_info, plan):
    while True:
        clear_screen()
        print_header()
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        print("Ready to restore:\n")
        print("File:")
        print(f" {os.path.basename(args.input)}\n")
        print("Input:")
        print(f" {img_info['width']} × {img_info['height']}\n")
        print("Mode:")
        print(f" Detail Restoration ({args.scale}×)")
        if args.face_enhance: print(" + Face Enhancement")
        if args.refine: print(" + Post-Processing Refinement")
        print()
        
        print("Device:")
        print(f" {plan['device'].upper()}\n")
        print("Tile:")
        print(f" {plan['tile_size']} × {plan['tile_size']}\n")
        
        est = "Normal"
        if plan['device'] == 'cpu' or img_info['pixels'] > 12000000:
            est = "Very slow"
            
        print("Estimated processing:")
        print(f" {est}")
        if est == "Very slow":
            print(" [WARNING] This image may take a long time to process on this system.\n")
        else:
            print()
            
        print("Output:")
        print(f" {os.path.basename(args.output)}\n")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        
        print("Start processing?\n")
        print("1. Yes")
        print("2. Cancel\n")
        
        choice = get_input("Select:\n> ")
        if choice == '1':
            execute_processing(args, img_info, plan)
            return
        elif choice == '2':
            return

def execute_processing(args, img_info, plan):
    clear_screen()
    print_header()
    print("Starting restoration...\n")
    
    start_time = time.time()
    temp_dir = None
    
    # Patch basicsr for newer PyTorch versions before doing any ML imports
    patch_basicsr_compat()
    
    try:
        report_progress("[1/6] Verifying AI models...")
        model_name = plan.get('model_name', "RealESRGAN_x4plus.pth")
        model_path = verify_and_download_model(model_name=model_name, models_dir="models")
        
        gfpgan_path = None
        if args.face_enhance:
            from src.face_enhance import verify_and_download_gfpgan
            gfpgan_path = verify_and_download_gfpgan(models_dir="models")
            
        report_progress("\n[2/6] Starting AI processing pipeline (Tiling & Inference)...")
        print(f"Device: {plan['device'].upper()}")
        print(f"Tile size: {plan['tile_size']}")
        processed_tiles, orig_dims, temp_dir = process_image(args.input, plan, model_path)
        
        report_progress("\n[3/6] Stitching and blending tiles...")
        final_image = reconstruct_image(processed_tiles, orig_dims, plan, SYS_INFO, temp_dir)
        
        if args.face_enhance and gfpgan_path:
            report_progress("\n[4/6] Applying Face Enhancement...")
            from src.face_enhance import apply_face_enhancement
            final_image = apply_face_enhancement(final_image, gfpgan_path, plan['device'])
        else:
            report_progress("\n[4/6] Face Enhancement skipped.")
            
        if args.refine:
            report_progress("\n[5/6] Applying subtle post-processing refinements...")
            final_image = apply_postprocessing(final_image, args)
        else:
            report_progress("\n[5/6] Post-processing skipped.")
            
        report_progress("\n[6/6] Finalizing and saving image...")
        final_image = restore_resolution(final_image, plan['original_width'], plan['original_height'], target_scale=args.scale)
        
        # Restore alpha channel if present
        import numpy as np
        import cv2
        alpha_path = os.path.join(temp_dir, "alpha.npy") if temp_dir else None
        if alpha_path and os.path.exists(alpha_path):
            report_progress("Restoring transparent background (alpha channel)...")
            alpha = np.load(alpha_path)
            fh, fw = final_image.shape[:2]
            alpha_resized = cv2.resize(alpha, (fw, fh), interpolation=cv2.INTER_LANCZOS4)
            b, g, r = cv2.split(final_image)
            final_image = cv2.merge((b, g, r, alpha_resized))
            
        save_image(final_image, args.output)
        
        elapsed = time.time() - start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        
        completion_screen(args, img_info, f"{mins}m {secs}s", plan['device'])
        
    except KeyboardInterrupt:
        print("\n[WARNING] Processing interrupted by user.")
        cleanup_temp(temp_dir)
        handle_error_menu()
    except Exception as e:
        print(f"\n[ERROR] Restoration failed.\n\nReason:\n{e}\n")
        cleanup_temp(temp_dir)
        handle_error_menu()

def cleanup_temp(temp_dir):
    if temp_dir and os.path.exists(temp_dir):
        print("Cleaning up temporary files...")
        shutil.rmtree(temp_dir, ignore_errors=True)

def handle_error_menu():
    print("\n1. Return to menu")
    print("2. Exit\n")
    choice = get_input("Select:\n> ")
    if choice == '2':
        sys.exit(1)

def completion_screen(args, img_info, processing_time, device):
    while True:
        clear_screen()
        print_header()
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("✓ RESTORATION COMPLETE")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        
        print("Input:")
        print(f" {os.path.basename(args.input)}\n")
        print("Output:")
        print(f" {os.path.basename(args.output)}\n")
        print("Original Resolution:")
        print(f" {img_info['width']} × {img_info['height']}\n")
        print("Final Scale:")
        print(f" {args.scale}×\n")
        print("Processing time:")
        print(f" {processing_time}\n")
        print("Device:")
        print(f" {device.upper()}\n")
        
        print("What next?\n")
        print("1. Open output location")
        print("2. Restore another image")
        print("3. Return to main menu")
        print("4. Exit\n")
        
        choice = get_input("Select:\n> ")
        
        if choice == '1':
            out_dir = os.path.abspath(os.path.dirname(args.output))
            if platform.system() == "Windows":
                os.startfile(out_dir)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", out_dir])
            else:
                subprocess.Popen(["xdg-open", out_dir])
        elif choice == '2':
            image_selection_menu()
            return
        elif choice == '3':
            return
        elif choice == '4':
            sys.exit(0)
