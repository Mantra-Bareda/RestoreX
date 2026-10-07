import os
import sys

def get_input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        sys.exit(0)

def run_interactive_wizard(probe_data: dict) -> dict:
    print("=" * 60, flush=True)
    print(" " * 15 + "RESTOREX v2 INTERACTIVE", flush=True)
    print("=" * 60, flush=True)
    
    print("\n[SYSTEM HARDWARE SCAN]", flush=True)
    print(f"  CPU Cores: {probe_data['cpu_cores']}", flush=True)
    print(f"  Available RAM: {probe_data['available_ram_gb']:.1f} GB", flush=True)
    if probe_data['has_gpu']:
        print(f"  GPU / Provider: {probe_data['provider']} ({probe_data['vram_gb']:.1f} GB VRAM)", flush=True)
    else:
        print("  GPU: None (CPU Fallback)", flush=True)
        
    print(f"  Assessed Profile: {probe_data['profile']}", flush=True)
    
    # Input Path
    input_path = ""
    print("\nSelect an input file (Image or GIF):\n")
    print("1. Enter file path manually")
    print("2. Open file dialog (GUI)\n")
    
    choice = get_input("Select:\n> ").strip()
    if choice == '2':
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            input_path = filedialog.askopenfilename(title="Select Image or GIF")
        except Exception:
            print("GUI dialog not available. Please enter path manually.")
            choice = '1'
            
    if choice == '1' or not input_path:
        input_path = get_input("\nEnter path to input image or GIF: ").strip()
        
    if not input_path or not os.path.exists(input_path):
        print("Invalid input path. Exiting.", flush=True)
        sys.exit(1)
        
    # Workflow
    print("\nSelect Workflow:\n")
    print("1. Detail Restoration")
    print("2. Background Removal")
    print("3. Both (Restore + Remove BG)")
    print("4. Animated GIF Processing\n")
    
    mode_choice = get_input("Choice [1-4] (default 1): ").strip()
    mode_map = {'1': 'restore', '2': 'remove-bg', '3': 'all', '4': 'restore'}
    mode = mode_map.get(mode_choice, 'restore')
    
    is_gif = mode_choice == '4' or input_path.lower().endswith('.gif')
    if is_gif:
        print("[INFO] GIF format detected/selected.", flush=True)
        
    # Model & Tier selection
    tier = 2 if probe_data['profile'] == 'OPTIMAL' else 1
    model = None
    
    if mode in ['restore', 'all'] and not is_gif:
        print("\nSelect AI Model for Upscaling:\n")
        print("1. realesr-general-x4v3 (Low / Lightweight)")
        print("2. RealESRGAN_x4plus (Standard)")
        print("3. 4x-UltraSharp (High / Heavy)\n")
        
        m_choice = get_input("Select Model [1-3] (default 2): ").strip()
        if m_choice == '1':
            model = "realesr-general-x4v3.pth"
            tier = 1
        elif m_choice == '3':
            model = "4x-UltraSharp.pth"
            tier = 2
        else:
            model = "RealESRGAN_x4plus.pth"
            tier = 2
            
    elif mode in ['remove-bg', 'all'] and not is_gif:
        print("\nSelect AI Model for Background Removal:\n")
        print("1. u2netp (Low / Lightweight)")
        print("2. bria-rmbg-1.4 (Standard / High)\n")
        
        m_choice = get_input("Select Model [1-2] (default 2): ").strip()
        if m_choice == '1':
            model = "u2netp.onnx"
            tier = 1
        else:
            model = "bria-rmbg-1.4.onnx"
            tier = 2
            
    if probe_data['profile'] == 'CONSTRAINED' and tier == 2:
        print("\n[WARNING] Your machine has constrained memory/compute.", flush=True)
        print("The selected model is Heavy. It may cause crashes or run very slowly.", flush=True)
        t_choice = get_input("Proceed anyway? (y/N): ").strip().lower()
        if t_choice != 'y':
            print("Aborting to prevent crash.", flush=True)
            sys.exit(1)
            
    # Output path
    output_path = get_input(f"\nEnter output file path (default: out_{os.path.basename(input_path)}): ").strip()
    if not output_path:
        output_path = f"out_{os.path.basename(input_path)}"
        
    # Extra flags
    keep_text = False
    scale = 1
    
    if mode in ['remove-bg', 'all'] and not is_gif:
        kt = get_input("Enable text preservation? (y/N): ").strip().lower()
        keep_text = (kt == 'y')
        
    if mode in ['restore', 'all'] or is_gif:
        sc = get_input("Output scale (1, 2, 4) [default 1]: ").strip()
        if sc in ['1', '2', '4']:
            scale = int(sc)
            
    face_enhance = False
    if mode in ['restore', 'all'] and not is_gif:
        fe = get_input("Enable Face Enhancement (GFPGAN)? (y/N): ").strip().lower()
        face_enhance = (fe == 'y')
            
    return {
        'input': input_path,
        'output': output_path,
        'mode': mode,
        'tier': tier,
        'model': model,
        'keep_text': keep_text,
        'scale': scale,
        'face_enhance': face_enhance,
        'device': 'auto'
    }
