import signal
import atexit
import gc
import sys
import os

_tmp_files = []

def register_temp_file(path: str):
    _tmp_files.append(path)

def cleanup_resources():
    # Remove tracked temporary files
    for f in _tmp_files:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass
    _tmp_files.clear()
    
    # Run garbage collection
    gc.collect()
    
    # Clear CUDA cache if present
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

def sigint_handler(signum, frame):
    print("\n[WARNING] Process interrupted by user! Cleaning up resources...", flush=True)
    cleanup_resources()
    sys.exit(1)

def init_cleanup():
    atexit.register(cleanup_resources)
    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)
