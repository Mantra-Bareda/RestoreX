import psutil
import json

def get_system_info():
    """
    Gathers system capabilities and returns them as a structured dictionary.
    Dynamic imports are used for heavy ML libraries to keep CLI launch fast.
    """
    sys_info = {}
    
    # CPU Info
    sys_info['cpu_cores_physical'] = psutil.cpu_count(logical=False)
    sys_info['cpu_cores_logical'] = psutil.cpu_count(logical=True)
    
    # RAM Info
    vm = psutil.virtual_memory()
    sys_info['ram_total_bytes'] = vm.total
    sys_info['ram_available_bytes'] = vm.available
    
    # PyTorch & Backend Info
    try:
        import torch
        sys_info['pytorch_version'] = torch.__version__
        sys_info['cuda_available'] = torch.cuda.is_available()
        
        # GPU Info
        sys_info['gpus'] = []
        if sys_info['cuda_available']:
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                gpu_info = {
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'vram_total_bytes': torch.cuda.get_device_properties(i).total_memory,
                }
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if i < len(gpus):
                        gpu_info['vram_free_bytes'] = gpus[i].memoryFree * 1024 * 1024
                except ImportError:
                    gpu_info['vram_free_bytes'] = None
                    
                sys_info['gpus'].append(gpu_info)
    except ImportError:
        # If torch isn't installed yet, fail gracefully for the interactive UI
        sys_info['pytorch_version'] = "Not Installed"
        sys_info['cuda_available'] = False
        sys_info['gpus'] = []

    return sys_info

def determine_execution_device(sys_info, requested_device="auto"):
    if requested_device == "cuda" and not sys_info.get('cuda_available', False):
        return "cpu"
    if requested_device == "cpu":
        return "cpu"
    if sys_info.get('cuda_available', False):
        return "cuda"
    return "cpu"

def get_system_report_json(sys_info, device):
    report = {
        "system_info": sys_info,
        "selected_device": device
    }
    return json.dumps(report, indent=2)
