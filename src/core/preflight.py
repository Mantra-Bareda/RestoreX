from typing import Dict, Any, Optional
from dataclasses import dataclass
from src.config.settings import CapabilityProfile

@dataclass
class PreflightResult:
    is_safe: bool
    execution_device: str
    suggested_tier: int
    warning_message: Optional[str]

def assess_task(task: str, requested_tier: int, probe_data: Dict[str, Any], **kwargs) -> PreflightResult:
    is_safe = True
    warning_message = None
    suggested_tier = requested_tier
    
    provider = probe_data['provider']
    execution_device = "cuda" if provider == "CUDA" else "cpu"
    profile = probe_data['profile']
    avail_ram = probe_data['available_ram_gb']

    if task == "restore":
        if requested_tier == 2 and avail_ram < 4.0:
            is_safe = False
            suggested_tier = 1
            warning_message = f"Warning: Available RAM is critically low ({avail_ram:.1f} GB) for Tier 2 restoration. Safety fallback to Tier 1."
            
    elif task == "remove-bg":
        if requested_tier == 2 and avail_ram < 4.0:
            # According to plan, we allow it but give a strong warning
            warning_message = f"Warning: Tier 2 background removal may take 10-15s on {avail_ram:.1f} GB RAM. UI may become unresponsive."
            
    elif task == "gif-process":
        frame_count = kwargs.get('frame_count', 0)
        if provider == "CPU" and frame_count > 60:
            warning_message = f"Notice: Processing {frame_count} frames on CPU will take time. Motion caching is automatically enforced."
        elif provider == "CPU":
            warning_message = "Notice: Processing GIFs on CPU. Motion caching enabled."

    return PreflightResult(
        is_safe=is_safe,
        execution_device=execution_device,
        suggested_tier=suggested_tier,
        warning_message=warning_message
    )
