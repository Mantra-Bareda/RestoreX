import os
import psutil
from typing import Dict, Any, Tuple
from src.config.settings import THRESHOLDS, CapabilityProfile

class SystemProbe:
    def __init__(self):
        pass
        
    def _get_ram_gb(self) -> Tuple[float, float]:
        vm = psutil.virtual_memory()
        total_gb = vm.total / (1024 ** 3)
        available_gb = vm.available / (1024 ** 3)
        return total_gb, available_gb

    def _get_cpu_cores(self) -> int:
        return psutil.cpu_count(logical=True) or 1

    def _check_gpu(self) -> Tuple[bool, float, str]:
        has_gpu = False
        vram_gb = 0.0
        provider = "CPU"
        
        try:
            import torch
            if torch.cuda.is_available():
                try:
                    # Safely test allocation
                    dummy = torch.zeros(1, device='cuda')
                    del dummy
                    has_gpu = True
                    provider = "CUDA"
                    # Try to get VRAM
                    try:
                        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    except Exception:
                        vram_gb = 0.0
                except Exception as e:
                    print(f"[Hardware Inspector] CUDA initialization failed: {e}", flush=True)
                    torch.cuda.empty_cache()
                    has_gpu = False
        except ImportError:
            pass

        if not has_gpu:
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                if 'DmlExecutionProvider' in providers:
                    provider = 'DirectML'
                    has_gpu = True
                elif 'OpenVINOExecutionProvider' in providers:
                    provider = 'OpenVINO'
                    has_gpu = True
            except ImportError:
                pass

        return has_gpu, vram_gb, provider

    def determine_capability_profile(self, available_ram_gb: float, vram_gb: float, provider: str) -> str:
        if available_ram_gb < THRESHOLDS.RAM_CONSTRAINED_GB:
            return CapabilityProfile.CONSTRAINED
        if provider == "CPU" and available_ram_gb < THRESHOLDS.RAM_MODERATE_GB:
            return CapabilityProfile.CONSTRAINED
            
        if available_ram_gb < THRESHOLDS.RAM_MODERATE_GB or (provider == "CUDA" and vram_gb > 0 and vram_gb < THRESHOLDS.VRAM_MODERATE_GB):
            return CapabilityProfile.MODERATE
            
        return CapabilityProfile.OPTIMAL

    def calculate_dynamic_tile_size(self, avail_ram_gb: float, cores: int, freq_ghz: float, has_gpu: bool, vram_gb: float, tier: int) -> int:
        """
        Dynamically calculate optimal tile size based on exact hardware capabilities and model tier.
        Uses a fluid scoring system to account for diverse and in-between hardware specifications.
        """
        # Calculate a unified hardware capability score
        # 1. RAM Score (~1.5 points per GB of available RAM)
        score = avail_ram_gb * 1.5
        
        # 2. CPU Score (Cores * Frequency gives total raw compute capacity)
        compute_units = cores * freq_ghz
        score += compute_units * 0.5
        
        # 3. GPU Score (VRAM heavily influences tiling capacity)
        if has_gpu and vram_gb > 0:
            score += vram_gb * 5.0
            
        # Map the continuous score to a standard base tile size
        if score >= 30.0:
            base_tile = 1024
        elif score >= 18.0:
            base_tile = 512
        elif score >= 9.0:
            base_tile = 256
        else:
            base_tile = 128
            
        # Adjust based on Model Tier (Heavy model requires smaller tiles to prevent OOM/Hangs)
        if tier == 2:
            base_tile = base_tile // 2
            
        # Hard clamps to prevent absurdly small or large tiles
        return max(128, min(1024, base_tile))

    def probe(self) -> Dict[str, Any]:
        total_ram, avail_ram = self._get_ram_gb()
        cpu_cores = self._get_cpu_cores()
        has_gpu, vram_gb, provider = self._check_gpu()
        
        # Get CPU Frequency (fall back to a reasonable guess if unavailable)
        try:
            freq = psutil.cpu_freq()
            freq_ghz = freq.current / 1000.0 if freq else 2.5
            if freq_ghz < 0.5: # sometimes current is very low when idle, use max if available
                freq_ghz = freq.max / 1000.0 if freq.max > 0 else 2.5
        except Exception:
            freq_ghz = 2.5
            
        profile = self.determine_capability_profile(avail_ram, vram_gb, provider)
        
        return {
            "total_ram_gb": total_ram,
            "available_ram_gb": avail_ram,
            "cpu_cores": cpu_cores,
            "cpu_freq_ghz": freq_ghz,
            "has_gpu": has_gpu,
            "vram_gb": vram_gb,
            "provider": provider,
            "profile": profile
        }
