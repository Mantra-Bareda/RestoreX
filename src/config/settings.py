from dataclasses import dataclass

@dataclass(frozen=True)
class HardwareThresholds:
    RAM_CONSTRAINED_GB: float = 4.0
    RAM_MODERATE_GB: float = 8.0
    VRAM_CONSTRAINED_GB: float = 2.0
    VRAM_MODERATE_GB: float = 4.0

@dataclass(frozen=True)
class TileSettings:
    TIER1_TILE_SIZE: int = 256
    TIER1_TILE_SIZE_CONSTRAINED: int = 128
    TIER2_TILE_SIZE: int = 256
    TIER2_TILE_SIZE_CONSTRAINED: int = 128
    TILE_OVERLAP: int = 48

class CapabilityProfile:
    CONSTRAINED = "CONSTRAINED"
    MODERATE = "MODERATE"
    OPTIMAL = "OPTIMAL"

THRESHOLDS = HardwareThresholds()
TILES = TileSettings()
