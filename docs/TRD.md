# TRD — RestoreX CLI

## 1. Architecture

```text
CLI
 │
 ▼
Argument Parser
 │
 ▼
System Analyzer
 │
 ├── CPU
 ├── RAM
 ├── GPU
 ├── VRAM
 └── Backend availability
 │
 ▼
Processing Planner
 │
 ├── Image analysis
 ├── Optional downscale
 ├── Tile size
 ├── overlap
 ├── batch size
 └── execution device
 │
 ▼
Preprocessor
 │
 ▼
Mandatory Tile Engine
 │
 ▼
Real-ESRGAN x4plus
 │
 ▼
Tile Reconstruction
 │
 ▼
Optional Face Restoration
 │
 ▼
Post-processing
 │
 ▼
Optional resize to target dimensions
 │
 ▼
Output
```

## 2. Language

**Python 3.10+ recommended.**

The implementation should remain compatible with CPU-only systems.

## 3. Main Dependencies

* PyTorch
* Real-ESRGAN
* BasicSR
* OpenCV
* Pillow
* NumPy
* psutil
* GPUtil or direct GPU detection where appropriate
* optional GFPGAN
* argparse

Real-ESRGAN's official implementation uses PyTorch/BasicSR and supports tiled inference.

## 4. AI Model

Primary model:

**RealESRGAN_x4plus**

Model file:

`RealESRGAN_x4plus.pth`

The application must automatically verify that the model exists before processing.

If missing, provide a clear setup/download instruction.

## 5. Device Selection

Priority:

```text
Compatible GPU
      ↓
CPU
```

The application must verify that GPU inference is actually available rather than assuming that a detected GPU can run PyTorch CUDA.

A GT 730 should therefore be treated as:

```text
GPU detected
      ↓
Can backend initialize?
      ↓
YES → GPU
NO  → CPU
```

## 6. Mandatory Tiling

Every image must be processed using tiles.

Never process the complete image tensor at once.

Recommended initial tile sizes:

```text
Very low RAM:
128–256

Low RAM:
256–384

Normal system:
384–512

Higher-memory GPU:
512–768
```

These are starting ranges, not hard-coded requirements.

The system analyzer must dynamically choose the safest tile size.

## 7. Tile Overlap

Tiles must overlap to reduce visible seams.

Default:

`32–64 pixels`

The overlap should be configurable internally.

Tiles must be blended/reconstructed carefully.

## 8. Dynamic Memory Management

Before processing:

```text
Detect available RAM
Detect available VRAM
Detect CPU cores
Detect backend
Analyze image dimensions
Estimate tensor memory
Select tile size
Select processing mode
```

If processing causes an out-of-memory condition:

```text
OOM
 ↓
reduce tile size
 ↓
clear cache
 ↓
retry tile
```

Example:

```text
512 → 384 → 256 → 192 → 128
```

The application should automatically retry with a smaller tile rather than crashing immediately.

## 9. Large Image Handling

Very large images should not automatically be sent to Real-ESRGAN at their full dimensions.

The planner should determine whether the image is unnecessarily large for meaningful restoration.

If appropriate:

```text
Original
 ↓
temporary working-resolution image
 ↓
AI restoration
 ↓
optional resize to requested output/original dimensions
```

However, **downscaling must never be performed merely because the image is large**.

The planner should preserve resolution when reducing it would destroy useful detail.

## 10. Resolution Policy

Default output:

**Same dimensions as input.**

The AI model may internally generate a higher-resolution result because Real-ESRGAN is a super-resolution model.

After restoration:

```text
AI output
 ↓
high-quality resize
 ↓
original dimensions
```

if the user selects the default detail-restoration mode.

## 11. CLI

Example:

```bash
python main.py input.jpg -o restored.jpg
```

Optional:

```bash
python main.py input.jpg \
    -o restored.jpg \
    --tile auto \
    --device auto \
    --detail high
```

Advanced:

```bash
python main.py input.jpg \
    -o restored.jpg \
    --tile 256 \
    --device cpu \
    --face-enhance
```

## 12. Progress System

All major operations must immediately print progress.

Example:

```text
[INFO] Loading image...
[INFO] Image: 7000x9300
[INFO] RAM available: 3.1 GB
[INFO] GPU: GT 730 detected
[INFO] GPU backend unavailable → using CPU
[INFO] Selected tile: 256x256
[INFO] Total tiles: 1092

[12.4%] Tile 136/1092
```

Python output must use:

```python
print(message, flush=True)
```

Long operations must never appear frozen.

## 13. Logging

Support:

```text
--verbose
```

and optionally write:

```text
logs/
    latest.log
```

## 14. Error Handling

Handle:

* Invalid image
* Unsupported format
* Missing model
* Insufficient RAM
* GPU initialization failure
* CUDA failure
* Out-of-memory
* Corrupt tile
* Output write failure
* Interrupted processing

The program must clean temporary files when possible.

## 15. Output Formats

Initially support:

* JPG
* PNG
* WEBP

Preserve alpha channels where applicable.

## 16. Performance Philosophy

**Reliability > speed.**

The application is explicitly allowed to be extremely slow on weak hardware.

The primary requirement is:

> If the system can eventually process the image without crashing, it is acceptable.

## 17. Important Constraint

Do not implement ordinary sharpening as the primary enhancement method.

AI restoration must produce the majority of perceived detail improvement.

Sharpening, if implemented, must be subtle and optional.
