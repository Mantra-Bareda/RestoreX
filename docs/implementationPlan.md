# IMPLEMENTATION PLAN — RestoreX CLI

## Phase 1 — Project Foundation

Create:

```text
restorex/
├── main.py
├── requirements.txt
├── README.md
├── config/
├── models/
├── src/
│   ├── cli.py
│   ├── system_info.py
│   ├── image_analyzer.py
│   ├── planner.py
│   ├── tiler.py
│   ├── inference.py
│   ├── reconstruction.py
│   ├── postprocess.py
│   ├── progress.py
│   └── utils.py
├── outputs/
└── logs/
```

Implement CLI argument parsing.

---

## Phase 2 — System Detection

Implement reliable detection of:

* CPU
* CPU cores
* system RAM
* available RAM
* GPU
* VRAM when available
* PyTorch version
* CUDA availability
* selected execution device

Output a readable system report.

---

## Phase 3 — Model Integration

Integrate:

**RealESRGAN_x4plus**

Download/load the model automatically or provide a clear first-run setup.

Verify model integrity before inference.

Do not load the model until system analysis is complete.

---

## Phase 4 — Dynamic Processing Planner

Create the planner.

Inputs:

```text
image dimensions
RAM
VRAM
CPU
GPU
backend
```

Outputs:

```text
device
tile size
tile overlap
batch size
precision
working resolution
```

Use conservative defaults.

The planner should never choose an aggressive configuration when uncertain.

---

## Phase 5 — Mandatory Tile Engine

Implement:

```text
Image
 ↓
Tile generator
 ↓
Overlapping tiles
 ↓
AI inference
 ↓
Tile result
```

Every image must use this system.

No full-image inference mode.

Implement configurable tile sizes.

---

## Phase 6 — Automatic OOM Recovery

Implement:

```text
try inference
 ↓
success → continue
 ↓
OOM → clear memory
 ↓
reduce tile
 ↓
retry
```

Use safe minimum tile size.

If even the minimum tile fails, produce a useful error explaining that the system cannot process the selected model/image combination.

---

## Phase 7 — Reconstruction

Implement overlapping tile reconstruction.

Use weighted blending around tile boundaries.

Test specifically for:

* seams
* brightness changes
* duplicate textures
* discontinuities

---

## Phase 8 — High-Resolution Image Strategy

Implement image-size analysis.

Do not blindly downscale.

Create a decision system:

```text
Is image extremely large?
       │
       ├── NO → process normally
       │
       └── YES
            │
            ▼
     Estimate workload
            │
            ▼
Would downscaling preserve
useful subject detail?
            │
       ┌────┴────┐
       YES       NO
        │         │
   temporary     keep
   downscale    original
```

The default final output should preserve original dimensions.

---

## Phase 9 — Post Processing

Implement optional:

* subtle sharpening
* local contrast
* artifact cleanup

Keep all effects conservative.

Never use sharpening to fake the main restoration result.

---

## Phase 10 — Face Enhancement

Add optional GFPGAN support.

Pipeline:

```text
Restored image
 ↓
Face detection
 ↓
GFPGAN
 ↓
Controlled blending
```

Default:

**OFF**

because the primary goal is general photo restoration.

Real-ESRGAN officially supports GFPGAN integration.

---

## Phase 11 — CLI Progress System

Implement a centralized progress reporter.

Example:

```text
[1/7] Analyzing system...
[2/7] Loading image...
[3/7] Planning processing...
[4/7] Processing tiles...
      Tile 421/1092 — 38.55%
[5/7] Reconstructing...
[6/7] Finalizing...
[7/7] Saving...
[DONE] Output saved successfully.
```

Every important print must use:

```python
print(message, flush=True)
```

Do not leave the terminal apparently frozen during inference.

---

## Phase 12 — Testing

Create tests for:

### Hardware

* CPU-only
* 128 MB GPU system
* 2 GB GPU
* normal desktop

### Images

* 500×500
* 1920×1080
* 4000×3000
* 7000×9300
* noisy image
* blurry image
* JPEG-compressed image
* portrait
* landscape
* PNG
* JPG
* WEBP

### Failure tests

* missing model
* invalid image
* insufficient RAM
* GPU failure
* CUDA failure
* forced OOM
* interrupted process
* invalid output path

---

## Phase 13 — Quality Benchmark

Create a test folder containing real photos from budget phones.

Compare:

```text
Original
Normal resize
Sharpened resize
Real-ESRGAN
Real-ESRGAN + subtle refinement
Real-ESRGAN + GFPGAN
```

Evaluate visually at:

```text
100%
200%
400%
```

The winning configuration should prioritize:

1. Real visible detail
2. Natural texture
3. Clean edges
4. Low noise
5. Low artifacts
6. No excessive sharpening

Resolution increase is not a primary metric.

---

## Phase 14 — Optimization

After correctness is established:

* Reduce RAM usage.
* Improve tile scheduling.
* Release tile tensors immediately.
* Use CPU/GPU memory cleanup.
* Optimize image loading.
* Avoid unnecessary copies.
* Optimize reconstruction.
* Add safe caching where useful.

Do **not** optimize by sacrificing restoration quality.

---

## Phase 15 — Final CLI

Final basic command:

```bash
python main.py input.jpg -o restored.jpg
```

Recommended command:

```bash
python main.py input.jpg -o restored.jpg --device auto --tile auto
```

Advanced:

```bash
python main.py input.jpg \
    -o restored.jpg \
    --device cpu \
    --tile 256 \
    --face-enhance \
    --verbose
```

---

# Final Implementation Rule

The developer must remember:

**This is NOT a resolution-increasing application.**

The application's definition of "upscale" is:

> **Recover and reconstruct useful visual detail so that an image looks clearer and remains useful when zoomed.**

A 7000×9300 image that looks terrible is still a terrible image.

A 4000×5300 image with substantially better visible detail is preferable.

Therefore:

**Detail quality > resolution > processing speed.**

The application is allowed to take a very long time.

It is not allowed to sacrifice quality or crash simply to finish faster.
