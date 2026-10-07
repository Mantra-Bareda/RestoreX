# HOW IT WORKS — RestoreX Internal Pipeline

## 1. Input

The user provides an image through the CLI.

```text
input.jpg
```

The application loads metadata first without immediately creating a huge processing tensor.

It determines:

* Width
* Height
* Channels
* Bit depth
* Format
* File size

## 2. System Analysis

The application analyzes the computer before loading the AI workload.

```text
CPU
RAM
GPU
VRAM
PyTorch backend
CUDA availability
```

The purpose is to determine the safest processing configuration.

## 3. Processing Planner

The planner receives:

```text
System capabilities
+
Image dimensions
+
Model requirements
```

and calculates:

```text
working resolution
tile size
tile overlap
device
precision
batch size
```

The system should favor a smaller tile when uncertain.

## 4. High-Resolution Decision

A 7000×9300 image does not automatically need to be processed at its full size.

The planner evaluates whether temporary downscaling would:

* reduce unnecessary computation,
* fit available memory,
* preserve meaningful subject detail.

If downscaling would remove important information, the original resolution is retained.

Any temporary downscale is never treated as the final output.

## 5. Tile Generation

The image is divided into overlapping tiles.

Example:

```text
┌─────┬─────┬─────┐
│  1  │  2  │  3  │
├─────┼─────┼─────┤
│  4  │  5  │  6  │
├─────┼─────┼─────┤
│  7  │  8  │  9  │
└─────┴─────┴─────┘
```

Each tile contains an overlap region.

The overlap provides contextual pixels around the tile boundary and helps prevent seams.

## 6. AI Processing

Each tile is passed independently into:

**RealESRGAN_x4plus**

```text
Low-detail tile
      ↓
Real-ESRGAN
      ↓
Restored/high-detail tile
```

Only one safe number of tiles should be held in memory at once.

For extremely weak machines:

```text
1 tile → inference → release memory
```

## 7. OOM Recovery

If a tile causes memory exhaustion:

```text
Inference failure
      ↓
clear model/cache
      ↓
reduce tile size
      ↓
retry
```

Example:

```text
512
 ↓ failure
384
 ↓ failure
256
 ↓
success
```

The new tile size is retained for subsequent processing.

## 8. Reconstruction

Processed tiles are placed back into their original positions.

Overlapping regions are blended.

The result must not contain:

* visible tile borders
* brightness discontinuities
* repeated textures
* obvious seams

## 9. Face Restoration

If enabled:

```text
Full restored image
       ↓
Face detection
       ↓
Face restoration
       ↓
Blend restored face with image
```

Face restoration should be optional because aggressive face models can alter identity/details.

## 10. Final Detail Refinement

Only very subtle refinement is allowed.

Possible operations:

* tiny amount of sharpening
* local contrast adjustment
* artifact reduction

The application must prioritize natural detail.

Avoid:

```text
oversharpening
haloing
plastic skin
fake texture
excessive contrast
```

## 11. Resolution Restoration

Default mode aims for the original image dimensions.

For example:

```text
INPUT
7000×9300

       ↓
AI restoration

INTERNAL
28000×37200

       ↓
High-quality resize

OUTPUT
7000×9300
```

The enormous intermediate image should preferably not be held entirely in memory.

The implementation should use memory-conscious processing wherever possible.

## 12. Output

The final image is written to the requested output path.

Only after successful writing should temporary data be deleted.

## 13. Progress

The CLI continuously reports:

```text
System analysis
Image analysis
Preprocessing
Tile planning
AI processing
Tile reconstruction
Face enhancement
Finalization
Saving
Complete
```

Every progress message should use immediate flushing:

```python
print(..., flush=True)
```

The user should always know that the application is still working.
