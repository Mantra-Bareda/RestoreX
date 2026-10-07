# PRD — RestoreX CLI

## 1. Product Overview

**RestoreX** is a Python-based CLI image restoration application designed to recover visually useful detail from poor-quality photographs, especially high-resolution photographs captured by budget smartphones.

The application does **not** primarily aim to increase image resolution. Its primary objective is to make images appear clearer, more detailed, naturally sharpened, and more visually useful when zoomed.

The core AI model will be **Real-ESRGAN x4plus**, used as a real-world image restoration/super-resolution model. Real-ESRGAN officially supports tiled inference, arbitrary output scaling, and optional GFPGAN face enhancement.

## 2. Problem

Budget smartphone cameras may produce extremely high-resolution files while containing limited useful detail due to:

* Small camera sensors
* Digital zoom
* Noise
* Motion blur
* Lens limitations
* Compression
* Aggressive camera processing
* Weak fine-texture capture

Simply increasing resolution does not solve this problem.

## 3. Product Goal

Transform:

`High-resolution but low-detail image`

into:

`Natural-looking, clearer, more detailed image`

The final output may remain at the original resolution.

## 4. Target Users

* Users with budget/older smartphones
* Users restoring zoomed photographs
* Users with low-end computers
* Users who prioritize image quality over output resolution
* CLI/technical users

## 5. Core Features

### Required

1. CLI-only interface.
2. Input image path.
3. Output image path.
4. Real-ESRGAN x4plus.
5. Mandatory tiled processing.
6. Automatic hardware/system detection.
7. Automatic tile-size selection.
8. CPU fallback.
9. GPU acceleration when safely available.
10. Memory-conscious processing.
11. Progress reporting.
12. `flush=True` for immediate terminal updates.
13. High-resolution image preprocessing.
14. Optional intelligent downscaling when beneficial.
15. Restoration-first processing.
16. Output can preserve original dimensions.
17. Temporary-file cleanup.
18. Graceful recovery from processing errors.
19. No GUI.
20. No cloud processing.

### Optional

* GFPGAN face restoration
* Denoising
* Multiple processing presets
* Manual tile size
* Manual output scale
* Verbose logging
* Batch processing

## 6. Design Principle

The application must never equate "upscaling" with simply increasing pixel dimensions.

The quality target is:

**clarity + recoverable detail + natural texture + controlled sharpening**

rather than:

**larger resolution + artificial sharpening**.

## 7. Example

Input:

`7000 × 9300`

Possible internal processing:

`7000 × 9300 → intelligent preprocessing → tiled AI restoration → final restoration → 7000 × 9300`

The output may have exactly the same dimensions while being significantly clearer.

## 8. Success Criteria

The application should:

* Work on very low-end CPUs.
* Work with approximately 128 MB GPU systems through CPU fallback.
* Utilize a 2 GB GT 730 when compatible and stable.
* Avoid memory crashes.
* Process extremely large images through tiles.
* Produce natural detail rather than excessive sharpening.
* Clearly report progress throughout processing.
* Never silently fail.
