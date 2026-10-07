# PRD_v2 — RestoreX CLI (Version 2.0)

## 1. Product Overview

**RestoreX v2** is a modular, hardware-adaptive, local CLI tool suite designed to perform high-fidelity visual tasks on consumer-grade and constrained hardware without crashing or freezing. 

Expanding beyond static photographic detail restoration, Version 2.0 introduces two major operational engines: **Salient Background Removal** and **Animated GIF Restoration/Editing**. The platform operates strictly on a **2-Tier Model Selection Architecture** across all modules:
* **Tier 1 (Lightweight / Fast):** Built for lower-spec hardware (low RAM, integrated graphics, budget dual/quad-core CPUs). Emphasizes rapid execution, minimal memory consumption, and aggressive algorithmic shortcuts.
* **Tier 2 (High Precision / Heavy + Optimization):** Built for higher-spec hardware (discrete GPUs, high RAM allocations, or patience-tolerant CPU runs). Emphasizes maximum edge matting, texture generation, and temporal consistency, backed by automated memory-safety throttles and chunked tiling.

The software remains 100% local, runs zero cloud calls, mandates strict memory-budget verification, and presents explicit hardware warnings before executing heavy pipelines.

---

## 2. Problem Statement

1. **Extreme Hardware Disparity:** Consumer machines range from budget laptops with 4 GB shared RAM and integrated GPUs to modern workstations with dedicated CUDA GPUs. Monolithic AI tools assume high-end VRAM and crash or freeze budget machines.
2. **Context Rupture in Background Removal:** Users attempt to apply image-upscaling tiling strategies ($128 \times 128$ chunks) to background removal models. Because salient object detection requires **global semantic awareness**, isolated tiles fail completely.
3. **Severe GIF Processing Bottlenecks:** Naive frame-by-frame super-resolution on GIFs (e.g., 96 frames) takes hours on CPUs and causes severe inter-frame jitter/flicker and massive file bloating due to improper color quantization.
4. **Subject vs. Text Ambiguity:** Traditional background removal algorithms delete floating textual elements, watermarks, or captions alongside backgrounds because they lack semantic classification for typography as an intended foreground object.

---

## 3. Product Goals & Core Tenets

1. **Universal Execution Without Crashing:** Any supported feature must complete without out-of-memory (OOM) crashes, even if it requires falling back to Tier 1 models or micro-tiling.
2. **2-Tier Standard Across All Workflows:** Every capability (Upscaling, Matting, GIF handling) must offer a clear binary choice between lightweight agility and heavyweight precision.
3. **Context-Preserving Downsample-and-Snap Matting:** Execute background removal without tiling by predicting downscaled global masks and snapping edges back to the native high-resolution canvas using classical linear-time edge-guided algorithms.
4. **Motion-Aware GIF Processing:** Minimize redundant computation in animated formats by caching identical regions between adjacent frames and processing only active bounding boxes.
5. **No Blind Rewrites:** The CLI must transparently warn users if a requested operation exceeds their system's safe operating envelope, allowing informed consent or automatic step-down.

---

## 4. User Personas & Target Hardware

* **Persona A (Budget Laptop User):** Runs an older dual/quad-core CPU with 4 GB to 8 GB total system RAM and integrated graphics (e.g., Intel HD/UHD Graphics, AMD Vega). Requires rapid background extraction and lightweight upscaling without locking the OS.
* **Persona B (Power/Archival User):** Operates modern multi-core CPUs with 16 GB+ RAM or entry-level discrete GPUs (e.g., GTX 1650, RTX 3050, or legacy GT 730). Seeks hairline-accurate alpha matting and deep texture reconstruction.
* **Persona C (Content Creator / Animator):** Batches animated reaction GIFs, requiring fast background cutouts, resolution improvements, and tiny output file sizes with clean 256-color palettes.

---

## 5. Feature Specifications & Modular Architecture

### Module 1: Image Super-Resolution & Detail Restoration (Static Images)
* **Tier 1 (Lightweight / Fast):**
  * **Model:** `realesr-general-x4v3` (Compact SRVGG architecture, ~15 MB).
  * **Characteristics:** Extremely low parameter count, direct CPU viability, minimal RAM footprint.
  * **Processing Mode:** Default larger tile size ($256 \times 256$ to $384 \times 384$).
* **Tier 2 (High Precision / Heavy):**
  * **Model:** `RealESRGAN_x4plus` or `4x-UltraSharp` (Deep RRDB architecture, ~67 MB).
  * **Characteristics:** Deep generative hallucination of fine textures and sharp borders.
  * **Processing Mode:** Mandatory dynamic micro-tiling ($128 \times 128$ to $256 \times 256$) with automatic OOM step-down and overlapping boundary blending.
* **Preservation Policy:** Target output defaults to original pixel dimensions via high-grade Lanczos/bicubic downsampling unless explicit upscaling scale factor (`--scale 2` or `--scale 4`) is passed.

### Module 2: Salient Background Removal (Alpha Matting)
* **Tier 1 (Lightweight / Fast):**
  * **Model:** `u2netp` (~4.7 MB weights) or `silueta` (~43 MB weights).
  * **Input Dimension:** Downscaled to $320 \times 320$.
  * **Target Latency:** $< 0.5$ seconds on CPU.
* **Tier 2 (High Precision / Studio Edge):**
  * **Model:** `bria-rmbg-1.4` (~170 MB) or `BiRefNet-lite` (~220 MB).
  * **Input Dimension:** Downscaled to $1024 \times 1024$.
  * **Target Latency:** $3$ to $8$ seconds on CPU, sub-second on GPU.
* **Non-Chunked Pipeline Execution:**
  1. High-resolution input image is cached in memory.
  2. A downsampled copy is passed to the selected SOD model to generate a 1-channel grayscale probability mask.
  3. The mask is scaled mathematically (bilinear/bicubic) back to full canvas dimensions.
  4. A **Fast Guided Filter** evaluates the high-resolution input image as a guide tensor to snap boundary transitions directly onto sharp pixel gradients without neural execution.
  5. Mask is composited with original RGB channels into an RGBA 32-bit PNG.
* **Text & Graphic Handling:**
  * **Default Behavior:** Model treats isolated floating captions as background and eliminates them.
  * **Flag `--keep-text`:** Bypasses pure neural matting for high-contrast islands or uniform backdrops via connected-component analysis and chroma thresholds to retain logos, floating typography, and subtitles.

### Module 3: Animated GIF Restoration & Editing
* **Tier 1 (Lightweight / Motion-Cached):**
  * **Model Engine:** `realesr-animevideov3` (~15 MB) for upscale; `u2netp` for background stripping.
  * **Difference Engine:** Frame-to-frame Mean Squared Error (MSE) comparison. Identical background regions are frozen; only dynamic bounding boxes are processed.
  * **Temporal Flow:** Optical flow skips static intervals, avoiding frame-by-frame full-pass inference.
* **Tier 2 (High Precision / Full Coherence):**
  * **Model Engine:** Exhaustive keyframe inference with temporal smoothing passes to eradicate inter-frame jitter.
* **Color Quantization Engine:**
  * Output GIFs are quantized using `libimagequant` (NeuQuant/Median Cut with Floyd-Steinberg dithering) to eliminate muddy palettes and restrict 1080p frame sequences to manageable file sizes.
  * Optional export flags allow dumping output as animated WebP or MP4 to circumvent the 256-color limitation entirely.

---

## 6. System Spec Verification & Safety Warnings

Before initiating any inference pass, the **Hardware Inspector** evaluates:
* Total System Physical RAM & Available Non-Paged RAM.
* Compute Backends: DirectML, CUDA, OpenVINO, or Fallback Native CPU.
* Total VRAM (if dedicated GPU initialized).
* Media Metrics: Resolution, frame count (for GIFs), color channels.

### Automated Thresholds & Action Matrix

| Target Task | Selected Tier | Detected Resource Level | Pipeline Action |
| :--- | :--- | :--- | :--- |
| **Upscale / Restore** | Tier 2 (`x4plus`) | Available RAM $< 4 \text{ GB}$ / No GPU | Display safety alert. Force tile size $\le 128 \times 128$. Recommend Tier 1. |
| **Background Removal** | Tier 2 (`BiRefNet`/`bria`) | Available RAM $< 4 \text{ GB}$ | Display warning: *Estimated processing 10-15s. UI may become unresponsive.* Allow user override or 1-key swap to `u2netp`. |
| **GIF Processing** | Any | Frame count $> 60$ on CPU | Display processing estimate. Automatically enable Motion Caching (skipping unchanged static blocks). |

---

## 7. Command-Line Interface (CLI) Specification

```bash
# General Syntax
restorex <command> <input_path> -o <output_path> [options]

# 1. Detail Restoration / Upscaling
restorex restore photo.jpg -o restored.png --tier 1
restorex restore photo.jpg -o restored_4x.png --tier 2 --scale 4 --tile auto

# 2. Background Removal
restorex remove-bg portrait.jpg -o cutout.png --tier 1
restorex remove-bg model.jpg -o cutout_hd.png --tier 2 --keep-text

# 3. GIF Processing
restorex gif-process anim.gif -o anim_clean.gif --action remove-bg --tier 1
restorex gif-process anim.gif -o anim_4x.gif --action upscale --tier 1 --format webp
```

### Argument Matrix

* `command`: `restore` | `remove-bg` | `gif-process`
* `--tier`: `1` (Fast / Lightweight) | `2` (High Quality / Heavy) [Default: Auto-determined by hardware]
* `--keep-text`: (Matting only) Preserves floating captions, watermarks, or graphic text.
* `--action`: (GIF only) `upscale` | `remove-bg`
* `--format`: (GIF only) `gif` | `webp` | `mp4`
* `--force`: Bypasses hardware safety advisories and prompts.

---

## 8. Success Metrics & Non-Functional Requirements

1. **Crash Prevention Rate:** 100% completion on valid inputs; zero unhandled OOM aborts across test matrices.
2. **Background Removal Speed:**
   * Tier 1 (`u2netp`): $\le 1.0 \text{ second}$ on 4-core mobile CPU.
   * Tier 2 (`bria-rmbg` / `BiRefNet-lite`): $\le 8.0 \text{ seconds}$ on 4-core mobile CPU.
3. **GIF Efficiency:** Motion difference caching must reduce total frame inference count by at least 40% on standard graphic/reaction GIFs.
4. **Edge Preservation:** Guided filter reconstruction on alpha masks must exhibit clean boundaries snapping to the original resolution image without manual polygon tracing.