# TRD_v2 — RestoreX CLI (Version 2.0 Technical Requirements)

## 1. System Architecture Overview

RestoreX v2 expands the original tile-based inference engine into a unified visual processing pipeline supporting three operational engines:
1. **Static Detail Restoration & Upscaling Engine**
2. **Salient Background Removal (Alpha Matting) Engine**
3. **Temporal GIF Processing Engine (Super-Resolution & Background Stripping)**

```text
                                ┌─────────────────────────┐
                                │       CLI Interface     │
                                └────────────┬────────────┘
                                             │
                                             ▼
                                ┌─────────────────────────┐
                                │ System & Hardware Probe │
                                └────────────┬────────────┘
                                             │
                                             ▼
                                ┌─────────────────────────┐
                                │   Execution Planner     │
                                └────────────┬────────────┘
                                             │
           ┌─────────────────────────────────┼─────────────────────────────────┐
           ▼                                 ▼                                 ▼
┌──────────────────────┐          ┌──────────────────────┐          ┌──────────────────────┐
│  Detail Restoration  │          │  Background Removal  │          │    GIF Processor     │
├──────────────────────┤          ├──────────────────────┤          ├──────────────────────┤
│ Tier 1: SRVGG x4v3   │          │ Tier 1: u2netp       │          │ Diff Cache & BBox    │
│ Tier 2: RRDB x4plus  │          │ Tier 2: BiRefNet-lite│          │ Model: animevideov3  │
│ (Dynamic Tiling)     │          │ (Guided Filter Snap) │          │ NeuQuant Palette Gen │
└──────────┬───────────┘          └──────────┬───────────┘          └──────────┬───────────┘
           │                                 │                                 │
           └─────────────────────────────────┼─────────────────────────────────┘
                                             │
                                             ▼
                                ┌─────────────────────────┐
                                │  Reconstruction & Save  │
                                └─────────────────────────┘
```

---

## 2. Technical Stack & Dependencies

* **Core Runtime:** Python 3.10+
* **Inference Runtimes:**
  * `torch` / `torchvision` (PyTorch C++ Backend)
  * `onnxruntime` (DirectML, CPU, OpenVINO, and CUDA execution providers)
* **Image & Array Processing:**
  * `opencv-python-headless` (Bilateral filtering, CLAHE, guided filtering, morphological ops)
  * `numpy` (Tensor arithmetic, channel permutation, diff maps)
  * `Pillow` (PIL) & `imageio` (GIF extraction, palette quantization, frame serialization)
* **System Telemetry:**
  * `psutil` (Non-paged pool memory, virtual memory metrics, core counts)

---

## 3. Model Inventory & Specifications

| Model Identifier | Architectural Family | Weights File Size | Native Tensor Resolution | Precision | Runtime Backend | Target Pipeline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`realesr-general-x4v3`** | Compact SRVGG | ~15.2 MB | Flexible (Tiled) | FP32 / FP16 | PyTorch / ONNX | Image Upscale (Tier 1) |
| **`RealESRGAN_x4plus`** | Deep RRDB (23 blocks) | ~67.1 MB | Flexible ($128\times 128$ min) | FP32 / FP16 | PyTorch | Image Upscale (Tier 2) |
| **`4x-UltraSharp`** | Fine-tuned RRDB | ~67.1 MB | Flexible ($128\times 128$ min) | FP32 / FP16 | PyTorch | Image Upscale (Tier 2 Alt) |
| **`u2netp`** | Nested U-Structure (Light) | ~4.7 MB | $320 \times 320 \times 3$ | FP32 | ONNX Runtime | BG Removal (Tier 1) |
| **`silueta`** | Compact Silhouette SOD | ~43.8 MB | $320 \times 320 \times 3$ | FP32 | ONNX Runtime | BG Removal (Tier 1 Alt) |
| **`BiRefNet-lite`** | Bilateral Refinement SOD | ~218 MB | $1024 \times 1024 \times 3$ | FP32 / FP16 | PyTorch / ONNX | BG Removal (Tier 2) |
| **`bria-rmbg-1.4`** | IS-Net Architecture | ~176 MB | $1024 \times 1024 \times 3$ | FP32 | ONNX Runtime | BG Removal (Tier 2 Alt) |
| **`realesr-animevideov3`**| Compact VGG Video | ~15.4 MB | Flexible ($2\times, 3\times, 4\times$) | FP32 / FP16 | PyTorch / ONNX | GIF Upscale (All Tiers) |

---

## 4. Hardware Inspection & Safety Planning

Before processing, `SystemAnalyzer` evaluates:
* Total System RAM (`psutil.virtual_memory().total`)
* Available Free RAM (`psutil.virtual_memory().available`)
* GPU Device Detection (`torch.cuda.is_available()`, device name, total VRAM)
* DirectML Device Availability via `onnxruntime.get_available_providers()`

### Automatic Allocation Logic

```text
Available RAM < 3.5 GB:
    Restoration: Force Tier 1 (realesr-general-x4v3), tile_size = 128
    BG Removal:  Force Tier 1 (u2netp)
    GIF:         Prompt warning; frame downsample default; enforce difference caching

3.5 GB <= Available RAM < 8.0 GB:
    Restoration: Tier 1 (tile_size = 256) OR Tier 2 (tile_size = 128, alert emitted)
    BG Removal:  Tier 1 default; Tier 2 permitted with warning
    GIF:         Standard processing with difference caching

Available RAM >= 8.0 GB or Dedicated GPU (VRAM >= 4.0 GB):
    Restoration: Tier 2 enabled (tile_size = 256–512)
    BG Removal:  Tier 2 enabled (BiRefNet-lite / bria-rmbg-1.4)
    GIF:         Full keyframe temporal batching
```

---

## 5. Algorithmic Pipelines

### 5.1 Salient Background Removal Pipeline (Non-Tiled)

Tiling breaks global semantic context in salient object detection. The background removal engine uses a **downsample-predict-upscale-snap** architecture:

```text
Original RGB Image (H x W x 3)
       │
       ├─── Hold Original Image in Memory as Guide
       │
       ▼
Pre-process: Resize to Model Dimension (320x320 or 1024x1024)
       │
       ▼
Normalize: Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225]
       │
       ▼
Inference Pass: ONNX Runtime / PyTorch Engine
       │
       ▼
Raw Output: 1-Channel Probability Map (h_model x w_model x 1) [Values: 0.0 - 1.0]
       │
       ▼
Classical Upscale: Bilinear Interpolation back to (H x W x 1)
       │
       ▼
Fast Guided Edge Filter Snap:
       Guide: Original Grayscale Image I (H x W)
       Source: Coarse Mask P (H x W)
       Radius r = 8, Regularization eps = 1e-4
       │
       ▼
Refined Alpha Matte Alpha (H x W) [Sharp transitions, zero haloing]
       │
       ▼
Composite: RGBA = Concatenate(Original_RGB, Alpha * 255)
       │
       ▼
Export PNG-32
```

#### Text & Caption Preservation Sub-Engine (`--keep-text`)
When the `--keep-text` flag is set:
1. Generate standard neural foreground mask $M_{\text{ai}}$.
2. Calculate background color estimate $C_{\text{bg}}$ via 4-corner sampling or modal color binning.
3. Compute delta distance map $D(x, y) = \|I(x, y) - C_{\text{bg}}\|_{2}$.
4. Detect high-frequency high-contrast text regions via adaptive thresholding:
   $$M_{\text{edge}} = \text{Canny}(I) \lor \text{AdaptiveThreshold}(I)$$
5. Isolate connected components not intersecting the primary neural subject.
6. Merge retained text mask: $M_{\text{final}} = M_{\text{ai}} \lor M_{\text{text}}$.

---

### 5.2 Animated GIF Processing Engine

```text
Input GIF (N Frames, Framerate F, Disposal Modes)
       │
       ▼
Deconstruct: Extract Frames {F_0, F_1, ..., F_{N-1}} and Delay Metadata
       │
       ▼
Frame Differencing & Motion Bounding Box Engine:
   For frame i in 1 to N-1:
       Diff_Map = |F_i - F_{i-1}|
       Thresholded = Diff_Map > delta_threshold (default = 12/255)
       If changed_pixel_ratio < 0.05:
           Copy previous upscale canvas; skip inference
       Else:
           Extract Bounding Box containing all changed pixels + 16px padding
           Process only sub-rectangle through AI model
           Paste processed patch into full-resolution canvas
       │
       ▼
Palette Optimization:
   Neural outputs emit 24-bit TrueColor. Convert back using NeuQuant / Median Cut
   Apply Floyd-Steinberg error diffusion dithering.
       │
       ▼
Encode: Save optimized GIF / WebP / MP4
```

---

## 6. Execution Flow & State Machine

```text
[State: Idle]
      │ (User input command)
      ▼
[State: System Check] ─── (RAM < Threshold && Selected Tier == 2) ──► [Prompt Warning / Confirm]
      │                                                                         │
      ├─────────────────────── User proceeds ◄──────────────────────────────────┘
      ▼
[State: Preprocessing]
      ├── Image Mode: Compute dimensions, working downscale check
      ├── BG Mode: Resize copy to native input tensor (320 or 1024)
      └── GIF Mode: Split frames, calculate inter-frame MSE
      ▼
[State: Model Execution]
      ├── Image: Chunked Overlap Tiling (with OOM Auto-Fallback: 512->256->128)
      ├── BG: Single-pass full-canvas inference
      └── GIF: Difference-cached bounded inference
      ▼
[State: Postprocessing & Refinement]
      ├── Image: Blended tile stitching, optional CLAHE / subtle unsharp
      ├── BG: Guided filter edge snapping (linear time CPU)
      └── GIF: Color quantization to 256-color palette via libimagequant
      ▼
[State: Write & Cleanup]
      ├── Flush buffers to disk
      ├── Memory release (gc.collect(), torch.cuda.empty_cache())
      └── Return exit code 0
```

---

## 7. Error Handling & Memory Boundaries

1. **Out of Memory (OOM) Interception:**
   * PyTorch: Catch `torch.cuda.OutOfMemoryError` or `RuntimeError("CUDA out of memory")`.
   * Action: Evacuate tensor cache, decrement active tile dimensions by 50% ($512 \rightarrow 256 \rightarrow 128$), and retry inference.
   * Host RAM Exhaustion: Monitor `psutil.virtual_memory().percent`. If $> 90\%$, stall queued tiles, swap tile buffers to temp mmap files on disk.

2. **Aspect Ratio & Dimension Preservation:**
   * All masks generated via SOD downsampling must match original coordinates during Guided Filter snapping. Coordinate offsets must use `align_corners=False` to prevent spatial drift across multiple frames.

3. **Temporal Jitter Mitigation:**
   * When upscaling animated GIFs in Tier 1, inter-frame flickers are avoided by enforcing `realesr-animevideov3` which has temporal consistency built into its training loss.