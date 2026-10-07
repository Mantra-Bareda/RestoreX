# IMPLEMENTATION PLAN v2 — RestoreX CLI (Version 2.0)

## Overview & Architecture Goals
Transition the single-purpose `RestoreX` upscaling engine into a modular 3-engine visual CLI suite (`restore`, `remove-bg`, `gif-process`) powered by a unified 2-tier execution strategy (Tier 1: Lightweight/Fast, Tier 2: Heavy/High Quality) with strict memory bounds, CPU fallback, and zero unhandled crashes.

---

## Phase 1: Modular Foundation & System Capability Engine
* **Goal:** Refactor project architecture into a multi-command layout and upgrade the system analyzer to support ONNX providers, RAM safety tiers, and pre-flight validation.
* **Core Tasks:**
  * Reorganize directory structure into modular subpackages: `src/core/`, `src/engines/` (`restoration/`, `matting/`, `gif/`), `src/models/`, and `src/utils/`.
  * Update argument parser (`src/cli.py`) to support subcommands: `restore`, `remove-bg`, and `gif-process` with shared flags (`--tier`, `--device`, `--verbose`, `--force`).
  * Extend `system_info.py` to probe host RAM, VRAM, and runtime providers (`CUDAExecutionProvider`, `DmlExecutionProvider`, `CPUExecutionProvider`).
  * Build the hardware pre-flight checker to compute safe operational tiers, display non-blocking user warnings for resource-heavy operations, and provide automatic fallbacks.
* **Verification Criteria:** Running `python main.py <command> --help` outputs consistent command schemas, and `python main.py probe` prints full hardware capabilities and tier recommendations.

---

## Phase 2: Static Detail Restoration Upgrades (Tier 1 & Tier 2)
* **Goal:** Integrate lightweight SRVGG models alongside existing RRDB architectures with adaptive tiling and automatic OOM step-down.
* **Core Tasks:**
  * Implement Tier 1 upscaler using `realesr-general-x4v3` (~15 MB) for CPU-friendly, low-RAM workloads.
  * Retain Tier 2 upscaler using `RealESRGAN_x4plus` and `4x-UltraSharp` (~67 MB) with mandatory chunked tiling.
  * Integrate automatic weight integrity verification and lazy on-demand downloads into `models/`.
  * Update dynamic tile planner to assign tile dimensions based on chosen tier and detected RAM ($128 \times 128$ for budget CPU, $256 \times 256$ to $384 \times 384$ for Tier 1 SRVGG).
  * Standardize dynamic resolution preservation: default to original dimensions via high-grade downsampling unless explicit `--scale` is requested.
* **Verification Criteria:** Processing a $4000 \times 3000$ image on CPU completes under Tier 1 without exceeding 1.5 GB peak RAM; OOM simulated injection triggers seamless fallback to $128 \times 128$ tiles.

---

## Phase 3: Background Removal Engine (Downsample-and-Snap)
* **Goal:** Deliver instant, context-aware salient object detection without chunking, using Fast Guided Filter boundary snapping.
* **Core Tasks:**
  * Build ONNX runtime inference harness supporting Tier 1 (`u2netp` ~4.7 MB, `silueta` ~43 MB) and Tier 2 (`bria-rmbg-1.4` ~170 MB, `BiRefNet-lite` ~220 MB).
  * Implement the non-chunked execution pipeline:
    1. Downsample input to model tensor ($320 \times 320$ or $1024 \times 1024$).
    2. Normalize and compute single-channel probability mask.
    3. Bilinear upscale mask back to original resolution.
  * Integrate classical Fast Guided Filter (OpenCV) using the high-resolution RGB image as guide to snap blurred mask boundaries directly onto source edges.
  * Implement RGBA composite serializer exporting 32-bit PNGs.
* **Verification Criteria:** Transparent PNG output achieves crisp edges around subjects in $< 1.0\text{s}$ on CPU (Tier 1) without blocky edge artifacts or tile boundary seams.

---

## Phase 4: Graphic & Caption Preservation Sub-Engine (`--keep-text`)
* **Goal:** Prevent accidental deletion of floating typography, watermarks, and subtitles during background removal.
* **Core Tasks:**
  * Implement uniform backdrop and color-key analyzer (corner sampling + modal color distribution).
  * Build high-contrast edge mask using adaptive thresholding and morphological edge linking.
  * Implement connected-component filtering to separate primary subject masks from isolated floating text islands.
  * Implement Boolean alpha combination logic ($M_{\text{final}} = M_{\text{ai}} \lor M_{\text{text}}$) triggered via `--keep-text`.
* **Verification Criteria:** An image containing a foreground subject and isolated captions over a contrasting background retains both the subject and the text when `--keep-text` is enabled.

---

## Phase 5: Animated GIF Engine (Motion Difference & Quantization)
* **Goal:** Enable rapid GIF upscaling and background removal without per-frame redundancy, temporal flickering, or palette degradation.
* **Core Tasks:**
  * Implement GIF/WebP reader and frame-delay extractor using `Pillow` and `imageio`.
  * Build inter-frame difference engine (Mean Squared Error thresholding):
    * Detect static background pixels across consecutive frames ($F_i$ vs $F_{i-1}$).
    * Calculate dynamic bounding boxes; route only active sub-rectangles to AI inference.
    * Cache and reuse static regions to bypass full-frame computation.
  * Integrate `realesr-animevideov3` for temporally stable video upscaling.
  * Add color quantization pipeline via `libimagequant` / NeuQuant with Floyd-Steinberg error diffusion to enforce high-quality 256-color palettes and small file sizes.
  * Add multi-format exporter supporting `.gif`, animated `.webp`, and `.mp4`.
* **Verification Criteria:** A 60-frame animated GIF processes with $\ge 40\%$ fewer model evaluations than naive frame-by-frame processing, with zero inter-frame color banding or file bloat.

---

## Phase 6: Unified Pipeline Integration, Stress Testing & Polishing
* **Goal:** Bind all engines into a unified CLI binary with real-time progress feedback, graceful error recovery, and comprehensive verification.
* **Core Tasks:**
  * Standardize centralized progress reporting across all modules with `flush=True` (reporting current phase, tile/frame percentages, ETA, and RAM usage).
  * Implement clean termination traps (`SIGINT`/Ctrl+C) ensuring all temporary mapped files and cached frame buffers are unlinked.
  * Build automated test suite (`tests/test_v2_suites.py`):
    * CPU-only stress test on minimum RAM environment.
    * Corrupt file and missing weight resilience tests.
    * Aspect ratio drift verification across multi-frame GIF and guided-mask pipelines.
  * Update interactive CLI terminal UI for zero-argument execution (`python main.py`).
* **Verification Criteria:** Full automated test suite passes on both CPU-only and GPU targets; interactive terminal cleanly navigates all three engines.