# RestoreX CLI

**RestoreX** is an intelligent, Python-based CLI application designed to recover visually useful detail from poor-quality photographs. It is specifically tailored for high-resolution photographs captured by budget smartphones, where the image is physically large (e.g., 64 Megapixels) but lacks genuine fine-texture detail.

Unlike standard upscalers, RestoreX does not aim to simply increase pixel dimensions. Its primary objective is to make images appear **clearer, more detailed, naturally sharpened, and visually useful when zoomed**, while maintaining memory safety on low-end hardware.

---

## 🚀 Core Features

- **Real-ESRGAN x4plus Integration:** Leverages state-of-the-art AI to hallucinate and reconstruct missing textures.
- **Dynamic Memory Planner:** Automatically analyzes your CPU RAM and GPU VRAM to determine the safest processing configuration, ensuring it can run on everything from a 128MB GPU to a 24GB RTX 4090.
- **Mandatory Tile Engine:** Slices massive images into overlapping chunks. It never attempts to process a huge image all at once.
- **Auto OOM (Out Of Memory) Recovery:** If a specific tile crashes your GPU, the engine automatically catches the error, cuts the tile size in half, clears the VRAM, and retries. **It will never crash halfway through a job.**
- **Smart Resolution Strategy:** Detects if an image is pointlessly massive (e.g., a blurry 50MP photo) and temporarily shrinks it to a working resolution to save processing time, before restoring it to the exact original dimensions.
- **GFPGAN Face Enhancement:** Optional targeted AI to fix "melted" or distorted faces in the background.
- **Memory-Mapped Reconstruction:** Uses disk-backed memory mapping to stitch images together, preventing RAM exhaustion on gigantic files.

---

## ⚙️ Installation

1. Ensure you have **Python 3.10+** installed.
2. Clone or download this repository.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
*(Note: The AI models (`RealESRGAN_x4plus.pth` and `GFPGANv1.3.pth`) will be automatically downloaded into the `models/` folder the first time you run the application.)*

---

## 🛠️ Usage & Features

### 1. Interactive Mode (Recommended)
Simply launch the application with no arguments:
```bash
python main.py
```
This opens a beautifully formatted, interactive terminal shell. You will be guided step-by-step to:
- Enter your image path or select it using a GUI file dialog.
- Choose your output scaling (1x, 2x, 4x).
- Add Face Enhancement or Post-Processing with a single keystroke.
- View real-time processing estimates and live progress.

### 2. Direct CLI Mode (For Scripts & Automation)
If you provide arguments, the application skips the interactive menu and runs headlessly.
```bash
python main.py input.jpg -o restored.jpg
```

#### Complete Command-Line Arguments

| Argument | Description |
| :--- | :--- |
| `input` | Path to the input image file. |
| `-o, --output` | **(Required)** Path to save the final restored image. |
| `--scale` | **(Optional)** Target output scale (1 for original size, 2 for 2x, 4 for 4x). Default is 1. |
| `--device` | **(Optional)** Forces the execution device. Options: `auto` (default), `cuda`, `cpu`. |
| `--tile` | **(Optional)** Forces a specific grid size (e.g., `256`). Auto-planner is recommended. |
| `--face-enhance` | **(Optional)** Enables GFPGAN face restoration. |
| `--refine` | **(Optional)** Enables subtle post-processing (CLAHE, Bilateral filtering, Unsharp mask). |
| `--verbose` | **(Optional)** Prints a highly detailed JSON report of system hardware and plan. |

### Feature Deep-Dive

#### 1. Face Enhancement (`--face-enhance`)
Because Real-ESRGAN doesn't understand human anatomy, it can sometimes distort faces in the background of photos. By passing `--face-enhance`, the app will:
1. Run a blazing-fast OpenCV Haar Cascade pre-scan on a shrunken version of your image.
2. If it detects a face, it loads `GFPGAN`.
3. It specifically targets the faces, restores eyes/teeth/skin, and seamlessly blends them back into the high-resolution canvas.
*(If no faces are found in the pre-scan, it intelligently skips GFPGAN to save time).*

#### 2. Subtle Post-Processing (`--refine`)
AI upscaling can sometimes flatten the lighting or create microscopic "plastic" noise on flat surfaces. The `--refine` flag runs a 3-step cleanup:
- **Bilateral Filter:** Melts away AI-generated noise in flat areas without destroying sharp edges.
- **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Restores natural lighting depth and shadows.
- **Unsharp Mask:** Adds a very delicate, mathematically restricted crispness to the micro-contrast.

---

## 🧪 Testing & Benchmarking

The application includes two automated scripts in the `scripts/` folder for testing and verifying quality.

### 1. Hardware & Edge-Case Testing
Ensures the application behaves correctly under stress, properly falls back to CPU, and handles missing files gracefully.
```bash
python scripts/run_tests.py
```
*(Requires a `trial.jpeg` in the root directory).*

### 2. Quality Benchmark & HTML Report Generator
Want to see exactly how much better the AI is compared to standard sharpening? 
1. Place some low-quality test photos into `tests/benchmark_images/`.
2. Run the generator:
   ```bash
   python scripts/benchmark_quality.py
   ```
3. The script will process your images 5 different ways (Original, OpenCV Sharpened, AI Default, AI + Refine, AI + GFPGAN) and automatically build a web page.
4. **Open the generated HTML files in `tests/benchmark_reports/`**. Hovering your mouse over any image in the browser will instantly apply a 200% CSS zoom, allowing you to seamlessly compare the fine details side-by-side!

---

## 🧠 Architecture Philosophy

**Detail quality > resolution > processing speed.**
This application is explicitly permitted to take a very long time to run on weak hardware. It is not allowed to sacrifice quality or crash simply to finish faster. By utilizing custom overlapping math, intelligent disk-backed memory mapping, and auto-OOM recovery, RestoreX guarantees that if a system can eventually process the image, it will succeed.
