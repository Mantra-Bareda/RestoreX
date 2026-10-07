# UPDATE_1.md — Interactive CLI Application Flow

## Purpose

This update changes the application from a command-only CLI into an **interactive CLI application**, similar in interaction style to modern coding agents such as Antigravity.

The existing image-processing engine, Real-ESRGAN pipeline, tiling system, hardware detection, adaptive tile sizing, restoration, and reconstruction logic should remain unchanged.

This update focuses on the **interactive application shell and user workflow** before final testing and optimization.

---

# 1. Application Launch

When the user launches:

```bash
python main.py
```

the application should open an interactive terminal interface.

It should NOT immediately require command-line arguments.

Example:

```text
╔══════════════════════════════════════════════╗
║              RESTOREX                  ║
║       AI Image Detail Restoration           ║
╚══════════════════════════════════════════════╝

[INFO] Initializing...

System:
  CPU: Intel ...
  RAM: 4.0 GB
  GPU: GT 730 2 GB
  Backend: CPU

What would you like to do?

  1. Upscale / Restore Image
  2. Exit

Select an option:
>
```

The exact visual design can be improved, but it must remain lightweight and work in a normal terminal.

---

# 2. Main Menu

Currently the application has only one user-facing feature:

```text
1. Upscale / Restore Image
2. Exit
```

Do NOT create fake options for features that do not exist.

When future features are implemented, they may be added to this menu.

---

# 3. Image Selection

After selecting:

```text
1. Upscale / Restore Image
```

show:

```text
How would you like to provide the image?

1. Enter file path manually
2. Select image using file dialog
3. Back

Select:
>
```

---

# 4. Manual File Path

If the user selects:

```text
1. Enter file path manually
```

show:

```text
Enter image file path:
>
```

Accept paths containing spaces.

Example:

```text
/home/user/Pictures/my photo.jpg
```

Windows paths must also be supported where applicable:

```text
C:\Users\User\Pictures\photo.jpg
```

The application should normalize and validate the path.

---

# 5. Supported Input Formats

Initially support:

```text
.jpg
.jpeg
.png
.webp
```

File extension checking should be case-insensitive.

Examples:

```text
photo.JPG
photo.Jpeg
photo.PNG
photo.WEBP
```

must work.

If an unsupported file is entered:

```text
[ERROR] Unsupported image format.

Supported formats:
  JPG / JPEG
  PNG
  WEBP

Please select another image.
```

Do not crash.

---

# 6. File Dialog

If the user selects:

```text
2. Select image using file dialog
```

open a native/simple Python file-selection dialog.

Use Python's standard GUI file dialog functionality, preferably:

```python
tkinter.filedialog.askopenfilename()
```

The dialog should filter:

```text
JPEG files
PNG files
WEBP files
All supported images
```

Example filter:

```text
Image files (*.jpg;*.jpeg;*.png;*.webp)
```

The application remains a CLI application.

The file dialog is only used when the user chooses the file-selection option.

---

# 7. Selected Image Confirmation

After selecting or entering a valid image:

```text
✓ Image selected

Path:
 /home/user/Pictures/my_photo.jpg

Name:
 my_photo.jpg

Size:
 7000 × 9300

Format:
 JPEG

What would you like to do?

1. Continue
2. Select another image
3. Back

Select:
>
```

The full path should be displayed.

If the path is extremely long, display a shortened visual representation if necessary while retaining the complete path internally.

---

# 8. Select Another

The option:

```text
2. Select another image
```

must return to the image-selection process.

Example:

```text
Current image:
photo1.jpg

1. Enter another path
2. Select another using file dialog
3. Continue with current image
4. Back
```

The user must not need to restart the application.

---

# 9. Multiple Images

The current application should remain **single-image processing** unless batch processing already exists in the current implementation.

Do not introduce batch processing merely for this UI update.

"Select another" means:

> Replace the currently selected image.

It does NOT mean queue multiple images.

---

# 10. Upscaling / Restoration Settings

After confirming the image, show:

```text
Image:
photo.jpg
Resolution:
7000 × 9300

Select restoration scale:

1. 1× — Restore detail, keep original resolution
2. 2× — Restore + 2× internal/output scaling
3. 4× — Restore + 4× internal/output scaling
4. Custom
5. Back

Select:
>
```

However, the application must distinguish between:

### Restoration strength

and

### Output resolution.

The default mode should prioritize restoration while preserving the original dimensions.

Recommended default:

```text
1× — Detail Restoration
```

This should internally use the Real-ESRGAN model but resize the final result back to the original dimensions if required by the existing processing design.

---

# 11. Recommended User-Facing Modes

If the current backend supports them, present:

```text
1. Detail Restoration — Recommended
2. Detail Restoration + 2×
3. Detail Restoration + 4×
4. Back
```

Do not expose technical terminology such as:

```text
RealESRGAN_x4plus
tile_size=256
FP16
tensor
CUDA
```

in the normal user menu.

Those are implementation details.

---

# 12. Processing Configuration

After selecting the restoration mode:

```text
Preparing restoration...

[1/5] Analyzing system...
[2/5] Analyzing image...
[3/5] Selecting processing configuration...
[4/5] Preparing tiles...
[5/5] Ready.
```

The existing automatic hardware-analysis system must determine:

* CPU/GPU
* available RAM
* available VRAM
* processing device
* tile size
* tile overlap
* batch size
* working resolution

The user should NOT need to manually configure these in normal mode.

---

# 13. Confirmation Before Processing

Before starting expensive processing:

```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ready to restore:

File:
photo.jpg

Input:
7000 × 9300

Mode:
Detail Restoration

Device:
CPU

Tile:
256 × 256

Estimated processing:
Very slow

Output:
restored_photo.jpg

Start processing?

1. Yes
2. Change settings
3. Cancel

Select:
>
```

The estimate does not need to be exact.

For very weak systems, clearly warn:

```text
[WARNING] This image may take a long time to process on this system.
```

---

# 14. Processing Screen

Once confirmed:

```text
Starting restoration...

Device: CPU
Tile size: 256
Total tiles: 1092

[00:32] Tile 137/1092
Progress: 12.54%
```

Progress must continuously update.

All important terminal output must use:

```python
print(..., flush=True)
```

The terminal must never appear frozen during long inference operations.

---

# 15. Completion

After processing:

```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ RESTORATION COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:
photo.jpg

Output:
restored_photo.jpg

Resolution:
7000 × 9300

Processing time:
18m 42s

Device:
CPU

What next?

1. Open output location
2. Restore another image
3. Return to main menu
4. Exit

Select:
>
```

"Open output location" should only be implemented if cross-platform support is handled correctly.

If not implemented, omit the option.

---

# 16. Output Naming

Default output naming:

```text
photo.jpg
→
photo_restored.jpg
```

If the file already exists:

```text
photo_restored_1.jpg
photo_restored_2.jpg
...
```

Never silently overwrite an existing user image.

---

# 17. Ctrl+C Handling

If the user presses:

```text
Ctrl+C
```

during processing:

```text
[WARNING] Processing interrupted.

Temporary files will be cleaned where possible.

1. Return to menu
2. Exit

Select:
>
```

Do not leave corrupted output presented as a successful result.

---

# 18. Error Handling

Any processing error should return to the interactive interface instead of immediately terminating.

Example:

```text
[ERROR] Restoration failed.

Reason:
Insufficient memory while processing tile.

The application automatically attempted smaller tile sizes.

1. Retry
2. Select another image
3. Return to main menu
4. Exit
```

The existing automatic OOM recovery system must remain active.

---

# 19. CLI Arguments

The interactive interface is now the default:

```bash
python main.py
```

Existing direct CLI functionality may be retained for advanced users:

```bash
python main.py input.jpg -o output.jpg
```

If direct CLI mode already exists, DO NOT remove it.

The application should detect whether arguments were supplied:

```text
No arguments
    ↓
Interactive mode

Arguments supplied
    ↓
Direct CLI mode
```

This preserves scripting/automation capability.

---

# 20. System Information

System detection should happen early, but expensive AI/model initialization should NOT happen until the user actually starts restoration.

Launch should remain reasonably fast.

Example:

```text
[INFO] CPU detected
[INFO] RAM detected
[INFO] GPU detected
[INFO] Backend checked
```

Model loading should occur later.

---

# 21. Important UX Rules

The application must:

* Never crash because of invalid user input.
* Always provide a way to go back.
* Never silently overwrite files.
* Never require technical configuration for normal use.
* Clearly communicate long processing times.
* Show continuous progress.
* Use `flush=True`.
* Keep the interface lightweight.
* Keep the application CLI-first.
* Use a file dialog only when explicitly requested.
* Never pretend that resolution increase equals detail restoration.

---

# 22. Updated Development Status

The project currently has:

```text
Core AI pipeline        ✓ DONE
Real-ESRGAN integration ✓ DONE
Tile processing         ✓ DONE
Hardware detection      ✓ DONE
Adaptive tile planning  ✓ DONE
Restoration pipeline    ✓ DONE

Interactive CLI         ← UPDATE_1
Testing                 → NEXT
Optimization            → NEXT
```

Therefore, implement this update **before beginning the final testing and optimization phase**.

After UPDATE_1 is complete:

```text
UPDATE_1
   ↓
Integration test
   ↓
Quality testing
   ↓
Memory testing
   ↓
Hardware testing
   ↓
Performance optimization
   ↓
Final release
```

# 23. Non-Goals

This update does NOT require:

* GUI application
* Web interface
* Cloud processing
* User accounts
* Database
* Batch processing
* New AI models
* New restoration algorithms
* Changing Real-ESRGAN
* Removing tiled processing

This update is specifically about making the existing application feel like a polished **interactive CLI tool** while preserving the existing restoration engine.
