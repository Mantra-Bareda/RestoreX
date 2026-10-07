import cv2
import numpy as np
from src.progress import report_progress

def apply_postprocessing(image, args):
    """
    Phase 9: Applies optional, conservative post-processing to the reconstructed image.
    Never uses sharpening to fake the main restoration result.
    """
    if not hasattr(args, 'refine') or not args.refine:
        return image
        
    report_progress("Applying subtle post-processing refinements...")
    
    # 1. Artifact Cleanup (Bilateral Filter)
    # A very subtle bilateral filter helps smooth out minor "plastic" AI artifacts in flat areas 
    # without destroying actual edges or recovered texture.
    report_progress(" - Reducing AI artifacts...")
    refined = cv2.bilateralFilter(image, d=5, sigmaColor=15, sigmaSpace=15)
    
    # 2. Local Contrast (CLAHE on L-channel)
    # Real-ESRGAN can sometimes "flatten" the lighting. CLAHE restores depth.
    report_progress(" - Enhancing local contrast...")
    lab = cv2.cvtColor(refined, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Very conservative clip limit (1.2). We don't want deep HDR crunch, just natural depth.
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    lab = cv2.merge((cl, a, b))
    refined = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 3. Subtle Sharpening (Unsharp Mask)
    # We use a very light touch (amount=0.2) just to add crispness to the micro-contrast.
    report_progress(" - Applying subtle sharpening...")
    blurred = cv2.GaussianBlur(refined, (0, 0), 2.0)
    # refined = refined * 1.2 - blurred * 0.2
    sharpened = cv2.addWeighted(refined, 1.2, blurred, -0.2, 0)
    
    report_progress("Post-processing complete.")
    return sharpened
