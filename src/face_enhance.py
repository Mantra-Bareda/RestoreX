import cv2
import os
import urllib.request
from src.progress import report_progress

GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
GFPGAN_NAME = "GFPGANv1.3.pth"

def verify_and_download_gfpgan(models_dir="models"):
    """
    Downloads the GFPGAN model if it doesn't exist.
    """
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, GFPGAN_NAME)
    
    if not os.path.exists(model_path):
        report_progress(f"GFPGAN model not found at {model_path}. Downloading...")
        try:
            req = urllib.request.Request(GFPGAN_URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(model_path, 'wb') as out_file:
                chunk_size = 8192
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
            report_progress("GFPGAN download complete.")
        except Exception as e:
            raise FileNotFoundError(f"GFPGAN could not be downloaded automatically: {e}")
            
    return model_path

def fast_face_detect(image):
    """
    Downscales the image to a manageable size and uses a lightweight Haar Cascade 
    to quickly check if any faces are actually present before loading massive AI models.
    """
    report_progress("Performing fast pre-scan for faces...")
    
    h, w = image.shape[:2]
    # Downscale heavily for speed (max 1024 on the longest side)
    scale = 1024.0 / max(h, w)
    if scale < 1.0:
        small_img = cv2.resize(image, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small_img = image.copy()
        
    gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    
    # Load OpenCV's built-in Haar cascade for frontal faces
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Very lenient detection: we'd rather have false positives than miss a face.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
    
    if len(faces) > 0:
        report_progress(f"Fast scan detected ~{len(faces)} potential face(s). Proceeding with GFPGAN.")
        return True
    
    report_progress("Fast scan detected no faces. Skipping GFPGAN to save processing time.")
    return False

def apply_face_enhancement(image, model_path, device):
    """
    Phase 10: Applies GFPGAN to the fully reconstructed image to restore facial details.
    """
    # Dynamic import so the main CLI doesn't crash if they never use --face-enhance
    try:
        from gfpgan import GFPGANer
    except ImportError:
        report_progress("GFPGAN module not installed. Please run: pip install gfpgan")
        return image
        
    # Phase 10 request: Detect if face is actually there first
    if not fast_face_detect(image):
        return image
        
    report_progress("Loading GFPGAN model into memory...")
    
    # We use upscale=1 because the image is ALREADY upscaled by Real-ESRGAN in Phase 7.
    # We set bg_upsampler=None to stop GFPGAN from pointlessly re-upscaling the background.
    face_enhancer = GFPGANer(
        model_path=model_path,
        upscale=1, 
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None,
        device=device
    )
    
    report_progress("Applying GFPGAN face restoration (this may take a moment)...")
    try:
        # The paste_back=True parameter automatically handles controlled blending and feathering
        # of the restored face back onto the original high-resolution background.
        _, _, restored_img = face_enhancer.enhance(
            image,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )
    except Exception as e:
        report_progress(f"GFPGAN enhancement encountered an error: {e}")
        return image
        
    report_progress("Face enhancement complete.")
    return restored_img
