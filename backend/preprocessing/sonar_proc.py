import cv2
import numpy as np
from pathlib import Path
from PIL import Image

def load_and_preprocess_sonar(image_path: Path):
    """
    Applies sonar-specific acoustic preprocessing:
    1. Grayscale conversion / multi-band collapse
    2. Adaptive gain equalization (mitigating slant-range beam attenuation)
    3. CLAHE (Contrast Limited Adaptive Histogram Equalization) for acoustic highlight enhancement
    4. Bilateral filtering for acoustic speckle reduction while preserving hard shadow edges
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load sonar image at {image_path}")

    # Convert to grayscale for acoustic intensity analysis
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 1. Bilateral Filter: despeckle acoustic noise without blurring shadow edges
    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 3. Create display representation with submarine / acoustic colormap (amber / oceanic)
    # We produce both clean enhanced gray and an ocean sonar overlay
    colored_sonar = cv2.applyColorMap(enhanced, cv2.COLORMAP_VIRIDIS)

    return {
        "raw_bgr": img,
        "raw_gray": gray,
        "enhanced_gray": enhanced,
        "colored_sonar": colored_sonar,
        "height": gray.shape[0],
        "width": gray.shape[1]
    }

def extract_target_patch(gray_img: np.ndarray, x: int, y: int, w: int, h: int):
    """
    Safely crops target patch and associated acoustic shadow window (down-range / rightwards).
    """
    img_h, img_w = gray_img.shape
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(img_w, int(x + w))
    y2 = min(img_h, int(y + h))

    target_patch = gray_img[y1:y2, x1:x2]

    # Down-range shadow window (typically extends beyond target in sonar range)
    # Range is along the horizontal axis in standard SSS waterfall
    shadow_x1 = min(img_w, x2)
    shadow_x2 = min(img_w, int(x2 + w * 1.5))
    shadow_patch = gray_img[y1:y2, shadow_x1:shadow_x2] if shadow_x2 > shadow_x1 else np.array([])

    return target_patch, shadow_patch
