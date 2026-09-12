import cv2
import numpy as np
from typing import Tuple, Optional

def to_acoustic_gray(image: np.ndarray) -> np.ndarray:
    """
    Ensure input is single-channel acoustic backscatter intensity.
    If 3-channel RGB or colorized sonar waterfall, converts to luminance.
    """
    if len(image.shape) == 2:
        return image.copy()
    elif len(image.shape) == 3:
        if image.shape[2] == 1:
            return image[:, :, 0].copy()
        elif image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported acoustic image shape: {image.shape}")

def dynamic_range_normalize(
    gray: np.ndarray, 
    p_low: float = 1.0, 
    p_high: float = 99.0
) -> np.ndarray:
    """
    Acoustic dynamic range normalization using percentile clipping.
    Preserves true specular highlights and acoustic cast shadows without
    letting extreme outlier speckle pings distort the dynamic range.
    """
    gray_float = gray.astype(np.float32)
    val_low = np.percentile(gray_float, p_low)
    val_high = np.percentile(gray_float, p_high)

    if val_high <= val_low:
        return np.clip(gray, 0, 255).astype(np.uint8)

    stretched = np.clip((gray_float - val_low) / (val_high - val_low) * 255.0, 0, 255.0)
    return stretched.astype(np.uint8)

def bilateral_despeckle(
    gray: np.ndarray, 
    d: int = 5, 
    sigma_color: float = 40.0, 
    sigma_space: float = 40.0
) -> np.ndarray:
    """
    Speckle-aware smoothing that preserves sharp acoustic specular highlight boundaries
    and cast-shadow drop-offs while dampening high-frequency acoustic speckle.
    """
    return cv2.bilateralFilter(gray, d, sigma_color, sigma_space)

def adaptive_clahe(
    gray: np.ndarray, 
    clip_limit: float = 2.5, 
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization specifically tuned for
    side-scan sonar backscatter variations across swath range.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)

def time_varied_gain_correction(gray: np.ndarray) -> np.ndarray:
    """
    Approximate empirical Time-Varied Gain (TVG) correction across the swath
    to compensate for acoustic transmission loss and spherical spreading.
    Normalizes column-wise average backscatter energy.
    """
    h, w = gray.shape
    col_means = np.mean(gray.astype(np.float32), axis=0) + 1e-5
    target_mean = np.median(col_means)
    gain_curve = target_mean / col_means
    # Dampen extreme gain factors at far range
    gain_curve = np.clip(gain_curve, 0.4, 2.5)
    
    corrected = gray.astype(np.float32) * gain_curve[np.newaxis, :]
    return np.clip(corrected, 0, 255).astype(np.uint8)
