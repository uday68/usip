import cv2
import numpy as np
from typing import Dict, Any

class SonarQualityControl:
    """
    Automated quality control for acoustic side-scan sonar data.
    Validates sensor data integrity before inference or training.
    """
    def __init__(self, min_contrast: float = 10.0, max_dead_ratio: float = 0.40):
        self.min_contrast = min_contrast
        self.max_dead_ratio = max_dead_ratio

    def assess_quality(self, gray: np.ndarray) -> Dict[str, Any]:
        """
        Calculates acoustic quality metrics:
        - SNR estimation (signal mean vs seabed noise std)
        - Dead pixel / dropped ping ratio
        - Dynamic range span
        - Passing status
        """
        h, w = gray.shape[:2]
        total_pixels = h * w
        
        # Dead pixels (near zero amplitude)
        dead_pixels = np.count_nonzero(gray <= 2)
        dead_ratio = float(dead_pixels) / float(total_pixels)

        # Saturated pixels (receiver clip)
        saturated_pixels = np.count_nonzero(gray >= 253)
        saturation_ratio = float(saturated_pixels) / float(total_pixels)

        # Mean and standard deviation
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))

        # Dynamic range (1st to 99th percentile)
        p1 = float(np.percentile(gray, 1))
        p99 = float(np.percentile(gray, 99))
        dynamic_range = p99 - p1

        # Estimated SNR in dB
        snr_db = 20.0 * np.log10(max(1.0, mean_val) / max(1.0, std_val))

        is_valid = True
        warnings = []

        if dead_ratio > self.max_dead_ratio:
            is_valid = False
            warnings.append(f"Excessive dead/dropped acoustic pings: {dead_ratio*100:.1f}%")

        if dynamic_range < self.min_contrast:
            is_valid = False
            warnings.append(f"Severely low acoustic dynamic range: {dynamic_range:.1f}")

        if saturation_ratio > 0.20:
            warnings.append(f"Acoustic receiver saturation detected: {saturation_ratio*100:.1f}%")

        return {
            "is_valid": is_valid,
            "mean_intensity": round(mean_val, 2),
            "std_intensity": round(std_val, 2),
            "dynamic_range": round(dynamic_range, 2),
            "dead_ratio": round(dead_ratio, 4),
            "saturation_ratio": round(saturation_ratio, 4),
            "estimated_snr_db": round(snr_db, 2),
            "warnings": warnings
        }
