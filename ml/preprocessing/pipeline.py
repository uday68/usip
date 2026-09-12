import cv2
import numpy as np
from typing import Dict, Any, Tuple
from ml.preprocessing.normalize import (
    to_acoustic_gray, 
    dynamic_range_normalize, 
    bilateral_despeckle, 
    adaptive_clahe, 
    time_varied_gain_correction
)
from ml.preprocessing.quality_control import SonarQualityControl
from ml.preprocessing.tiling import SonarTiler

class SonarPreprocessingPipeline:
    """
    Configurable acoustic preprocessing pipeline.
    Preserves true sonar physics while suppressing speckle and enhancing low-contrast backscatter.
    """
    def __init__(
        self,
        enable_tvg: bool = False,
        enable_despeckle: bool = True,
        enable_clahe: bool = True,
        clahe_clip: float = 2.5,
        clahe_grid: Tuple[int, int] = (8, 8),
        p_low: float = 1.0,
        p_high: float = 99.0,
        tile_size: int = 640
    ):
        self.enable_tvg = enable_tvg
        self.enable_despeckle = enable_despeckle
        self.enable_clahe = enable_clahe
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.p_low = p_low
        self.p_high = p_high
        self.qc = SonarQualityControl()
        self.tiler = SonarTiler(tile_size=tile_size)

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Execute acoustic pipeline:
        1. Acoustic luminance extraction
        2. Sensor quality control
        3. Dynamic range normalization
        4. Optional TVG correction
        5. Bilateral edge-preserving despeckling
        6. Acoustic CLAHE contrast enhancement
        """
        gray = to_acoustic_gray(image)
        qc_report = self.qc.assess_quality(gray)

        # 1. Dynamic range percentile normalization
        norm_gray = dynamic_range_normalize(gray, self.p_low, self.p_high)

        # 2. Time-Varied Gain (TVG) correction
        if self.enable_tvg:
            norm_gray = time_varied_gain_correction(norm_gray)

        # 3. Bilateral despeckle filter
        if self.enable_despeckle:
            despeckled = bilateral_despeckle(norm_gray)
        else:
            despeckled = norm_gray

        # 4. Adaptive CLAHE
        if self.enable_clahe:
            enhanced = adaptive_clahe(despeckled, self.clahe_clip, self.clahe_grid)
        else:
            enhanced = despeckled

        return enhanced, qc_report
