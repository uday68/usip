import cv2
import numpy as np
import random
from typing import Tuple

class AcousticSafeAugmenter:
    """
    Acoustic-safe data augmentation.
    Specifically respects side-scan sonar physical geometry:
    - Horizontal Flip (Port <-> Starboard channel symmetry is acoustically valid).
    - Multiplicative Rayleigh Speckle Noise injection (physically matches acoustic backscatter coherence).
    - Along-track gain fluctuation (simulates vehicle speed and altitude variations).
    - Cross-range gain attenuation (simulates acoustic spreading loss across swath).
    - FORBIDDEN: Arbitrary 90-degree rotations (flips across-track range physics with along-track time).
    - FORBIDDEN: Vertical flips (inverts sonar grazing shadow geometry relative to towfish).
    """
    def __init__(
        self, 
        p_flip: float = 0.5, 
        p_speckle: float = 0.3, 
        p_gain: float = 0.3,
        p_attenuation: float = 0.2
    ):
        self.p_flip = p_flip
        self.p_speckle = p_speckle
        self.p_gain = p_gain
        self.p_attenuation = p_attenuation

    def __call__(
        self, 
        image: np.ndarray, 
        bboxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        image: H x W or H x W x C
        bboxes: Nx5 array of [class_id, x_center, y_center, w, h] normalized in [0, 1]
        """
        aug_image = image.copy()
        aug_boxes = bboxes.copy() if len(bboxes) > 0 else np.zeros((0, 5))

        # 1. Horizontal Flip (port/starboard swap)
        if random.random() < self.p_flip:
            aug_image = cv2.flip(aug_image, 1)
            if len(aug_boxes) > 0:
                # Invert normalized x_center
                aug_boxes[:, 1] = 1.0 - aug_boxes[:, 1]

        # 2. Acoustic Multiplicative Speckle Noise (Rayleigh or Gaussian)
        if random.random() < self.p_speckle:
            h, w = aug_image.shape[:2]
            # Rayleigh noise normalized around mean 1.0: E[Rayleigh(s)] = s * sqrt(pi/2)
            scale = 0.08
            rayleigh_raw = np.random.rayleigh(scale=scale, size=(h, w)).astype(np.float32)
            noise = (rayleigh_raw - scale * np.sqrt(np.pi / 2.0))
            if len(aug_image.shape) == 3:
                noise = noise[:, :, np.newaxis]
            noisy = aug_image.astype(np.float32) * (1.0 + noise)
            aug_image = np.clip(noisy, 0, 255).astype(np.uint8)

        # 3. Along-track gain fluctuation (gradual brightness shift along Y-axis)
        if random.random() < self.p_gain:
            h = aug_image.shape[0]
            gain_factor = random.uniform(0.85, 1.15)
            ramp = np.linspace(1.0, gain_factor, h, dtype=np.float32)[:, np.newaxis]
            if len(aug_image.shape) == 3:
                ramp = ramp[:, :, np.newaxis]
            aug_image = np.clip(aug_image.astype(np.float32) * ramp, 0, 255).astype(np.uint8)

        # 4. Cross-range acoustic gain attenuation (gradual roll-off along swath X-axis)
        if random.random() < self.p_attenuation:
            w = aug_image.shape[1]
            atten_factor = random.uniform(0.80, 1.20)
            range_ramp = np.linspace(1.0, atten_factor, w, dtype=np.float32)[np.newaxis, :]
            if len(aug_image.shape) == 3:
                range_ramp = range_ramp[:, :, np.newaxis]
            aug_image = np.clip(aug_image.astype(np.float32) * range_ramp, 0, 255).astype(np.uint8)

        return aug_image, aug_boxes
