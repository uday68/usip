# Sonar-Specific Acoustic Preprocessing Pipeline

**Module**: `ml/preprocessing/`  
**Platform**: USIP (SIH26057)

---

## 1. Acoustic Integrity Principles
Standard computer vision transformations (e.g. random vertical flips, arbitrary 90° rotations, heavy color jitter) destroy physical sonar geometry:
- **Across-Track Range vs Along-Track Time**: Sonar rows represent sequential along-track acoustic pulses; columns represent slant-range acoustic travel time.
- **Port/Starboard Symmetry**: Horizontal flip is the only physically invariant reflection (swapping port and starboard acoustic channels).
- **Acoustic Shadow Occlusion**: Shadows always fall away from the sonar transducer/flight path.

---

## 2. Implemented Preprocessing Stages

### A. Dynamic Range Normalization (`normalize.py`)
- Standard min-max normalization is vulnerable to solitary extreme acoustic speckle spikes.
- We implement percentile-based clipping ($P_1$ to $P_{99}$), ensuring the full 8-bit dynamic range is allocated to genuine specular highlights and acoustic cast shadows.

### B. Bilateral Despeckle Filtering (`normalize.py`)
- Acoustic side-scan imagery suffers from multiplicative Rayleigh speckle noise.
- Standard Gaussian or box blurring smears the critical high-frequency boundary between a hard target highlight and its acoustic shadow.
- Bilateral filtering smooths uniform seabed sediment while strictly preserving sharp highlight edges.

### C. Contrast Limited Adaptive Histogram Equalization (CLAHE) (`normalize.py`)
- Compensates for acoustic transmission loss across the swath without over-amplifying background noise in low-return sediment zones.

### D. Adaptive Overlap Tiling (`tiling.py`)
- Sonar waterfall strips often exceed 5,000 pixels in range while having only 500 pings in length.
- Resizing a 5000×500 strip directly into 640×640 would obliterate small anthropogenic debris (downsampling them to sub-pixel noise).
- The `SonarTiler` crops the strip into 640×640 patches with a 15–20% overlap along-track and across-track, and maintains coordinate inversion mappings back to global ping space.

### E. Sensor Quality Control (`quality_control.py`)
- Automatically assesses acoustic signal-to-noise ratio (SNR in dB), dead ping ratio, and receiver saturation.

