import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

class SonarTiler:
    """
    Adaptive tiling for extreme aspect-ratio side-scan sonar waterfall strips.
    Splits long continuous acoustic records (e.g., 5000x500 or 6749x1728) into square
    model-ready crops (e.g. 640x640) with along-track and across-track overlap.
    """
    def __init__(self, tile_size: int = 640, overlap_ratio: float = 0.20):
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.step = int(tile_size * (1.0 - overlap_ratio))

    def generate_tiles(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generate tiled crops with their pixel bounding coordinates in the source image.
        """
        h, w = image.shape[:2]
        tiles = []

        y_starts = list(range(0, max(1, h - self.tile_size + 1), self.step))
        if len(y_starts) == 0 or (y_starts[-1] + self.tile_size < h):
            y_starts.append(max(0, h - self.tile_size))
        y_starts = sorted(list(set(y_starts)))

        x_starts = list(range(0, max(1, w - self.tile_size + 1), self.step))
        if len(x_starts) == 0 or (x_starts[-1] + self.tile_size < w):
            x_starts.append(max(0, w - self.tile_size))
        x_starts = sorted(list(set(x_starts)))

        for y0 in y_starts:
            for x0 in x_starts:
                x1 = min(w, x0 + self.tile_size)
                y1 = min(h, y0 + self.tile_size)
                
                # Crop
                tile_crop = image[y0:y1, x0:x1]
                
                # If smaller than tile_size, pad with zero (acoustic silence/seabed baseline)
                crop_h, crop_w = tile_crop.shape[:2]
                if crop_h < self.tile_size or crop_w < self.tile_size:
                    if len(image.shape) == 2:
                        padded = np.zeros((self.tile_size, self.tile_size), dtype=image.dtype)
                        padded[:crop_h, :crop_w] = tile_crop
                    else:
                        padded = np.zeros((self.tile_size, self.tile_size, image.shape[2]), dtype=image.dtype)
                        padded[:crop_h, :crop_w, :] = tile_crop
                    tile_crop = padded

                tiles.append({
                    "tile_image": tile_crop,
                    "crop_coords": (x0, y0, x1, y1),
                    "valid_size": (crop_w, crop_h)
                })

        return tiles

    def map_bbox_to_global(
        self, 
        tile_bbox: Tuple[float, float, float, float], 
        crop_coords: Tuple[int, int, int, int]
    ) -> Tuple[float, float, float, float]:
        """
        Convert a bounding box from local tile coordinates (x, y, w, h)
        back to global source image coordinates.
        """
        x_local, y_local, w_local, h_local = tile_bbox
        x0, y0, _, _ = crop_coords
        return (x0 + x_local, y0 + y_local, w_local, h_local)
