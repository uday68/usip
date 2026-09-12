import pytest
import numpy as np
from ml.preprocessing.pipeline import SonarPreprocessingPipeline
from ml.preprocessing.tiling import SonarTiler
from ml.preprocessing.quality_control import SonarQualityControl
from ml.preprocessing.augmentation import AcousticSafeAugmenter

def test_preprocessing_pipeline_execution():
    # Synthetic sonar test pattern (speckle + highlight + shadow)
    np.random.seed(42)
    synthetic_sonar = np.random.normal(100, 20, (1000, 500)).astype(np.uint8)
    # Add specular highlight
    synthetic_sonar[200:230, 200:240] = 245
    # Add acoustic shadow
    synthetic_sonar[200:230, 240:290] = 10

    pipeline = SonarPreprocessingPipeline()
    enhanced, qc = pipeline.process(synthetic_sonar)

    assert enhanced.shape == synthetic_sonar.shape
    assert enhanced.dtype == np.uint8
    assert "is_valid" in qc
    assert "estimated_snr_db" in qc
    assert qc["is_valid"] is True

def test_tiler_and_coordinate_mapping():
    tiler = SonarTiler(tile_size=640, overlap_ratio=0.2)
    dummy_strip = np.zeros((1500, 600), dtype=np.uint8)
    tiles = tiler.generate_tiles(dummy_strip)

    assert len(tiles) > 0
    first_tile = tiles[0]
    assert first_tile["tile_image"].shape == (640, 640)

    # Test coordinate re-mapping
    local_box = (50.0, 50.0, 100.0, 80.0)
    crop_coords = first_tile["crop_coords"]
    global_box = tiler.map_bbox_to_global(local_box, crop_coords)
    assert global_box[0] == crop_coords[0] + 50.0
    assert global_box[1] == crop_coords[1] + 50.0

def test_acoustic_augmentation():
    aug = AcousticSafeAugmenter(p_flip=1.0, p_speckle=1.0, p_gain=1.0)
    dummy_img = np.ones((100, 100), dtype=np.uint8) * 128
    dummy_boxes = np.array([[0, 0.2, 0.5, 0.1, 0.1]])

    aug_img, aug_boxes = aug(dummy_img, dummy_boxes)
    assert aug_img.shape == dummy_img.shape
    # Flipped horizontally: 1.0 - 0.2 = 0.8
    assert np.isclose(aug_boxes[0, 1], 0.8)
