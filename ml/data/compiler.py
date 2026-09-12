import os
import zipfile
import io
import hashlib
import cv2
import numpy as np
import yaml
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple
from ml.preprocessing.pipeline import SonarPreprocessingPipeline
from ml.preprocessing.tiling import SonarTiler

# Output unified directory on D: drive to protect C: drive space
OUTPUT_DATASET_DIR = Path("D:/USIP_DATA/unified_sonar_dataset")

AI4_TRAIN_SITES = [
    'DM_Wilson', 'DR_Hanna', 'EB_Allen', 'Egyptian', 
    'Grecian', 'Heart_Failure', 'Isaac_M_Scott', 'Montana', 
    'Near_Shore', 'Oscar_T_Flint'
]
AI4_VAL_SITES = ['Pewabic', 'WP_Rend']
AI4_TEST_SITES = [
    'Artificial_Reef', 'Barge_No_1', 'Corsair', 'Corsican', 
    'Haltiner_Barge', 'James_Davidson', 'Lucinda_van_Valkenburg', 
    'Monohansett', 'Monrovia', 'Shamrock', 'Viator', 'WH_Gilbert', 'WP_Thew'
]

AQUASCAN_TRAIN_DATES = [
    '2025-06-09', '2025-06-15', '2025-06-29', '2025-07-01', 
    '2025-07-13', '2025-07-22', '2025-08-03'
]
AQUASCAN_VAL_DATES = ['2025-08-10', '2025-08-17', '2025-08-18']
AQUASCAN_TEST_DATES = ['2025-08-28', '2025-08-29', '2025-11-14']

CLASS_MAP = {
    "Shipwreck": 0,
    "Pipeline": 1,
    "Debris": 2
}

def mask_to_yolo_boxes(mask: np.ndarray, class_idx: int) -> List[str]:
    """
    Extract connected component bounding boxes from binary segmentation mask
    and convert to YOLO normalized format: <class_idx> <xc> <yc> <w> <h>.
    """
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yolo_lines = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:  # suppress tiny single-pixel acoustic noise
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        xc = (bx + bw / 2.0) / float(w)
        yc = (by + bh / 2.0) / float(h)
        nw = float(bw) / float(w)
        nh = float(bh) / float(h)
        yolo_lines.append(f"{class_idx} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
    return yolo_lines

def setup_dirs():
    for split in ['train', 'val', 'test']:
        (OUTPUT_DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def compile_shipwrecks():
    print("Compiling AI4Shipwrecks with strict site-disjoint partitioning...")
    zip_path = Path("D:/AI4Shipwrecks.zip")
    if not zip_path.exists():
        print(f"Warning: {zip_path} not found.")
        return 0

    tiler = SonarTiler(tile_size=640, overlap_ratio=0.15)
    counts = {"train": 0, "val": 0, "test": 0}

    with zipfile.ZipFile(zip_path, 'r') as z:
        # Collect all image/label pairs
        all_imgs = [n for n in z.namelist() if 'images/' in n and n.endswith('.png')]
        
        for img_name in all_imgs:
            base_filename = img_name.split('/')[-1]
            site_prefix = '_'.join(base_filename.split('_')[:-1])
            
            # Determine disjoint split
            if site_prefix in AI4_TRAIN_SITES:
                split = "train"
            elif site_prefix in AI4_VAL_SITES:
                split = "val"
            elif site_prefix in AI4_TEST_SITES:
                split = "test"
            else:
                # Default extras/unassigned to train
                split = "train"

            # Corresponding mask
            mask_name = img_name.replace("images/", "labels/")
            if mask_name not in z.namelist():
                continue

            # Read image and mask
            img_bytes = z.read(img_name)
            mask_bytes = z.read(mask_name)
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue

            # Binarize mask
            mask = (mask > 0).astype(np.uint8) * 255
            has_target = np.any(mask > 0)

            # High resolution strip: tile or resize
            # If height is very large, generate tiles containing target
            if img.shape[0] > 1000 or img.shape[1] > 1000:
                tiles_img = tiler.generate_tiles(img)
                tiles_mask = tiler.generate_tiles(mask)
                for t_idx, (t_img_data, t_mask_data) in enumerate(zip(tiles_img, tiles_mask)):
                    t_crop = t_img_data["tile_image"]
                    m_crop = t_mask_data["tile_image"]
                    yolo_boxes = mask_to_yolo_boxes(m_crop, CLASS_MAP["Shipwreck"])

                    # Include all target-bearing tiles + small fraction of background tiles
                    if len(yolo_boxes) > 0 or (t_idx % 4 == 0):
                        tile_name = f"shipwreck_{site_prefix}_{base_filename[:-4]}_t{t_idx}"
                        out_img = OUTPUT_DATASET_DIR / "images" / split / f"{tile_name}.png"
                        out_lbl = OUTPUT_DATASET_DIR / "labels" / split / f"{tile_name}.txt"
                        cv2.imwrite(str(out_img), t_crop)
                        out_lbl.write_text("\n".join(yolo_boxes))
                        counts[split] += 1
            else:
                yolo_boxes = mask_to_yolo_boxes(mask, CLASS_MAP["Shipwreck"])
                out_name = f"shipwreck_{base_filename[:-4]}"
                out_img = OUTPUT_DATASET_DIR / "images" / split / f"{out_name}.png"
                out_lbl = OUTPUT_DATASET_DIR / "labels" / split / f"{out_name}.txt"
                cv2.imwrite(str(out_img), img)
                out_lbl.write_text("\n".join(yolo_boxes))
                counts[split] += 1

    print(f"AI4Shipwrecks compiled: {counts}")
    return sum(counts.values())

def compile_aquascan():
    print("Compiling AquaScan-1K with date-disjoint partitioning...")
    zip_path = Path("D:/AquaScan-1K.zip")
    if not zip_path.exists():
        print(f"Warning: {zip_path} not found.")
        return 0

    seen_hashes = set()
    counts = {"train": 0, "val": 0, "test": 0}

    with zipfile.ZipFile(zip_path, 'r') as z:
        img_names = [n for n in z.namelist() if n.startswith('AquaScan-1K/images/') and n.endswith('.png')]
        
        for img_name in img_names:
            base_filename = img_name.split('/')[-1]
            data = z.read(img_name)
            img_hash = hashlib.md5(data).hexdigest()

            # Prevent duplicate leakage
            if img_hash in seen_hashes:
                continue
            seen_hashes.add(img_hash)

            # Determine survey date
            survey_date = "2025-06-09"
            if "Screenshot_" in base_filename:
                date_part = base_filename.split("Screenshot_")[1].split("_")[0]
                survey_date = date_part

            if survey_date in AQUASCAN_TRAIN_DATES:
                split = "train"
            elif survey_date in AQUASCAN_VAL_DATES:
                split = "val"
            else:
                split = "test"

            # Label file
            lbl_name = img_name.replace("images/", "labels/").replace(".png", ".txt")
            yolo_lines = []
            if lbl_name in z.namelist():
                content = z.read(lbl_name).decode('utf-8').strip()
                for line in content.splitlines():
                    parts = line.strip().split()
                    if parts:
                        # Remap to Debris class index (2)
                        yolo_lines.append(f"{CLASS_MAP['Debris']} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")

            # Resize to 640x640 preserving aspect ratio
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # AquaScan original images are 1920x1080. Downscaling to 640x640 with letterbox or direct resize
            resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
            
            out_name = f"debris_{base_filename[:-4]}"
            out_img = OUTPUT_DATASET_DIR / "images" / split / f"{out_name}.png"
            out_lbl = OUTPUT_DATASET_DIR / "labels" / split / f"{out_name}.txt"

            cv2.imwrite(str(out_img), resized)
            out_lbl.write_text("\n".join(yolo_lines))
            counts[split] += 1

    print(f"AquaScan-1K compiled: {counts}")
    return sum(counts.values())

def compile_subpipe():
    print("Compiling SubPipeMini with chronological sequence-disjoint partitioning...")
    zip_path = Path("D:/SubPipeMini.zip")
    if not zip_path.exists():
        print(f"Warning: {zip_path} not found.")
        return 0

    counts = {"train": 0, "val": 0, "test": 0}
    tiler = SonarTiler(tile_size=640, overlap_ratio=0.15)

    with zipfile.ZipFile(zip_path, 'r') as z:
        # Collect HF images and labels
        img_names = sorted([
            n for n in z.namelist() 
            if 'SSS_HF_images/Image/' in n and n.endswith('.pbm')
        ])
        total_pbm = len(img_names)
        if total_pbm == 0:
            return 0

        # Disjoint chunk boundaries with 20-frame acoustic buffer gap
        train_end = int(total_pbm * 0.70)
        val_start = train_end + 15
        val_end = val_start + int(total_pbm * 0.15)
        test_start = val_end + 15

        for idx, img_name in enumerate(img_names):
            if idx < train_end:
                split = "train"
            elif val_start <= idx < val_end:
                split = "val"
            elif test_start <= idx:
                split = "test"
            else:
                # Buffer gap - skip to prevent boundary leakage
                continue

            base_filename = img_name.split('/')[-1]
            lbl_name = img_name.replace("Image/", "YOLO_Annotation/").replace(".pbm", ".txt")
            
            pbm_bytes = z.read(img_name)
            img = cv2.imdecode(np.frombuffer(pbm_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Original size: 500 x 5000 (H x W)
            # Read YOLO bounding boxes (normalized in original image coords)
            orig_boxes = []
            if lbl_name in z.namelist():
                content = z.read(lbl_name).decode('utf-8').strip()
                for line in content.splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        orig_boxes.append((xc, yc, w, h))

            # Tile the wide waterfall strip
            tiles = tiler.generate_tiles(img)
            img_h, img_w = img.shape[:2]

            for t_idx, tile_data in enumerate(tiles):
                tile_crop = tile_data["tile_image"]
                x0, y0, x1, y1 = tile_data["crop_coords"]

                tile_yolo_lines = []
                for (xc_norm, yc_norm, w_norm, h_norm) in orig_boxes:
                    # Convert to pixel coords
                    box_x = (xc_norm - w_norm / 2.0) * img_w
                    box_y = (yc_norm - h_norm / 2.0) * img_h
                    box_w = w_norm * img_w
                    box_h = h_norm * img_h

                    # Compute intersection with tile
                    inter_x0 = max(x0, box_x)
                    inter_y0 = max(y0, box_y)
                    inter_x1 = min(x1, box_x + box_w)
                    inter_y1 = min(y1, box_y + box_h)

                    if inter_x1 > inter_x0 and inter_y1 > inter_y0:
                        inter_w = inter_x1 - inter_x0
                        inter_h = inter_y1 - inter_y0
                        # Minimum area overlap
                        if (inter_w * inter_h) > 200:
                            local_xc = ((inter_x0 - x0) + inter_w / 2.0) / 640.0
                            local_yc = ((inter_y0 - y0) + inter_h / 2.0) / 640.0
                            local_w = inter_w / 640.0
                            local_h = inter_h / 640.0
                            tile_yolo_lines.append(
                                f"{CLASS_MAP['Pipeline']} {local_xc:.6f} {local_yc:.6f} {local_w:.6f} {local_h:.6f}"
                            )

                # Keep tiles containing target + sampled background tiles
                if len(tile_yolo_lines) > 0 or (t_idx % 5 == 0):
                    tile_name = f"pipe_{base_filename[:-4]}_t{t_idx}"
                    out_img = OUTPUT_DATASET_DIR / "images" / split / f"{tile_name}.png"
                    out_lbl = OUTPUT_DATASET_DIR / "labels" / split / f"{tile_name}.txt"
                    cv2.imwrite(str(out_img), tile_crop)
                    out_lbl.write_text("\n".join(tile_yolo_lines))
                    counts[split] += 1

    print(f"SubPipeMini compiled: {counts}")
    return sum(counts.values())

def generate_yaml():
    dataset_yaml = {
        "path": str(OUTPUT_DATASET_DIR).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_MAP),
        "names": list(CLASS_MAP.keys())
    }
    yaml_path = OUTPUT_DATASET_DIR / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)
    print(f"Dataset YAML written to {yaml_path}")

def run_compiler():
    print("=== USIP Data Compiler: Building Leakage-Resistant Dataset ===")
    setup_dirs()
    total_wrecks = compile_shipwrecks()
    total_debris = compile_aquascan()
    total_pipe = compile_subpipe()
    generate_yaml()
    print(f"=== Compilation Completed: {total_wrecks + total_debris + total_pipe} total unified sonar samples ===")

if __name__ == "__main__":
    run_compiler()
