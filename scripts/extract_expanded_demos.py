import os
import zipfile
import cv2
import tifffile
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEMO_DIR = BASE_DIR / "demo_samples"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

ARCHIVES = {
    "shipwreck": "D:/AI4Shipwrecks.zip",
    "debris": "D:/AquaScan-1K.zip",
    "pipe": "D:/SubPipeMini.zip"
}

def extract_expanded_demos():
    print("Extracting rich, diverse demo samples from SSS archives...")

    # 1. More Shipwrecks: Isaac M. Scott, Montana, Pewabic, Egyptian
    shipwreck_zip = Path(ARCHIVES["shipwreck"])
    if shipwreck_zip.exists():
        with zipfile.ZipFile(shipwreck_zip, 'r') as z:
            named_wrecks = {
                "sample_shipwreck_isaac_scott.png": "AI4Shipwrecks/train/images/Isaac_M_Scott_01.png",
                "sample_shipwreck_montana.png": "AI4Shipwrecks/train/images/Montana_01.png",
                "sample_shipwreck_pewabic.png": "AI4Shipwrecks/train/images/Pewabic_01.png",
                "sample_shipwreck_grecian.png": "AI4Shipwrecks/train/images/Grecian_01.png"
            }
            for out_name, zip_path in named_wrecks.items():
                if zip_path in z.namelist():
                    with z.open(zip_path) as src:
                        (DEMO_DIR / out_name).write_bytes(src.read())
                    print(f" Extracted shipwreck: {out_name}")

    # 2. More Debris / Anthropogenic targets from AquaScan-1K
    debris_zip = Path(ARCHIVES["debris"])
    if debris_zip.exists():
        with zipfile.ZipFile(debris_zip, 'r') as z:
            sample_candidates = [
                f for f in z.namelist()
                if f.startswith("AquaScan-1K/images/") and (f.endswith(".png") or f.endswith(".jpg"))
            ]
            for i, member in enumerate(sample_candidates[5:10]):
                ext = Path(member).suffix
                out_name = f"sample_debris_{i+6}{ext}"
                with z.open(member) as src:
                    (DEMO_DIR / out_name).write_bytes(src.read())
                print(f" Extracted debris target: {out_name}")

    # 3. More Pipeline frames from SubPipeMini
    pipe_zip = Path(ARCHIVES["pipe"])
    if pipe_zip.exists():
        with zipfile.ZipFile(pipe_zip, 'r') as z:
            pipe_candidates = [
                f for f in z.namelist()
                if "Cam0_images" in f and (f.endswith(".jpg") or f.endswith(".png"))
            ]
            for i, member in enumerate(pipe_candidates[15:20]):
                ext = Path(member).suffix
                out_name = f"sample_pipe_{i+6}{ext}"
                with z.open(member) as src:
                    (DEMO_DIR / out_name).write_bytes(src.read())
                print(f" Extracted pipeline frame: {out_name}")

    # 4. Genuine Unlabelled SSS Patches from D:\USIP_SSL_SAMPLE
    ssl_dir = Path("D:/USIP_SSL_SAMPLE")
    if ssl_dir.exists():
        tiffs = list(ssl_dir.glob("*.tiff"))
        for i, tiff_path in enumerate(tiffs[:4]):
            try:
                img = tifffile.imread(str(tiff_path))
                if len(img.shape) == 3:
                    gray = img[:, :, 0]
                else:
                    gray = img
                out_name = f"sample_unlabelled_ssl_{i+1}.png"
                cv2.imwrite(str(DEMO_DIR / out_name), gray)
                print(f" Converted unlabelled SSS SSL patch to demo sample: {out_name}")
            except Exception as e:
                print(f"Error converting {tiff_path}: {e}")

    total_samples = list(DEMO_DIR.glob("*.*"))
    print(f"\nTotal demo samples now in {DEMO_DIR}: {len(total_samples)}")

if __name__ == "__main__":
    extract_expanded_demos()
