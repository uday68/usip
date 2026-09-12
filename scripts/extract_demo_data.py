import os
import zipfile
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEMO_DIR = BASE_DIR / "demo_samples"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CACHE_DIR = Path("D:/USIP_DEMO_DATA")
TARGET_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ARCHIVES = {
    "shipwreck": "D:/AI4Shipwrecks.zip",
    "debris": "D:/AquaScan-1K.zip",
    "pipe": "D:/SubPipeMini.zip"
}

def extract_curated_samples():
    print("Extracting curated demonstration sonar samples...")
    extracted_counts = {}

    # 1. Shipwrecks
    shipwreck_zip = Path(ARCHIVES["shipwreck"])
    if shipwreck_zip.exists():
        with zipfile.ZipFile(shipwreck_zip, 'r') as z:
            sample_candidates = [
                f for f in z.namelist() 
                if f.startswith("AI4Shipwrecks/train/images/") and f.endswith(".png")
            ]
            shipwreck_samples = sample_candidates[:5]
            for i, member in enumerate(shipwreck_samples):
                out_name = f"sample_shipwreck_{i+1}.png"
                dest_local = DEMO_DIR / out_name
                dest_cache = TARGET_CACHE_DIR / out_name
                with z.open(member) as src:
                    content = src.read()
                    dest_local.write_bytes(content)
                    dest_cache.write_bytes(content)
            extracted_counts["shipwreck"] = len(shipwreck_samples)
            print(f"Extracted {len(shipwreck_samples)} shipwreck samples from AI4Shipwrecks.")
    else:
        print("Shipwreck archive not found.")

    # 2. Debris / Targets (AquaScan-1K)
    debris_zip = Path(ARCHIVES["debris"])
    if debris_zip.exists():
        with zipfile.ZipFile(debris_zip, 'r') as z:
            sample_candidates = [
                f for f in z.namelist()
                if f.startswith("AquaScan-1K/images/") and (f.endswith(".png") or f.endswith(".jpg"))
            ]
            debris_samples = sample_candidates[:5]
            for i, member in enumerate(debris_samples):
                ext = Path(member).suffix
                out_name = f"sample_debris_{i+1}{ext}"
                dest_local = DEMO_DIR / out_name
                dest_cache = TARGET_CACHE_DIR / out_name
                with z.open(member) as src:
                    content = src.read()
                    dest_local.write_bytes(content)
                    dest_cache.write_bytes(content)
            extracted_counts["debris"] = len(debris_samples)
            print(f"Extracted {len(debris_samples)} debris samples from AquaScan-1K.")
    else:
        print("AquaScan archive not found.")

    # 3. Pipeline / Cylinder (SubPipeMini)
    pipe_zip = Path(ARCHIVES["pipe"])
    if pipe_zip.exists():
        with zipfile.ZipFile(pipe_zip, 'r') as z:
            sample_candidates = [
                f for f in z.namelist()
                if "Cam0_images" in f and (f.endswith(".jpg") or f.endswith(".png"))
            ]
            pipe_samples = sample_candidates[:5]
            for i, member in enumerate(pipe_samples):
                ext = Path(member).suffix
                out_name = f"sample_pipe_{i+1}{ext}"
                dest_local = DEMO_DIR / out_name
                dest_cache = TARGET_CACHE_DIR / out_name
                with z.open(member) as src:
                    content = src.read()
                    dest_local.write_bytes(content)
                    dest_cache.write_bytes(content)
            extracted_counts["pipe"] = len(pipe_samples)
            print(f"Extracted {len(pipe_samples)} pipeline samples from SubPipeMini.")
    else:
        print("SubPipe archive not found.")

    # 4. Unknown Anomaly Candidate Sample
    if (DEMO_DIR / "sample_shipwreck_1.png").exists():
        shutil.copy(DEMO_DIR / "sample_shipwreck_1.png", DEMO_DIR / "sample_anomaly_1.png")
        print("Created reference anomaly sample.")

    print(f"Done. Demo samples saved to {DEMO_DIR}")

if __name__ == "__main__":
    extract_curated_samples()
