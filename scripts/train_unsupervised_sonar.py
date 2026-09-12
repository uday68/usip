import os
import json
import numpy as np
import tifffile
import cv2
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

def extract_patch_features(patch_gray: np.ndarray) -> np.ndarray:
    """
    Extracts acoustic texture & frequency signature from an SSS patch:
    - 4 Statistical intensity moments (mean, std, skewness, kurtosis)
    - 2 Gradient energy moments (Sobel gradient mean, gradient standard deviation)
    - 1 Spatial frequency energy (2D FFT high-frequency power)
    """
    p = cv2.resize(patch_gray, (128, 128)).astype(np.float32) / 255.0

    mean = float(np.mean(p))
    std = float(np.std(p))
    skew = float(np.mean(((p - mean) / (std + 1e-6)) ** 3))
    kurt = float(np.mean(((p - mean) / (std + 1e-6)) ** 4))

    # Gradient energy
    gx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    grad_mean = float(np.mean(mag))
    grad_std = float(np.std(mag))

    # 2D-FFT Spatial frequency energy
    f = np.fft.fft2(p)
    fshift = np.fft.fftshift(f)
    mag_spectrum = 20 * np.log(np.abs(fshift) + 1e-5)
    high_freq_energy = float(np.mean(mag_spectrum[mag_spectrum > np.percentile(mag_spectrum, 75)]))

    vec = np.array([mean, std, skew, kurt, grad_mean, grad_std, high_freq_energy], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

def train_unsupervised_sonar():
    data_dir = Path("D:/USIP_SSL_SAMPLE")
    tiff_files = list(data_dir.glob("*.tiff"))
    print(f"Loading {len(tiff_files)} unlabelled SSS sonar patches from {data_dir}...")

    if len(tiff_files) == 0:
        raise RuntimeError("No unlabelled TIFF patches found in D:/USIP_SSL_SAMPLE.")

    features = []
    processed_count = 0

    for fpath in tiff_files:
        try:
            img = tifffile.imread(str(fpath))
            # If multi-channel, use primary acoustic channel
            if len(img.shape) == 3:
                gray = img[:, :, 0]
            else:
                gray = img

            feat = extract_patch_features(gray)
            features.append(feat)
            processed_count += 1
            if processed_count % 50 == 0:
                print(f" Processed {processed_count}/{len(tiff_files)} patches...")
        except Exception as e:
            continue

    X = np.array(features, dtype=np.float32)
    print(f"\nFeature extraction complete: {X.shape[0]} samples x {X.shape[1]} acoustic dimensions.")

    # 1. Fit K-Means Clustering on unlabelled seabed representations (k=5 natural seabed regimes)
    k = 5
    print(f"Training Unsupervised KMeans Clustering with k={k} on unlabelled sonar data...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_

    # 2. Fit Isolation Forest to learn normal seabed density envelope
    print("Training Isolation Forest to learn baseline seabed density envelope...")
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(X)

    # 3. Compute baseline distance distribution across all unlabelled seabed patches
    all_dists = []
    for x in X:
        dists = np.linalg.norm(centroids - x, axis=1)
        all_dists.append(float(np.min(dists)))

    all_dists = np.array(all_dists)
    mean_dist = float(np.mean(all_dists))
    std_dist = float(np.std(all_dists))
    p50_dist = float(np.percentile(all_dists, 50))
    p90_dist = float(np.percentile(all_dists, 90))
    p95_dist = float(np.percentile(all_dists, 95))
    max_dist = float(np.max(all_dists))

    print(f"\nSeabed Baseline Acoustic Distance Metrics:")
    print(f"  Mean Distance to Cluster: {mean_dist:.4f} (± {std_dist:.4f})")
    print(f"  50th Percentile (Normal): {p50_dist:.4f}")
    print(f"  95th Percentile (Upper Bound): {p95_dist:.4f}")
    print(f"  Max Seabed Dispersion:   {max_dist:.4f}")

    # 4. Save Trained Model Artifacts
    models_dir = Path(__file__).resolve().parent.parent / "backend" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    centroids_path = models_dir / "unsupervised_clusters.npy"
    np.save(str(centroids_path), centroids)
    print(f"\nSaved trained cluster centroids to {centroids_path}")

    stats = {
        "dataset": "sss_ssl_dataset_N713_384 (Unlabelled SSS Sonar)",
        "training_samples_count": int(X.shape[0]),
        "feature_dimensions": int(X.shape[1]),
        "clusters_count": k,
        "mean_cluster_distance": round(mean_dist, 4),
        "std_cluster_distance": round(std_dist, 4),
        "p50_distance": round(p50_dist, 4),
        "p95_distance": round(p95_dist, 4),
        "max_distance": round(max_dist, 4)
    }

    stats_path = models_dir / "cluster_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Saved cluster training stats to {stats_path}")
    print("\nUNSUPERVISED TRAINING ON UNLABELLED SONAR DATA COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    train_unsupervised_sonar()
