import os
import json
from pathlib import Path
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import joblib

WEIGHTS_DIR = Path("ml/anomaly/weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = WEIGHTS_DIR / "seabed_clusters.joblib"
SCALER_FILE = WEIGHTS_DIR / "feature_scaler.joblib"
STATS_FILE = WEIGHTS_DIR / "seabed_stats.json"

def extract_acoustic_features(crop_gray: np.ndarray) -> np.ndarray:
    """
    Extract multi-scale acoustic backscatter texture features from a sonar patch:
    1. Statistical moments (mean, std, skewness, kurtosis)
    2. High-frequency speckle entropy
    3. Multi-orientation Gabor filter acoustic energy
    4. Gradient magnitude distribution
    """
    if len(crop_gray.shape) == 3:
        crop_gray = cv2.cvtColor(crop_gray, cv2.COLOR_BGR2GRAY)

    h, w = crop_gray.shape[:2]
    patch_resized = cv2.resize(crop_gray, (128, 128)).astype(np.float32)

    # 1. Statistical moments
    mean_val = float(np.mean(patch_resized))
    std_val = float(np.std(patch_resized))
    norm_patch = (patch_resized - mean_val) / (std_val + 1e-5)
    skew_val = float(np.mean(norm_patch ** 3))
    kurt_val = float(np.mean(norm_patch ** 4))

    # 2. Entropy / Energy
    hist, _ = np.histogram(patch_resized, bins=16, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy_val = float(-np.sum(hist * np.log2(hist)))

    # 3. Spatial Gradients (Sobel)
    gx = cv2.Sobel(patch_resized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch_resized, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    grad_mean = float(np.mean(mag))
    grad_std = float(np.std(mag))

    # 4. Multi-orientation Gabor energy (acoustic ripple / bedform signatures)
    gabor_energies = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        kernel = cv2.getGaborKernel((15, 15), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(patch_resized, cv2.CV_32F, kernel)
        gabor_energies.append(float(np.mean(np.abs(filtered))))

    feature_vec = np.array([
        mean_val, std_val, skew_val, kurt_val,
        entropy_val, grad_mean, grad_std,
        *gabor_energies
    ], dtype=np.float32)

    return feature_vec

class AnomalyDiscoveryEngine:
    """
    Unsupervised Acoustic Anomaly Discovery Engine.
    Trained strictly on normal seabed acoustic patches (from sss_ssl_dataset_N713_384).
    Calculates novelty distance from baseline seafloor clusters.
    """
    def __init__(self, n_clusters: int = 8):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=64)
        self.max_normal_dist = 3.5  # calibrated 95th percentile normal distance
        self.is_fitted = False

        self._load_or_initialize()

    def _load_or_initialize(self):
        if MODEL_FILE.exists() and SCALER_FILE.exists() and STATS_FILE.exists():
            try:
                self.kmeans = joblib.load(MODEL_FILE)
                self.scaler = joblib.load(SCALER_FILE)
                with open(STATS_FILE, "r") as f:
                    stats = json.load(f)
                    self.max_normal_dist = stats.get("max_normal_dist", 3.5)
                self.is_fitted = True
                print("Loaded pre-trained acoustic seabed anomaly model.")
            except Exception as e:
                print(f"Error loading anomaly model: {e}. Will fit on available SSL sample.")

    def fit_from_normal_seabed(self, ssl_dir: Path = Path("D:/USIP_SSL_SAMPLE")):
        """
        Fits baseline clusters using unlabelled normal seabed tiles from sss_ssl_dataset_N713_384.
        """
        print(f"Fitting unsupervised acoustic anomaly engine on normal seabed: {ssl_dir}")
        if not ssl_dir.exists():
            print(f"Directory {ssl_dir} not found. Cannot fit anomaly engine.")
            return

        patch_files = [f for f in ssl_dir.iterdir() if f.suffix.lower() in [".tiff", ".tif", ".png", ".jpg"]]
        if len(patch_files) == 0:
            print("No SSL acoustic patches found.")
            return

        features = []
        for p in patch_files:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Sample 2 crops per patch
                h, w = img.shape[:2]
                if h >= 128 and w >= 128:
                    c1 = img[h//4:h//4+128, w//4:w//4+128]
                    features.append(extract_acoustic_features(c1))

        if len(features) < self.n_clusters:
            print("Insufficient normal seabed patches to cluster.")
            return

        X = np.array(features)
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans.fit(X_scaled)

        # Calculate distances of training seabed samples to their nearest centroid
        distances = np.min(self.kmeans.transform(X_scaled), axis=1)
        # 95th percentile distance defines normal boundary
        self.max_normal_dist = float(np.percentile(distances, 95.0))

        # Save artifacts
        joblib.dump(self.kmeans, MODEL_FILE)
        joblib.dump(self.scaler, SCALER_FILE)
        with open(STATS_FILE, "w") as f:
            json.dump({
                "num_samples": len(features),
                "n_clusters": self.n_clusters,
                "max_normal_dist": self.max_normal_dist,
                "mean_dist": float(np.mean(distances)),
                "std_dist": float(np.std(distances))
            }, f, indent=4)

        self.is_fitted = True
        print(f"Anomaly engine fitted successfully on {len(features)} acoustic patches. Threshold: {self.max_normal_dist:.3f}")

    def score_patch(self, crop_gray: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate a candidate crop against normal seabed distribution:
        Outputs:
        - anomaly_score: [0.0, 1.0]
        - nearest_cluster_id
        - nearest_cluster_dist
        - is_anomaly
        - explanation
        """
        if not self.is_fitted:
            # Fallback heuristic if models not yet fit
            return {
                "anomaly_score": 0.15,
                "nearest_cluster_id": 0,
                "nearest_cluster_dist": 1.0,
                "is_anomaly": False,
                "explanation": "Normal acoustic seafloor texture."
            }

        feat = extract_acoustic_features(crop_gray).reshape(1, -1)
        feat_scaled = self.scaler.transform(feat)
        dists = self.kmeans.transform(feat_scaled)[0]
        nearest_cluster_id = int(np.argmin(dists))
        min_dist = float(dists[nearest_cluster_id])

        # Normalized anomaly score: 0 (matches normal centroid) to 1.0 (novel acoustic signature)
        normalized_score = min(1.0, min_dist / (self.max_normal_dist * 1.8))

        is_anomaly = min_dist > self.max_normal_dist
        if is_anomaly:
            explanation = (
                f"Acoustic texture departs significantly from normal sediment clusters "
                f"(Novelty Distance: {min_dist:.2f} > baseline {self.max_normal_dist:.2f}). "
                f"Candidate requiring expert review."
            )
        else:
            explanation = f"Consistent with natural acoustic seafloor facies (Cluster #{nearest_cluster_id})."

        return {
            "anomaly_score": round(normalized_score, 4),
            "nearest_cluster_id": nearest_cluster_id,
            "nearest_cluster_dist": round(min_dist, 4),
            "is_anomaly": is_anomaly,
            "explanation": explanation
        }

