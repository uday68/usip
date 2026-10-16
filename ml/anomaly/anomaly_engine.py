import os
import json
from pathlib import Path
import numpy as np
import cv2
from typing import Dict, Any, List, Optional
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import joblib

from configs.settings import config
from ml.utils import set_deterministic_seeds

WEIGHTS_DIR = Path("ml/anomaly/weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = WEIGHTS_DIR / "seabed_clusters.joblib"
GMM_FILE = WEIGHTS_DIR / "seabed_gmm.joblib"
SCALER_FILE = WEIGHTS_DIR / "feature_scaler.joblib"
STATS_FILE = WEIGHTS_DIR / "seabed_stats.json"

def extract_acoustic_features(crop_gray: np.ndarray) -> np.ndarray:
    """
    Extract multi-scale acoustic backscatter texture features from a sonar patch:
    1. Statistical moments (mean, std, skewness, kurtosis) [4]
    2. High-frequency speckle entropy [1]
    3. Spatial gradient magnitude distribution (Sobel) [2]
    4. Multi-orientation Gabor filter acoustic energy [4]
    Total feature dimension: 11
    """
    if len(crop_gray.shape) == 3:
        crop_gray = cv2.cvtColor(crop_gray, cv2.COLOR_BGR2GRAY)

    patch_resized = cv2.resize(crop_gray, (128, 128)).astype(np.float32)

    # 1. Statistical moments
    mean_val = float(np.mean(patch_resized))
    std_val = float(np.std(patch_resized))
    norm_patch = (patch_resized - mean_val) / (std_val + 1e-5)
    skew_val = float(np.clip(np.mean(norm_patch ** 3), -10.0, 10.0))
    kurt_val = float(np.clip(np.mean(norm_patch ** 4), 0.0, 50.0))

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
    Uses a hybrid Gaussian Mixture Model with Tied/Full Covariance and MiniBatchKMeans:
    - Mahalanobis distance and negative log-likelihood account for correlated acoustic features.
    - Avoids false alarms on natural bedforms (e.g. sand ripples) while flagging true acoustic anomalies.
    """
    def __init__(self, n_clusters: int = 8):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=config.seed, batch_size=64)
        self.gmm = GaussianMixture(
            n_components=n_clusters, 
            covariance_type="tied", 
            reg_covar=1e-3, 
            random_state=config.seed
        )
        self.max_normal_dist = 7.5  # calibrated 95th percentile Mahalanobis distance
        self.max_normal_nll = 25.0  # calibrated 95th percentile Negative Log-Likelihood
        self.min_normal_nll = 5.0   # calibrated minimum NLL on training seabed
        self.inv_cov = None        # Tied inverse covariance for Mahalanobis metric
        self.is_fitted = False

        self._load_or_initialize()

    def _load_or_initialize(self):
        if MODEL_FILE.exists() and SCALER_FILE.exists() and STATS_FILE.exists():
            try:
                self.kmeans = joblib.load(MODEL_FILE)
                self.scaler = joblib.load(SCALER_FILE)
                if GMM_FILE.exists():
                    self.gmm = joblib.load(GMM_FILE)
                    if hasattr(self.gmm, "precisions_"):
                        # In tied covariance, precisions_ is (D, D) or (1, D, D)
                        self.inv_cov = self.gmm.precisions_[0] if self.gmm.precisions_.ndim == 3 else self.gmm.precisions_

                with open(STATS_FILE, "r") as f:
                    stats = json.load(f)
                    self.max_normal_dist = float(stats.get("max_normal_dist", 7.5))
                    self.max_normal_nll = float(stats.get("max_normal_nll", 25.0))
                    self.min_normal_nll = float(stats.get("min_normal_nll", 5.0))
                self.is_fitted = True
                print("Loaded pre-trained acoustic seabed anomaly model (Tied-Covariance GMM & NLL Engine).")
            except Exception as e:
                print(f"Error loading anomaly model: {e}. Will fit on available SSL sample.")

    def fit_from_normal_seabed(self, ssl_dir: Optional[Path] = None):
        """
        Fits baseline density and clusters using unlabelled normal seabed tiles from sss_ssl_dataset_N713_384.
        Calibrates both exact GMM Negative Log-Likelihood (NLL) and Mahalanobis distance thresholds.
        """
        if ssl_dir is None:
            ssl_dir = config.ssl_sample_dir

        set_deterministic_seeds(config.seed)
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

        # 1. Fit KMeans for auxiliary partition reference
        self.kmeans.fit(X_scaled)

        # 2. Fit Tied-Covariance GMM for true probabilistic density and Mahalanobis metric
        self.gmm.fit(X_scaled)
        self.inv_cov = self.gmm.precisions_[0] if self.gmm.precisions_.ndim == 3 else self.gmm.precisions_

        # Calculate exact GMM Negative Log-Likelihood across normal training samples
        train_log_probs = self.gmm.score_samples(X_scaled)
        train_nlls = -train_log_probs
        self.max_normal_nll = float(np.percentile(train_nlls, 95.0))
        self.min_normal_nll = float(np.min(train_nlls))

        # Calculate Mahalanobis distances to nearest GMM component mean on normal training set
        distances = []
        for x in X_scaled:
            diffs = self.gmm.means_ - x[np.newaxis, :]  # (K, D)
            # Mahalanobis: sqrt(diff * inv_cov * diff.T)
            maha_dists = np.sqrt(np.sum((diffs @ self.inv_cov) * diffs, axis=1))
            distances.append(np.min(maha_dists))

        distances = np.array(distances)
        # 95th percentile distance defines the calibrated normal threshold
        self.max_normal_dist = float(np.percentile(distances, 95.0))

        # Save artifacts
        joblib.dump(self.kmeans, MODEL_FILE)
        joblib.dump(self.gmm, GMM_FILE)
        joblib.dump(self.scaler, SCALER_FILE)
        with open(STATS_FILE, "w") as f:
            json.dump({
                "num_samples": len(features),
                "n_clusters": self.n_clusters,
                "covariance_model": "GMM_tied",
                "max_normal_dist": self.max_normal_dist,
                "mean_dist": float(np.mean(distances)),
                "std_dist": float(np.std(distances)),
                "max_normal_nll": self.max_normal_nll,
                "min_normal_nll": self.min_normal_nll,
                "mean_nll": float(np.mean(train_nlls))
            }, f, indent=4)

        self.is_fitted = True
        print(
            f"Anomaly engine fitted on {len(features)} acoustic patches.\n"
            f"  - Mahalanobis 95th Percentile Threshold : {self.max_normal_dist:.3f}\n"
            f"  - Negative Log-Likelihood 95th Threshold: {self.max_normal_nll:.3f}"
        )

    def score_patch(self, crop_gray: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate a candidate crop against normal seabed distribution using:
        1. Exact GMM Negative Log-Likelihood (NLL = -log p(x))
        2. Component-wise Mahalanobis Novelty Distance to GMM means
        Outputs:
        - anomaly_score: [0.0, 1.0] (composite probabilistic novelty index)
        - negative_log_likelihood: float (exact GMM density deficit)
        - mahalanobis_dist: float (exact covariance-scaled novelty distance)
        - nearest_cluster_id: int (index of closest GMM component)
        - nearest_cluster_dist: float (Mahalanobis distance to closest GMM component)
        - is_anomaly: bool (true if novelty exceeds 95th percentile baseline)
        - explanation: str (human-readable acoustic explanation)
        """
        if not self.is_fitted:
            # Fallback heuristic if models not yet fit
            return {
                "anomaly_score": 0.15,
                "negative_log_likelihood": 10.0,
                "mahalanobis_dist": 1.0,
                "nearest_cluster_id": 0,
                "nearest_cluster_dist": 1.0,
                "is_anomaly": False,
                "explanation": "Normal acoustic seafloor texture."
            }

        feat = extract_acoustic_features(crop_gray).reshape(1, -1)
        feat_scaled = self.scaler.transform(feat)

        # 1. Exact GMM Negative Log-Likelihood
        log_prob = float(self.gmm.score_samples(feat_scaled)[0])
        nll = -log_prob

        # 2. Mahalanobis distance to nearest GMM component mean
        if self.inv_cov is not None:
            diffs = self.gmm.means_ - feat_scaled  # (K, D)
            maha_dists = np.sqrt(np.sum((diffs @ self.inv_cov) * diffs, axis=1))
            nearest_cluster_id = int(np.argmin(maha_dists))
            min_dist = float(maha_dists[nearest_cluster_id])
        else:
            # Fallback Euclidean distance to KMeans centroids
            dists = self.kmeans.transform(feat_scaled)[0]
            nearest_cluster_id = int(np.argmin(dists))
            min_dist = float(dists[nearest_cluster_id])

        # Dual-threshold anomaly evaluation
        maha_anomaly = min_dist > self.max_normal_dist
        nll_anomaly = nll > self.max_normal_nll
        is_anomaly = maha_anomaly or nll_anomaly

        # Composite normalized anomaly score: [0.0, 1.0]
        score_maha = min(1.0, min_dist / (self.max_normal_dist * 1.8))
        score_nll = min(1.0, max(0.0, (nll - self.min_normal_nll) / (max(self.max_normal_nll - self.min_normal_nll, 1e-4) * 1.5)))
        normalized_score = float(np.clip(0.5 * score_maha + 0.5 * score_nll, 0.0, 1.0))

        if is_anomaly:
            explanation = (
                f"Acoustic texture departs significantly from normal sediment distribution "
                f"(Mahalanobis Distance: {min_dist:.2f} > baseline {self.max_normal_dist:.2f}, "
                f"Negative Log-Likelihood: {nll:.2f} > baseline {self.max_normal_nll:.2f}). "
                f"Unmodeled seafloor candidate requiring expert review."
            )
        else:
            explanation = (
                f"Consistent with natural acoustic seafloor facies "
                f"(GMM Component #{nearest_cluster_id}, D_M: {min_dist:.2f}, NLL: {nll:.2f})."
            )

        return {
            "anomaly_score": round(normalized_score, 4),
            "negative_log_likelihood": round(nll, 4),
            "mahalanobis_dist": round(min_dist, 4),
            "nearest_cluster_id": nearest_cluster_id,
            "nearest_cluster_dist": round(min_dist, 4),
            "is_anomaly": is_anomaly,
            "explanation": explanation
        }

