import os
from pathlib import Path
from pydantic import BaseModel
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent

def _resolve_dataset_root() -> Path:
    # 1. Explicit environment variable override
    env_path = os.getenv("DATASET_ROOT")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    
    # 2. Repository-local standard location
    local = BASE_DIR / "dataset"
    if local.exists():
        return local
    
    # 3. Dedicated fast-storage paths (Windows / Linux / macOS fallbacks)
    external_candidates = [
        Path("D:/USIP_DATA/unified_sonar_dataset"),
        Path("/data/USIP_DATA/unified_sonar_dataset"),
        Path.home() / "USIP_DATA" / "unified_sonar_dataset"
    ]
    for ext in external_candidates:
        if ext.exists():
            return ext
            
    return local

def _resolve_ssl_dir() -> Path:
    # 1. Explicit environment variable override
    env_path = os.getenv("SSL_SAMPLE_DIR")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    
    # 2. Repository-local standard location
    local = BASE_DIR / "ssl_sample"
    if local.exists():
        return local
        
    # 3. Dedicated fast-storage paths (Windows / Linux / macOS fallbacks)
    external_candidates = [
        Path("D:/USIP_SSL_SAMPLE"),
        Path("/data/USIP_SSL_SAMPLE"),
        Path.home() / "USIP_SSL_SAMPLE"
    ]
    for ext in external_candidates:
        if ext.exists():
            return ext
            
    return local

class SystemConfig(BaseModel):
    dataset_root: Path = _resolve_dataset_root()
    ssl_sample_dir: Path = _resolve_ssl_dir()
    raw_datasets_dir: Path = Path(os.getenv("RAW_DATASETS_DIR", str(BASE_DIR)))
    model_path: Path = BASE_DIR / os.getenv("MODEL_PATH", "ml/detection/weights/best.pt")
    anomaly_model_dir: Path = BASE_DIR / os.getenv("ANOMALY_MODEL_DIR", "ml/anomaly/weights")
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./usip_sonar.db")
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    frontend_url: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
    demo_samples_dir: Path = BASE_DIR / os.getenv("DEMO_SAMPLES_DIR", "demo_samples")
    device: str = os.getenv("DEVICE", "cuda")
    batch_size: int = int(os.getenv("BATCH_SIZE", "4"))
    img_size: int = int(os.getenv("IMG_SIZE", "640"))
    seed: int = int(os.getenv("SEED", "42"))
    strict_model: bool = os.getenv("STRICT_MODEL", "false").lower() in ("true", "1", "yes")

config = SystemConfig()

