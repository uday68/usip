import os
import random
import numpy as np
import torch

def set_deterministic_seeds(seed: int = 42) -> None:
    """
    Sets global deterministic seeds across Python, NumPy, PyTorch (CPU & CUDA).
    Ensures bit-exact reproducibility across experiment training and evaluation.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[USIP Seeds] Global deterministic seeds locked to {seed} (CUDA deterministic mode active).")
    else:
        print(f"[USIP Seeds] Global deterministic seeds locked to {seed} (CPU mode).")
