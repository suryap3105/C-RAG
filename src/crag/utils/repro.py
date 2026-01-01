
import os
import random
import numpy as np
import torch
import sys

def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch (CPU & GPU)
    to ensure reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # print(f"[Repro] Global seed set to {seed}")

def log_env():
    """
    Logs critical environment versions to ensure reproducibility of the environment.
    """
    import transformers
    import faiss
    
    env_info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else "None",
        "transformers": transformers.__version__,
        "faiss": faiss.__version__ if hasattr(faiss, "__version__") else "Installed (unknown version)"
    }
    
    print(f"[Repro] Environment: {env_info}")
    return env_info
