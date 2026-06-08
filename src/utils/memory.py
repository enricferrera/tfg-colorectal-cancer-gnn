import torch
import gc

def clean_vram():
    """
    Forcefully cleans VRAM by triggering GC and clearing the CUDA cache.
    """
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
