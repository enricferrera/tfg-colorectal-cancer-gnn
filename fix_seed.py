import torch
import numpy as np
import random

def fix_seeds(r_seed=123):
  """
    Fixes the random seeds for reproducibility across PyTorch, NumPy, and Python.

    This ensures that random operations like weight initialization, data shuffling,
    and GPU computations yield the same results across different runs.

    Args:
    r_seed (int): The seed value to use. Defaults to 123.
  """

  torch.manual_seed(r_seed)
  np.random.seed(r_seed)
  random.seed(r_seed)
  torch.cuda.manual_seed(r_seed)
  torch.backends.cudnn.enabled=False
  torch.backends.cudnn.deterministic=True