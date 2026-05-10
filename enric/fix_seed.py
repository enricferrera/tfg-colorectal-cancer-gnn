import torch
import numpy as np
import random

def fix_seeds(r_seed=123):
  """
    Loads classification images from the specified directory.

      Args:
        path (str): The absolute path to the dataset folder.
        batch_size (int, optional): Number of images per batch. Defaults to 32.

      Returns:
        np.ndarray: A NumPy array containing the loaded image batch.
  """

  torch.manual_seed(r_seed)
  np.random.seed(r_seed)
  random.seed(r_seed)
  torch.cuda.manual_seed(r_seed)
  torch.backends.cudnn.enabled=False
  torch.backends.cudnn.deterministic=True