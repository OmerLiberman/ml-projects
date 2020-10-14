"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

MNIST_CNVRG.py
==============================================================================
"""
from torch.utils.data.dataset import Dataset

class RealMnist(Dataset):
	"""
	MNIST dataset with real images.
	"""
	def __init__(self, path, root_dir, transform=None):
		"""
		"""

