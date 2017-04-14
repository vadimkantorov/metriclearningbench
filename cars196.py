import os
import torch
import torch.utils.data as data
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10

class Cars196(ImageFolder, CIFAR10):
	base_folder = 'car_ims'
	url = 'http://imagenet.stanford.edu/internal/car196/car_ims.tgz'
	filename = 'cars_ims.tgz'
	tgz_md5 = 'd5c8f0aa497503f355e17dc7886c3f14'

	base_folder_devkit = 'devkit'
	url_devkit = 'http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz'
	filename_devkit = 'cars_devkit.tgz'
	tgz_md5_devkit = 'c3b158d763b6e2245038c8ad08e45376'

	train_list = []
	test_list = []

	def download(self):
		pass
	
	def __init__(self, root, train=False, transform=None, target_transform=None, download=False, **kwargs):
		self.root = root
		if download:
			self.download()

		if not self._check_integrity():
			raise RuntimeError('Dataset not found or corrupted.' +
							   ' You can use download=True to download it')
		ImageFolder.__init__(self, os.path.join(root, self.base_folder), transform = transform, target_transform = target_transform, **kwargs)
