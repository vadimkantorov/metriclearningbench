import os
import scipy.io
import torch
import torch.utils.data as data
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10
from torchvision.datasets.folder import default_loader

class Cars196(ImageFolder, CIFAR10):
	base_folder = 'car_ims'
	url = 'http://imagenet.stanford.edu/internal/car196/car_ims.tgz'
	filename = 'cars_ims.tgz'
	tgz_md5 = 'd5c8f0aa497503f355e17dc7886c3f14'

	base_folder_devkit = 'devkit'
	url_devkit = 'http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz'
	filename_devkit = 'cars_devkit.tgz'
	tgz_md5_devkit = 'c3b158d763b6e2245038c8ad08e45376'

	train_list = [
		['000001.jpg', '2d44a28f071aeaac9c0802fddcde452e'],
		['000002.jpg', '531fbde520bee33dcbfced6ae588c8f9']
	]
	test_list = [
		['014981.jpg', 'e7238ce05218a6e4dc92cda5e8971f17'],
		['014982.jpg', 'b2a95af89329d32d5fe2b74f0922378e']
	]
	
	def __init__(self, root, train=False, transform=None, target_transform=None, download=False, **kwargs):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.loader = default_loader

		if download:
			base_folder, url, filename, tgz_md5 = self.base_folder, self.url, self.filename, self.tgz_md5
			self.base_folder, self.url, self.filename, self.tgz_md5 = self.base_folder_devkit, self.url_devkit, self.filename_devkit, self.tgz_md5_devkit
			self.download()
			self.base_folder, self.url, self.filename, self.tgz_md5	= base_folder, url, filename, tgz_md5
			self.download()

		if not self._check_integrity():
			raise RuntimeError('Dataset not found or corrupted.' +
							   ' You can use download=True to download it')

		self.imgs = [(os.path.join(root, self.base_folder, '0' + a[-1][0]), int(a[-2][0]) - 1) for a in scipy.io.loadmat(os.path.join(root, self.base_folder_devkit, 'cars_{}_annos.mat'.format('train' if train else 'test')))['annotations'][0]]
