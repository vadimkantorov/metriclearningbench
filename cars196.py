import os
import scipy.io
import torch
import torch.utils.data as data
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url


class Cars196MetricLearning(ImageFolder, CIFAR10):
	base_folder_devkit = 'devkit'
	url_devkit = 'http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz'
	filename_devkit = 'cars_devkit.tgz'
	tgz_md5_devkit = 'c3b158d763b6e2245038c8ad08e45376'

	base_folder_trainims = 'cars_train'
	url_trainims = 'http://imagenet.stanford.edu/internal/car196/cars_train.tgz'
	filename_trainims = 'cars_ims_train.tgz'
	tgz_md5_trainims = '065e5b463ae28d29e77c1b4b166cfe61'
	
	base_folder_testims = 'cars_test'
	url_testims = 'http://imagenet.stanford.edu/internal/car196/cars_test.tgz'
	filename_testims = 'cars_ims_test.tgz'
	tgz_md5_testims = '4ce7ebf6a94d07f1952d94dd34c4d501'
	
	url_testanno = 'http://imagenet.stanford.edu/internal/car196/cars_test_annos_withlabels.mat'
	filename_testanno = 'cars_test_annos_withlabels.mat'
	mat_md5_testanno = 'b0a2b23655a3edd16d84508592a98d10'
	
	filename_trainanno = 'cars_train_annos.mat'
	
	train_list = [
		['000001.jpg', '2d44a28f071aeaac9c0802fddcde452e'],
		['000002.jpg', '531fbde520bee33dcbfced6ae588c8f9']
	]
	test_list = [
		['014981.jpg', 'e7238ce05218a6e4dc92cda5e8971f17'],
		['014982.jpg', 'b2a95af89329d32d5fe2b74f0922378e']
	]
	num_training_classes = 98
	
	def __init__(self, root, train=False, transform=None, target_transform=None, download=False, **kwargs):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.loader = default_loader

		if download:
			self.url, self.filename, self.tgz_md5 = self.url_devkit, self.filename_devkit, self.tgz_md5_devkit
			self.download()
			
			self.url, self.filename, self.tgz_md5 = self.url_trainims, self.filename_trainims, self.tgz_md5_trainims
			self.download()
			
			self.url, self.filename, self.tgz_md5 = self.url_testims, self.filename_testims, self.tgz_md5_testims
			self.download()
			
			download_url(self.url_testanno, os.path.join(root, self.base_folder_devkit), self.filename_testanno, self.mat_md5_testanno)
			
		if not self._check_integrity():
			raise RuntimeError('Dataset not found or corrupted.' +
							   ' You can use download=True to download it')

		self.imgs = [(os.path.join(root, self.base_folder, self.base_folder_trainims, a[-1][0]), int(a[-2][0]) - 1) for filename in [self.filename_trainanno] for a in scipy.io.loadmat(os.path.join(root, self.base_folder_devkit, filename))['annotations'][0] if (int(a[-2][0]) - 1 < self.num_training_classes) == train] + [(os.path.join(root, self.base_folder_testims, a[-1][0]), int(a[-2][0]) - 1) for filename in [self.filename_testanno] for a in scipy.io.loadmat(os.path.join(root, self.base_folder_devkit, filename))['annotations'][0] if (int(a[-2][0]) - 1 < self.num_training_classes) == train]
