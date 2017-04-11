import os
import torch
import torch.utils.data as data
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10

class Cub2011(ImageFolder, CIFAR10):
	base_folder = 'CUB_200_2011/images'
	url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
	filename = 'CUB_200_2011.tgz'
	tgz_md5 = '97eceeb196236b17998738112f37df78'
	
	train_list = [
		['180.Wilson_Warbler/Wilson_Warbler_0002_175571.jpg', '0c763ca2ad60ed3ae43e76a04df63983']
	]
	test_list = []
	
	def __init__(self, root, transform=None, target_transform=None, download=False, **kwargs):
		self.root = root
		if download:
			self.download()

		if not self._check_integrity():
			raise RuntimeError('Dataset not found or corrupted.' +
							   ' You can use download=True to download it')
		ImageFolder.__init__(self, os.path.join(root, self.base_folder), transform = transform, target_transform = target_transform, **kwargs)

	def recall(self, embeddings, labels, K = 1):
		norm = embeddings.mul(embeddings).sum(1).expand(embeddings.size(0), embeddings.size(0))
		D = norm + norm.t() - 2 * torch.mm(embeddings, embeddings.t())
		knn_inds = D.topk(1 + K, dim = 1, largest = False)[1][:, 1:]
		return torch.Tensor([labels[knn_inds[i]].eq(labels[i]).max() for i in range(len(embeddings))]).mean()
