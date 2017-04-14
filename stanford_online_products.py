import os
import torch
import torch.utils.data as data
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import download_url

class StanfordOnlineProducts(ImageFolder, CIFAR10):
	base_folder = 'Stanford_Online_Products'
	url = 'ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip'
	filename = 'Stanford_Online_Products.zip'
	zip_md5 = '7f73d41a2f44250d4779881525aea32e'

	train_list = [
		['bicycle_final/111265328556_0.JPG', '77420a4db9dd9284378d7287a0729edb']
		['chair_final/111182689872_0.JPG', 'ce78d10ed68560f4ea5fa1bec90206ba']
	]
	test_list = [
		['table_final/111194782300_0.JPG', '8203e079b5c134161bbfa7ee2a43a0a1'],
		['toaster_final/111157129195_0.JPG', 'd6c24ee8c05d986cafffa6af82ae224e']
	]

	def __init__(self, root, train=None, transform=None, target_transform=None, download=False, **kwargs):
		self.root = root
		if download:
			self.download()

		if not self._check_integrity():
			raise RuntimeError('Dataset not found or corrupted.' +
							   ' You can use download=True to download it')
	
	def download(self):
		import zipfile

		if self._check_integrity():
			print('Files already downloaded and verified')
			return

		root = self.root
		download_url(self.url, root, self.filename, self.zip_md5)

		# extract file
		cwd = os.getcwd()
		zip = zipfile.open(os.path.join(root, self.filename), "r")
		os.chdir(root)
		zip.extractall()
		zip.close()
		os.chdir(cwd)
