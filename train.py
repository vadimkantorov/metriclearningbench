import os
import sys
import random
import itertools
import hickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable

import googlenet
import cub2011

assert os.environ.get('CUDA_VISIBLE_DEVICES')

opts = dict(
	DATA_DIR = 'data',
	BASE_MODEL_WEIGHTS = 'data/googlenet.h5',
	LOG = 'data/log.txt',
	SEED = 1,
	NUM_THREADS = 16,
	TRAIN_CLASSES = 100,
	NUM_EPOCHS = 30,
	BATCH_SIZE = 16
)

def pairwise_euclidean_distance(A, eps = 1e-4):
	prod = torch.mm(A, A.t())
	norm = prod.diag().unsqueeze(1).expand_as(prod)
	return torch.sqrt((norm + norm.t() - 2 * prod).clamp(min = 0) + eps) + eps

class LiftedStruct(nn.Module):
	def __init__(self, base_model, embedding_size = 128):
		super(LiftedStruct, self).__init__()
		self.base_model = base_model
		self.embedder = nn.Linear(base_model.output_size, embedding_size)

	def forward(self, input):
		return self.embedder(self.base_model(input).view(input.size(0), -1))

	def criterion(self, input, labels, margin = 1.0, eps = 1e-4):
		d = pairwise_euclidean_distance(input, eps = eps)
		pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(input)
		m_d = margin - d
		max_elem = m_d.max().unsqueeze(1).expand_as(m_d)
		neg_i = torch.mul((m_d - max_elem).exp(), 1 - pos).sum(1).expand_as(d)
		return torch.sum(torch.mul(pos.triu(1), torch.log(neg_i + neg_i.t()) + d + max_elem).clamp(min = 0).pow(2)) / (pos.sum() - len(d))

	def sampler(self, batch_size, dataset, train_classes):
		'''lazy sampling, not like in lifted_struct. they add to the pool all postiive combinations, then compute the average number of positive pairs per image, then sample for every image the same number of negative pairs'''
		images_by_class = {class_label_ind_train : [example_idx for example_idx, (image_file_name, class_label_ind) in enumerate(dataset.imgs) if class_label_ind == class_label_ind_train] for class_label_ind_train in range(train_classes)}
		sample_from_class = lambda class_label_ind: images_by_class[class_label_ind][random.randrange(len(images_by_class[class_label_ind]))]
		while True:
			example_indices = []
			for i in range(0, batch_size, 2):
				perm = random.sample(xrange(train_classes), 2)
				example_indices += [sample_from_class(perm[0]), sample_from_class(perm[0 if i == 0 or random.random() > 0.5 else 1])]
			yield example_indices
	
	optim_algo = optim.SGD
	optim_params = dict(lr = 1e-5, momentum = 0.9, weight_decay = 2e-4, dampening = 0.9)

def adapt_sampler(batch_size, dataset, sampler, **kwargs):
	return type('', (), dict(
		__len__ = dataset.__len__,
		__iter__ = lambda _: itertools.chain.from_iterable(sampler(batch_size, dataset, **kwargs))
	))()

for set_random_seed in [random.seed, torch.manual_seed, torch.cuda.manual_seed_all]:
	set_random_seed(opts['SEED'])

base_model = googlenet.GoogLeNet()
base_model_weights = hickle.load(opts['BASE_MODEL_WEIGHTS'])
base_model.load_state_dict({k : torch.from_numpy(v) for k, v in base_model_weights.items()})
model = LiftedStruct(base_model)

normalize = transforms.Compose([
	transforms.ToTensor(),
	transforms.Lambda(lambda x: x * 255.0),
	transforms.Normalize(mean = base_model.rgb_mean, std = base_model.rgb_std),
	transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])])
])

dataset_train = cub2011.Cub2011(opts['DATA_DIR'], transforms.Compose([
	transforms.RandomSizedCrop(base_model.input_side),
	transforms.RandomHorizontalFlip(),
	normalize
]), download = True)
dataset_eval = cub2011.Cub2011(opts['DATA_DIR'], transforms.Compose([
	transforms.Scale(256),
	transforms.CenterCrop(base_model.input_side),
	normalize
]), download = True)

loader_train = torch.utils.data.DataLoader(dataset_train, sampler = adapt_sampler(opts['BATCH_SIZE'], dataset_train, model.sampler, train_classes = opts['TRAIN_CLASSES']), num_workers = opts['NUM_THREADS'], batch_size = opts['BATCH_SIZE'])
loader_eval = torch.utils.data.DataLoader(dataset_eval, sampler = [example_idx for example_idx in range(len(dataset_eval)) if dataset_eval.imgs[example_idx][1] >= opts['TRAIN_CLASSES']], shuffle = False, num_workers = opts['NUM_THREADS'], batch_size = opts['BATCH_SIZE'])

model.cuda()
optimizer = model.optim_algo(model.parameters(), **model.optim_params)

log = open(opts['LOG'], 'w', 0)
for epoch in range(opts['NUM_EPOCHS']):
	model.train()
	loss_all = []
	for batch_idx, batch in enumerate(loader_train):
		images, labels = [Variable(tensor.cuda()) for tensor in batch]
		loss = model.criterion(model(images), labels)
		loss_all.append(loss.data[0])
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print('train {:>3}.{:05}  loss  {:.06}'.format(epoch, batch_idx, loss.data[0]))
	log.write('loss epoch {}: {}\n'.format(epoch, torch.Tensor(loss_all).mean()))
	
	model.eval()
	embeddings_all, labels_all = [], []
	for batch_idx, batch in enumerate(loader_eval):
		images, labels = [Variable(tensor.cuda(), volatile = True) for tensor in batch]
		output = model(images)
		embeddings_all.append(output.data.cpu())
		labels_all.append(labels.data.cpu())
		print('eval  {:>3}.{:05}'.format(epoch, batch_idx))
	log.write('recall@1 epoch {}: {}\n'.format(epoch, dataset_eval.recall(torch.cat(embeddings_all, 0), torch.cat(labels_all, 0))))
