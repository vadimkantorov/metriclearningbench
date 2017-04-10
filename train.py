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

opts = dict(
	BASE_MODEL_WEIGHTS = 'data/googlenet.h5',
	DATA_DIR = 'data',
	NUM_EPOCHS = 1,
	BATCH_SIZE = 64,
	RETRIEVAL_K = 1,
	TRAIN_CLASSES = 100,
	NUM_THREADS = 16
)

def pairwise_euclidean_distance(A, eps = 1e-6):
	norm = A.mul(A).sum(1).expand(A.size(0), A.size(0))
	return torch.sqrt(2 * norm - 2 * torch.mm(A, A.t()) + eps)

class LiftedStruct(nn.Module):
	def __init__(self, base_model, embedding_size = 128):
		super(LiftedStruct, self).__init__()
		self.base_model = base_model
		self.embedder = nn.Linear(1024, embedding_size)

	def forward(self, input):
		return self.embedder(self.base_model(input).view(input.size(0), -1))

	def criterion(self, input, labels, margin = 1.0, eps = 1e-6):
		pos = torch.eq(*[labels.unsqueeze(d).expand(len(labels), len(labels)) for d in [0, 1]]).type_as(input)
		d = pairwise_euclidean_distance(input, eps = eps)

		margin_d = margin - d
		neg_i = torch.mul(torch.exp(margin_d), 1 - pos).sum(0)
		v = torch.log(neg_i.view(1, -1).expand_as(margin_d) + neg_i.view(-1, 1).expand_as(margin_d) + eps)
		return torch.sum(torch.mul(pos, d + v).clamp(min = 0).pow(2)) / (2 * pos.sum())

	def sampler(self, batch_size, dataset, train_classes):
		'''lazy sampling, not like in lifted_struct. they add to the pool all postiive combinations, then compute the average number of positive pairs per image, then sample for every image the same number of negative pairs'''
		images_by_class = {class_label_ind_train : [example_idx for example_idx, (image_file_name, class_label_ind) in enumerate(dataset.imgs) if class_label_ind == class_label_ind_train] for class_label_ind_train in range(train_classes)}
		sample_from_class = lambda class_label_ind: images_by_class[class_label_ind][random.randrange(len(images_by_class[class_label_ind]))]
		while True:
			example_indices = []
			for i in range(0, batch_size, 2):
				perm = random.sample(xrange(train_classes), 2)
				example_indices += [sample_from_class(perm[0]), sample_from_class(perm[0 if random.random() > 0.5 else 1])]
			yield example_indices
	
	optim_algo = optim.SGD
	optim_params = dict(lr = 1e-5, momentum = 0.9, weight_decay = 2e-4, dampening = 0.9)

class PDDM(nn.Module):
	def __init__(self, base_model):
		super(PDDM, self).__init__()
		self.base_model = base_model

def SequentialSampler(batch_size, dataset):
	batch_idx = 0
	while True:
		yield range(batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(dataset)))
		batch_idx += 1

def ShuffleSampler(batch_size, dataset):
	while True:
		yield random.sample(xrange(len(dataset)), batch_size)
		
def adapt_sampler(batch_size, dataset, sampler, **kwargs):
	return type('', (), dict(
		__len__ = dataset.__len__,
		__iter__ = lambda _: itertools.chain.from_iterable(sampler(batch_size, dataset, **kwargs))
	))()

base_model = googlenet.GoogLeNet()
base_model_weights = hickle.load(opts['BASE_MODEL_WEIGHTS'])
base_model.load_state_dict({k : torch.from_numpy(v) for k, v in base_model_weights.items()})
model = LiftedStruct(base_model)

dataset = cub2011.Cub2011(opts['DATA_DIR'], transforms.Compose([
	transforms.Scale(256),
	transforms.CenterCrop(227),
	transforms.ToTensor(),
	transforms.Lambda(lambda x: x * 255.0),
	transforms.Normalize(mean = [122.7717, 115.9465, 102.9801], std = [1, 1, 1]),
	transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])])
]), download = True)

loader_train = torch.utils.data.DataLoader(dataset, sampler = adapt_sampler(opts['BATCH_SIZE'], dataset, model.sampler, train_classes = opts['TRAIN_CLASSES']), num_workers = opts['NUM_THREADS'], batch_size = opts['BATCH_SIZE'])
loader_eval = torch.utils.data.DataLoader(dataset, sampler = [example_idx for example_idx in range(len(dataset)) if dataset.imgs[example_idx][1] >= opts['TRAIN_CLASSES']], shuffle = False, num_workers = opts['NUM_THREADS'], batch_size = opts['BATCH_SIZE'])

model.cuda()
optimizer = model.optim_algo(model.parameters(), **model.optim_params)

for epoch in range(opts['NUM_EPOCHS']):
	model.train()
	for batch_idx, batch in enumerate(loader_train):
		images, labels = [Variable(tensor.cuda()) for tensor in batch]
		loss = model.criterion(model(images), labels)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print('train {:>3}.{:05}  loss  {:.06}'.format(epoch, batch_idx, loss.data[0]))
	
	model.eval()
	embeddings_all, labels_all = [], []
	for batch_idx, batch in enumerate(loader_eval):
		images, labels = [Variable(tensor.cuda(), volatile = True) for tensor in batch]
		output = model(images)
		embeddings_all.append(output.data.cpu())
		labels_all.append(labels.data.cpu())
		print('eval  {:>3}.{:05}'.format(epoch, batch_idx))
	print(dataset.recall(torch.cat(embeddings, 0), torch.cat(labels, 0), opts['RETRIEVAL_K']))
