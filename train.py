import os
import sys
import random
import argparse
import itertools
import hickle
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable

import cub2011
import cars196
import stanford_online_products
import googlenet
import model
import sampler

assert os.getenv('CUDA_VISIBLE_DEVICES')

parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__ = lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))
parser.add_argument('--DATASET', choices = dict(CUB2011 = cub2011.CUB2011MetricLearning, CARS196 = cars196.Cars196, STANFORD_ONLINE_PRODUCTS = stanford_online_products.StanfordOnlineProducts), default = cub2011.CUB2011MetricLearning, action = LookupChoices)
parser.add_argument('--BASE_MODEL', choices = dict(GOOGLENET = googlenet.GoogLeNet), default = googlenet.GoogLeNet, action = LookupChoices)
parser.add_argument('--MODEL', choices = dict(LIFTED_STRUCT = model.LiftedStruct, TRIPLET = model.Triplet, TRIPLET_RATIO = model.TripletRatio, PDDM = model.Pddm, UNTRAINED = model.Untrained), default = model.LiftedStruct, action = LookupChoices)
parser.add_argument('--DATA_DIR', default = 'data')
parser.add_argument('--BASE_MODEL_WEIGHTS', default = 'data/googlenet.h5')
parser.add_argument('--LOG', default = 'data/log.txt')
parser.add_argument('--SEED', default = 1, type = int)
parser.add_argument('--NUM_THREADS', default = 16, type = int)
parser.add_argument('--NUM_EPOCHS', default = 200, type = int)
parser.add_argument('--BATCH_SIZE', default = 64, type = int)
opts = parser.parse_args()

for set_random_seed in [random.seed, torch.manual_seed, torch.cuda.manual_seed_all]:
	set_random_seed(opts.SEED)

def recall(embeddings, labels, K = 1):
	prod = torch.mm(embeddings, embeddings.t())
	norm = prod.diag().unsqueeze(1).expand_as(prod)
	D = norm + norm.t() - 2 * prod
	knn_inds = D.topk(1 + K, dim = 1, largest = False)[1][:, 1:]
	return torch.Tensor([labels[knn_inds[i]].eq(labels[i]).max() for i in range(len(embeddings))]).mean()

normalize = transforms.Compose([
	transforms.ToTensor(),
	transforms.Lambda(lambda x: x * 255.0),
	transforms.Normalize(mean = base_model.rgb_mean, std = base_model.rgb_std),
	transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])])
])

dataset_train = opts.DATASET(opts.DATA_DIR, train = True, transform = transforms.Compose([
	transforms.RandomSizedCrop(base_model.input_side),
	transforms.RandomHorizontalFlip(),
	normalize
]), download = True)
dataset_eval = opts.DATASET(opts.DATA_DIR, train = False, transform = transforms.Compose([
	transforms.Scale(256),
	transforms.CenterCrop(base_model.input_side),
	normalize
]), download = True)

adapt_sampler = lambda batch_size, dataset, sampler, **kwargs: type('', (), dict(__len__ = dataset.__len__, __iter__ = lambda _: itertools.chain.from_iterable(sampler(batch_size, dataset, **kwargs))))()
loader_train = torch.utils.data.DataLoader(dataset_train, sampler = adapt_sampler(opts.BATCH_SIZE, dataset_train, sampler.simple), num_workers = opts.NUM_THREADS, batch_size = opts.BATCH_SIZE, drop_last = True)
loader_eval = torch.utils.data.DataLoader(dataset_eval, shuffle = False, num_workers = opts.NUM_THREADS, batch_size = opts.BATCH_SIZE)

base_model = opts.BASE_MODEL()
base_model_weights = hickle.load(opts.BASE_MODEL_WEIGHTS)
base_model.load_state_dict({k : torch.from_numpy(v) for k, v in base_model_weights.items()})
model = opts.MODEL(base_model).cuda()

weights, biases = [[p for k, p in model.named_parameters() if p.requires_grad and ('bias' in k) == is_bias] for is_bias in [False, True]]
optimizer = model.optim_algo([dict(params = weights), dict(params = biases, weight_decay = 0.0)], **model.optim_params)

log = open(opts.LOG, 'w', 0)
for epoch in range(opts.NUM_EPOCHS):
	model.adjust_learning_rate(epoch, optimizer)
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
	log.write('recall@1 epoch {}: {}\n'.format(epoch, recall(torch.cat(embeddings_all, 0), torch.cat(labels_all, 0))))
