import os
import sys
import random
import argparse
import itertools
import hickle
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torch.optim.lr_scheduler

import cub2011
import cars196
import stanford_online_products
import googlenet
import model
import sampler

assert os.getenv('CUDA_VISIBLE_DEVICES')

parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__ = lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))
parser.add_argument('--dataset', choices = dict(CUB2011 = cub2011.CUB2011MetricLearning, CARS196 = cars196.Cars196, STANFORD_ONLINE_PRODUCTS = stanford_online_products.StanfordOnlineProducts), default = cub2011.CUB2011MetricLearning, action = LookupChoices)
parser.add_argument('--base_model', choices = dict(GOOGLENET = googlenet.GoogLeNet), default = googlenet.GoogLeNet, action = LookupChoices)
parser.add_argument('--model', choices = dict(LIFTED_STRUCT = model.LiftedStruct, TRIPLET = model.Triplet, TRIPLET_RATIO = model.TripletRatio, PDDM = model.Pddm, UNTRAINED = model.Untrained, MARGIN = model.Margin), default = model.Triplet, action = LookupChoices)
parser.add_argument('--sampler', choices = dict(SIMPLE = sampler.simple, TRIPLET = sampler.triplet, NPAIRS = sampler.npairs), default = sampler.npairs, action = LookupChoices)
parser.add_argument('--data_dir', default = 'data')
parser.add_argument('--base_model_weights', default = 'data/googlenet.h5')
parser.add_argument('--log', default = 'data/log.txt')
parser.add_argument('--seed', default = 1, type = int)
parser.add_argument('--threads', default = 16, type = int)
parser.add_argument('--epochs', default = 200, type = int)
parser.add_argument('--batch_size', default = 128, type = int)
opts = parser.parse_args()

for set_random_seed in [random.seed, torch.manual_seed, torch.cuda.manual_seed_all]:
	set_random_seed(opts.seed)

def recall(embeddings, labels, K = 1):
	prod = torch.mm(embeddings, embeddings.t())
	norm = prod.diag().unsqueeze(1).expand_as(prod)
	D = norm + norm.t() - 2 * prod
	knn_inds = D.topk(1 + K, dim = 1, largest = False)[1][:, 1:]
	return torch.Tensor([labels[knn_inds[i]].eq(labels[i]).max() for i in range(len(embeddings))]).mean()

base_model = opts.base_model()
base_model.load_state_dict({k : torch.from_numpy(v) for k, v in hickle.load(opts.base_model_weights).items()})

normalize = transforms.Compose([
	transforms.ToTensor(),
	transforms.Lambda(lambda x: x * 255.0),
	transforms.Normalize(mean = base_model.rgb_mean, std = base_model.rgb_std),
	transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])])
])

dataset_train = opts.dataset(opts.data_dir, train = True, transform = transforms.Compose([
	transforms.RandomSizedCrop(base_model.input_side),
	transforms.RandomHorizontalFlip(),
	normalize
]), download = True)
dataset_eval = opts.dataset(opts.data_dir, train = False, transform = transforms.Compose([
	transforms.Scale(256),
	transforms.CenterCrop(base_model.input_side),
	normalize
]), download = True)

adapt_sampler = lambda batch_size, dataset, sampler, **kwargs: type('', (), dict(__len__ = dataset.__len__, __iter__ = lambda _: itertools.chain.from_iterable(sampler(batch_size, dataset, **kwargs))))()
loader_train = torch.utils.data.DataLoader(dataset_train, sampler = adapt_sampler(opts.batch_size, dataset_train, opts.sampler), num_workers = opts.threads, batch_size = opts.batch_size, drop_last = True)
loader_eval = torch.utils.data.DataLoader(dataset_eval, shuffle = False, num_workers = opts.threads, batch_size = opts.batch_size)

model = opts.model(base_model, dataset_train.num_training_classes).cuda()
weights, biases = [[p for k, p in model.named_parameters() if p.requires_grad and ('bias' in k) == is_bias] for is_bias in [False, True]]
optimizer = model.optim_algo([dict(params = weights), dict(params = biases, weight_decay = 0.0)], **model.optim_params)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = model.optim_params_annealed['epoch'], gamma = model.optim_params_annealed['gamma'])

log = open(opts.log, 'w', 0)
for epoch in range(opts.epochs):
	scheduler.step()
	model.train()
	loss_all, norm_all = [], []
	for batch_idx, batch in enumerate(loader_train if model.criterion is not None else []):
		images, labels = [torch.autograd.Variable(tensor.cuda()) for tensor in batch]
		loss = model.criterion(model(images), labels)
		loss_all.append(loss.data[0])
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print('train {:>3}.{:05}  loss  {:.06f}'.format(epoch, batch_idx, loss_all[-1]))
	log.write('loss epoch {}: {}\n'.format(epoch, torch.Tensor(loss_all or [0.0]).mean()))
	
	model.eval()
	embeddings_all, labels_all = [], []
	for batch_idx, batch in enumerate(loader_eval):
		images, labels = [torch.autograd.Variable(tensor.cuda(), volatile = True) for tensor in batch]
		output = model(images)
		embeddings_all.append(output.data.cpu())
		labels_all.append(labels.data.cpu())
		print('eval  {:>3}.{:05}'.format(epoch, batch_idx))
	log.write('recall@1 epoch {}: {}\n'.format(epoch, recall(torch.cat(embeddings_all), torch.cat(labels_all))))
