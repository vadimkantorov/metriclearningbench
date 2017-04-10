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

def SequentialSampler(batch_size, dataset):
	batch_idx = 0
	while True:
		yield range(batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(dataset)))
		batch_idx += 1

def ShuffleSampler(batch_size, dataset):
	while True:
		yield random.sample(xrange(len(dataset)), batch_size)
		
def adapt_sampler(batch_size, dataset, sampler):
	return type('', (), dict(
		__len__ = dataset.__len__,
		__iter__ = lambda _: itertools.chain.from_iterable(sampler(batch_size, dataset))
	))()

class LiftedStruct(nn.Module):
	def __init__(self, base_model):
		super(LiftedStruct, self).__init__()
		self.base_model = base_model
	
	criterion = nn.BCELoss(size_average = True)
	optim_algo = optim.SGD
	optim_params = dict(lr = 5e-3, momentum = 0.9, weight_decay = 5e-4, dampening = 0.9)
	optim_params_annealed = dict(lr = 5e-4, epoch = 10)

class PDDM(nn.Module):
	def __init__(self, base_model):
		super(PDDM, self).__init__()
		self.base_model = base_model

opts = dict(
	BASE_MODEL_WEIGHTS = 'data/googlenet.h5',
	DATA_DIR = 'data',
	NUM_EPOCHS = 1,
	BATCH_SIZE = 64
)

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

loader_train = torch.utils.data.DataLoader(dataset, sampler = adapt_sampler(opts['BATCH_SIZE'], dataset, SequentialSampler), batch_size = opts['BATCH_SIZE'])

model.cuda()
model.criterion.cuda()
optimizer = model.optim_algo(model.parameters(), **model.optim_params)

for epoch in range(opts['NUM_EPOCHS']):
	for batch_idx, batch in enumerate(loader_train):
		images, labels = [Variable(tensor.cuda()) for tensor in batch]
		loss = model.criterion(model(images), labels)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		print('epoch  %{:<3}  batch  %{:<5}  {:%.06}'.format(epoch, batch_idx, loss))
