import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
	
class Model(nn.Module):
	def __init__(self, base_model, embedding_size = 128):
		super(Model, self).__init__()
		self.base_model = base_model
		self.embedder = nn.Linear(base_model.output_size, embedding_size)

	def forward(self, input):
		return self.embedder(self.base_model(input).view(input.size(0), -1))
	
	def adjust_learning_rate(self, epoch, optimizer):
		if epoch > self.optim_params_annealed['epoch']:
			for param_group in optimizer.param_groups:
				param_group.update(self.optim_params_annealed)
	
	optim_algo = optim.SGD
	optim_params = dict(lr = 1e-5, momentum = 0.9, weight_decay = 2e-4, dampening = 0.9)
	optim_params_annealed = dict(epoch = float('nan'))

def pdist(A, squared = False, eps = 1e-4):
	prod = torch.mm(A, A.t())
	norm = prod.diag().unsqueeze(1).expand_as(prod)
	res = (norm + norm.t() - 2 * prod).clamp(min = 0)
	return res if squared else (res + eps).sqrt() + eps 
		
class Untrained(Model):
	def forward(self, input):
		return self.base_model(input).view(input.size(0), -1)

class LiftedStruct(Model):
	def criterion(self, features, labels, margin = 1.0, eps = 1e-4):
		d = pdist(features, squared = False, eps = eps)
		pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(features)
		neg_i = torch.mul((margin - d).exp(), 1 - pos).sum(1).expand_as(d)
		return torch.sum(F.relu(pos.triu(1) * ((neg_i + neg_i.t()).log() + d)).pow(2)) / (pos.sum() - len(d))

class Triplet(Model):
	def criterion(self, features, labels, margin = 1.0):
		d = pdist(features, squared = False)
		pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(features)
		T = d.unsqueeze(1).expand(*(len(d),) * 3)
		M = pos.unsqueeze(1).expand_as(T) * (1 - pos.unsqueeze(2).expand_as(T))
		return (M * F.relu(T - T.transpose(1, 2) + margin)).sum() / M.sum()
	
	optim_params = dict(lr = 2e-5, momentum = 0.9, weight_decay = 5e-4)

class TripletRatio(Model):
	def criterion(self, features, labels, margin = 0.1, eps = 1e-4):
		d = pdist(features, squared = False, eps = eps)
		pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(features)
		T = d.unsqueeze(1).expand(*(len(d),) * 3)
		M = pos.unsqueeze(1).expand_as(T) * (1 - pos.unsqueeze(2).expand_as(T))
		return (M * T.div(T.transpose(1, 2) + margin)).sum() / M.sum()

class Npairs(Model):
	pass

class Pddm(Model):
	def __init__(self, base_model):
		nn.Module.__init__(self)
		self.base_model = base_model
		d = base_model.output_size
		
		self.wu = nn.Linear(d, d)
		self.wv = nn.Linear(d, d)
		self.wc = nn.Linear(2 * d, d)
		self.ws = nn.Linear(d, 1)
		self.dropout = nn.Dropout(p = 0.5)
	
	def forward(self, input):
		return F.normalize(self.base_model(input).view(input.size(0), -1))

	def criterion(self, features, labels, Alpha = 0.5, Beta = 1.0, Lambda = 0.5):
		d = pdist(features, squared = True)
		pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(features)

		f1, f2 = [features.detach().unsqueeze(dim).expand(len(features), *features.size()) for dim in [0, 1]]
		u = (f1 - f2).abs()
		v = (f1 + f2) / 2
		u_ = F.normalize(F.relu(self.dropout(self.wu(u.view(-1, u.size(-1))))))
		v_ = F.normalize(F.relu(self.dropout(self.wv(v.view(-1, v.size(-1))))))
		s = self.ws(F.relu(self.dropout(self.wc(torch.cat((u_, v_), -1))))).view_as(d)
		
		sneg = s * (1 - pos)
		i, j = min([(s.data[i, j], (i, j)) for i, j in pos.data.nonzero() if i != j])[1]
		k, l = sneg.max(1)[1].data.squeeze(1)[torch.cuda.LongTensor([i, j])]
		assert pos[i, j] == 1 and pos[i, k] == 0 and pos[j, l] == 0

		smin, smax = torch.min(sneg[i], sneg[j]).min().detach(), torch.max(sneg[i], sneg[j]).max().detach()
		s = (s - smin.expand_as(s)) / (smax - smin).expand_as(s)

		E_m = F.relu(Alpha + s[i, k] - s[i, j]) + F.relu(Alpha + s[j, l] - s[i, j])
		E_e = F.relu(Beta + d[i, j] - d[i, k]) + F.relu(Beta + d[i, j] - d[j, l])

		return E_m + Lambda * E_e
	
	optim_params = dict(lr = 1e-4, momentum = 0.9, weight_decay = 5e-4)
	#optim_params_annealed = dict(lr = 1e-5, epoch = 5)
