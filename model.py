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
		return self.embedder(self.base_model(input).view(len(input), -1))
	
	criterion = None
	optim_algo = optim.SGD
	optim_params = dict(lr = 1e-5, momentum = 0.9, weight_decay = 2e-4, dampening = 0.9)
	optim_params_annealed = dict(epoch = float('inf'), gamma = 0.1)

def pdist(A, squared = False, eps = 1e-4):
	prod = torch.mm(A, A.t())
	norm = prod.diag().unsqueeze(1).expand_as(prod)
	res = (norm + norm.t() - 2 * prod).clamp(min = 0)
	return res if squared else res.clamp(min = eps).sqrt() + eps 
		
class Untrained(Model):
	def forward(self, input):
		return self.base_model(input).view(input.size(0), -1).detach()

class LiftedStruct(Model):
	def criterion(self, embeddings, labels, margin = 1.0, eps = 1e-4):
		d = pdist(embeddings, squared = False, eps = eps)
		pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(embeddings)
		neg_i = torch.mul((margin - d).exp(), 1 - pos).sum(1).expand_as(d)
		return torch.sum(F.relu(pos.triu(1) * ((neg_i + neg_i.t()).log() + d)).pow(2)) / (pos.sum() - len(d))

class Triplet(Model):
	def criterion(self, embeddings, labels, margin = 1.0):
		d = pdist(embeddings, squared = False)
		pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(embeddings) - torch.autograd.Variable(torch.eye(len(d))).type_as(embeddings)
		T = d.unsqueeze(1).expand(*(len(d),) * 3)
		M = pos.unsqueeze(1).expand_as(T) * (1 - pos.unsqueeze(2).expand_as(T))
		return (M * F.relu(T - T.transpose(1, 2) + margin)).sum() / M.sum()
	
	optim_params = dict(lr = 1e-4, momentum = 0.9, weight_decay = 5e-4)
	optim_params_annealed = dict(epoch = 30, gamma = 0.5)

class TripletRatio(Model):
	def criterion(self, embeddings, labels, margin = 0.1, eps = 1e-4):
		d = pdist(embeddings, squared = False, eps = eps)
		pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(embeddings)
		T = d.unsqueeze(1).expand(*(len(d),) * 3)
		M = pos.unsqueeze(1).expand_as(T) * (1 - pos.unsqueeze(2).expand_as(T))
		return (M * T.div(T.transpose(1, 2) + margin)).sum() / M.sum()

def topk_mask(input, dim, K = 10):
	index = input.topk(max(1, min(K, input.size(dim))), dim = dim)[1]
	mask = zeros_like(input).scatter(dim, index, 1.0)
	return mask

def zeros_like(input):
	return torch.autograd.Variable(torch.zeros_like(input.data))

class Pddm(Model):
	def __init__(self, base_model, d = 1024):
		nn.Module.__init__(self)
		self.base_model = base_model
		#self.embedder = nn.Linear(base_model.output_size, d)
		self.embedder = lambda x: x #nn.Linear(base_model.output_size, d)
		self.wu = nn.Linear(d, d)
		self.wv = nn.Linear(d, d)
		self.wc = nn.Linear(2 * d, d)
		self.ws = nn.Linear(d, 1)
	
	def forward(self, input):
		#return F.normalize(self.base_model(input).view(len(input), -1))
		return F.normalize(self.embedder(self.base_model(input).view(len(input), -1)))

	def criterion(self, embeddings, labels, Alpha = 0.5, Beta = 1.0, Lambda = 0.5):
		#embeddings = embeddings * topk_mask(embeddings, dim = 1, K = 512)
		d = pdist(embeddings, squared = True)
		pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(embeddings) - torch.autograd.Variable(torch.eye(len(d))).type_as(embeddings)

		f1, f2 = [embeddings.unsqueeze(dim).expand(len(embeddings), *embeddings.size()) for dim in [0, 1]]
		u = (f1 - f2).abs()
		v = (f1 + f2) / 2
		u_ = F.normalize(F.relu(F.dropout(self.wu(u.view(-1, u.size(-1))), training = self.training)))
		v_ = F.normalize(F.relu(F.dropout(self.wv(v.view(-1, v.size(-1))), training = self.training)))
		s = self.ws(F.relu(F.dropout(self.wc(torch.cat((u_, v_), -1)), training = self.training))).view_as(d)
		
		sneg = s * (1 - pos)
		i, j = min([(s.data[i, j], (i, j)) for i, j in pos.data.nonzero()])[1]
		k, l = sneg.data.max(1, keepdim = True)[1][[i, j], ...].squeeze(1)

		E_m = F.relu(Alpha - s[i, j] + s[i, k]) + F.relu(Alpha - s[i, j] + s[j, l])
		E_e = F.relu(Beta + d[i, j] - d[i, k]) + F.relu(Beta + d[i, j] - d[j, l])

		return E_m + Lambda * E_e
	
	optim_params = dict(lr = 1e-4, momentum = 0.9, weight_decay = 5e-4)
	optim_params_annealed = dict(epoch = 10, gamma = 0.1)
