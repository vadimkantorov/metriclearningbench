import torch
import torch.nn as nn
import torch.nn.functional as F

def topk_mask(input, dim, K = 10, **kwargs):
	index = input.topk(max(1, min(K, input.size(dim))), dim = dim, **kwargs)[1]
	return torch.zeros_like(input).scatter(dim, index, 1.0)

def pdist(A, squared = False, eps = 1e-4):
	prod = torch.mm(A, A.t())
	norm = prod.diag().unsqueeze(1).expand_as(prod)
	res = (norm + norm.t() - 2 * prod).clamp(min = 0)
	return res if squared else res.clamp(min = eps).sqrt()
	
class Model(nn.Module):
	def __init__(self, base_model, num_classes, embedding_size = 128):
		super(Model, self).__init__()
		self.base_model = base_model
		self.num_classes = num_classes
		self.embedder = nn.Linear(base_model.output_size, embedding_size)

	def forward(self, input):
		return self.embedder(F.relu(self.base_model(input).view(len(input), -1)))
	
	criterion = None
	optimizer = torch.optim.SGD
	optimizer_params = dict(lr = 1e-4, momentum = 0.9, weight_decay = 2e-4)
	lr_scheduler_params = dict(step_size = float('inf'), gamma = 0.1)

class Untrained(Model):
	def forward(self, input):
		return self.base_model(input).view(input.size(0), -1).detach()

class LiftedStruct(Model):
	def criterion(self, embeddings, labels, margin = 1.0, eps = 1e-4):
		d = pdist(embeddings, squared = False, eps = eps)
		pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(d)
		neg_i = torch.mul((margin - d).exp(), 1 - pos).sum(1).expand_as(d)
		return torch.sum(F.relu(pos.triu(1) * ((neg_i + neg_i.t()).log() + d)).pow(2)) / (pos.sum() - len(d))

class Triplet(Model):
	def criterion(self, embeddings, labels, margin = 1.0):
		d = pdist(embeddings)
		pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(d) - torch.autograd.Variable(torch.eye(len(d))).type_as(d)
		T = d.unsqueeze(1).expand(*(len(d),) * 3)
		M = pos.unsqueeze(1).expand_as(T) * (1 - pos.unsqueeze(2).expand_as(T))
		return (M * F.relu(T - T.transpose(1, 2) + margin)).sum() / M.sum()
	
	optimizer_params = dict(lr = 1e-4, momentum = 0.9, weight_decay = 5e-4)
	lr_scheduler_params = dict(step_size = 30, gamma = 0.5)

class TripletRatio(Model):
	def criterion(self, embeddings, labels, margin = 0.1, eps = 1e-4):
		d = pdist(embeddings, squared = False, eps = eps)
		pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(d)
		T = d.unsqueeze(1).expand(*(len(d),) * 3)
		M = pos.unsqueeze(1).expand_as(T) * (1 - pos.unsqueeze(2).expand_as(T))
		return (M * T.div(T.transpose(1, 2) + margin)).sum() / M.sum()

class Pddm(Model):
	def __init__(self, base_model, num_classes, d = 1024):
		nn.Module.__init__(self)
		self.base_model = base_model
		#self.embedder = nn.Linear(base_model.output_size, d)
		self.embedder = lambda x: x #nn.Linear(base_model.output_size, d)
		self.wu = nn.Linear(d, d)
		self.wv = nn.Linear(d, d)
		self.wc = nn.Linear(2 * d, d)
		self.ws = nn.Linear(d, 1)
	
	def forward(self, input):
		return F.normalize(Model.forward(self, input))

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
		i, j = min([(s[i, j], (i, j)) for i, j in pos.nonzero()])[1]
		k, l = sneg.max(1, keepdim = True)[1][[i, j], ...].squeeze(1)

		E_m = F.relu(Alpha - s[i, j] + s[i, k]) + F.relu(Alpha - s[i, j] + s[j, l])
		E_e = F.relu(Beta + d[i, j] - d[i, k]) + F.relu(Beta + d[i, j] - d[j, l])

		return E_m + Lambda * E_e
	
	optimizer_params = dict(lr = 1e-4, momentum = 0.9, weight_decay = 5e-4)
	lr_scheduler_params = dict(step_size = 10, gamma = 0.1)

class Margin(Model):
	def forward(self, input):
		return F.normalize(Model.forward(self, input))

	def criterion(self, embeddings, labels, alpha = 0.2, beta = 1.2, distance_threshold = 0.5, inf = 1e6, eps = 1e-6, distance_weighted_sampling = False):
		d = pdist(embeddings)
		pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(d) - torch.eye(len(d)).type_as(d)
		num_neg = int(pos.sum() / len(pos))
		if distance_weighted_sampling:
			neg = torch.zeros_like(pos).scatter_(1, torch.multinomial((d.clamp(min = distance_threshold).pow(embeddings.size(-1) - 2) * (1 - d.clamp(min = distance_threshold).pow(2) / 4).pow(0.5 * (embeddings.size(-1) - 3))).reciprocal().masked_fill_(pos + torch.eye(len(d)).type_as(d) > 0, eps), replacement = False, num_samples = num_neg), 1)
		else:
			neg = topk_mask(d  + inf * ((pos > 0) + (d < distance_threshold)).type_as(d), dim = 1, largest = False, K = num_neg)
		L = F.relu(alpha + (pos * 2 - 1) * (d - beta))
		M = ((pos + neg > 0) * (L > 0)).float()
		return (M * L).sum() / M.sum()

	optimizer = torch.optim.Adam
	optimizer_params = dict(lr = 1e-3, weight_decay = 1e-4, base_model_lr_mult = 1e-2)
	#optimizer_params = dict(lr = 1e-3, momentum = 0.9, weight_decay = 5e-4, base_model_lr_mult = 1)
	#lr_scheduler_params = dict(step_size = 10, gamma = 0.5)
