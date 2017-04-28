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
	
	optim_algo = optim.SGD
	optim_params = dict(lr = 1e-5, momentum = 0.9, weight_decay = 2e-4, dampening = 0.9)

def pdist(A, squared = False, eps = 1e-4):
	prod = torch.mm(A, A.t())
	norm = prod.diag().unsqueeze(1).expand_as(prod)
	res = (norm + norm.t() - 2 * prod).clamp(min = 0)
	return res if squared else (res + eps).sqrt() + eps 
		
def l2_normalize(A, eps = 1e-4):
	return A / (A.norm(p = 2, dim = -1).expand_as(A) + eps)

class LiftedStruct(Model):
	def criterion(self, features, labels, margin = 1.0, eps = 1e-4):
		d = pdist(features, squared = False, eps = eps)
		pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(features)
		neg_i = torch.mul((margin - d).exp(), 1 - pos).sum(1).expand_as(d)
		return torch.sum(torch.mul(pos.triu(1), torch.log(neg_i + neg_i.t()) + d).clamp(min = 0).pow(2)) / (pos.sum() - len(d))

class TripletWorking(Model):
	def criterion(self, features, labels, margin = 1.0):
		features = features.view(features.size(0) / 3, 3, *features.size()[1:])
		anchor, positive, negative = features.select(1, 0), features.select(1, 1), features.select(1, 2)
		return F.triplet_margin_loss(anchor, positive, negative, margin = margin)

	optim_params = dict(lr = 1e-5, momentum = 0.9, weight_decay = 5e-4)

class Triplet(Model):
	def criterion(self, features, labels, margin = 1.0):
		d = pdist(features, squared = False)
		pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(features)
		T = d.unsqueeze(1).expand(*(len(d),) * 3)
		M = pos.unsqueeze(1).expand_as(T) * (1 - pos.unsqueeze(2).expand_as(T))
		return (M * torch.clamp(T - T.transpose(1, 2) + margin, min = 0)).sum() / M.sum()
	
	optim_params = dict(lr = 2e-5, momentum = 0.9, weight_decay = 5e-4)

class TripletRatio(Model):
	def criterion(self, features, labels, margin = 0.1, eps = 1e-4):
		d = pdist(features, squared = False, eps = eps)
		pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(features)
		T = d.unsqueeze(1).expand(*(len(d),) * 3) # [i][k][j]
		M = pos.unsqueeze(1).expand_as(T) * (1 - pos.unsqueeze(2).expand_as(T))
		return (M * T.div(T.transpose(1, 2) + margin)).sum() / M.sum() #[i][k][j] = 

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
		self.dropout = nn.Dropout()
	
	def forward(self, input):
		return l2_normalize(self.base_model(input).view(input.size(0), -1))

	def criterion(self, features, labels, Alpha = 0.5, Beta = 1.0, Lambda = 0.5):
		d = pdist(features, squared = False)
		pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(features)

		f1, f2 = [features.unsqueeze(dim).expand(len(features), *features.size()) for dim in [0, 1]]
		u = (f1 - f2).abs()
		v = (f1 + f2) / 2
		u_ = l2_normalize(F.relu(self.dropout(self.wu(u.view(-1, u.size(-1))))))
		v_ = l2_normalize(F.relu(self.dropout(self.wv(v.view(-1, v.size(-1))))))
		s = self.ws(F.relu(self.dropout(self.wc(torch.cat((u_, v_), -1))))).view(len(features), len(features))
		s = (s - s.min().expand_as(s)) / (s.max() - s.min()).expand_as(s)
		
		i, j = min([(s[i, j], (i, j)) for i, j in pos.data.nonzero()])[1]
		k, l = (s * (1 - pos)).max(1)[1].data.squeeze(1)[torch.cuda.LongTensor([i, j])]

		E_m = torch.clamp(Alpha + s[i, k] - s[i, j], min = 0) + torch.clamp(Alpha + s[j, l] - s[i, j], min = 0)
		E_e = torch.clamp(Beta + d[i, j] - d[i, k], min = 0) + torch.clamp(Beta + d[i, j] - d[j, l], min = 0)

		return E_m + Lambda * E_e
		#return E_e
		#return E_m
	
	optim_params = dict(lr = 1e-5, momentum = 0.9, weight_decay = 5e-4)
