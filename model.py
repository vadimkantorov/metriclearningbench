import torch
import torch.nn as nn
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

class LiftedStruct(Model):
	def criterion(self, input, labels, margin = 1.0, eps = 1e-4):
		d = pdist(input, squared = False, eps = eps)
		pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(input)
		neg_i = torch.mul((margin - d).exp(), 1 - pos).sum(1).expand_as(d)
		return torch.sum(torch.mul(pos.triu(1), torch.log(neg_i + neg_i.t()) + d).clamp(min = 0).pow(2)) / (pos.sum() - len(d))

class Triplet(Model):
	def criterion(self, input, labels, margin = 1.0):
		d = pdist(input, squared = True)
		pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(input)
		T = d.unsqueeze(1).expand((len(d),) * 3) # [i][k][j]
		M = pos.unsqueeze(1).expand_as(T) * (1 - pos.unsqueeze(2).expand_as(T))
		return (M * torch.clamp(T + T.transpose(1, 2) - margin, min = 0)).sum() / M.sum() #[i][k][j] = 

class TripletRatio(Model):
	def criterion(self, input, labels, margin = 0.1, eps = 1e-4):
		d = pdist(input, squared = False, eps = eps)
		pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(input)
		T = d.unsqueeze(1).expand((len(d),) * 3) # [i][k][j]
		M = pos.unsqueeze(1).expand_as(T) * (1 - pos.unsqueeze(2).expand_as(T))
		return (M * T.div(T.transpose(1, 2) + margin)).sum() / M.sum() #[i][k][j] = 

class Npairs(Model):
	pass

class Pddm(nn.Module):
	def __init__(self, base_model, embedding_size = 128):
		d = base_model.output_size
		self.wu = nn.Linear(d, d)
		self.wv = nn.Linear(d, d)
		self.wc = nn.Linear(2 * d, d)
		self.ws = nn.Linear(d, 1)
		self.base_model = base_model
	
	def forward(self, input):
		l2normalize = lambda input, eps = 1e-4: input / (input.norm(p = 2, dim = -1).expand_as(input) + eps)

		f = l2normalize(self.base_model(input.view(len(input), -1)))
		f1, f2 = [f.unsqueeze(dim).expand(len(f), *f.size()) for dim in [0, 1]]
		
		u = (f1 - f2).abs()
		v = (f1 + f2) / 2
		u_ = l2normalize(F.relu(F.dropout(self.wu(u.view(-1, u.size(-1))))))
		v_ = l2normalize(F.relu(F.dropout(self.wv(v.view(-1, v.size(-1))))))

		c = F.relu(F.dropout(self.wc(torch.cat((u_, v_), -1))))
		s = self.ws(c).view(len(f), len(f))
