import random
import torch
import torch.nn as nn
import torch.optim as optim
	
class EmbedderSimple(nn.Module):
	def __init__(self, base_model, embedding_size = 128):
		super(EmbedderSimple, self).__init__()
		self.base_model = base_model
		self.embedder = nn.Linear(base_model.output_size, embedding_size)

	def forward(self, input):
		return self.embedder(self.base_model(input).view(input.size(0), -1))
	
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

def pdist_squared(A):
	prod = torch.mm(A, A.t())
	norm = prod.diag().unsqueeze(1).expand_as(prod)
	return (norm + norm.t() - 2 * prod).clamp(min = 0)

class LiftedStruct(EmbedderSimple):
	def criterion(self, input, labels, margin = 1.0, eps = 1e-4):
		d = (pdist_squared(input) + eps).sqrt() + eps
		pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(input)
		neg_i = torch.mul((margin - d).exp(), 1 - pos).sum(1).expand_as(d)
		return torch.sum(torch.mul(pos.triu(1), torch.log(neg_i + neg_i.t()) + d).clamp(min = 0).pow(2)) / (pos.sum() - len(d))

class Triplet(EmbedderSimple):
	def criterion(self, input, labels, margin = 1.0):
		d = pdist_squared(input)
		pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(input)
		T = d.unsqueeze(1).expand((len(d),) * 3) # [i][k][j]
		M = pos.unsqueeze(1).expand_as(T) * (1 - pos.unsqueeze(2).expand_as(T))
		return (M * torch.clamp(T + T.transpose(1, 2) - margin, min = 0)).sum() / M.sum() #[i][k][j] = 

class TripletRatio(EmbedderSimple):
	def criterion(self, input, labels, margin = 0.1, eps = 1e-4):
		d = (pdist_squared(input) + eps).sqrt() + eps
		pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(input)
		T = d.unsqueeze(1).expand((len(d),) * 3) # [i][k][j]
		M = pos.unsqueeze(1).expand_as(T) * (1 - pos.unsqueeze(2).expand_as(T))
		return (M * T.div(T.transpose(1, 2) + margin)).sum() / M.sum() #[i][k][j] = 
