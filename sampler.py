import math
import random
import itertools

def index_dataset(dataset):
	return {c : [example_idx for example_idx, (image_file_name, class_label_ind) in enumerate(dataset.imgs) if class_label_ind == c] for c in set(dict(dataset.imgs).values())}

def sample_from_class(images_by_class, class_label_ind):
	return images_by_class[class_label_ind][random.randrange(len(images_by_class[class_label_ind]))]

def simple(batch_size, dataset, prob_other = 0.5):
	'''lazy sampling, not like in lifted_struct. they add to the pool all postiive combinations, then compute the average number of positive pairs per image, then sample for every image the same number of negative pairs'''
	images_by_class = index_dataset(dataset)
	for batch_idx in xrange(int(math.ceil(len(dataset) * 1.0 / batch_size))):
		example_indices = []
		for i in range(0, batch_size, 2):
			perm = random.sample(images_by_class.keys(), 2)
			example_indices += [sample_from_class(images_by_class, perm[0]), sample_from_class(images_by_class, perm[0 if i == 0 or random.random() > prob_other else 1])]
		yield example_indices[:batch_size]

def triplet(batch_size, dataset):
	images_by_class = index_dataset(dataset)
	for batch_idx in xrange(int(math.ceil(len(dataset) * 1.0 / batch_size))):
		example_indices = []
		for i in range(0, batch_size, 3):
			perm = random.sample(images_by_class.keys(), 2)
			example_indices += [sample_from_class(images_by_class, perm[0]), sample_from_class(images_by_class, perm[0]), sample_from_class(images_by_class, perm[1])]
		yield example_indices[:batch_size]

def npairs(batch_size, dataset, K = 4):
	images_by_class = index_dataset(dataset)
	for batch_idx in xrange(int(math.ceil(len(dataset) * 1.0 / batch_size))):
		example_indices = [sample_from_class(images_by_class, class_label_ind) for k in range(int(math.ceil(batch_size * 1.0 / K))) for class_label_ind in [random.choice(images_by_class.keys())] for i in range(K)]
		yield example_indices[:batch_size]
