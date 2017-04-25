import random
import itertools

def simple(batch_size, dataset):
	'''lazy sampling, not like in lifted_struct. they add to the pool all postiive combinations, then compute the average number of positive pairs per image, then sample for every image the same number of negative pairs'''
	class_label_inds = sorted(set(dict(dataset.imgs).values()))
	images_by_class = {c : [example_idx for example_idx, (image_file_name, class_label_ind) in enumerate(dataset.imgs) if class_label_ind == c] for c in class_label_inds}
	sample_from_class = lambda class_label_ind: images_by_class[class_label_ind][random.randrange(len(images_by_class[class_label_ind]))]
	while True:
		example_indices = []
		for i in range(0, batch_size, 2):
			perm = random.sample(class_label_inds, 2)
			example_indices += [sample_from_class(perm[0]), sample_from_class(perm[0 if i == 0 or random.random() > 0.5 else 1])]
		yield example_indices
