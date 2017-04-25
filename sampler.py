import random

def simple(batch_size, dataset, train_classes):
	'''lazy sampling, not like in lifted_struct. they add to the pool all postiive combinations, then compute the average number of positive pairs per image, then sample for every image the same number of negative pairs'''
	images_by_class = {class_label_ind_train : [example_idx for example_idx, (image_file_name, class_label_ind) in enumerate(dataset.imgs) if class_label_ind == class_label_ind_train] for class_label_ind_train in range(train_classes)}
	sample_from_class = lambda class_label_ind: images_by_class[class_label_ind][random.randrange(len(images_by_class[class_label_ind]))]
	while True:
		example_indices = []
		for i in range(0, batch_size, 2):
			perm = random.sample(xrange(train_classes), 2)
			example_indices += [sample_from_class(perm[0]), sample_from_class(perm[0 if i == 0 or random.random() > 0.5 else 1])]
		yield example_indices
