from collections import OrderedDict
import torch.nn as nn
import torchvision

class resnet50(nn.Sequential):
	output_size = 2048
	input_side = 224
	rescale = 1
	rgb_mean = [0.485, 0.456, 0.406]
	rgb_std = [0.229, 0.224, 0.225]

	def __init__(self, dilation = False):
		super(resnet50, self).__init__()
		pretrained = torchvision.models.resnet50(pretrained = True)
		for module in filter(lambda m: type(m) == nn.BatchNorm2d, pretrained.modules()):
			module.eval()
			module.train = lambda _: None
		
		if dilation:
			pretrained.layer4[0].conv1.dilation = (2, 2)
			pretrained.layer4[0].conv1.padding = (2, 2)
			pretrained.layer4[0].conv1.stride = (1, 1)
			pretrained.layer4[0].downsample[0].stride = (1, 1)
		
		for module_name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']:
			self.add_module(module_name, getattr(pretrained, module_name))
