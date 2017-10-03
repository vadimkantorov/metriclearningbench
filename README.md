# [WIP] Metric learning [models](./model.py) in PyTorch
1. [Lifted structured embedding](https://arxiv.org/abs/1511.06452)
2. [Triplet loss](https://arxiv.org/abs/1503.03832)
3. [Margin contrastive loss](https://arxiv.org/abs/1706.07567)

# Datasets supported
1. [CUB2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
2. [CARS196](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)
3. [Stanford Online Products](http://cvgl.stanford.edu/projects/lifted_struct/)

# Examples
```shell
# train margin contrastive loss on CUB2011 using ResNet-50
python train.py --dataset CUB2011 --model margin --base resnet50

# download GoogLeNet weights and train using LiftedStruct loss
wget -P ./data https://github.com/vadimkantorov/metriclearningbench/releases/download/data/googlenet.h5
python train.py --dataset CUB2011 --model liftedstruct --base inception_v1_googlenet
```

# Results @ epoch100
| | CUB2011
|:---:|:---:|
| Margin contrastive | 0.58 |
