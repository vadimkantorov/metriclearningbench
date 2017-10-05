# [WIP] Metric learning [models](./model.py) in PyTorch
1. [Lifted structured embedding](https://arxiv.org/abs/1511.06452)
2. [Triplet loss](https://arxiv.org/abs/1503.03832)
3. [Margin contrastive loss](https://arxiv.org/abs/1706.07567)

# Examples
```shell
# train margin contrastive loss on CUB2011 using ResNet-50
python train.py --dataset cub2011 --model margin --base resnet50

# download GoogLeNet weights and train using LiftedStruct loss
wget -P ./data https://github.com/vadimkantorov/metriclearningbench/releases/download/data/googlenet.h5
python train.py --dataset cub2011 --model liftedstruct --base inception_v1_googlenet
```

# Results (recall@1)
| |[CUB2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) | [CARS196](http://ai.stanford.edu/~jkrause/cars/car_dataset.html) | [Stanford Online Products](http://cvgl.stanford.edu/projects/lifted_struct/)
|:---:|:---:|:---:|:---:|
| Margin contrastive, semi-hard | [0.58](./data/log.txt.margin) @ epoch60 | | [0.7526](./data/log.txt.margin_stanfordonlineproducts) @ epoch90 | 
