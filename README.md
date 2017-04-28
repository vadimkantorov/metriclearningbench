# [WIP] Metric learning [models](./model.py) in PyTorch
1. [Lifted structured embedding](https://arxiv.org/abs/1511.06452)
2. [Triplet loss](https://arxiv.org/abs/1503.03832)
3. [Triplet ratio loss](https://arxiv.org/abs/1502.05908)
4. [PDDM](https://arxiv.org/abs/1610.08904) [WIP]

# Datasets supported
1. [CUB2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
2. [CARS196](http://ai.stanford.edu/~jkrause/cars/car_dataset.html) [WIP]
3. [Stanford Online Products](http://cvgl.stanford.edu/projects/lifted_struct/) [WIP]

### Examples
```shell
# train lifted structured embedding on CUB2011
python train.py --MODEL=LIFTED_STRUCT --BASE_MODEL=GOOGLENET --DATASET=CUB2011

# train triplet loss embedding on CUB2011
python train.py --MODEL=TRIPLET --BASE_MODEL=GOOGLENET --DATASET=CUB2011
```

### CUB2011 results
| model | R1 @ epoch0 | R1 @ epoch10 | R1 @ epoch30 |
| --- | --- | --- | --- |
| LIFTED_STRUCT | | | |
| TRIPLET | | | | |
| UNTRAINED | | | |
