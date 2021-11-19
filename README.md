
# Introduction

Codes for 2021 CCF HUAWEI MindSpore competition

These codes were developed under `mindspore` deep learning framework, and were well tested under local GPU machine (NVIDIA RTX TITAN) and *HUAWEI cloud modelarts* GPU machine, and *HUAWEI cloud modelarts* Ascend machine. I used a simple EfficientNet-b0 structure to deal with image classification problem. The EfficientNet-b0 and pretrained weights are available in `mindspore-hub`'s github repository.

# Directory structure

|---data
|    |---train
|    |    |-class1
|    |    |    |-1.jpg
|    |    |    |-...
|    |    |-class2
|    |    |-...
|    |
|    |---test
|         |-class1
|         |    |-1.jpg
|         |    |-...
|         |-class2
|         |-...
|
|-Recoder.py
|-Solver.py

# Prerequisite

- `mindspore-hub`: Please follow [https://github.com/mindspore-ai/hub](https://github.com/mindspore-ai/hub)

# Data Preprocessing

```python
python Recoder.py
```

This will compress train images and test images into mindspore record format, saved as `./data/{train|test}.msrec` and `./data/{train|test}.msrec.db`

# Training

```python
python Solver.py
```

Training configs can be set in `Solver.py`. Available arguments include learning rate, batch size, mixup probability, mixup alpha, dropout rate, image size and number of epochs. Data augmentations were also writen following `mindspore`'s recommendation. Learning rate scheduler was defined as iter-based cosine-annealing with warmup. 

The training logs and hyper-parameters will be saved in `./logs/{version}`, and `version` can be set in `Solver.py`.

The training time of each machine:

- (fastest) HUAWEI modelarts 1x Ascend **with fp16 level O3** (4h/10epochs)

- local NVIDIA TITAN RTX GPU & HUAWEI modelarts 1x V100 **without fp16** (6h/10epochs)

- HUAWEI modelarts 1x Ascend **without fp16** (7h/10epochs)

- (slowest) local NVIDIA TITAN RTX GPU **with fp16 level O2** (9h/10epochs)

# Results

|version|test accuracy|
|:-:|:-:|
|local GPU 10 epochs, mixup alpha = 0.4, prob = 0.5|0.9721|
|local GPU 20 epochs, mixup alpha = 0.4, prob = 0.5|0.9725|
|local GPU 30 epochs, mixup alpha = 0.4, prob = 0.5|0.9732|
|Ascend 10 epochs, mixup alpha = 0.4, prob = 0.5|0.9709|

