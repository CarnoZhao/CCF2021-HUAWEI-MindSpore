gpus = "0"
import os
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import time
import glob
import datetime
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict

import mindspore as mind
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
import mindspore.dataset as Dataset
import mindspore.nn.probability.distribution as msd
import mindspore_hub as mshub
import mindspore.dataset.transforms.c_transforms as T
import mindspore.dataset.vision.c_transforms as C
from mindspore import context, Model
from mindspore.nn import Accuracy

from mindspore_lightning.callbacks import EvalCallBack, LossCallback, CallbackList
context.set_context(mode = context.GRAPH_MODE, device_target = "GPU")
mind.set_seed(0); Dataset.config.set_seed(0)


class Train(object):
    def __init__(self, **args):
        for k, v in args.items():
            self.__setattr__(k, v)
        self.args = args
        self.prepare_data()
        self.model = mshub.load(self.model_name, force_reload = False)
        if self.model_name.split("/")[-1].startswith("eff"):
            self.model.drop_out = nn.Dropout(keep_prob = 1 - self.drop_rate)
            self.model.classifier = nn.Dense(in_channels = self.model.classifier.in_channels, out_channels = self.num_classes)
        else:
            self.model.head.fc = nn.SequentialCell(
                nn.Dropout(keep_prob = 1 - self.drop_rate), 
                nn.Dense(in_channels = self.model.head.fc.in_channels, out_channels = self.num_classes))

        self.criterion = nn.SoftmaxCrossEntropyWithLogits(sparse = False, reduction = 'mean')
        self.optimizer = self.configure_optimizers()
        
    def prepare_data(self):
        onehot = T.OneHot(self.num_classes)
        tofloat32 = T.TypeCast(mind.float32)
        mixup = T.RandomApply([C.MixUpBatch(alpha = self.mixup_alpha)], prob = self.mixup_prob)
        self.dl_train = Dataset.MindDataset("./data/train.msrec", ["data", "label"], shuffle = True)\
                    .map(operations = C.Decode(), input_columns = ["data"], num_parallel_workers = 8)\
                    .map(operations = self.trans_train, input_columns = ["data"], num_parallel_workers = 8)\
                    .map(operations = onehot, input_columns = ["label"])\
                    .batch(self.batch_size, drop_remainder = True)\
                    .map(operations = mixup, input_columns = ["data", "label"])\
                    .map(operations = tofloat32, input_columns = ["label"])
        self.dl_valid = Dataset.MindDataset("./data/test.msrec", ["data", "label"])\
                    .map(operations = C.Decode(), input_columns = ["data"], num_parallel_workers = 8)\
                    .map(operations = self.trans_valid, input_columns = ["data"], num_parallel_workers = 8)\
                    .map(operations = onehot, input_columns = ["label"])\
                    .batch(self.batch_size)\
                    .map(operations = tofloat32, input_columns = ["label"])

    def train_dataloader(self):
        return self.dl_train

    def val_dataloader(self):
        return self.dl_valid

    def configure_optimizers(self):
        self.total_steps = self.dl_train.get_dataset_size() * self.num_epochs
        warmup_steps = int(0.3 * self.total_steps)
        annealing_steps = self.total_steps - warmup_steps
        scheduler = np.linspace(0, self.learning_rate, warmup_steps).tolist()
        scheduler += (1e-6 + (self.learning_rate - 1e-6) * (1 + np.cos(np.arange(1, annealing_steps + 1) / annealing_steps * np.pi)) / 2).tolist()
        optimizer = nn.AdamWeightDecay(self.model.trainable_params(), learning_rate = scheduler, weight_decay = 2e-5)
        return optimizer


args = dict(
    model_name = "mindspore/ascend/1.1/efficientnet_v1.1",
    batch_size = 128,
    image_size = 224,
    num_epochs = 30,
    drop_rate = 0.3,
    mixup_alpha = 0.4,
    mixup_prob = 0.5,
    num_classes = 2388,
    learning_rate = 1e-3,
    monitor = "Accuracy",
    monitor_mode = "max",
    version = "0.4mx-30ep",
)
args["trans_train"] = T.Compose([
    C.Resize((args['image_size'], args['image_size'])),
    C.RandomHorizontalFlip(),
    T.RandomApply([C.RandomColorAdjust(0.2, 0.2, 0.2, 0.2)], prob = 0.5),
    T.RandomApply([C.RandomAffine(0, translate = (0, 0.1), scale = (0.9, 1.1))], prob = 1),
    C.Rescale(1.0 / 255.0, 0.0),
    C.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
    C.HWC2CHW()
])
args["trans_valid"] = T.Compose([
    C.Resize((args['image_size'], args['image_size'])),
    C.Rescale(1.0 / 255.0, 0.0),
    C.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
    C.HWC2CHW()
])

if __name__ == "__main__":
    train = Train(**args)
    model = Model(
        train.model,
        train.criterion,
        train.optimizer, 
        metrics = {"Accuracy": Accuracy()})

    logger = CallbackList(args, [
        LossCallback(50),
        EvalCallBack(model, train.val_dataloader())
    ])
    model.train(
        args["num_epochs"], 
        train.train_dataloader(), 
        callbacks = [logger], 
        dataset_sink_mode = False)
