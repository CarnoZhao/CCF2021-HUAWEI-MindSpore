import os
import time
import datetime
import numpy as np
import yaml
from collections import defaultdict

import mindspore as mind
from mindspore.train.callback import Callback
from mindspore.common.tensor import Tensor
from mindspore.train.serialization import save_checkpoint



class CallbackList(Callback):
    def __init__(self, args, callbacks = []):
        super(CallbackList, self).__init__()
        for k, v in args.items():
            self.__setattr__(k, v)
        self.args = args
        self.name = (datetime.datetime.now() + datetime.timedelta(hours = 8)).strftime("%m%d-%H%M%S-%f")
        self.log_dir = os.path.join("./logs", self.version)
        self.ckpt_dir = os.path.join(self.log_dir, "checkpoints")
        self.log_file = os.path.join(self.log_dir, f"{self.name}.log")
        self.yaml_file = os.path.join(self.log_dir, f"{self.name}.yaml")
        os.system(f"mkdir -p {self.log_dir}")
        os.system(f"mkdir -p {self.ckpt_dir}")
        self.callbacks = callbacks
        self.best_score = None
        self.previous_ckpt_file = None
        with open(self.yaml_file, "w") as f:
            yaml.dump(args, f)

    def log(self, output):
        if not isinstance(output, dict) or "text" not in output: 
            return
        output = output["text"]
        cur_time = (datetime.datetime.utcnow() + datetime.timedelta(hours = 8)).strftime("%m-%d %H:%M:%S")
        output = cur_time + ", " + output
        print(output)
        with open(self.log_file, "a") as f:
            f.write(output + "\n")

    def save_ckpt(self, output, run_context):
        if not isinstance(output, dict) or "metric" not in output:
            return
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        output = output["metric"][self.monitor]
        if self.best_score is None or ((output > self.best_score) ^ (self.monitor_mode != "max")):
            ckpt_file = os.path.join(self.ckpt_dir, f"epoch={cur_epoch}_metric={output:.3f}.ckpt")
            self.best_score = output
            save_checkpoint(cb_params.train_network, ckpt_file)
            if self.previous_ckpt_file is not None:
                os.system(f"rm {self.previous_ckpt_file}")
            self.previous_ckpt_file = ckpt_file
        if cur_epoch == cb_params.epoch_num:
            ckpt_file = os.path.join(self.ckpt_dir, f"last.ckpt")
            save_checkpoint(cb_params.train_network, ckpt_file)

    def epoch_end(self, run_context):
        for callback in self.callbacks:
            output = callback.epoch_end(run_context)
            self.log(output)
            self.save_ckpt(output, run_context)

    def step_end(self, run_context):
        for callback in self.callbacks:
            output = callback.step_end(run_context)
            self.log(output)

class EvalCallBack(Callback):
    def __init__(self, model, eval_dataset, eval_per_epoch = 1):
        super(EvalCallBack, self).__init__()
        self.model = model
        self.eval_dataset = eval_dataset
        self.eval_per_epoch = eval_per_epoch
        self.epoch_per_eval = defaultdict(list)

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        if cur_epoch % self.eval_per_epoch == 0:
            metric = self.model.eval(self.eval_dataset, dataset_sink_mode = False)
            self.epoch_per_eval["epoch"].append(cur_epoch)
            self.epoch_per_eval["acc"].append(metric["Accuracy"])
            self.model.train_network.set_train(True)
            self.model.train_network.phase = "train"
            output = f"val: [{cb_params.cur_epoch_num}], " + ", ".join([f"{k}: {v:.4f}" for k, v in metric.items()])
            return {"text": output, "metric": metric}

class LossCallback(Callback):
    def __init__(self, per_print_times = 1):
        super(LossCallback, self).__init__()
        self._per_print_times = per_print_times
        self.t0 = time.time()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            loss = cb_params.net_outputs
            if isinstance(loss, (tuple, list)):
                if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                    loss = loss[0]
            if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
                loss = np.mean(loss.asnumpy())
            cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
            total_steps = cb_params.batch_num * cb_params.epoch_num
            t1 = time.time()
            eta = str(datetime.timedelta(seconds = (t1 - self.t0) / cb_params.cur_step_num * (total_steps - cb_params.cur_step_num) + 2))
            output = (f"train: [{cb_params.cur_epoch_num:2d}], " + 
                  f"step: [{cur_step_in_epoch:^4d}/{cb_params.batch_num:^4d}], " +
                  f"eta: {eta[:-7] if eta[:-7] else '0:00:00'}, " + 
                  f"loss: {loss:.4f}")
            return {"text": output}

