"""
Misc Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import sys
import os
import shutil
import time
import torch
import torch.utils.data
from collections import OrderedDict

if sys.version_info >= (3, 10):
    pass
else:
    pass
from pointcept.utils.timer import Timer
from pointcept.utils.comm import is_main_process
import pointcept.utils.comm as comm
from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class IterationTimer(HookBase):
    def __init__(self, warmup_iter=1):
        self._warmup_iter = warmup_iter
        self._start_time = time.perf_counter()
        self._iter_timer = Timer()
        self._remain_iter = 0

    def before_train(self):
        self._start_time = time.perf_counter()
        _remain_epoch = self.trainer.max_epoch - self.trainer.start_epoch
        self._remain_iter = _remain_epoch * len(self.trainer.train_loader)

    def before_epoch(self):
        self._iter_timer.reset()

    def before_step(self):
        data_time = self._iter_timer.seconds()
        self.trainer.storage.put_scalar("data_time", data_time)

    def after_step(self):
        batch_time = self._iter_timer.seconds()
        self._iter_timer.reset()
        self.trainer.storage.put_scalar("batch_time", batch_time)
        self._remain_iter -= 1
        remain_time = self._remain_iter * self.trainer.storage.history("batch_time").avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))
        if "iter_info" in self.trainer.comm_info.keys():
            info = "Data {data_time_val:.3f} ({data_time_avg:.3f}) Batch {batch_time_val:.3f} ({batch_time_avg:.3f}) Remain {remain_time} ".format(
                data_time_val=self.trainer.storage.history("data_time").val,
                data_time_avg=self.trainer.storage.history("data_time").avg,
                batch_time_val=self.trainer.storage.history("batch_time").val,
                batch_time_avg=self.trainer.storage.history("batch_time").avg,
                remain_time=remain_time,
            )
            self.trainer.comm_info["iter_info"] += info
        if self.trainer.comm_info["iter"] <= self._warmup_iter:
            self.trainer.storage.history("data_time").reset()
            self.trainer.storage.history("batch_time").reset()


@HOOKS.register_module()
class InformationWriter(HookBase):
    def __init__(self):
        self.curr_iter = 0
        self.model_output_keys = []

    def before_train(self):
        self.trainer.comm_info["iter_info"] = ""
        self.curr_iter = self.trainer.start_epoch * len(self.trainer.train_loader)

    def before_step(self):
        self.curr_iter += 1
        info = "Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] ".format(
            epoch=self.trainer.epoch + 1,
            max_epoch=self.trainer.max_epoch,
            iter=self.trainer.comm_info["iter"] + 1,
            max_iter=len(self.trainer.train_loader),
        )
        self.trainer.comm_info["iter_info"] += info

    def after_step(self):
        if "model_output_dict" in self.trainer.comm_info.keys():
            model_output_dict = self.trainer.comm_info["model_output_dict"]
            self.model_output_keys = model_output_dict.keys()
            for key in self.model_output_keys:
                self.trainer.storage.put_scalar(key, model_output_dict[key].item())
        for key in self.model_output_keys:
            self.trainer.comm_info["iter_info"] += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).val
            )
        lr = self.trainer.optimizer.state_dict()["param_groups"][0]["lr"]
        self.trainer.comm_info["iter_info"] += "Lr: {lr:.5f}".format(lr=lr)
        self.trainer.logger.info(self.trainer.comm_info["iter_info"])
        self.trainer.comm_info["iter_info"] = ""
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("params/lr", lr, self.curr_iter)
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train_batch/" + key,
                    self.trainer.storage.history(key).val,
                    self.curr_iter,
                )

    def after_epoch(self):
        epoch_info = "Train result: "
        for key in self.model_output_keys:
            epoch_info += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).avg
            )
        self.trainer.logger.info(epoch_info)
        if self.trainer.writer is not None:
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train/" + key,
                    self.trainer.storage.history(key).avg,
                    self.trainer.epoch + 1,
                )


@HOOKS.register_module()
class CheckpointSaver(HookBase):
    def __init__(self, save_freq=None, save_freq_steps=None):
        self.save_freq = save_freq
        self.save_freq_steps = save_freq_steps
        self.save_disabled = False

    def after_step(self):
        if self.save_freq_steps is None or self.save_disabled:
            return
        if not is_main_process():
            return
        step = int(self.trainer.comm_info.get("iter", 0)) + 1
        if step % int(self.save_freq_steps) != 0:
            return
        filename = os.path.join(
            self.trainer.cfg.save_path, "model", "model_step_last.pth"
        )
        tmp_filename = filename + ".tmp"
        torch.save(
            {
                "epoch": self.trainer.epoch + 1,
                "step": step,
                "state_dict": self.trainer.model.state_dict(),
                "optimizer": self.trainer.optimizer.state_dict(),
                "scheduler": self.trainer.scheduler.state_dict(),
                "scaler": self.trainer.scaler.state_dict()
                if self.trainer.cfg.enable_amp
                else None,
                "best_metric_value": self.trainer.best_metric_value,
            },
            tmp_filename,
        )
        os.replace(tmp_filename, filename)
        self.trainer.logger.info("Saving step checkpoint to: " + filename)

    def after_epoch(self):
        if is_main_process():
            if self.save_disabled:
                return
            is_best = False
            if self.trainer.cfg.evaluate:
                current_metric_value = self.trainer.comm_info["current_metric_value"]
                current_metric_name = self.trainer.comm_info["current_metric_name"]
                if current_metric_value > self.trainer.best_metric_value:
                    self.trainer.best_metric_value = current_metric_value
                    is_best = True
                    self.trainer.logger.info(
                        "Best validation {} updated to: {:.4f}".format(
                            current_metric_name, current_metric_value
                        )
                    )
                self.trainer.logger.info(
                    "Currently Best {}: {:.4f}".format(
                        current_metric_name, self.trainer.best_metric_value
                    )
                )
            filename = os.path.join(
                self.trainer.cfg.save_path, "model", "model_last.pth"
            )
            self.trainer.logger.info("Saving checkpoint to: " + filename)
            tmp_filename = filename + ".tmp"
            try:
                torch.save(
                    {
                        "epoch": self.trainer.epoch + 1,
                        "state_dict": self.trainer.model.state_dict(),
                        "optimizer": self.trainer.optimizer.state_dict(),
                        "scheduler": self.trainer.scheduler.state_dict(),
                        "scaler": self.trainer.scaler.state_dict()
                        if self.trainer.cfg.enable_amp
                        else None,
                        "best_metric_value": self.trainer.best_metric_value,
                    },
                    tmp_filename,
                )
                os.replace(tmp_filename, filename)
                if is_best:
                    shutil.copyfile(
                        filename,
                        os.path.join(
                            self.trainer.cfg.save_path, "model", "model_best.pth"
                        ),
                    )
                if self.save_freq and (self.trainer.epoch + 1) % self.save_freq == 0:
                    shutil.copyfile(
                        filename,
                        os.path.join(
                            self.trainer.cfg.save_path,
                            "model",
                            f"model_eval{self.trainer.epoch + 1}.pth",
                        ),
                    )
            except Exception:
                self.save_disabled = True
                if os.path.exists(tmp_filename):
                    try:
                        os.remove(tmp_filename)
                    except OSError:
                        pass
                self.trainer.logger.exception(
                    "Checkpoint save failed; disabling further checkpoint writes for this run."
                )


@HOOKS.register_module()
class CheckpointLoader(HookBase):
    def __init__(self, keywords="", replacement=None, strict=True):
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.strict = strict

    def before_train(self):
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        if self.trainer.cfg.weight and not os.path.isfile(self.trainer.cfg.weight):
            raise FileNotFoundError(self.trainer.cfg.weight)
        if (
            self.trainer.cfg.resume or self.trainer.cfg.get("eval_only", False)
        ) and not self.trainer.cfg.weight:
            raise ValueError("resume/eval_only requires a checkpoint weight")
        if self.trainer.cfg.weight and os.path.isfile(self.trainer.cfg.weight):
            self.trainer.logger.info(f"Loading weight at: {self.trainer.cfg.weight}")
            checkpoint = torch.load(
                self.trainer.cfg.weight,
                map_location=lambda storage, loc: storage.cuda(),
                weights_only=False,
            )
            self.trainer.logger.info(
                f"Loading layer weights with keyword: {self.keywords}, replace keyword with: {self.replacement}"
            )
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("module."):
                    key = "module." + key
                if self.keywords in key:
                    key = key.replace(self.keywords, self.replacement, 1)
                if comm.get_world_size() == 1:
                    key = key[7:]
                weight[key] = value
            load_state_info = self.trainer.model.load_state_dict(
                weight, strict=self.strict
            )
            self.trainer.logger.info(f"Missing keys: {load_state_info[0]}")
            self.trainer.logger.info(f"Unexpected keys: {load_state_info[1]}")
            if self.trainer.cfg.resume:
                self.trainer.logger.info(
                    f"Resuming train at eval epoch: {checkpoint['epoch']}"
                )
                self.trainer.start_epoch = checkpoint["epoch"]
                self.trainer.best_metric_value = checkpoint["best_metric_value"]
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])
                if self.trainer.cfg.enable_amp:
                    self.trainer.scaler.load_state_dict(checkpoint["scaler"])
        else:
            self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")
