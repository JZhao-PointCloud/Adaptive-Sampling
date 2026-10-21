"""
Trainer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import weakref
import torch
import torch.nn as nn
import torch.utils.data
from packaging import version
from functools import partial

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter
from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.events import EventStorage, ExceptionWriter, JSONWriter
from pointcept.utils.registry import Registry

TRAINERS = Registry("trainers")
AMP_DTYPE = dict(float16=torch.float16, bfloat16=torch.bfloat16)


def _detach_output_dict(output_dict):
    return {
        key: value.detach() if torch.is_tensor(value) else value
        for key, value in output_dict.items()
    }


class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.model = None
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter
        self.json_writer = None
        self.json_iter = 0

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        with EventStorage() as self.storage:
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.before_epoch()
                for self.comm_info["iter"], self.comm_info[
                    "input_dict"
                ] in self.data_iterator:
                    self.before_step()
                    self.run_step()
                    self.after_step()
                self.after_epoch()
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        if comm.is_main_process() and self.json_writer is not None:
            self.storage.iter = self.json_iter
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()
        if comm.is_main_process() and self.json_writer is not None:
            self.json_writer.write()
            self.json_iter += 1

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if comm.is_main_process():
            if self.writer is not None:
                self.writer.close()
            if self.json_writer is not None:
                self.json_writer.close()


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)
        self._gradient_accumulation_counter = 0

    def run_pretrain_validation(self):
        if not self.cfg.evaluate or self.val_loader is None:
            return
        self.logger.info(">>>>>>>>>>>>>>>> Start Pre-Train Evaluation >>>>>>>>>>>>>>>>")
        ran_evaluator = False
        self.comm_info["eval_filename_tag"] = "pretrain"
        for h in self.hooks:
            eval_fn = getattr(h, "eval", None)
            if callable(eval_fn):
                ran_evaluator = True
                eval_fn()
        self.comm_info.pop("eval_filename_tag", None)
        if not ran_evaluator:
            self.logger.warning("No evaluator hook found, skip pre-train evaluation.")
        self.logger.info("<<<<<<<<<<<<<<<<< End Pre-Train Evaluation <<<<<<<<<<<<<<<<<")

    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():
            self.before_train()
            if comm.is_main_process() and self.json_writer is not None:
                self.json_iter = self.start_epoch * (len(self.train_loader) + 1)
            self.run_pretrain_validation()
            if getattr(self.cfg, "eval_only", False):
                self.logger.info(
                    ">>>>>>>>>>>>>>>> Eval-only mode, skip training loop >>>>>>>>>>>>>>>>"
                )
                self.after_train()
                return
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()
                for self.comm_info["iter"], self.comm_info[
                    "input_dict"
                ] in self.data_iterator:
                    self.before_step()
                    self.run_step()
                    self.after_step()
                self.after_epoch()
            self.after_train()

    def run_step(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            auto_cast = partial(torch.amp.autocast, device_type="cuda")
        else:
            auto_cast = torch.cuda.amp.autocast
        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
        if self._gradient_accumulation_counter == 0:
            self.optimizer.zero_grad()
        with auto_cast(
            enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype]
        ):
            output_dict = self.model(input_dict)
            loss = output_dict["loss"] / self.cfg.gradient_accumulation_steps
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        self._gradient_accumulation_counter += 1
        if self._gradient_accumulation_counter >= self.cfg.gradient_accumulation_steps:
            if self.cfg.enable_amp:
                self.scaler.unscale_(self.optimizer)
                if self.cfg.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.clip_grad
                    )
                self.scaler.step(self.optimizer)
                scale = self.scaler.get_scale()
                self.scaler.update()
                if scale <= self.scaler.get_scale():
                    self.scheduler.step()
            else:
                if self.cfg.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.clip_grad
                    )
                self.optimizer.step()
                self.scheduler.step()
            self._gradient_accumulation_counter = 0
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = _detach_output_dict(output_dict)

    def after_epoch(self):
        if comm.is_main_process() and self.json_writer is not None:
            self.storage.iter = self.json_iter
        for h in self.hooks:
            h.after_epoch()
        if comm.is_main_process() and self.json_writer is not None:
            self.json_writer.write()
            self.json_iter += 1
        self.storage.reset_histories()
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum((p.numel() for p in model.parameters() if p.requires_grad))
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
            fp16_compression=getattr(self.cfg, "fp16_compression", False),
        )
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        if writer is not None:
            self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        if self.cfg.enable_json_log and comm.is_main_process():
            json_log_path = os.path.join(self.cfg.save_path, self.cfg.json_log_file)
            if not self.cfg.resume and os.path.exists(json_log_path):
                os.remove(json_log_path)
            self.json_writer = JSONWriter(
                json_log_path, window_size=self.cfg.json_log_window_size
            )
            self.logger.info(f"JSON writer logging file: {json_log_path}")
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)
        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None
        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=train_sampler is None,
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.val_num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
                persistent_workers=True,
                prefetch_factor=2,
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = (
            len(self.train_loader)
            * self.cfg.eval_epoch
            // self.cfg.gradient_accumulation_steps
        )
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            grad_scaler = partial(torch.amp.GradScaler, device="cuda")
        else:
            grad_scaler = torch.cuda.amp.GradScaler
        scaler = grad_scaler() if self.cfg.enable_amp else None
        return scaler
