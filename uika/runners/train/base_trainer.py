# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import time
import math
import shutil
import torch
import safetensors

from typing import Optional, Literal
from omegaconf import OmegaConf
from abc import abstractmethod
from contextlib import contextmanager
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed

from uika.utils.config import parse_configs
from uika.utils.logging import configure_logger
from uika.utils.compile import configure_dynamo
from uika.runners.abstract import Runner


logger = get_logger(__name__)


class Trainer(Runner):
    def __init__(self):
        super().__init__()

        self.cfg = parse_configs()
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")

        self.accelerator = Accelerator(
            mixed_precision=self.cfg.train.mixed_precision,
            gradient_accumulation_steps=self.cfg.train.accum_steps,
            log_with=tuple(self.cfg.logger.trackers),
            project_config=ProjectConfiguration(
                logging_dir=self.cfg.logger.tracker_root,
            ),
            use_seedable_sampler=True,
            kwargs_handlers=[
                DistributedDataParallelKwargs(
                    find_unused_parameters=self.cfg.train.find_unused_parameters,
                ),
            ],
        )
        set_seed(self.cfg.experiment.seed, device_specific=True)
        with self.accelerator.main_process_first():
            configure_logger(
                stream_level=self.cfg.logger.stream_level,
                log_level=self.cfg.logger.log_level,
                file_path=os.path.join(
                    self.cfg.logger.log_root,
                    self.cfg.experiment.parent, self.cfg.experiment.child,
                    f"{self.timestamp}.log",
                ) if self.accelerator.is_main_process else None,
            )
        logger.info(self.accelerator.state, main_process_only=False, in_order=True)
        configure_dynamo(dict(self.cfg.compile))

        self._log_config()

        # mkdirs for img logging
        self.img_log_path = os.path.join(
            self.cfg.logger.img_log_root,
            self.cfg.experiment.parent, 
            self.cfg.experiment.child,
            f"{self.timestamp}",
        )
        if self.accelerator.is_main_process:
            os.makedirs(self.img_log_path, exist_ok=True)

        # attributes with defaults
        self.model : torch.nn.Module = None
        self.optimizer: torch.optim.Optimizer = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler = None
        self.train_loader: DataLoader = None
        self.val_loader: DataLoader = None
        self.prepare_dataloaders_with_accelerator: bool = True
        self.N_max_global_steps: int = None
        self.N_global_steps_per_epoch: int = None
        self.global_step: int = 0
        self.current_epoch: int = 0

    def __enter__(self):
        self.accelerator.init_trackers(
            project_name=f"{self.cfg.experiment.parent}/{self.cfg.experiment.child}",
        )
        self.prepare_everything()
        self.log_inital_info()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.accelerator.end_training()

    # @staticmethod
    def control(option: str = None, synchronized: bool = False):
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                if option is None or hasattr(self.accelerator, option):
                    accelerated_func = getattr(self.accelerator, option)(func) if option is not None else func
                    result = accelerated_func(self, *args, **kwargs)
                    if synchronized:
                        self.accelerator.wait_for_everyone()
                    return result
                else:
                    raise AttributeError(f"Accelerator has no attribute {option}")
            return wrapper
        return decorator

    @contextmanager
    def exec_in_order(self):
        for rank in range(self.accelerator.num_processes):
            try:
                if self.accelerator.process_index == rank:
                    yield
            finally:
                self.accelerator.wait_for_everyone()

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self) -> bool:
        return self.accelerator.num_processes > 1

    @property
    def train_loader_for_stats(self):
        if not self.prepare_dataloaders_with_accelerator:
            loader = getattr(self, "real_train_loader", None)
            return loader if loader is not None else self.train_loader
        return self.train_loader

    @property
    def val_loader_for_stats(self):
        if not self.prepare_dataloaders_with_accelerator:
            loader = getattr(self, "real_val_loader", None)
            return loader if loader is not None else self.val_loader
        return self.val_loader

    def prepare_everything(self, is_dist_validation: bool = True):
        # prepare with accelerator
        train_loader_for_stats = self.train_loader_for_stats
        val_loader_for_stats = self.val_loader_for_stats
        logger.info(
            f"Len of Actual Loaders Before Prepare: "
            f"Train -> {len(train_loader_for_stats)}, Val -> {len(val_loader_for_stats)}"
        )
        if self.prepare_dataloaders_with_accelerator and is_dist_validation:
            self.model, self.optimizer, self.train_loader, self.val_loader = \
                self.accelerator.prepare(
                    self.model, self.optimizer, self.train_loader, self.val_loader,
                )
        elif self.prepare_dataloaders_with_accelerator:
            self.model, self.optimizer, self.train_loader = \
                self.accelerator.prepare(
                    self.model, self.optimizer, self.train_loader,
                )
        else:
            self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        train_loader_for_stats = self.train_loader_for_stats
        val_loader_for_stats = self.val_loader_for_stats
        logger.info(
            f"Len of Actual Loaders After Prepare: "
            f"Train -> {len(train_loader_for_stats)}, Val -> {len(val_loader_for_stats)}"
        )
        logger.info(f"Accelerate dataloader prepare: {self.prepare_dataloaders_with_accelerator}")

        self.accelerator.register_for_checkpointing(self.scheduler)
        # prepare stats
        # N_total_batch_size = self.cfg.train.batch_size * self.accelerator.num_processes * self.cfg.train.accum_steps
        self.N_global_steps_per_epoch = math.ceil(len(train_loader_for_stats) / self.cfg.train.accum_steps)
        self.N_max_global_steps = self.N_global_steps_per_epoch * self.cfg.train.epochs
        if self.cfg.train.debug_global_steps is not None:
            logger.warning(f"Overriding max global steps from {self.N_max_global_steps} to {self.cfg.train.debug_global_steps}")
            self.N_max_global_steps = self.cfg.train.debug_global_steps
        logger.info(f"======== Statistics ========")
        logger.info(f"** N_max_global_steps: {self.N_max_global_steps}")
        # logger.info(f"** N_total_batch_size: {N_total_batch_size}")
        logger.info(f"** Max_img_per_gpu: {self.cfg.data.loader.max_img_per_gpu}")
        logger.info(f"** N_epochs: {self.cfg.train.epochs}")
        logger.info(f"** N_global_steps_per_epoch: {self.N_global_steps_per_epoch}")
        logger.debug(f"** Actual train loader length: {len(train_loader_for_stats)}")
        logger.info(f"** Distributed validation: {is_dist_validation}")
        logger.info(f"============================")
        logger.info(f"======== Trainable parameters ========")
        logger.info(f"** Total: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        for sub_name, sub_module in self.accelerator.unwrap_model(self.model).named_children():
            logger.info(f"** {sub_name}: {sum(p.numel() for p in sub_module.parameters() if p.requires_grad)}")
        logger.info(f"=====================================")
        self.accelerator.wait_for_everyone()
        # load checkpoint or model
        self.load_ckpt_or_auto_resume_(self.cfg)
        # register hooks
        self.register_hooks()

    @abstractmethod
    def register_hooks(self):
        pass

    def auto_resume_(self, cfg) -> bool:
        ckpt_root = os.path.join(
            cfg.saver.checkpoint_root,
            cfg.experiment.parent, cfg.experiment.child,
        )
        if not os.path.exists(ckpt_root):
            return False
        ckpt_dirs = os.listdir(ckpt_root)
        if len(ckpt_dirs) == 0:
            return False
        ckpt_dirs.sort()
        latest_ckpt = ckpt_dirs[-1]
        latest_ckpt_dir = os.path.join(ckpt_root, latest_ckpt)
        logger.info(f"======== Auto-resume from {latest_ckpt_dir} ========")
        self.accelerator.load_state(latest_ckpt_dir)
        self.global_step = int(latest_ckpt)
        self.current_epoch = self.global_step // self.N_global_steps_per_epoch
        return True

    def load_model_(self, cfg):
        logger.info(f"======== Loading model from {cfg.saver.load_model} ========")
        safetensors.torch.load_model(
            self.accelerator.unwrap_model(self.model),
            cfg.saver.load_model,
            strict=False,
        )
        logger.info(f"======== Model loaded ========")
        return True

    @control(synchronized=True)
    def load_ckpt_or_auto_resume_(self, cfg):
        # auto resume has higher priority, load model from path if auto resume is not available
        # cfg.saver.auto_resume and cfg.saver.load_model
        if cfg.saver.auto_resume:
            successful_resume = self.auto_resume_(cfg)
            if successful_resume:
                return
        if cfg.saver.load_model:
            successful_load = self.load_model_(cfg)
            if successful_load:
                return
        logger.debug(f"======== No checkpoint or model is loaded ========")

    @control('on_main_process', synchronized=True)
    def save_checkpoint(self):
        ckpt_dir = os.path.join(
            self.cfg.saver.checkpoint_root,
            self.cfg.experiment.parent, self.cfg.experiment.child,
            f"{self.global_step:06d}",
        )
        self.accelerator.save_state(output_dir=ckpt_dir, safe_serialization=True)
        logger.info(f"======== Saved checkpoint at global step {self.global_step} ========")
        # manage stratified checkpoints
        ckpt_dirs = os.listdir(os.path.dirname(ckpt_dir))
        ckpt_dirs.sort()
        max_ckpt = int(ckpt_dirs[-1])
        ckpt_base = int(self.cfg.saver.checkpoint_keep_level)
        ckpt_period = self.cfg.saver.checkpoint_global_steps
        logger.debug(f"Checkpoint base: {ckpt_base}")
        logger.debug(f"Checkpoint period: {ckpt_period}")

        # Ensure the math.log argument is positive.
        if max_ckpt // ckpt_period <= 0:
            logger.info(f"Current max checkpoint {max_ckpt} is less than the first period {ckpt_period}. Skipping cleanup.")
            return

        cur_order = ckpt_base ** math.floor(math.log(max_ckpt // ckpt_period, ckpt_base))
        cur_idx = 0
        while cur_order > 0:
            cur_digit = max_ckpt // ckpt_period // cur_order % ckpt_base
            while cur_idx < len(ckpt_dirs) and int(ckpt_dirs[cur_idx]) // ckpt_period // cur_order % ckpt_base < cur_digit:
                if int(ckpt_dirs[cur_idx]) // ckpt_period % cur_order != 0:
                    shutil.rmtree(os.path.join(os.path.dirname(ckpt_dir), ckpt_dirs[cur_idx]))
                    logger.info(f"Removed checkpoint {ckpt_dirs[cur_idx]}")
                cur_idx += 1
            cur_order //= ckpt_base

    @property
    def global_step_in_epoch(self):
        return self.global_step % self.N_global_steps_per_epoch

    @abstractmethod
    def _build_model(self):
        pass

    @abstractmethod
    def _build_optimizer(self):
        pass

    @abstractmethod
    def _build_scheduler(self):
        pass

    @abstractmethod
    def _build_dataloader(self):
        pass

    @abstractmethod
    def _build_loss_fn(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @staticmethod
    def _get_str_progress(epoch: int = None, step: int = None):
        if epoch is not None:
            log_type = 'epoch'
            log_progress = epoch
        elif step is not None:
            log_type = 'step'
            log_progress = step
        else:
            raise ValueError('Either epoch or step must be provided')
        return log_type, log_progress

    @control('on_main_process')
    def log_scalar_kwargs(self, epoch: int = None, step: int = None, split: str = None, **scalar_kwargs):
        log_type, log_progress = self._get_str_progress(epoch, step)
        split = f'/{split}' if split else ''
        for key, value in scalar_kwargs.items():
            self.accelerator.log({f'{key}{split}/{log_type}': value}, log_progress)

    @control('on_main_process')
    def log_images(self, values: dict, step: Optional[int] = None, log_kwargs: Optional[dict] = {}):
        for tracker in self.accelerator.trackers:
            if hasattr(tracker, 'log_images'):
                tracker.log_images(values, step=step, **log_kwargs.get(tracker.name, {}))

    @control('on_main_process')
    def log_optimizer(self, epoch: int = None, step: int = None, attrs: list[str] = [], group_ids: list[int] = []):
        log_type, log_progress = self._get_str_progress(epoch, step)
        assert self.optimizer is not None, 'Optimizer is not initialized'
        if not attrs:
            logger.warning('No optimizer attributes are provided, nothing will be logged')
        if not group_ids:
            logger.warning('No optimizer group ids are provided, nothing will be logged')
        for attr in attrs:
            assert attr in ['lr', 'momentum', 'weight_decay'], f'Invalid optimizer attribute {attr}'
            for group_id in group_ids:
                self.accelerator.log({f'opt/{attr}/{group_id}': self.optimizer.param_groups[group_id][attr]}, log_progress)

    @control('on_main_process')
    def log_inital_info(self):
        assert self.model is not None, 'Model is not initialized'
        assert self.optimizer is not None, 'Optimizer is not initialized'
        assert self.scheduler is not None, 'Scheduler is not initialized'
        self.accelerator.log({'Config': "```\n" + OmegaConf.to_yaml(self.cfg) + "\n```"})
        self.accelerator.log({'Model': "```\n" + str(self.model) + "\n```"})
        self.accelerator.log({'Optimizer': "```\n" + str(self.optimizer) + "\n```"})
        self.accelerator.log({'Scheduler': "```\n" + str(self.scheduler) + "\n```"})
    
    @control('on_main_process')
    def log_image_monitor(
            self, epoch: int = None, step: int = None,
            split: Literal['val', 'train'] = 'val',
            renders: torch.Tensor = None, gts: torch.Tensor = None,
            renders_refine: Optional[torch.Tensor] = None,
        ):
        """
        Log images to local dir
        Args:
            renders:        (B, V, C, H, W)
            gts:            (B, V, C, H, W)
            renders_refine: (B, V, C, H, W)
        """
        log_type, log_progress = self._get_str_progress(epoch, step)
        img_name = f'{split}_{log_type}_{log_progress}.png'
        img_path = os.path.join(self.img_log_path, img_name)

        M = renders.shape[1]
        print(renders.shape)
        if renders_refine is not None:
            merged = torch.stack([renders, renders_refine, gts], dim=1).view(-1, *renders.shape[2:])  # (N, C, H, W)
        else:
            merged = torch.stack([renders, gts], dim=1).view(-1, *renders.shape[2:])  # (N, C, H, W)
        save_image(merged, img_path, nrow=M)

    def _format_config(self):
        """Format the config string"""
        separator = "=" * 80
        title = "TRAINING CONFIGURATION"
        config_yaml = OmegaConf.to_yaml(self.cfg, resolve=True)
        return f"\n\n{separator}\n{title:^80}\n{separator}\n{config_yaml}{separator}\n"
    
    @control('on_main_process')
    def _log_config(self):
        """Log && Print"""
        config_str = self._format_config()
        logger.info(config_str)
        print(config_str)

    def run(self):
        self.train()
