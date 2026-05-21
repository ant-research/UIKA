import os
import math
import torch

from tqdm.auto import tqdm
from typing import Optional
from abc import abstractmethod
from torch.utils.data import DataLoader
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from .base_trainer import Trainer
from uika.datasets import MixerDataset, DynamicDataManager
from uika.utils.profiler import DummyProfiler


logger = get_logger(__name__)


class DynamicViewTrainer(Trainer):
    loss_name_dict: list[str] = []

    def __init__(self):
        super().__init__()
        self.prepare_dataloaders_with_accelerator = False
        self.model = self._build_model(self.cfg)
        self.optimizer = self._build_optimizer(self.model, self.cfg)
        self.real_train_loader, self.real_val_loader, self.train_loader, self.val_loader = self._build_dataloader(self.cfg)
        self.scheduler = self._build_scheduler(self.optimizer, self.cfg)
        self._build_loss_fn(self.cfg)

        if not self.loss_name_dict:
            raise ValueError(f"{self.__class__.__name__} must define loss_name_dict")

    def _build_optimizer(self, model: torch.nn.Module, cfg):
        no_decay_params = []

        # add all bias and LayerNorm params to no_decay_params
        for _, module in model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                no_decay_params.extend([p for p in module.parameters()])
            elif hasattr(module, 'bias') and module.bias is not None:
                no_decay_params.append(module.bias)

        # add remaining parameters to decay_params
        _no_decay_ids = set(map(id, no_decay_params))
        decay_params = [p for p in model.parameters() if id(p) not in _no_decay_ids]

        # filter out parameters with no grad
        decay_params = list(filter(lambda p: p.requires_grad, decay_params))
        no_decay_params = list(filter(lambda p: p.requires_grad, no_decay_params))

        # monitor this to make sure we don't miss any parameters
        logger.info("======== Weight Decay Parameters ========")
        logger.info(f"Total: {len(decay_params)}")
        logger.info("======== No Weight Decay Parameters ========")
        logger.info(f"Total: {len(no_decay_params)}")

        opt_groups = [
            {'params': decay_params, 'weight_decay': cfg.train.optim.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(
            opt_groups,
            lr=cfg.train.optim.lr,
            betas=(cfg.train.optim.beta1, cfg.train.optim.beta2),
        )

        return optimizer

    def _build_scheduler(self, optimizer, cfg):
        steps_per_epoch = math.ceil(len(self.real_train_loader) / self.cfg.train.accum_steps)
        total_global_batches = cfg.train.epochs * steps_per_epoch
        effective_warmup_iters = cfg.train.scheduler.warmup_real_iters
        logger.debug(f"======== Scheduler effective max iters: {total_global_batches} ========")
        logger.debug(f"======== Scheduler effective warmup iters: {effective_warmup_iters} ========")
        if cfg.train.scheduler.type == 'cosine':
            from uika.utils.scheduler import CosineWarmupScheduler
            scheduler = CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_iters=effective_warmup_iters,
                max_iters=total_global_batches,
            )
        else:
            raise NotImplementedError(f"Scheduler type {cfg.train.scheduler.type} not implemented")
        return scheduler

    def _build_dataloader(self, cfg):
        patch_size = self._patch_size_for_encoder(cfg.model.encoder_type)

        train_dataset = MixerDataset(
            split="train",
            subsets=cfg.data.subsets,
            sample_side_views=cfg.data.dataset.sample_side_views,
            source_image_res=cfg.data.dataset.source_image_res,
            render_image_res_low=cfg.data.dataset.render_image.low,
            render_image_res_high=cfg.data.dataset.render_image.high,
            render_region_size=cfg.data.dataset.render_image.region,
            repeat_num=cfg.data.dataset.repeat_num,
            aspect_standard=cfg.data.dataset.aspect_standard,
            enlarge_ratio=cfg.data.dataset.enlarge_ratio,
            multiply=patch_size,
        )
        val_dataset = MixerDataset(
            split="val",
            subsets=cfg.data.subsets,
            sample_side_views=cfg.data.dataset.sample_side_views,
            source_image_res=cfg.data.dataset.source_image_res,
            render_image_res_low=cfg.data.dataset.render_image.low,
            render_image_res_high=cfg.data.dataset.render_image.high,
            render_region_size=cfg.data.dataset.render_image.region,
            repeat_num=cfg.data.dataset.repeat_num,
            aspect_standard=cfg.data.dataset.aspect_standard,
            enlarge_ratio=cfg.data.dataset.enlarge_ratio,
            multiply=patch_size,
        )

        logger.info("======== Len of Datasets ========")
        logger.info(f"Train: {len(train_dataset)}")
        logger.info(f"Val: {len(val_dataset)}")

        train_data_manager = DynamicDataManager(
            dataset=train_dataset,
            max_img_per_gpu=cfg.data.loader.max_img_per_gpu,
            max_batch=cfg.data.loader.max_batch,
            image_num=cfg.data.loader.image_num,
            num_workers=cfg.data.loader.num_workers,
            pin_memory=cfg.data.loader.pin_memory,
            shuffle=cfg.data.loader.shuffle,
            drop_last=cfg.data.loader.drop_last,
            persistent_workers=cfg.data.loader.persistent_workers,
            seed=cfg.experiment.seed,
        )
        val_data_manager = DynamicDataManager(
            dataset=val_dataset,
            max_img_per_gpu=cfg.data.loader.max_img_per_gpu,
            max_batch=cfg.data.loader.max_batch,
            image_num=cfg.data.loader.image_num,
            num_workers=cfg.data.loader.num_workers,
            pin_memory=cfg.data.loader.pin_memory,
            shuffle=cfg.data.loader.shuffle,
            drop_last=cfg.data.loader.drop_last,
            persistent_workers=cfg.data.loader.persistent_workers,
            seed=cfg.experiment.seed,
        )

        real_train_loader = train_data_manager.get_loader(self.accelerator.process_index)
        real_val_loader = val_data_manager.get_loader(self.accelerator.process_index)

        # Legacy Trainer aliases. Dynamic loaders remain the canonical loaders.
        train_loader = real_train_loader
        val_loader = real_val_loader

        return real_train_loader, real_val_loader, train_loader, val_loader

    @staticmethod
    def _patch_size_for_encoder(encoder_type: str) -> int:
        if encoder_type == 'dinov2_fusion':
            return 14
        if encoder_type == 'dinov3_fusion':
            return 16
        raise ValueError(f"encoder_type: {encoder_type} not supported")

    def register_hooks(self):
        pass

    def _update_loader_by_epoch(self, epoch: int):
        set_seed(self.cfg.experiment.seed + epoch * 100, device_specific=True)

        self.real_train_loader.batch_sampler.set_epoch(epoch)
        self.real_val_loader.batch_sampler.set_epoch(epoch)

        if hasattr(self.real_train_loader.dataset, "epoch"):
            self.real_train_loader.dataset.epoch = epoch
        if hasattr(self.real_train_loader.dataset, "set_epoch"):
            self.real_train_loader.dataset.set_epoch(epoch)

        if hasattr(self.real_val_loader.dataset, "epoch"):
            self.real_val_loader.dataset.epoch = epoch
        if hasattr(self.real_val_loader.dataset, "set_epoch"):
            self.real_val_loader.dataset.set_epoch(epoch)

        logger.info("======== Len of Real Datasets ========")
        logger.info(f"Len of real train loader: {len(self.real_train_loader)}")
        logger.info(f"Len of real val loader: {len(self.real_val_loader)}")

    def train_epoch(self, pbar: tqdm, loader: DataLoader, profiler: torch.profiler.profile):
        self.model.train()

        local_step_losses = []
        global_step_losses = []

        logger.debug(f"======== Starting epoch {self.current_epoch} ========")
        for data in loader:

            logger.debug(f"======== Starting global step {self.global_step} ========")
            with self.accelerator.accumulate(self.model):
                data = self._move_to_device(data)

                outs, loss_dict = self.forward_loss_local_step(data)
                loss = loss_dict['total_loss']

                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients and self.cfg.train.optim.clip_grad_norm > 0.:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.train.optim.clip_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                local_step_losses.append(self._stack_losses(loss_dict, detach=True))

            if self.accelerator.sync_gradients:
                profiler.step()
                self.scheduler.step()
                logger.debug(f"======== Scheduler step ========")
                self.global_step += 1
                global_step_loss = self.accelerator.gather(torch.stack(local_step_losses)).mean(dim=0).cpu()
                loss_kwargs = self._loss_kwargs_from_tensor(global_step_loss)

                self.log_scalar_kwargs(
                    step=self.global_step, split='train',
                    **loss_kwargs
                )
                self.log_optimizer(step=self.global_step, attrs=['lr'], group_ids=[0, 1])
                local_step_losses = []
                global_step_losses.append(global_step_loss)

                pbar.update(1)
                description = {
                    **loss_kwargs,
                    'lr': self.optimizer.param_groups[0]['lr'],
                }
                description = '[TRAIN STEP]' + \
                    ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in description.items() if not math.isnan(v))
                pbar.set_description(description)

                if self.global_step % self.cfg.saver.checkpoint_global_steps == 0:
                    self.save_checkpoint()
                if self.global_step % self.cfg.val.global_step_period == 0:
                    self.evaluate()
                    self.model.train()
                if self.global_step % self.cfg.logger.image_monitor.train_global_steps == 0:
                    self._log_image_monitor_from_outputs(
                        step=self.global_step, split='train',
                        data=data, outs=outs,
                    )

                if self.global_step >= self.N_max_global_steps:
                    self.accelerator.set_trigger()
                    break

        self.current_epoch += 1
        epoch_losses = torch.stack(global_step_losses).mean(dim=0)
        epoch_loss_dict = self._loss_kwargs_from_tensor(epoch_losses)

        self.log_scalar_kwargs(
            epoch=self.current_epoch, split='train',
            **epoch_loss_dict,
        )
        logger.info(
            f'[TRAIN EPOCH] {self.current_epoch}/{self.cfg.train.epochs}: ' + \
                ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in epoch_loss_dict.items() if not math.isnan(v))
        )

    def train(self):
        starting_local_step_in_epoch = self.global_step_in_epoch * self.cfg.train.accum_steps
        if starting_local_step_in_epoch > 0:
            logger.warning(
                "Resume is inside an epoch, but dynamic loader batch skipping is not implemented yet. "
                f"Restarting epoch {self.current_epoch} from its first local batch."
            )
        else:
            logger.info("======== Starting from the first local batch of the epoch ========")

        self.evaluate()

        with tqdm(
            range(0, self.N_max_global_steps),
            initial=self.global_step,
            disable=(not self.accelerator.is_main_process),
        ) as pbar:

            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=10, warmup=10, active=100,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(
                    self.cfg.logger.tracker_root,
                    self.cfg.experiment.parent, self.cfg.experiment.child,
                )),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) if self.cfg.logger.enable_profiler else DummyProfiler()

            with profiler:
                self.optimizer.zero_grad()
                for _ in range(self.current_epoch, self.cfg.train.epochs):
                    self._update_loader_by_epoch(self.current_epoch)
                    self.train_epoch(pbar=pbar, loader=self.real_train_loader, profiler=profiler)
                    if self.accelerator.check_trigger():
                        break

            logger.info(f"======== Training finished at global step {self.global_step} ========")

            self.save_checkpoint()
            self.evaluate()

    @torch.no_grad()
    @torch.compiler.disable
    def evaluate(self, epoch: Optional[int] = None):
        self.model.eval()

        max_val_batches = self.cfg.val.debug_batches or len(self.real_val_loader)
        running_losses = []
        sample_data, sample_outs = None, None

        cur_epoch = epoch if epoch is not None else self.current_epoch
        self._update_loader_by_epoch(cur_epoch)

        for data in tqdm(self.real_val_loader, disable=(not self.accelerator.is_main_process), total=max_val_batches):
            data = self._move_to_device(data)

            if len(running_losses) >= max_val_batches:
                logger.info(f"======== Early stop validation at {len(running_losses)} batches ========")
                break

            outs, loss_dict = self.forward_loss_local_step(data)
            sample_data, sample_outs = data, outs

            running_losses.append(self._stack_losses(loss_dict, detach=False))

        total_losses = self.accelerator.gather(torch.stack(running_losses)).mean(dim=0).cpu()
        total_loss_dict = self._loss_kwargs_from_tensor(total_losses)

        if epoch is not None:
            self.log_scalar_kwargs(
                epoch=epoch, split='val',
                **total_loss_dict,
            )
            logger.info(
                f'[VAL EPOCH] {epoch}/{self.cfg.train.epochs}: ' + \
                    ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in total_loss_dict.items() if not math.isnan(v))
            )
            self._log_image_monitor_from_outputs(
                epoch=epoch, split='val',
                data=sample_data, outs=sample_outs,
            )
        else:
            self.log_scalar_kwargs(
                step=self.global_step, split='val',
                **total_loss_dict,
            )
            logger.info(
                f'[VAL STEP] {self.global_step}/{self.N_max_global_steps}: ' + \
                    ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in total_loss_dict.items() if not math.isnan(v))
            )
            self._log_image_monitor_from_outputs(
                step=self.global_step, split='val',
                data=sample_data, outs=sample_outs,
            )

    def _move_to_device(self, data):
        if torch.is_tensor(data):
            return data.to(self.device)
        if isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._move_to_device(v) for v in data]
        if isinstance(data, tuple):
            return tuple(self._move_to_device(v) for v in data)
        return data

    def _stack_losses(self, loss_dict: dict, detach: bool) -> torch.Tensor:
        return torch.stack([
            self._loss_value_to_tensor(loss_dict.get(loss_name), detach=detach)
            for loss_name in self.loss_name_dict
        ])

    def _loss_value_to_tensor(self, value, detach: bool) -> torch.Tensor:
        if value is None:
            return torch.tensor(float('nan'), device=self.device)
        if torch.is_tensor(value):
            return value.detach() if detach else value
        return torch.tensor(float(value), device=self.device)

    def _loss_kwargs_from_tensor(self, losses: torch.Tensor) -> dict[str, float]:
        return {
            loss_name: value.item()
            for loss_name, value in zip(self.loss_name_dict, losses.unbind())
        }

    def _log_image_monitor_from_outputs(self, *, data, outs, epoch: int = None, step: int = None, split: str):
        image_kwargs = self._image_monitor_kwargs(data, outs)
        image_kwargs = {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in image_kwargs.items()
        }
        self.log_image_monitor(epoch=epoch, step=step, split=split, **image_kwargs)

    @abstractmethod
    def forward_loss_local_step(self, data):
        pass

    @abstractmethod
    def _image_monitor_kwargs(self, data, outs) -> dict:
        pass
