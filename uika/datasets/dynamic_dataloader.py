# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This file contains code derived from VGGT and is subject to the VGGT License.
# See LICENSES/VGGT-LICENSE.txt in this repository for details.

# -----------------------------
# copied from VGGT's Code: https://github.com/facebookresearch/vggt/blob/main/training/data/dynamic_dataloader.py
# -----------------------------

from typing import Callable, Optional

import random
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from abc import ABC, abstractmethod
from typing import List

from uika.datasets.worker_fn import get_worker_init_fn


__all__ = ['DynamicDataManager']


class DynamicDataManager(ABC):
    def __init__(
        self,
        dataset: Dataset,
        max_img_per_gpu: int,
        max_batch: int,
        image_num: List[int],
        num_workers: int,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = True,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
        persistent_workers: bool = False,
        seed: int = 42,
    ) -> None:
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        self.persistent_workers = persistent_workers
        self.seed = seed
        self.max_img_per_gpu = max_img_per_gpu
        self.max_batch = max_batch
        self.dataset = dataset

        # Extract aspect ratio and image number ranges from the configuration
        # self.aspect_ratio_range = common_config.augs.aspects  # e.g., [0.5, 1.0]

        self.image_num_range = image_num    # e.g., [2, 24]
        # self.epoch_len = common_config.epoch_len  # e.g., 100000

        # Validate the aspect ratio and image number ranges
        # if len(self.aspect_ratio_range) != 2 or self.aspect_ratio_range[0] > self.aspect_ratio_range[1]:
        #     raise ValueError(f"aspect_ratio_range must be [min, max] with min <= max, got {self.aspect_ratio_range}")

        if len(self.image_num_range) != 2 or self.image_num_range[0] < 1 or self.image_num_range[0] > self.image_num_range[1]:
            raise ValueError(f"image_num_range must be [min, max] with 1 <= min <= max, got {self.image_num_range}")

        # Create samplers
        self._create_samplers()

        return
    
    def _create_samplers(self):
        self.sampler = DynamicDistributedSampler(self.dataset, seed=self.seed, shuffle=self.shuffle)

        self.batch_sampler = DynamicBatchSampler(
            self.sampler,
            self.image_num_range,
            # self.epoch_len,
            seed=self.seed,
            max_img_per_gpu=self.max_img_per_gpu,
            max_batch=self.max_batch,
        )
    
    def get_loader(self, epoch: int = 0):
        # print("Building dynamic dataloader with epoch:", epoch)

        # Set the epoch for the sampler
        # self.sampler.set_epoch(epoch)
        self.batch_sampler.set_epoch(epoch)

        if hasattr(self.dataset, "epoch"):
            self.dataset.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

        # Create and return the dataloader
        return DataLoader(
            self.dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_sampler=self.batch_sampler,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            worker_init_fn=get_worker_init_fn(
                seed=self.seed,
                num_workers=self.num_workers,
                epoch=epoch,
                worker_init_fn=self.worker_init_fn,
            ),
            timeout=600,
        )


class DynamicBatchSampler(Sampler):
    """
    A custom batch sampler that dynamically adjusts batch size and image number for each sample.
    Batches within a sample share the same image number.
    """
    def __init__(self,
                 sampler,
                 image_num_range: List[int],
                 epoch: int = 0,
                 dummy_epoch_len: Optional[int] = None,
                 seed: int = 42,
                 max_img_per_gpu: int = 48,
                 max_batch: int = 8,
        ):
        """
        Initializes the dynamic batch sampler.

        Args:
            sampler: Instance of DynamicDistributedSampler.
            image_num_range: List containing [min_images, max_images] per sample.
            epoch: Current epoch number.
            dummy_epoch_len: A large dummy length of the epoch.
            seed: Random seed for reproducibility.
            max_img_per_gpu: Maximum number of images to fit in GPU memory.
        """
        self.sampler = sampler
        self.image_num_range = image_num_range
        self.seed = seed
        self.epoch = epoch
        # self.rng = random.Random()
        # self.rng = np.random.RandomState(self.seed)
        
        # Set the epoch for the sampler
        self.set_epoch(self.epoch)

        # Uniformly sample from the range of possible image numbers
        # For any image number, the weight is 1.0 (uniform sampling). You can set any different weights here.
        self.image_num_weights = {num_images: 1.0 for num_images in range(image_num_range[0], image_num_range[1]+1)}

        # Possible image numbers, e.g., [2, 3, 4, ..., 24]
        self.possible_nums = np.array([n for n in self.image_num_weights.keys()
                                       if self.image_num_range[0] <= n <= self.image_num_range[1]])

        # Normalize weights for sampling
        weights = [self.image_num_weights[n] for n in self.possible_nums]
        self.normalized_weights = np.array(weights) / sum(weights)

        # Maximum image number per GPU
        self.max_img_per_gpu = max_img_per_gpu
        self.max_batch = max_batch

        # Calculate mean Batch Number
        # batch_sizes = [max(1, int(self.max_img_per_gpu // num_imgs)) for num_imgs in self.possible_nums]
        batch_sizes = [min(max(1, int(self.max_img_per_gpu // num_imgs)), self.max_batch) for num_imgs in self.possible_nums]
        self.batch_size = int(sum([batch_sizes[i] * self.normalized_weights[i] for i in range(len(self.possible_nums))]))

        # Precalculate batch sizes for each possible image number
        self.batch_size_map = {num_imgs: min(max(1, int(self.max_img_per_gpu // num_imgs)), self.max_batch) for num_imgs in self.possible_nums}
        
        # epoch length
        self.epoch_len = len(self.sampler) // self.batch_size if dummy_epoch_len is None else dummy_epoch_len

    def set_epoch(self, epoch: int):
        """
        Sets the epoch for this sampler, affecting the random sequence.

        Args:
            epoch: The epoch number.
        """
        self.epoch = epoch
        self.sampler.set_epoch(epoch)
        self.rng = np.random.RandomState(self.seed + epoch * 100)

    def __iter__(self):
        """
        Yields batches of samples with synchronized dynamic parameters.

        Returns:
            Iterator yielding batches of indices with associated parameters.
        """
        sampler_iterator = iter(self.sampler)

        while True:
            try:
                # Sample random image number
                # random_image_num = int(np.random.choice(self.possible_nums, p=self.normalized_weights))
                random_image_num = int(self.rng.choice(self.possible_nums, p=self.normalized_weights))
                # random_aspect_ratio = round(self.rng.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1]), 2)

                # Update sampler parameters
                self.sampler.update_parameters(image_num=random_image_num)

                # Calculate batch size based on max images per GPU and current image number
                # batch_size = self.max_img_per_gpu / random_image_num
                # batch_size = np.floor(batch_size).astype(int)
                # batch_size = max(1, batch_size)  # Ensure batch size is at least 1
                batch_size = self.batch_size_map[random_image_num]

                # Collect samples for the current batch
                current_batch = []
                for _ in range(batch_size):
                    try:
                        item = next(sampler_iterator)  # item is (idx, aspect_ratio, image_num)
                        current_batch.append(item)
                    except StopIteration:
                        break  # No more samples

                if not current_batch:
                    break  # No more data to yield

                yield current_batch

            except StopIteration:
                break  # End of sampler's iterator

    def __len__(self) -> int:
        # Return a mean batch num or a large dummy length
        return self.epoch_len


class DynamicDistributedSampler(DistributedSampler):
    """
    Extends PyTorch's DistributedSampler to include dynamic image_num parameters, 
    which can be passed into the dataset's __getitem__ method.
    """
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )
        self.image_num = None

    def __iter__(self):
        """
        Yields a sequence of (index, image_num).
        Relies on the parent class's logic for shuffling/distributing
        the indices across replicas, then attaches extra parameters.
        """
        indices_iter = super().__iter__()

        for idx in indices_iter:
            yield (idx, self.image_num)

    def update_parameters(self, image_num: List[int]):
        """
        Updates dynamic parameters for each new epoch or iteration.

        Args:
            image_num: The number of images to set.
        """
        self.image_num = image_num


if __name__ == '__main__':
    from uika.datasets.mv_video_head import MV_VideoHeadDataset

    # root_dir = "./train_data/nersemble_v2/export"
    # meta_path = "./train_data/nersemble_v2/label/local_total_ids.json"
    root_dir = "./train_data/synth_mv/export"
    meta_path = "./train_data/synth_mv/label/local_total_ids.json"
    
    dataset = MV_VideoHeadDataset(
        root_dirs=root_dir, meta_path=meta_path, sample_side_views=15,
        render_image_res_low=512, render_image_res_high=512,
        render_region_size=(512, 512), source_image_res=512,
        enlarge_ratio=[0.8, 1.2],
        debug=False, is_val=False
    )

    max_input = 16

    sampler = Sampler(dataset)
    batch_sampler = DynamicBatchSampler(
        sampler,
        image_num_range=[1, max_input],
        seed=42,
        max_img_per_gpu=max_input,
        max_batch=8,
    )

    pass
