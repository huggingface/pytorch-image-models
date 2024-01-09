""" Dataset reader for HF IterableDataset
"""
import math
import os
from itertools import repeat, chain
from typing import Optional

import torch
import torch.distributed as dist
from PIL import Image

try:
    import datasets
    from datasets.distributed import split_dataset_by_node
    from datasets.splits import SplitInfo
except ImportError as e:
    print("Please install Hugging Face datasets package `pip install datasets`.")
    raise e


from .class_map import load_class_map
from .reader import Reader
from .shared_count import SharedCount


SHUFFLE_SIZE = int(os.environ.get('HFIDS_SHUFFLE_SIZE', 4096))


class ReaderHfids(Reader):
    def __init__(
            self,
            name: str,
            root: Optional[str] = None,
            split: str = 'train',
            is_training: bool = False,
            batch_size: int = 1,
            download: bool = False,
            repeats: int = 0,
            seed: int = 42,
            class_map: Optional[dict] = None,
            input_key: str = 'image',
            input_img_mode: str = 'RGB',
            target_key: str = 'label',
            target_img_mode: str = '',
            shuffle_size: Optional[int] = None,
            num_samples: Optional[int] = None,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.is_training = is_training
        self.batch_size = batch_size
        self.download = download
        self.repeats = repeats
        self.common_seed = seed  # a seed that's fixed across all worker / distributed instances
        self.shuffle_size = shuffle_size or SHUFFLE_SIZE

        self.input_key = input_key
        self.input_img_mode = input_img_mode
        self.target_key = target_key
        self.target_img_mode = target_img_mode

        self.builder = datasets.load_dataset_builder(name, cache_dir=root)
        if download:
            self.builder.download_and_prepare()

        split_info: Optional[SplitInfo] = None
        if self.builder.info.splits and split in self.builder.info.splits:
            if isinstance(self.builder.info.splits[split], SplitInfo):
                split_info: Optional[SplitInfo] = self.builder.info.splits[split]

        if num_samples:
            self.num_samples = num_samples
        elif split_info and split_info.num_examples:
            self.num_samples = split_info.num_examples
        else:
            raise ValueError(
                "Dataset length is unknown, please pass `num_samples` explicitely. "
                "The number of steps needs to be known in advance for the learning rate scheduler."
            )

        self.remap_class = False
        if class_map:
            self.class_to_idx = load_class_map(class_map)
            self.remap_class = True
        else:
            self.class_to_idx = {}

        # Distributed world state
        self.dist_rank = 0
        self.dist_num_replicas = 1
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            self.dist_rank = dist.get_rank()
            self.dist_num_replicas = dist.get_world_size()

        # Attributes that are updated in _lazy_init
        self.worker_info = None
        self.worker_id = 0
        self.num_workers = 1
        self.global_worker_id = 0
        self.global_num_workers = 1

        # Initialized lazily on each dataloader worker process
        self.ds: Optional[datasets.IterableDataset] = None
        self.epoch = SharedCount()

    def set_epoch(self, count):
        # to update the shuffling effective_seed = seed + epoch
        self.epoch.value = count

    def set_loader_cfg(
            self,
            num_workers: Optional[int] = None,
    ):
        if self.ds is not None:
            return
        if num_workers is not None:
            self.num_workers = num_workers
            self.global_num_workers = self.dist_num_replicas * self.num_workers

    def _lazy_init(self):
        """ Lazily initialize worker (in worker processes)
        """
        if self.worker_info is None:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                self.worker_info = worker_info
                self.worker_id = worker_info.id
                self.num_workers = worker_info.num_workers
            self.global_num_workers = self.dist_num_replicas * self.num_workers
            self.global_worker_id = self.dist_rank * self.num_workers + self.worker_id

        if self.download:
            dataset = self.builder.as_dataset(split=self.split)
            # to distribute evenly to workers
            ds = dataset.to_iterable_dataset(num_shards=self.global_num_workers)
        else:
            # in this case the number of shard is determined by the number of remote files
            ds = self.builder.as_streaming_dataset(split=self.split)

        if self.is_training:
            # will shuffle the list of shards and use a shuffle buffer
            ds = ds.shuffle(seed=self.common_seed, buffer_size=self.shuffle_size)

        # Distributed:
        # The dataset has a number of shards that is a factor of `dist_num_replicas` (i.e. if `ds.n_shards % dist_num_replicas == 0`),
        # so the shards are evenly assigned across the nodes.
        # If it's not the case for dataset streaming, each node keeps 1 example out of `dist_num_replicas`, skipping the other examples.

        # Workers:
        # In a node, datasets.IterableDataset assigns the shards assigned to the node as evenly as possible to workers.
        self.ds = split_dataset_by_node(ds, rank=self.dist_rank, world_size=self.dist_num_replicas)

    def _num_samples_per_worker(self):
        num_worker_samples = \
            max(1, self.repeats) * self.num_samples / max(self.global_num_workers, self.dist_num_replicas)
        if self.is_training or self.dist_num_replicas > 1:
            num_worker_samples = math.ceil(num_worker_samples)
        if self.is_training and self.batch_size is not None:
            num_worker_samples = math.ceil(num_worker_samples / self.batch_size) * self.batch_size
        return int(num_worker_samples)

    def __iter__(self):
        if self.ds is None:
            self._lazy_init()
        self.ds.set_epoch(self.epoch.value)

        target_sample_count = self._num_samples_per_worker()
        sample_count = 0

        if self.is_training:
            ds_iter = chain.from_iterable(repeat(self.ds))
        else:
            ds_iter = iter(self.ds)
        for sample in ds_iter:
            input_data: Image.Image = sample[self.input_key]
            if self.input_img_mode and input_data.mode != self.input_img_mode:
                input_data = input_data.convert(self.input_img_mode)
            target_data = sample[self.target_key]
            if self.target_img_mode:
                assert isinstance(target_data, Image.Image), "target_img_mode is specified but target is not an image"
                if target_data.mode != self.target_img_mode:
                    target_data = target_data.convert(self.target_img_mode)
            elif self.remap_class:
                target_data = self.class_to_idx[target_data]
            yield input_data, target_data
            sample_count += 1
            if self.is_training and sample_count >= target_sample_count:
                break

    def __len__(self):
        num_samples = self._num_samples_per_worker() * self.num_workers
        return num_samples

    def _filename(self, index, basename=False, absolute=False):
        assert False, "Not supported"  # no random access to examples

    def filenames(self, basename=False, absolute=False):
        """ Return all filenames in dataset, overrides base"""
        if self.ds is None:
            self._lazy_init()
        names = []
        for sample in self.ds:
            if 'file_name' in sample:
                name = sample['file_name']
            elif 'filename' in sample:
                name = sample['filename']
            elif 'id' in sample:
                name = sample['id']
            elif 'image_id' in sample:
                name = sample['image_id']
            else:
                assert False, "No supported name field present"
            names.append(name)
        return names