""" Dataset reader for HF IterableDataset
"""
import os
from typing import Optional

import torch
import torch.distributed as dist
from PIL import Image

try:
    import datasets
    from datasets.distributed import split_dataset_by_node
except ImportError:
    print("Please install Hugging Face datasets package `pip install datasets`.")
    exit(1)


from .class_map import load_class_map
from .reader import Reader


SHUFFLE_SIZE = int(os.environ.get('HFIDS_SHUFFLE_SIZE', 8192))


class ReaderHfids(Reader):
    def __init__(
            self,
            root: str,
            name: str,
            split: str,
            is_training: bool = False,
            batch_size: Optional[int] = None,
            download: bool =False,
            repeats: int = 0,
            seed: int = 42,
            class_map: Optional[dict] = None,
            input_name: str = 'image',
            input_img_mode: str = 'RGB',
            target_name: str = 'label',
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

        self.input_name = input_name
        self.input_img_mode = input_img_mode
        self.target_key = target_name
        self.target_img_mode = target_img_mode

        self.builder = datasets.load_dataset_builder(name, cache_dir=root)
        if download:
            self.builder.download_and_prepare()
        
        self.num_samples = num_samples
        if self.builder.info.splits and split in self.builder.info.splits:
            self.num_samples = self.builder.info.splits[split].num_examples

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

        # DataPipeline is lazy init, majority of WDS DataPipeline could be init here, BUT, shuffle seed
        # is not handled in manner where it can be deterministic for each worker AND initialized up front
        self.ds: Optional[datasets.IterableDataset] = None

    def set_epoch(self, count):
        self.ds.set_epoch(self.epoch_count.value)

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
            self.num_samples = len(dataset)
            # optimized to distribute evenly to workers
            ds: datasets.IterableDataset = dataset.to_iterable_dataset(num_shards=self.global_num_workers)
        else:
            # in this case the number of shard is fixed, and it may be less optimized
            ds: datasets.IterableDataset = self.builder.as_streaming_dataset(split=self.split)

        if self.is_training:
            # will shuffle the list of shards and use shuffle buffer
            ds = ds.shuffle(seed=self.common_seed, buffer_size=self.shuffle_size)

        # Distributed:

        # The dataset has a number of shards that is a factor of `dist_num_replicas` (i.e. if `ds.n_shards % dist_num_replicas == 0`),
        # so the shards are evenly assigned across the nodes, which is the most optimized.
        # If it's not the case for dataset streaming, each node keeps 1 example out of `dist_num_replicas`, skipping the other examples.

        # Workers:

        # In a node, datasets.IterableDataset assigns the shards assigned to the node as evenly as possible to workers.

        self.ds = split_dataset_by_node(ds, rank=self.dist_rank, world_size=self.dist_num_replicas)

    def __iter__(self):
        if self.ds is None:
            self._lazy_init()

        # TODO(lhoestq): take batch_size into account to use to unsure total samples % batch_size == 0 in training across all dis nodes
        for sample in self.ds:
            input_data: Image.Image = sample[self.input_name]
            if self.input_img_mode and input_data.mode != self.input_img_mode:
                input_data = input_data.convert(self.input_img_mode)
            target_data = sample[self.target_name]
            if self.target_img_mode:
                assert isinstance(target_data, Image.Image), "target_img_mode is specified but target is not an image"
                if target_data.mode != self.target_img_mode:
                    target_data = target_data.convert(self.target_img_mode)
            elif self.remap_class:
                target_data = self.class_to_idx[target_data]
            yield input_data, target_data

    def __len__(self):
        if self.num_samples is None:
            raise RuntimeError("Dataset length is unknown, please pass `num_samples` explicitely")
        return self.num_samples

    def _filename(self, index, basename=False, absolute=False):
        assert False, "Not supported"  # no random access to examples

    def filenames(self, basename=False, absolute=False):
        """ Return all filenames in dataset, overrides base"""
        if self.ds is None:
            self._lazy_init()
        names = []
        for sample in self.ds:
            if len(names) > self.num_samples:
                break  # safety for ds.repeat() case
            if 'file_name' in sample:
                name = sample['file_name']
            elif 'filename' in sample:
                name = sample['filename']
            elif 'id' in sample:
                name = sample['id']
            else:
                assert False, "No supported name field present"
            names.append(name)
        return names
