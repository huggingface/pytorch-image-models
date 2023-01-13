""" Dataset reader for webdataset

Hacked together by / Copyright 2022 Ross Wightman
"""
import io
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from functools import partial
from itertools import islice
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import yaml
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, get_worker_info

try:
    import webdataset as wds
    from webdataset.filters import _shuffle
    from webdataset.shardlists import expand_urls
    from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample
except ImportError:
    wds = None
    expand_urls = None

from .reader import Reader
from .shared_count import SharedCount

_logger = logging.getLogger(__name__)

SHUFFLE_SIZE = int(os.environ.get('WDS_SHUFFLE_SIZE', 8192))


def _load_info(root, basename='info'):
    info_json = os.path.join(root, basename + '.json')
    info_yaml = os.path.join(root, basename + '.yaml')
    err_str = ''
    try:
        with wds.gopen.gopen(info_json) as f:
            info_dict = json.load(f)
        return info_dict
    except Exception as e:
        err_str = str(e)
    try:
        with wds.gopen.gopen(info_yaml) as f:
            info_dict = yaml.safe_load(f)
        return info_dict
    except Exception:
        pass
    _logger.warning(
        f'Dataset info file not found at {info_json} or {info_yaml}. Error: {err_str}. '
        'Falling back to provided split and size arg.')
    return {}


@dataclass
class SplitInfo:
    num_samples: int
    filenames: Tuple[str]
    shard_lengths: Tuple[int] = ()
    alt_label: str = ''
    name: str = ''


def _parse_split_info(split: str, info: Dict):
    def _info_convert(dict_info):
        return SplitInfo(
            num_samples=dict_info['num_samples'],
            filenames=tuple(dict_info['filenames']),
            shard_lengths=tuple(dict_info['shard_lengths']),
            alt_label=dict_info.get('alt_label', ''),
            name=dict_info['name'],
        )

    if 'tar' in split or '..' in split:
        # split in WDS string braceexpand format, sample count can be included with a | separator
        # ex: `dataset-split-{0000..9999}.tar|100000` for 9999 shards, covering 100,000 samples
        split = split.split('|')
        num_samples = 0
        split_name = ''
        if len(split) > 1:
            num_samples = int(split[1])
        split = split[0]
        if '::' not in split:
            split_parts = split.split('-', 3)
            split_idx = len(split_parts) - 1
            if split_idx and 'splits' in info and split_parts[split_idx] in info['splits']:
                split_name = split_parts[split_idx]

        split_filenames = expand_urls(split)
        if split_name:
            split_info = info['splits'][split_name]
            if not num_samples:
                _fc = {f: c for f, c in zip(split_info['filenames'], split_info['shard_lengths'])}
                num_samples = sum(_fc[f] for f in split_filenames)
                split_info['filenames'] = tuple(_fc.keys())
                split_info['shard_lengths'] = tuple(_fc.values())
                split_info['num_samples'] = num_samples
            split_info = _info_convert(split_info)
        else:
            split_info = SplitInfo(
                name=split_name,
                num_samples=num_samples,
                filenames=split_filenames,
            )
    else:
        if split not in info['splits']:
            raise RuntimeError(f"split {split} not found in info ({info['splits'].keys()})")
        split = split
        split_info = info['splits'][split]
        split_info = _info_convert(split_info)

    return split_info


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    _logger.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def _decode(
        sample,
        image_key='jpg',
        image_format='RGB',
        target_key='cls',
        alt_label=''
):
    """ Custom sample decode
    * decode and convert PIL Image
    * cls byte string label to int
    * pass through JSON byte string (if it exists) without parse
    """
    # decode class label, skip if alternate label not valid
    if alt_label:
        # alternative labels are encoded in json metadata
        meta = json.loads(sample['json'])
        class_label = int(meta[alt_label])
        if class_label < 0:
            # skipped labels currently encoded as -1, may change to a null/None value
            return None
    else:
        class_label = int(sample[target_key])

    # decode image
    with io.BytesIO(sample[image_key]) as b:
        img = Image.open(b)
        img.load()
    if image_format:
        img = img.convert(image_format)

    # json passed through in undecoded state
    decoded = dict(jpg=img, cls=class_label, json=sample.get('json', None))
    return decoded


def _decode_samples(
        data,
        image_key='jpg',
        image_format='RGB',
        target_key='cls',
        alt_label='',
        handler=log_and_continue):
    """Decode samples with skip."""
    for sample in data:
        try:
            result = _decode(
                sample,
                image_key=image_key,
                image_format=image_format,
                target_key=target_key,
                alt_label=alt_label
            )
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break

        # null results are skipped
        if result is not None:
            if isinstance(sample, dict) and isinstance(result, dict):
                result["__key__"] = sample.get("__key__")
            yield result


def pytorch_worker_seed():
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour the seed already created for pytorch dataloader workers if it exists
        return worker_info.seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


if wds is not None:
    # conditional to avoid mandatory wds import (via inheritance of wds.PipelineStage)
    class detshuffle2(wds.PipelineStage):
        def __init__(
                self,
                bufsize=1000,
                initial=100,
                seed=0,
                epoch=-1,
        ):
            self.bufsize = bufsize
            self.initial = initial
            self.seed = seed
            self.epoch = epoch

        def run(self, src):
            if isinstance(self.epoch, SharedCount):
                epoch = self.epoch.value
            else:
                # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
                # situation as different workers may wrap at different times (or not at all).
                self.epoch += 1
                epoch = self.epoch

            if self.seed < 0:
                seed = pytorch_worker_seed() + epoch
            else:
                seed = self.seed + epoch
            # _logger.info(f'shuffle seed: {self.seed}, {seed}, epoch: {epoch}')  # FIXME temporary
            rng = random.Random(seed)
            return _shuffle(src, self.bufsize, self.initial, rng)

else:
    detshuffle2 = None


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=True,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls = wds.shardlists.expand_urls(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = pytorch_worker_seed if worker_seed is None else worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedCount):
            epoch = self.epoch.value
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch

        if self.deterministic:
            # reset seed w/ epoch if deterministic, worker seed should be deterministic due to arg.seed
            self.rng = random.Random(self.worker_seed() + epoch)

        for _ in range(self.nshards):
            index = self.rng.randint(0, len(self.urls) - 1)
            yield dict(url=self.urls[index])


class ReaderWds(Reader):
    def __init__(
            self,
            root,
            name,
            split,
            is_training=False,
            batch_size=None,
            repeats=0,
            seed=42,
            input_name='jpg',
            input_image='RGB',
            target_name='cls',
            target_image='',
            prefetch_size=None,
            shuffle_size=None,
    ):
        super().__init__()
        if wds is None:
            raise RuntimeError(
                'Please install webdataset 0.2.x package `pip install git+https://github.com/webdataset/webdataset`.')
        self.root = root
        self.is_training = is_training
        self.batch_size = batch_size
        self.repeats = repeats
        self.common_seed = seed  # a seed that's fixed across all worker / distributed instances
        self.shard_shuffle_size = 500
        self.sample_shuffle_size = shuffle_size or SHUFFLE_SIZE

        self.image_key = input_name
        self.image_format = input_image
        self.target_key = target_name
        self.filename_key = 'filename'
        self.key_ext = '.JPEG'  # extension to add to key for original filenames (DS specific, default ImageNet)

        self.info = _load_info(self.root)
        self.split_info = _parse_split_info(split, self.info)
        self.num_samples = self.split_info.num_samples
        if not self.num_samples:
            raise RuntimeError(f'Invalid split definition, no samples found.')

        # Distributed world state
        self.dist_rank = 0
        self.dist_num_replicas = 1
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            self.dist_rank = dist.get_rank()
            self.dist_num_replicas = dist.get_world_size()

        # Attributes that are updated in _lazy_init
        self.worker_info = None
        self.worker_id = 0
        self.worker_seed = seed  # seed unique to each worker instance
        self.num_workers = 1
        self.global_worker_id = 0
        self.global_num_workers = 1
        self.init_count = 0
        self.epoch_count = SharedCount()

        # DataPipeline is lazy init, majority of WDS DataPipeline could be init here, BUT, shuffle seed
        # is not handled in manner where it can be deterministic for each worker AND initialized up front
        self.ds = None

    def set_epoch(self, count):
        self.epoch_count.value = count

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
                self.worker_seed = worker_info.seed
                self.num_workers = worker_info.num_workers
            self.global_num_workers = self.dist_num_replicas * self.num_workers
            self.global_worker_id = self.dist_rank * self.num_workers + self.worker_id

        # init data pipeline
        abs_shard_filenames = [os.path.join(self.root, f) for f in self.split_info.filenames]
        pipeline = [wds.SimpleShardList(abs_shard_filenames)]
        # at this point we have an iterator over all the shards
        if self.is_training:
            pipeline.extend([
                detshuffle2(self.shard_shuffle_size, seed=self.common_seed, epoch=self.epoch_count),
                self._split_by_node_and_worker,
                # at this point, we have an iterator over the shards assigned to each worker
                wds.tarfile_to_samples(handler=log_and_continue),
                wds.shuffle(
                    self.sample_shuffle_size,
                    rng=random.Random(self.worker_seed)),  # this is why we lazy-init whole DataPipeline
            ])
        else:
            pipeline.extend([
                self._split_by_node_and_worker,
                # at this point, we have an iterator over the shards assigned to each worker
                wds.tarfile_to_samples(handler=log_and_continue),
            ])
        pipeline.extend([
            partial(
                _decode_samples,
                image_key=self.image_key,
                image_format=self.image_format,
                alt_label=self.split_info.alt_label
            )
        ])
        self.ds = wds.DataPipeline(*pipeline)

    def _split_by_node_and_worker(self, src):
        if self.global_num_workers > 1:
            for s in islice(src, self.global_worker_id, None, self.global_num_workers):
                yield s
        else:
            for s in src:
                yield s

    def _num_samples_per_worker(self):
        num_worker_samples = self.num_samples / max(self.global_num_workers, self.dist_num_replicas)
        if self.is_training or self.dist_num_replicas > 1:
            num_worker_samples = math.ceil(num_worker_samples)
        if self.is_training and self.batch_size is not None:
            num_worker_samples = math.ceil(num_worker_samples / self.batch_size) * self.batch_size
        return int(num_worker_samples)

    def __iter__(self):
        if self.ds is None:
            self._lazy_init()

        num_worker_samples = self._num_samples_per_worker()
        if self.is_training or self.dist_num_replicas > 1:
            # NOTE: doing distributed validation w/ WDS is messy, hard to meet constraints that
            # same # of batches needed across all replicas w/ seeing each sample once.
            # with_epoch() is simple but could miss a shard's worth of samples in some workers,
            # and duplicate in others. Best to keep num DL workers low and a divisor of #val shards.
            ds = self.ds.with_epoch(num_worker_samples)
        else:
            ds = self.ds

        i = 0
        # _logger.info(f'start {i}, {self.worker_id}')  # FIXME temporary debug
        for sample in ds:
            yield sample[self.image_key], sample[self.target_key]
            i += 1
        # _logger.info(f'end {i}, {self.worker_id}')  # FIXME temporary debug

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
            if self.filename_key in sample:
                name = sample[self.filename_key]
            elif '__key__' in sample:
                name = sample['__key__'] + self.key_ext
            else:
                assert False, "No supported name field present"
            names.append(name)
            if len(names) >= self.num_samples:
                break  # safety for ds.repeat() case
        return names
