""" Dataset parser interface that wraps TFDS datasets

Wraps many (most?) TFDS image-classification datasets
from https://github.com/tensorflow/datasets
https://www.tensorflow.org/datasets/catalog/overview#image_classification

Hacked together by / Copyright 2020 Ross Wightman
"""
import os
import io
import math
import torch
import torch.distributed as dist
from PIL import Image

try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')  # Hands off my GPU! (or pip install tensorflow-cpu)
    import tensorflow_datasets as tfds
except ImportError as e:
    print(e)
    print("Please install tensorflow_datasets package `pip install tensorflow-datasets`.")
    exit(1)
from .parser import Parser


MAX_TP_SIZE = 8  # maximum TF threadpool size, only doing jpeg decodes and queuing activities
SHUFFLE_SIZE = 16834  # samples to shuffle in DS queue
PREFETCH_SIZE = 4096  # samples to prefetch


def even_split_indices(split, n, num_samples):
    partitions = [round(i * num_samples / n) for i in range(n + 1)]
    return [f"{split}[{partitions[i]}:{partitions[i+1]}]" for i in range(n)]


class ParserTfds(Parser):
    """ Wrap Tensorflow Datasets for use in PyTorch

    There several things to be aware of:
      * To prevent excessive samples being dropped per epoch w/ distributed training or multiplicity of
         dataloader workers, the train iterator wraps to avoid returning partial batches that trigger drop_last
         https://github.com/pytorch/pytorch/issues/33413
      * With PyTorch IterableDatasets, each worker in each replica operates in isolation, the final batch
        from each worker could be a different size. For training this is worked around by option above, for
        validation extra samples are inserted iff distributed mode is enabled so that the batches being reduced
        across replicas are of same size. This will slightly alter the results, distributed validation will not be
        100% correct. This is similar to common handling in DistributedSampler for normal Datasets but a bit worse
        since there are up to N * J extra samples with IterableDatasets.
      * The sharding (splitting of dataset into TFRecord) files imposes limitations on the number of
        replicas and dataloader workers you can use. For really small datasets that only contain a few shards
        you may have to train non-distributed w/ 1-2 dataloader workers. This is likely not a huge concern as the
        benefit of distributed training or fast dataloading should be much less for small datasets.
      * This wrapper is currently configured to return individual, decompressed image samples from the TFDS
        dataset. The augmentation (transforms) and batching is still done in PyTorch. It would be possible
        to specify TF augmentation fn and return augmented batches w/ some modifications to other downstream
        components.

    """
    def __init__(self, root, name, split='train', shuffle=False, is_training=False, batch_size=None, repeats=0):
        super().__init__()
        self.root = root
        self.split = split
        self.shuffle = shuffle
        self.is_training = is_training
        if self.is_training:
            assert batch_size is not None,\
                "Must specify batch_size in training mode for reasonable behaviour w/ TFDS wrapper"
        self.batch_size = batch_size
        self.repeats = repeats
        self.subsplit = None

        self.builder = tfds.builder(name, data_dir=root)
        # NOTE: please use tfds command line app to download & prepare datasets, I don't want to call
        # download_and_prepare() by default here as it's caused issues generating unwanted paths.
        self.num_samples = self.builder.info.splits[split].num_examples
        self.ds = None  # initialized lazily on each dataloader worker process

        self.worker_info = None
        self.dist_rank = 0
        self.dist_num_replicas = 1
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            self.dist_rank = dist.get_rank()
            self.dist_num_replicas = dist.get_world_size()

    def _lazy_init(self):
        """ Lazily initialize the dataset.

        This is necessary to init the Tensorflow dataset pipeline in the (dataloader) process that
        will be using the dataset instance. The __init__ method is called on the main process,
        this will be called in a dataloader worker process.

        NOTE: There will be problems if you try to re-use this dataset across different loader/worker
        instances once it has been initialized. Do not call any dataset methods that can call _lazy_init
        before it is passed to dataloader.
        """
        worker_info = torch.utils.data.get_worker_info()

        # setup input context to split dataset across distributed processes
        split = self.split
        num_workers = 1
        if worker_info is not None:
            self.worker_info = worker_info
            num_workers = worker_info.num_workers
            global_num_workers = self.dist_num_replicas * num_workers
            worker_id = worker_info.id

            # FIXME I need to spend more time figuring out the best way to distribute/split data across
            # combo of distributed replicas + dataloader worker processes
            """
            InputContext will assign subset of underlying TFRecord files to each 'pipeline' if used.
            My understanding is that using split, the underling TFRecord files will shuffle (shuffle_files=True)
            between the splits each iteration, but that understanding could be wrong.
            Possible split options include:
              * InputContext for both distributed & worker processes (current)
              * InputContext for distributed and sub-splits for worker processes
              * sub-splits for both
            """
            # split_size = self.num_samples // num_workers
            # start = worker_id * split_size
            # if worker_id == num_workers - 1:
            #     split = split + '[{}:]'.format(start)
            # else:
            #     split = split + '[{}:{}]'.format(start, start + split_size)
            if not self.is_training and '[' not in self.split:
                # If not training, and split doesn't define a subsplit, manually split the dataset
                # for more even samples / worker
                self.subsplit = even_split_indices(self.split, global_num_workers, self.num_samples)[
                    self.dist_rank * num_workers + worker_id]

        if self.subsplit is None:
            input_context = tf.distribute.InputContext(
                num_input_pipelines=self.dist_num_replicas * num_workers,
                input_pipeline_id=self.dist_rank * num_workers + worker_id,
                num_replicas_in_sync=self.dist_num_replicas  # FIXME does this arg have any impact?
            )
        else:
            input_context = None

        read_config = tfds.ReadConfig(
            shuffle_seed=42,
            shuffle_reshuffle_each_iteration=True,
            input_context=input_context)
        ds = self.builder.as_dataset(
            split=self.subsplit or self.split, shuffle_files=self.shuffle, read_config=read_config)
        # avoid overloading threading w/ combo fo TF ds threads + PyTorch workers
        ds.options().experimental_threading.private_threadpool_size = max(1, MAX_TP_SIZE // num_workers)
        ds.options().experimental_threading.max_intra_op_parallelism = 1
        if self.is_training or self.repeats > 1:
            # to prevent excessive drop_last batch behaviour w/ IterableDatasets
            # see warnings at https://pytorch.org/docs/stable/data.html#multi-process-data-loading
            ds = ds.repeat()  # allow wrap around and break iteration manually
        if self.shuffle:
            ds = ds.shuffle(min(self.num_samples // self._num_pipelines, SHUFFLE_SIZE), seed=0)
        ds = ds.prefetch(min(self.num_samples // self._num_pipelines, PREFETCH_SIZE))
        self.ds = tfds.as_numpy(ds)

    def __iter__(self):
        if self.ds is None:
            self._lazy_init()
        # compute a rounded up sample count that is used to:
        #   1. make batches even cross workers & replicas in distributed validation.
        #     This adds extra samples and will slightly alter validation results.
        #   2. determine loop ending condition in training w/ repeat enabled so that only full batch_size
        #     batches are produced (underlying tfds iter wraps around)
        target_sample_count = math.ceil(max(1, self.repeats) * self.num_samples / self._num_pipelines)
        if self.is_training:
            # round up to nearest batch_size per worker-replica
            target_sample_count = math.ceil(target_sample_count / self.batch_size) * self.batch_size
        sample_count = 0
        for sample in self.ds:
            img = Image.fromarray(sample['image'], mode='RGB')
            yield img, sample['label']
            sample_count += 1
            if self.is_training and sample_count >= target_sample_count:
                # Need to break out of loop when repeat() is enabled for training w/ oversampling
                # this results in extra samples per epoch but seems more desirable than dropping
                # up to N*J batches per epoch (where N = num distributed processes, and J = num worker processes)
                break
        if not self.is_training and self.dist_num_replicas and 0 < sample_count < target_sample_count:
            # Validation batch padding only done for distributed training where results are reduced across nodes.
            # For single process case, it won't matter if workers return different batch sizes.
            # FIXME if using input_context or % based subsplits, sample count can vary by more than +/- 1 and this
            # approach is not optimal
            yield img, sample['label']  # yield prev sample again
            sample_count += 1

    @property
    def _num_workers(self):
        return 1 if self.worker_info is None else self.worker_info.num_workers

    @property
    def _num_pipelines(self):
        return self._num_workers * self.dist_num_replicas

    def __len__(self):
        # this is just an estimate and does not factor in extra samples added to pad batches based on
        # complete worker & replica info (not available until init in dataloader).
        return math.ceil(max(1, self.repeats) * self.num_samples / self.dist_num_replicas)

    def _filename(self, index, basename=False, absolute=False):
        assert False, "Not supported" # no random access to samples

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
