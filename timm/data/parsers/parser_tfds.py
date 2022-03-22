""" Dataset parser interface that wraps TFDS datasets

Wraps many (most?) TFDS image-classification datasets
from https://github.com/tensorflow/datasets
https://www.tensorflow.org/datasets/catalog/overview#image_classification

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import torch
import torch.distributed as dist
from PIL import Image

try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')  # Hands off my GPU! (or pip install tensorflow-cpu)
    import tensorflow_datasets as tfds
    try:
        tfds.even_splits('', 1, drop_remainder=False)  # non-buggy even_splits has drop_remainder arg
        has_buggy_even_splits = False
    except TypeError:
        print("Warning: This version of tfds doesn't have the latest even_splits impl. "
              "Please update or use tfds-nightly for better fine-grained split behaviour.")
        has_buggy_even_splits = True
    # NOTE uncomment below if having file limit issues on dataset build (or alter your OS defaults)
    # import resource
    # low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
    # resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))
except ImportError as e:
    print(e)
    print("Please install tensorflow_datasets package `pip install tensorflow-datasets`.")
    exit(1)
from .parser import Parser


MAX_TP_SIZE = 8  # maximum TF threadpool size, only doing jpeg decodes and queuing activities
SHUFFLE_SIZE = 8192  # examples to shuffle in DS queue
PREFETCH_SIZE = 2048  # examples to prefetch


def even_split_indices(split, n, num_examples):
    partitions = [round(i * num_examples / n) for i in range(n + 1)]
    return [f"{split}[{partitions[i]}:{partitions[i + 1]}]" for i in range(n)]


def get_class_labels(info):
    if 'label' not in info.features:
        return {}
    class_label = info.features['label']
    class_to_idx = {n: class_label.str2int(n) for n in class_label.names}
    return class_to_idx


class ParserTfds(Parser):
    """ Wrap Tensorflow Datasets for use in PyTorch

    There several things to be aware of:
      * To prevent excessive examples being dropped per epoch w/ distributed training or multiplicity of
         dataloader workers, the train iterator wraps to avoid returning partial batches that trigger drop_last
         https://github.com/pytorch/pytorch/issues/33413
      * With PyTorch IterableDatasets, each worker in each replica operates in isolation, the final batch
        from each worker could be a different size. For training this is worked around by option above, for
        validation extra examples are inserted iff distributed mode is enabled so that the batches being reduced
        across replicas are of same size. This will slightly alter the results, distributed validation will not be
        100% correct. This is similar to common handling in DistributedSampler for normal Datasets but a bit worse
        since there are up to N * J extra examples with IterableDatasets.
      * The sharding (splitting of dataset into TFRecord) files imposes limitations on the number of
        replicas and dataloader workers you can use. For really small datasets that only contain a few shards
        you may have to train non-distributed w/ 1-2 dataloader workers. This is likely not a huge concern as the
        benefit of distributed training or fast dataloading should be much less for small datasets.
      * This wrapper is currently configured to return individual, decompressed image examples from the TFDS
        dataset. The augmentation (transforms) and batching is still done in PyTorch. It would be possible
        to specify TF augmentation fn and return augmented batches w/ some modifications to other downstream
        components.

    """

    def __init__(
            self,
            root,
            name,
            split='train',
            is_training=False,
            batch_size=None,
            download=False,
            repeats=0,
            seed=42,
            input_name='image',
            input_image='RGB',
            target_name='label',
            target_image='',
            prefetch_size=None,
            shuffle_size=None,
            max_threadpool_size=None
    ):
        """ Tensorflow-datasets Wrapper

        Args:
            root: root data dir (ie your TFDS_DATA_DIR. not dataset specific sub-dir)
            name: tfds dataset name (eg `imagenet2012`)
            split: tfds dataset split (can use all TFDS split strings eg `train[:10%]`)
            is_training: training mode, shuffle enabled, dataset len rounded by batch_size
            batch_size: batch_size to use to unsure total examples % batch_size == 0 in training across all dis nodes
            download: download and build TFDS dataset if set, otherwise must use tfds CLI
            repeats: iterate through (repeat) the dataset this many times per iteration (once if 0 or 1)
            seed: common seed for shard shuffle across all distributed/worker instances
            input_name: name of Feature to return as data (input)
            input_image: image mode if input is an image (currently PIL mode string)
            target_name: name of Feature to return as target (label)
            target_image: image mode if target is an image (currently PIL mode string)
            prefetch_size: override default tf.data prefetch buffer size
            shuffle_size: override default tf.data shuffle buffer size
            max_threadpool_size: override default threadpool size for tf.data
        """
        super().__init__()
        self.root = root
        self.split = split
        self.is_training = is_training
        if self.is_training:
            assert batch_size is not None, \
                "Must specify batch_size in training mode for reasonable behaviour w/ TFDS wrapper"
        self.batch_size = batch_size
        self.repeats = repeats
        self.common_seed = seed  # a seed that's fixed across all worker / distributed instances

        # performance settings
        self.prefetch_size = prefetch_size or PREFETCH_SIZE
        self.shuffle_size = shuffle_size or SHUFFLE_SIZE
        self.max_threadpool_size = max_threadpool_size or MAX_TP_SIZE

        # TFDS builder and split information
        self.input_name = input_name  # FIXME support tuples / lists of inputs and targets and full range of Feature
        self.input_image = input_image
        self.target_name = target_name
        self.target_image = target_image
        self.builder = tfds.builder(name, data_dir=root)
        # NOTE: the tfds command line app can be used download & prepare datasets if you don't enable download flag
        if download:
            self.builder.download_and_prepare()
        self.class_to_idx = get_class_labels(self.builder.info) if self.target_name == 'label' else {}
        self.split_info = self.builder.info.splits[split]
        self.num_examples = self.split_info.num_examples

        # Distributed world state
        self.dist_rank = 0
        self.dist_num_replicas = 1
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            self.dist_rank = dist.get_rank()
            self.dist_num_replicas = dist.get_world_size()

        # Attributes that are updated in _lazy_init, including the tf.data pipeline itself
        self.global_num_workers = 1
        self.worker_info = None
        self.worker_seed = 0  # seed unique to each work instance
        self.subsplit = None  # set when data is distributed across workers using sub-splits
        self.ds = None  # initialized lazily on each dataloader worker process

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
        num_workers = 1
        global_worker_id = 0
        if worker_info is not None:
            self.worker_info = worker_info
            self.worker_seed = worker_info.seed
            num_workers = worker_info.num_workers
            self.global_num_workers = self.dist_num_replicas * num_workers
            global_worker_id = self.dist_rank * num_workers + worker_info.id

            """ Data sharding
            InputContext will assign subset of underlying TFRecord files to each 'pipeline' if used.
            My understanding is that using split, the underling TFRecord files will shuffle (shuffle_files=True)
            between the splits each iteration, but that understanding could be wrong.

            I am currently using a mix of InputContext shard assignment and fine-grained sub-splits for distributing
            the data across workers. For training InputContext is used to assign shards to nodes unless num_shards
            in dataset < total number of workers. Otherwise sub-split API is used for datasets without enough shards or
            for validation where we can't drop examples and need to avoid minimize uneven splits to avoid padding.
            """
            should_subsplit = self.global_num_workers > 1 and (
                    self.split_info.num_shards < self.global_num_workers or not self.is_training)
            if should_subsplit:
                # split the dataset w/o using sharding for more even examples / worker, can result in less optimal
                # read patterns for distributed training (overlap across shards) so better to use InputContext there
                if has_buggy_even_splits:
                    # my even_split workaround doesn't work on subsplits, upgrade tfds!
                    if not isinstance(self.split_info, tfds.core.splits.SubSplitInfo):
                        subsplits = even_split_indices(self.split, self.global_num_workers, self.num_examples)
                        self.subsplit = subsplits[global_worker_id]
                else:
                    subsplits = tfds.even_splits(self.split, self.global_num_workers)
                    self.subsplit = subsplits[global_worker_id]

        input_context = None
        if self.global_num_workers > 1 and self.subsplit is None:
            # set input context to divide shards among distributed replicas
            input_context = tf.distribute.InputContext(
                num_input_pipelines=self.global_num_workers,
                input_pipeline_id=global_worker_id,
                num_replicas_in_sync=self.dist_num_replicas  # FIXME does this arg have any impact?
            )
        read_config = tfds.ReadConfig(
            shuffle_seed=self.common_seed,
            shuffle_reshuffle_each_iteration=True,
            input_context=input_context)
        ds = self.builder.as_dataset(
            split=self.subsplit or self.split, shuffle_files=self.is_training, read_config=read_config)
        # avoid overloading threading w/ combo of TF ds threads + PyTorch workers
        options = tf.data.Options()
        thread_member = 'threading' if hasattr(options, 'threading') else 'experimental_threading'
        getattr(options, thread_member).private_threadpool_size = max(1, self.max_threadpool_size // num_workers)
        getattr(options, thread_member).max_intra_op_parallelism = 1
        ds = ds.with_options(options)
        if self.is_training or self.repeats > 1:
            # to prevent excessive drop_last batch behaviour w/ IterableDatasets
            # see warnings at https://pytorch.org/docs/stable/data.html#multi-process-data-loading
            ds = ds.repeat()  # allow wrap around and break iteration manually
        if self.is_training:
            ds = ds.shuffle(min(self.num_examples, self.shuffle_size) // self.global_num_workers, seed=self.worker_seed)
        ds = ds.prefetch(min(self.num_examples // self.global_num_workers, self.prefetch_size))
        self.ds = tfds.as_numpy(ds)

    def __iter__(self):
        if self.ds is None:
            self._lazy_init()

        # Compute a rounded up sample count that is used to:
        #   1. make batches even cross workers & replicas in distributed validation.
        #     This adds extra examples and will slightly alter validation results.
        #   2. determine loop ending condition in training w/ repeat enabled so that only full batch_size
        #     batches are produced (underlying tfds iter wraps around)
        target_example_count = math.ceil(max(1, self.repeats) * self.num_examples / self.global_num_workers)
        if self.is_training:
            # round up to nearest batch_size per worker-replica
            target_example_count = math.ceil(target_example_count / self.batch_size) * self.batch_size

        # Iterate until exhausted or sample count hits target when training (ds.repeat enabled)
        example_count = 0
        for example in self.ds:
            input_data = example[self.input_name]
            if self.input_image:
                input_data = Image.fromarray(input_data, mode=self.input_image)
            target_data = example[self.target_name]
            if self.target_image:
                target_data = Image.fromarray(target_data, mode=self.target_image)
            yield input_data, target_data
            example_count += 1
            if self.is_training and example_count >= target_example_count:
                # Need to break out of loop when repeat() is enabled for training w/ oversampling
                # this results in extra examples per epoch but seems more desirable than dropping
                # up to N*J batches per epoch (where N = num distributed processes, and J = num worker processes)
                break

        # Pad across distributed nodes (make counts equal by adding examples)
        if not self.is_training and self.dist_num_replicas > 1 and self.subsplit is not None and \
                0 < example_count < target_example_count:
            # Validation batch padding only done for distributed training where results are reduced across nodes.
            # For single process case, it won't matter if workers return different batch sizes.
            # If using input_context or % based splits, sample count can vary significantly across workers and this
            # approach should not be used (hence disabled if self.subsplit isn't set).
            while example_count < target_example_count:
                yield input_data, target_data  # yield prev sample again
                example_count += 1

    def __len__(self):
        # this is just an estimate and does not factor in extra examples added to pad batches based on
        # complete worker & replica info (not available until init in dataloader).
        return math.ceil(max(1, self.repeats) * self.num_examples / self.dist_num_replicas)

    def _filename(self, index, basename=False, absolute=False):
        assert False, "Not supported"  # no random access to examples

    def filenames(self, basename=False, absolute=False):
        """ Return all filenames in dataset, overrides base"""
        if self.ds is None:
            self._lazy_init()
        names = []
        for sample in self.ds:
            if len(names) > self.num_examples:
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
