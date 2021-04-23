import math
import torch
from torch.utils.data import Sampler
import torch.distributed as dist


class OrderedDistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


class VariableDistributedSampler(Sampler):
    """Sampler that distributes the dataset to each GPU according to the workload specified by the callery.
       It adjusts the dataset slice and batch size. 
       Note: Sampling now occurs in slices of the dataset; no longer by stepping through it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        gpu_load: GPU workload distribution list
        batch_size: Average batch size for the overall system
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
    """
            
    def __init__(self, dataset, gpu_load, batch_size, shuffle = True, seed = 0):
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
            
        world_size = dist.get_world_size()
        rank       = dist.get_rank()
            
        if (len(gpu_load) != world_size):
            raise ValueError("Number of gpu_load entries not equal to world size")

        if (sum(gpu_load) != world_size):
            raise ValueError("Total gpu_load weights not equal to world size")
                        
        self.dataset = dataset
        self.num_replicas = world_size
        self.rank = rank
        self.epoch = 0

        self.num_samples  = [None for _ in range(world_size)]
        self.index_offset = [None for _ in range(world_size)]
        self.batch_size   = [None for _ in range(world_size)]
        self.num_batches  = [None for _ in range(world_size)]
        
        # calculate the dataset slice size for each GPU
        for i in range(world_size):
            self.num_samples[i] = int(math.ceil(len(self.dataset) / self.num_replicas * gpu_load[i]))
            self.batch_size[i]  = int(math.ceil(batch_size * gpu_load[i]))
            self.num_batches[i] = int(math.ceil(self.num_samples[i] / self.batch_size[i]))
                        
        for i in range(1, world_size):
            if (self.num_batches[i] != self.num_batches[i-1]):
                raise ValueError("Number of batches mismatch: ", self.num_batches)                
            
        # calculcate the dataset offset of each GPU slice
        self.index_offset[0] = 0
        for i in range(1, world_size):
            self.index_offset[i] = self.index_offset[i-1] + self.num_samples[i-1]
        
        self.total_size = sum(self.num_samples)
        
        if (rank == 0):
            print('VariableDistributedSampler: Number of samples: ', self.num_samples)
            print('VariableDistributedSampler: Index offsets    : ', self.index_offset)
            print('VariableDistributedSampler: Batch sizes      : ', self.batch_size)
            print('VariableDistributedSampler: Number of batches: ', self.num_batches)
        
        self.shuffle = shuffle
        self.seed = seed

    def get_batch_size(self):
        return self.batch_size[self.rank]
    
    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        # add extra samples to make it evenly divisible
        padding_size = self.total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        assert len(indices) == self.total_size

        # subsample
        #indices = indices[self.rank:self.total_size:self.num_replicas]
        indices = indices[self.index_offset[self.rank]:self.index_offset[self.rank] + self.num_samples[self.rank]]
        assert len(indices) == self.num_samples[self.rank]

        return iter(indices)

    def __len__(self):
        return self.num_samples[self.rank]

    def set_epoch(self, epoch: int):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
        