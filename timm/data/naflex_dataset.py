""" Dynamic Sequence Length Datasets for Variable Resolution Image Processing

Implements two dataset wrappers:
1. NaFlexMapDatasetWrapper - Map-style dataset that returns batches with variable sequence lengths
TODO: 2. NaFlexIterableDatasetWrapper - Iterable dataset that yields batches with variable sequence lengths

Both support:
- Pre-initialized transforms for efficiency
- Distributed training
- Multiple workers
- Variable batch sizes based on sequence length

Hacked together by / Copyright 2025, Ross Wightman, Hugging Face
"""

import math
import random
import warnings
from functools import partial
from typing import Any, Iterator, List, Tuple, Dict, Optional, Union, Callable

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from PIL import Image

from .naflex_transforms import Patchify
from timm.layers import to_2tuple


def calculate_naflex_batch_size(
        tokens_per_batch: int,
        seq_len: int,
        max_size: Optional[int] = None,
        divisor: int = 1,
        rounding: str = 'floor',
) -> int:
    """Calculate batch size based on sequence length with divisibility constraints.

    Args:
        tokens_per_batch: Target number of tokens per batch.
        seq_len: Sequence length for this batch.
        max_size: Optional maximum batch size.
        divisor: Ensure batch size is divisible by this value.
        rounding: Rounding method ('floor', 'ceil', 'round').

    Returns:
        Calculated batch size.
    """
    # Calculate raw batch size based on sequence length
    raw_batch_size = tokens_per_batch / seq_len

    # Apply divisibility with specified rounding method
    if divisor > 1:
        if rounding == 'floor':
            batch_size = math.floor(raw_batch_size / divisor) * divisor
        elif rounding == 'ceil':
            batch_size = math.ceil(raw_batch_size / divisor) * divisor
        else:  # 'round' is the default
            batch_size = round(raw_batch_size / divisor) * divisor
    else:
        # If no divisor specified, just use integer division
        batch_size = int(raw_batch_size)

    # Ensure batch size is valid
    batch_size = max(1, batch_size)  # At least 1

    if max_size is not None:
        batch_size = min(batch_size, max_size)

    return batch_size


class NaFlexCollator:
    """Custom collator for batching NaFlex-style variable-resolution images."""

    def __init__(
            self,
            max_seq_len: Optional[int] = None,
    ) -> None:
        """Initialize NaFlexCollator.

        Args:
            max_seq_len: Maximum sequence length for batching.
        """
        self.max_seq_len = max_seq_len or 576  # Default ViT-B/16 sequence length (577 = 24*24)

    def __call__(self, batch: List[Tuple[Dict[str, torch.Tensor], Union[int, torch.Tensor]]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Collate batch of NaFlex samples.

        Args:
            batch: List of tuples (patch_dict, target).

        Returns:
            A tuple of (input_dict, targets) where input_dict contains:
                - patches: Padded tensor of patches
                - patch_coord: Coordinates for each patch (y, x)
                - patch_valid: Valid indicators
        """
        assert isinstance(batch[0], tuple)
        batch_size = len(batch)

        # Extract targets
        targets = [item[1] for item in batch]
        if isinstance(targets[0], torch.Tensor):
            targets = torch.stack(targets)
        else:
            targets = torch.tensor(targets, dtype=torch.int64)

        # Get patch dictionaries
        patch_dicts = [item[0] for item in batch]

        # If we have a maximum sequence length constraint, ensure we don't exceed it
        if self.max_seq_len is not None:
            max_patches = self.max_seq_len
        else:
            # Find the maximum number of patches in this batch
            max_patches = max(item['patches'].shape[0] for item in patch_dicts)

        # Check if patches are flattened or unflattened
        patches_tensor = patch_dicts[0]['patches']
        is_unflattened = patches_tensor.ndim == 4  # [N, Ph, Pw, C]

        if is_unflattened:
            # Patches are [N, Ph, Pw, C] - variable patch size mode
            _, ph, pw, c = patches_tensor.shape
            patches = torch.zeros((batch_size, max_patches, ph, pw, c), dtype=torch.float32)
        else:
            # Patches are [N, P*P*C] - normal mode
            patch_dim = patches_tensor.shape[1]
            patches = torch.zeros((batch_size, max_patches, patch_dim), dtype=torch.float32)

        # Prepare other tensors
        patch_coord = torch.zeros((batch_size, max_patches, 2), dtype=torch.int64)  # [B, N, 2] for (y, x)
        patch_valid = torch.zeros((batch_size, max_patches), dtype=torch.bool)

        # Fill in the tensors
        for i, patch_dict in enumerate(patch_dicts):
            num_patches = min(patch_dict['patches'].shape[0], max_patches)

            patches[i, :num_patches] = patch_dict['patches'][:num_patches]
            patch_coord[i, :num_patches] = patch_dict['patch_coord'][:num_patches]
            patch_valid[i, :num_patches] = patch_dict['patch_valid'][:num_patches]

        result = {
            'patches': patches,
            'patch_coord': patch_coord,
            'patch_valid': patch_valid,
            'seq_len': max_patches,
        }

        return result, targets


def _resolve_patch_cfg(
        patch_size: Optional[Union[int, Tuple[int, int]]],
        patch_size_choices: Optional[List[int]],
        patch_size_choice_probs: Optional[List[float]],
) -> Tuple[List[Tuple[int, int]], List[float], bool]:
    """Resolve patch size configuration.

    Args:
        patch_size: Single patch size to use.
        patch_size_choices: List of patch sizes to choose from.
        patch_size_choice_probs: Probabilities for each patch size choice.

    Returns:
        Tuple of (sizes, probs, variable) where sizes is list of patch size tuples,
        probs is list of probabilities, and variable indicates if patch size varies.
    """
    # If both are None, default to patch_size=16
    if patch_size is None and patch_size_choices is None:
        patch_size = 16

    if (patch_size is None) == (patch_size_choices is None):
        raise ValueError(
            "Specify exactly one of `patch_size` or `patch_size_choices`."
        )

    if patch_size is not None:
        sizes = [to_2tuple(patch_size)]
        probs = [1.0]
        variable = False
    else:
        sizes = [to_2tuple(p) for p in patch_size_choices]
        if patch_size_choice_probs is None:
            probs = [1.0 / len(sizes)] * len(sizes)
        else:
            if len(patch_size_choice_probs) != len(sizes):
                raise ValueError("`patch_size_choice_probs` length mismatch.")
            s = float(sum(patch_size_choice_probs))
            if s <= 0:
                raise ValueError("`patch_size_choice_probs` sum to zero.")
            probs = [p / s for p in patch_size_choice_probs]
        variable = True
    return sizes, probs, variable


class NaFlexMapDatasetWrapper(IterableDataset):
    """
    IterableDataset wrapper for a map-style base dataset.

    Yields batches with variable sequence lengths. It calculates a canonical
    batch schedule (sequence length, batch size pairs) once based on the
    total dataset size (padded for distribution). Each epoch, it shuffles
    the order of this canonical schedule and the dataset indices.
    This ensures a consistent number of batches and samples per epoch
    across all ranks. Handles distributed training and multiple workers.
    """

    def __init__(
            self,
            base_dataset: Dataset,
            patch_size: Optional[Union[int, Tuple[int, int]]] = None,
            patch_size_choices: Optional[List[int]] = None,
            patch_size_choice_probs: Optional[List[float]] = None,
            seq_lens: Tuple[int, ...] = (128, 256, 576, 784, 1024),
            max_tokens_per_batch: int = 4096 * 4,
            transform_factory: Optional[Callable] = None,
            mixup_fn: Optional[Callable] = None,
            seed: int = 42,
            shuffle: bool = True,
            distributed: bool = False,
            rank: int = 0,
            world_size: int = 1,
            epoch: int = 0,
            batch_divisor: int = 8,
    ) -> None:
        """Initialize NaFlexMapDatasetWrapper.

        Args:
            base_dataset: Map-style dataset to wrap.
            patch_size: Single patch size to use.
            patch_size_choices: List of patch sizes to randomly select from.
            patch_size_choice_probs: Probabilities for each patch size.
            seq_lens: Sequence lengths to use for batching.
            max_tokens_per_batch: Target tokens per batch.
            transform_factory: Factory function for creating transforms.
            mixup_fn: Optional mixup function.
            seed: Random seed.
            shuffle: Whether to shuffle data.
            distributed: Whether using distributed training.
            rank: Process rank for distributed training.
            world_size: Total number of processes.
            epoch: Starting epoch.
            batch_divisor: Ensure batch size is divisible by this.
        """
        super().__init__()
        if not hasattr(base_dataset, '__len__') or not hasattr(base_dataset, '__getitem__'):
            raise TypeError("base_dataset must be a map-style dataset (implement __len__ and __getitem__)")

        self.base_dataset = base_dataset
        self.seq_lens = sorted(list(set(seq_lens))) # Ensure unique and sorted
        self.max_tokens_per_batch = max_tokens_per_batch
        self.seed = seed
        self.shuffle = shuffle
        self.distributed = distributed
        self.rank = rank if distributed else 0
        self.world_size = world_size if distributed else 1
        self.epoch = epoch
        self.batch_divisor = batch_divisor

        # Resolve patch size configuration
        self.patch_sizes, self.patch_size_probs, self.variable_patch_size = _resolve_patch_cfg(
            patch_size,
            patch_size_choices,
            patch_size_choice_probs
        )

        # Pre-initialize transforms and collate fns for each (seq_len, patch_idx) combination
        self.transforms: Dict[Tuple[int, int], Optional[Callable]] = {}
        self.collate_fns: Dict[int, Callable] = {}
        self.patchifiers: List[Callable] = []

        for seq_len in self.seq_lens:
            self.collate_fns[seq_len] = NaFlexCollator(seq_len)

        for patch_idx, patch_size_tuple in enumerate(self.patch_sizes):
            # Pre-initialize patchifiers for each patch size (indexed by patch_idx)
            self.patchifiers.append(Patchify(
                patch_size=patch_size_tuple,
                flatten_patches=not self.variable_patch_size
            ))

            # Create transforms for each (seq_len, patch_idx) combination
            for seq_len in self.seq_lens:
                key = (seq_len, patch_idx)
                if transform_factory:
                    self.transforms[key] = transform_factory(max_seq_len=seq_len, patch_size=patch_size_tuple)
                else:
                    self.transforms[key] = None # No transform

        self.mixup_fn = mixup_fn

        # Canonical Schedule Calculation (Done Once)
        self._canonical_batch_schedule: List[Tuple[int, int]] = []
        self._num_batches_per_rank: int = 0
        self._padded_samples_per_rank: int = 0
        self._create_canonical_schedule() # Calculate schedule based on padded size

        # Per-Epoch State
        # Stores (seq_len, list_of_indices) for the current epoch, specific to this rank
        self._epoch_batches: List[Tuple[int, List[int]]] = []
        self._prepare_epoch_batches(self.epoch)  # setup for initial epoch

    def _create_canonical_schedule(self):
        """
        Calculates the canonical batch schedule (seq_len, batch_size pairs)
        based on the dataset size, padded for distributed training.
        This schedule is the *same* for all ranks and ensures consistent
        epoch length. It is calculated once during initialization.
        """
        total_len = len(self.base_dataset)
        padded_total_len = total_len
        num_samples_per_rank = total_len

        if self.distributed and self.world_size > 1:
            # Calculate padding needed for even distribution
            if total_len % self.world_size != 0:
                 pad_size = self.world_size - (total_len % self.world_size)
                 padded_total_len += pad_size
                 print(f"Rank {self.rank}: Padding dataset with {pad_size} samples for distributed training (total size {padded_total_len}).")
            else:
                 pad_size = 0

            if padded_total_len % self.world_size != 0:
                 # This should not happen with the padding logic, but safeguard
                 raise RuntimeError(f"Internal Error: Padded total length {padded_total_len} not divisible by world size {self.world_size}")

            num_samples_per_rank = padded_total_len // self.world_size
        elif self.distributed and self.world_size <= 1:
             # Distributed flag set but world_size is 1, treat as non-distributed
             pass # num_samples_per_rank remains total_len

        self._padded_samples_per_rank = num_samples_per_rank

        if num_samples_per_rank == 0:
             self._canonical_batch_schedule = []
             self._num_batches_per_rank = 0
             return

        # Use a fixed seed for generating the canonical schedule structure
        g = torch.Generator()
        g.manual_seed(self.seed) # Use base seed, NOT epoch seed

        current_schedule: List[Tuple[int, int]] = []
        remaining_samples = num_samples_per_rank
        total_scheduled_samples = 0

        while remaining_samples > 0:
            # Sample sequence length deterministically based on base seed
            seq_idx = torch.randint(0, len(self.seq_lens), (1,), generator=g).item()
            seq_len = self.seq_lens[seq_idx]

            # Calculate batch size
            batch_size = calculate_naflex_batch_size(
                tokens_per_batch=self.max_tokens_per_batch,
                seq_len=seq_len,
                # max_size should be remaining_samples to avoid overshooting
                max_size=remaining_samples,
                divisor=self.batch_divisor,
                rounding='floor',
            )
            # Ensure batch size is positive and doesn't exceed remaining samples
            batch_size = max(1, batch_size)
            batch_size = min(batch_size, remaining_samples)

            if batch_size <= 0:
                 warnings.warn(f"Calculated batch size <= 0 (seq_len={seq_len}, remaining={remaining_samples}). Stopping schedule generation early.")
                 break # Avoid infinite loop if something goes wrong

            current_schedule.append((seq_len, batch_size))
            remaining_samples -= batch_size
            total_scheduled_samples += batch_size

        # Sanity check: Ensure the schedule covers all samples for the rank
        if total_scheduled_samples != num_samples_per_rank:
            warnings.warn(
                f"Rank {self.rank}: Canonical schedule accounts for {total_scheduled_samples} samples, "
                f"but expected {num_samples_per_rank} samples per rank. "
                f"This might happen if min_batch_size or batch_divisor constraints prevent utilizing all samples. "
                f"Check parameters. Remaining samples: {remaining_samples}"
            )
            # Adjust if needed? Could add a final small batch, but might violate constraints.
            # Current behavior: some samples might be dropped if schedule logic fails.

        self._canonical_batch_schedule = current_schedule
        self._num_batches_per_rank = len(current_schedule)
        print(f"Rank {self.rank}: Created canonical schedule with {self._num_batches_per_rank} batches for {self._padded_samples_per_rank} samples/rank.")


    def _prepare_epoch_batches(self, epoch: int):
        """
        Prepares the batches for the current epoch by:
        1. Shuffling the full dataset indices (using epoch seed).
        2. Applying padding if in distributed mode.
        3. Selecting indices for the current rank.
        4. Shuffling the *order* of the canonical batch schedule (using epoch seed).
        5. Assigning the rank's indices to the shuffled batches.
        """
        g = torch.Generator()
        g.manual_seed(self.seed + epoch) # Epoch-specific seed for shuffling

        # 1. Get shuffled global indices
        total_len = len(self.base_dataset)
        if self.shuffle:
            all_indices_shuffled = torch.randperm(total_len, generator=g).tolist()
        else:
            all_indices_shuffled = list(range(total_len))

        # 2. Apply padding for distributed mode
        indices_for_ranks = all_indices_shuffled
        if self.distributed and self.world_size > 1:
            padded_total_len = self._padded_samples_per_rank * self.world_size
            if padded_total_len > total_len:
                pad_size = padded_total_len - total_len
                # Repeat initial elements from the *shuffled* list for padding
                indices_for_ranks = all_indices_shuffled + all_indices_shuffled[:pad_size]
            # Ensure length matches expectation
            if len(indices_for_ranks) != padded_total_len:
                 raise RuntimeError(f"Internal Error: Padded index list length {len(indices_for_ranks)} does not match expected {padded_total_len}")

        # 3. Select indices for the current rank
        if self.distributed and self.world_size > 1:
            indices_this_rank = indices_for_ranks[self.rank::self.world_size]
        else: # Non-distributed or world_size=1
            indices_this_rank = indices_for_ranks

        # Sanity check length
        if len(indices_this_rank) != self._padded_samples_per_rank:
             # This might happen if canonical schedule generation had warnings/issues
             warnings.warn(
                 f"Rank {self.rank}: Number of indices for this rank ({len(indices_this_rank)}) "
                 f"does not match expected padded samples per rank ({self._padded_samples_per_rank}). "
                 f"Epoch generation might be inconsistent."
              )
             # Adjust expected samples? Or truncate/pad indices? Let's proceed but warn.
             # Using min() prevents IndexError later if indices are fewer than expected.
             effective_samples_this_rank = min(len(indices_this_rank), self._padded_samples_per_rank)
             indices_this_rank = indices_this_rank[:effective_samples_this_rank]

        else:
             effective_samples_this_rank = self._padded_samples_per_rank

        # 4. Shuffle the order of the canonical batch schedule for this epoch
        if self.shuffle:
            schedule_perm = torch.randperm(self._num_batches_per_rank, generator=g).tolist()
            shuffled_schedule = [self._canonical_batch_schedule[i] for i in schedule_perm]
        else:
            shuffled_schedule = list(self._canonical_batch_schedule) # Keep original order

        # 5. Assign indices to the shuffled batches
        self._epoch_batches = []
        idx_pos = 0
        scheduled_samples_count = 0
        for seq_len, bs in shuffled_schedule:
            # Ensure we don't try to grab more indices than available for the rank
            actual_bs = min(bs, effective_samples_this_rank - idx_pos)
            if actual_bs <= 0:
                 if scheduled_samples_count < effective_samples_this_rank:
                     # This indicates mismatch between schedule total and actual samples
                     warnings.warn(f"Rank {self.rank}: Ran out of samples ({idx_pos}/{effective_samples_this_rank}) before processing entire schedule. Check schedule generation.")
                 break # Stop if no more indices or batch size is zero

            batch_indices = indices_this_rank[idx_pos : idx_pos + actual_bs]
            self._epoch_batches.append((seq_len, batch_indices))
            idx_pos += actual_bs
            scheduled_samples_count += actual_bs

        # Final check
        if scheduled_samples_count != effective_samples_this_rank:
             warnings.warn(
                f"Rank {self.rank}: Assigned {scheduled_samples_count} samples to batches, "
                f"but expected {effective_samples_this_rank} effective samples this epoch. "
                f"Indices remaining: {effective_samples_this_rank - scheduled_samples_count}."
             )

    def set_epoch(self, epoch: int) -> None:
        """Updates the epoch, regenerating the epoch-specific batches.

        Args:
            epoch: New epoch number.
        """
        # Only regenerate if the epoch actually changes
        if epoch != self.epoch:
            self.epoch = epoch
            self._prepare_epoch_batches(epoch)

    def __len__(self) -> int:
        """Returns the number of batches per worker for the current epoch.

        Returns:
            Number of batches this worker will process.
        """
        return self._num_batches_per_rank

    def __iter__(self) -> Iterator[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        """Iterates through pre-calculated batches for the current epoch.

        Yields:
            Tuple of (input_dict, targets) for each batch.
        """
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        # Distribute pre-calculated batches among workers for this rank
        # Each worker processes a slice of the batches prepared in _prepare_epoch_batches
        batches_for_worker = self._epoch_batches[worker_id::num_workers]
        for seq_len, indices in batches_for_worker:
            if not indices: # Skip if a batch ended up with no indices (shouldn't happen often)
                 continue

            # Select patch size for this batch
            patch_idx = 0
            if self.variable_patch_size:
                # Use torch multinomial for weighted random choice
                patch_idx = torch.multinomial(torch.tensor(self.patch_size_probs), 1).item()

            # Get the pre-initialized transform and patchifier using patch_idx
            transform_key = (seq_len, patch_idx)
            transform = self.transforms.get(transform_key)
            batch_patchifier = self.patchifiers[patch_idx]

            batch_imgs = []
            batch_targets = []
            for idx in indices:
                try:
                    # Get original image and label from map-style dataset
                    img, label = self.base_dataset[idx]

                    # Apply transform if available
                    # Handle cases where transform might return None or fail
                    processed_img = transform(img) if transform else img
                    if processed_img is None:
                        warnings.warn(f"Transform returned None for index {idx}. Skipping sample.")
                        continue

                    batch_imgs.append(processed_img)
                    batch_targets.append(label)

                except IndexError:
                     warnings.warn(f"IndexError encountered for index {idx} (possibly due to padding/repeated indices). Skipping sample.")
                     continue
                except Exception as e:
                    # Log other potential errors during data loading/processing
                    warnings.warn(f"Error processing sample index {idx}. Error: {e}. Skipping sample.")
                    continue # Skip problematic sample

            if self.mixup_fn is not None:
                batch_imgs, batch_targets = self.mixup_fn(batch_imgs, batch_targets)

            batch_imgs = [batch_patchifier(img) for img in batch_imgs]
            batch_samples = list(zip(batch_imgs, batch_targets))
            if batch_samples: # Only yield if we successfully processed samples
                # Collate the processed samples into a batch
                yield self.collate_fns[seq_len](batch_samples)

            # If batch_samples is empty after processing 'indices', an empty batch is skipped.
