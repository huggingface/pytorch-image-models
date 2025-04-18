"""
Dynamic Sequence Length Datasets for Variable Resolution Image Processing

Implements two dataset wrappers:
1. DynamicSeqMapDataset - Map-style dataset that returns batches with variable sequence lengths
2. DynamicSeqIterDataset - Iterable dataset that yields batches with variable sequence lengths

Both support:
- Pre-initialized transforms for efficiency
- Distributed training
- Multiple workers
- Variable batch sizes based on sequence length
"""

import math
import random
import warnings
from functools import partial
from typing import Any, Iterator, List, Tuple, Dict, Optional, Union, Callable

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision import transforms
from PIL import Image


from .naflex_transforms import Patchify, patchify


def calculate_batch_size(
        tokens_per_batch: int,
        seq_len: int,
        max_size: Optional[int] = None,
        divisor: int = 1,
        rounding: str ='floor',
):
    """Calculate batch size based on sequence length with divisibility constraints."""
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


def _collate_batch(
    batch_samples: List[Tuple[Dict[str, torch.Tensor], Any]],
    target_seq_len: int
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Collates processed samples into a batch, padding/truncating to target_seq_len."""
    batch_patch_data = [item[0] for item in batch_samples]
    batch_labels = [item[1] for item in batch_samples]

    if not batch_patch_data:
         return {}, torch.empty(0)

    batch_size = len(batch_patch_data)
    patch_dim = batch_patch_data[0]['patches'].shape[1]

    # Initialize tensors with target sequence length
    patches_batch = torch.zeros((batch_size, target_seq_len, patch_dim), dtype=torch.float32)
    patch_coord_batch = torch.zeros((batch_size, target_seq_len, 2), dtype=torch.int64)
    patch_valid_batch = torch.zeros((batch_size, target_seq_len), dtype=torch.bool) # Use bool

    for i, data in enumerate(batch_patch_data):
        num_patches = data['patches'].shape[0]
        # Take min(num_patches, target_seq_len) patches
        n_copy = min(num_patches, target_seq_len)

        patches_batch[i, :n_copy] = data['patches'][:n_copy]
        patch_coord_batch[i, :n_copy] = data['patch_coord'][:n_copy]
        patch_valid_batch[i, :n_copy] = data['patch_valid'][:n_copy] # Copy validity flags

    # Create the final input dict
    input_dict = {
        'patches': patches_batch,
        'patch_coord': patch_coord_batch,
        'patch_valid': patch_valid_batch, # Boolean mask
        # Note: 'seq_length' might be ambiguous. The target length is target_seq_len.
        # The actual number of valid patches per sample varies.
        # 'patch_valid' mask is the most reliable source of truth.
    }

    # Attempt to stack labels if they are tensors, otherwise return list
    try:
        if isinstance(batch_labels[0], torch.Tensor):
            labels_tensor = torch.stack(batch_labels)
        else:
            # Convert numerical types to tensor, keep others as list (or handle specific types)
            if isinstance(batch_labels[0], (int, float)):
                 labels_tensor = torch.tensor(batch_labels)
            else:
                 # Cannot convert non-numerical labels easily, return as list
                 # Or handle specific conversion if needed
                 # For FakeDataset, labels are ints, so this works
                 labels_tensor = torch.tensor(batch_labels) # Assuming labels are numerical
    except Exception:
        # Fallback if stacking fails (e.g., different shapes, types)
        print("Warning: Could not stack labels into a tensor. Returning list of labels.")
        labels_tensor = batch_labels # Return as list

    return input_dict, labels_tensor


class VariableSeqMapWrapper(IterableDataset):
    """
    IterableDataset wrapper for a map-style base dataset.

    Yields batches with variable sequence lengths. It calculates a canonical
    batch schedule (sequence length, batch size pairs) once based on the
    total dataset size (padded for distribution). Each epoch, it shuffles
    the *order* of this canonical schedule and the dataset indices.
    This ensures a consistent number of batches and samples per epoch
    across all ranks. Handles distributed training and multiple workers.
    """

    def __init__(
            self,
            base_dataset: Dataset,
            patch_size: Union[int, Tuple[int, int]] = 16,
            seq_lens: List[int] = (128, 256, 576, 784, 1024),
            max_tokens_per_batch: int = 4096 * 4, # Example: 16k tokens
            transform_factory: Optional[Callable] = None,
            seed: int = 42,
            shuffle: bool = True,
            distributed: bool = False,
            rank: int = 0,
            world_size: int = 1,
            epoch: int = 0,
            batch_divisor: int = 8, # Ensure batch size is divisible by this
    ):
        super().__init__()
        if not hasattr(base_dataset, '__len__') or not hasattr(base_dataset, '__getitem__'):
            raise TypeError("base_dataset must be a map-style dataset (implement __len__ and __getitem__)")

        self.base_dataset = base_dataset
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.seq_lens = sorted(list(set(seq_lens))) # Ensure unique and sorted
        self.max_tokens_per_batch = max_tokens_per_batch
        self.seed = seed
        self.shuffle = shuffle
        self.distributed = distributed
        self.rank = rank if distributed else 0
        self.world_size = world_size if distributed else 1
        self.epoch = epoch
        self.batch_divisor = batch_divisor

        # Pre-initialize transforms for each sequence length
        self.transforms: Dict[int, Optional[Callable]] = {}
        if transform_factory:
            for seq_len in self.seq_lens:
                self.transforms[seq_len] = transform_factory(max_seq_len=seq_len, patch_size=self.patch_size)
        else:
             for seq_len in self.seq_lens:
                 self.transforms[seq_len] = None # No transform

        self.patchifier = Patchify(self.patch_size)

        # --- Canonical Schedule Calculation (Done Once) ---
        self._canonical_batch_schedule: List[Tuple[int, int]] = []
        self._num_batches_per_rank: int = 0
        self._padded_samples_per_rank: int = 0
        self._create_canonical_schedule() # Calculate schedule based on padded size

        # --- Per-Epoch State ---
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
            batch_size = calculate_batch_size(
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

    def set_epoch(self, epoch: int):
        """Updates the epoch, regenerating the epoch-specific batches."""
        # Only regenerate if the epoch actually changes
        if epoch != self.epoch:
            self.epoch = epoch
            self._prepare_epoch_batches(epoch)

    def __len__(self) -> int:
        """
        Returns the number of batches per **worker** for the current epoch.
        Calculated based on the fixed number of batches per rank divided by
        the number of workers.
        """
        return self._num_batches_per_rank

    def __iter__(self) -> Iterator[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        """
        Iterates through the pre-calculated batches for the current epoch,
        distributing them among workers.
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

            # Get the pre-initialized transform for this sequence length
            transform = self.transforms.get(seq_len)

            batch_samples = []
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

                    # Apply patching
                    patch_data = self.patchifier(processed_img)
                    batch_samples.append((patch_data, label))

                except IndexError:
                     warnings.warn(f"IndexError encountered for index {idx} (possibly due to padding/repeated indices). Skipping sample.")
                     continue
                except Exception as e:
                    # Log other potential errors during data loading/processing
                    warnings.warn(f"Error processing sample index {idx}. Error: {e}. Skipping sample.")
                    continue # Skip problematic sample

            # Collate the processed samples into a batch
            if batch_samples: # Only yield if we successfully processed samples
                yield _collate_batch(batch_samples, seq_len)

            # If batch_samples is empty after processing 'indices', an empty batch is skipped.
