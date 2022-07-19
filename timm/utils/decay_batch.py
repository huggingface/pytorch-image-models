""" Batch size decay and retry helpers.

Copyright 2022 Ross Wightman
"""
import math


def decay_batch_step(batch_size, num_intra_steps=2, no_odd=False):
    """ power of two batch-size decay with intra steps

    Decay by stepping between powers of 2:
    * determine power-of-2 floor of current batch size (base batch size)
    * divide above value by num_intra_steps to determine step size
    * floor batch_size to nearest multiple of step_size (from base batch size)
    Examples:
     num_steps == 4 --> 64, 56, 48, 40, 32, 28, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1
     num_steps (no_odd=True) == 4 --> 64, 56, 48, 40, 32, 28, 24, 20, 16, 14, 12, 10, 8, 6, 4, 2
     num_steps == 2 --> 64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1
     num_steps == 1 --> 64, 32, 16, 8, 4, 2, 1
    """
    if batch_size <= 1:
        # return 0 for stopping value so easy to use in loop
        return 0
    base_batch_size = int(2 ** (math.log(batch_size - 1) // math.log(2)))
    step_size = max(base_batch_size // num_intra_steps, 1)
    batch_size = base_batch_size + ((batch_size - base_batch_size - 1) // step_size) * step_size
    if no_odd and batch_size % 2:
        batch_size -= 1
    return batch_size


def check_batch_size_retry(error_str):
    """ check failure error string for conditions where batch decay retry should not be attempted
    """
    error_str = error_str.lower()
    if 'required rank' in error_str:
        # Errors involving phrase 'required rank' typically happen when a conv is used that's
        # not compatible with channels_last memory format.
        return False
    if 'illegal' in error_str:
        # 'Illegal memory access' errors in CUDA typically leave process in unusable state
        return False
    return True
