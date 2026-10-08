""" Exponential Moving Average (EMA) of model updates

Hacked together by / Copyright 2020 Ross Wightman
"""
import logging
from collections import OrderedDict
from copy import deepcopy
from typing import List, Optional

import torch
import torch.nn as nn

_logger = logging.getLogger(__name__)


class ModelEma:
    """ Model Exponential Moving Average (DEPRECATED)

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This version is deprecated, it does not work with scripted models. Will be removed eventually.

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.9999, device='', resume=''):
        # make a copy of the model for accumulating moving average of weights
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if device:
            self.ema.to(device=device)
        self.ema_has_module = hasattr(self.ema, 'module')
        if resume:
            self._load_checkpoint(resume)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _load_checkpoint(self, checkpoint_path):
        # local import to avoid a circular import (timm.utils -> timm.models)
        from timm.models._helpers import _torch_load
        checkpoint = _torch_load(checkpoint_path, map_location='cpu', weights_only=True)
        assert isinstance(checkpoint, dict)
        if 'state_dict_ema' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict_ema'].items():
                # ema model may have been wrapped by DataParallel, and need module prefix
                if self.ema_has_module:
                    name = 'module.' + k if not k.startswith('module') else k
                else:
                    name = k
                new_state_dict[name] = v
            self.ema.load_state_dict(new_state_dict)
            _logger.info("Loaded state_dict_ema")
        else:
            _logger.warning("Failed to find state_dict_ema, starting from loaded model weights")

    def update(self, model):
        # correct a mismatch in state dict keys
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)


class ModelEmaV2(nn.Module):
    """ Model Exponential Moving Average V2

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.9999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _lerp_stochastic_round_bf16_(
        ema_values: List[torch.Tensor],
        model_values: List[torch.Tensor],
        weight: float,
        chunk_numel: int = 2 ** 22,
        generator: Optional[torch.Generator] = None,
) -> None:
    """In-place EMA update of bfloat16 tensors with stochastic rounding.

    The update is computed in float32, then random bits are added below the bfloat16 mantissa before truncating,
    so the result rounds up with probability equal to the discarded fraction. Updates far below bfloat16
    resolution (the usual case with decay ~0.9998) then accumulate in expectation instead of rounding away.

    Tensors are flattened and processed in chunks of at most chunk_numel elements to bound the transient
    float32 memory (~10 bytes / element) and keep the number of kernel launches low with many small tensors.
    """
    def _update_chunk(chunk_ema: List[torch.Tensor], chunk_model: List[torch.Tensor]):
        x = torch.cat([v.reshape(-1) for v in chunk_ema]).float()
        x.mul_(1. - weight).add_(torch.cat([v.reshape(-1) for v in chunk_model]), alpha=weight)
        bits = x.view(torch.int32)
        bits.add_(torch.randint(
            0, 1 << 16, bits.shape, dtype=torch.int32, device=bits.device, generator=generator,
        ))
        bits.bitwise_and_(-(1 << 16))  # truncate to bfloat16 precision (exactly representable in bfloat16)
        x_split = [xi.view(v.shape) for xi, v in zip(x.split([v.numel() for v in chunk_ema]), chunk_ema)]
        if hasattr(torch, '_foreach_copy_'):
            torch._foreach_copy_(chunk_ema, x_split)
        else:
            for ema_v, xi in zip(chunk_ema, x_split):
                ema_v.copy_(xi)

    chunk_ema, chunk_model, numel = [], [], 0
    for ema_v, model_v in zip(ema_values, model_values):
        if numel and numel + ema_v.numel() > chunk_numel:
            _update_chunk(chunk_ema, chunk_model)
            chunk_ema, chunk_model, numel = [], [], 0
        chunk_ema.append(ema_v)
        chunk_model.append(model_v)
        numel += ema_v.numel()
    if chunk_ema:
        _update_chunk(chunk_ema, chunk_model)


class ModelEmaV3(nn.Module):
    """ Model Exponential Moving Average V3

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V3 of this module leverages for_each and in-place operations for faster performance.

    Decay warmup based on code by @crowsonkb, her comments:
      If inv_gamma=1 and power=1, implements a simple average. inv_gamma=1, power=2/3 are
      good values for models you plan to train for a million or more steps (reaches decay
      factor 0.999 at 31.6K steps, 0.9999 at 1M steps), inv_gamma=1, power=3/4 for models
      you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
      215.4k steps).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    Low precision (bfloat16 / float16) weights cannot hold a plain EMA update, (1 - decay) * (model - ema)
    is far below their resolution at typical decay values and rounds to zero. By default bfloat16 EMA
    tensors are updated with stochastic rounding (stochastic_rounding=True) so the updates accumulate in
    expectation. Alternatively set dtype=torch.float32 to keep the EMA in float32 (2x the EMA memory) for a
    low precision model, this is the only option for float16.

    Stochastic rounding uses a private RNG seeded with stochastic_rounding_seed, the global RNG is untouched
    and all ranks in distributed training round identically (keeping EMA weights in sync) given the same seed.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(
            self,
            model,
            decay: float = 0.9999,
            min_decay: float = 0.0,
            update_after_step: int = 0,
            use_warmup: bool = False,
            warmup_gamma: float = 1.0,
            warmup_power: float = 2/3,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            stochastic_rounding: bool = True,
            stochastic_rounding_seed: int = 0,
            foreach: bool = True,
            exclude_buffers: bool = False,
    ):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_warmup = use_warmup
        self.warmup_gamma = warmup_gamma
        self.warmup_power = warmup_power
        self.foreach = foreach
        self.device = device  # perform ema on different device from model if set
        self.dtype = dtype  # keep ema in a different (e.g. higher precision) dtype from model if set
        self.stochastic_rounding = stochastic_rounding  # stochastic rounding for bfloat16 ema tensors
        self.stochastic_rounding_seed = stochastic_rounding_seed
        self.exclude_buffers = exclude_buffers
        if self.device is not None and device != next(model.parameters()).device:
            self.foreach = False  # cannot use foreach methods with different devices
            self.module.to(device=device)
        if self.dtype is not None:
            self.module.to(dtype=dtype)
        # private RNG for stochastic rounding, leaves global RNG untouched and keeps ranks in sync
        self._sr_generator = torch.Generator(device=next(self.module.parameters()).device)
        self._sr_generator.manual_seed(stochastic_rounding_seed)

    def get_decay(self, step: Optional[int] = None) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        if step is None:
            return self.decay

        step = max(0, step - self.update_after_step - 1)
        if step <= 0:
            return 0.0

        if self.use_warmup:
            decay = 1 - (1 + step / self.warmup_gamma) ** -self.warmup_power
            decay = max(min(decay, self.decay), self.min_decay)
        else:
            decay = self.decay

        return decay

    @torch.no_grad()
    def update(self, model, step: Optional[int] = None):
        decay = self.get_decay(step)
        if self.exclude_buffers:
            self.apply_update_no_buffers_(model, decay)
        else:
            self.apply_update_(model, decay)

    def _lerp_(self, ema_v: torch.Tensor, model_v: torch.Tensor, weight: float) -> None:
        # single tensor update, tensors on same device, model_v cast to ema dtype as needed
        if self.stochastic_rounding and ema_v.dtype == torch.bfloat16:
            _lerp_stochastic_round_bf16_([ema_v], [model_v], weight, generator=self._sr_generator)
        else:
            ema_v.lerp_(model_v.to(dtype=ema_v.dtype), weight=weight)

    def _foreach_lerp_(self, ema_values: List[torch.Tensor], model_values: List[torch.Tensor], weight: float) -> None:
        # multi tensor update, tensors on same device, handles mixed ema / model dtypes
        if self.stochastic_rounding:
            sr_ema_values, sr_model_values = [], []
            lerp_ema_values, lerp_model_values = [], []
            for ema_v, model_v in zip(ema_values, model_values):
                if ema_v.dtype == torch.bfloat16:
                    sr_ema_values.append(ema_v)
                    sr_model_values.append(model_v)
                else:
                    lerp_ema_values.append(ema_v)
                    lerp_model_values.append(model_v)
            if sr_ema_values:
                _lerp_stochastic_round_bf16_(sr_ema_values, sr_model_values, weight, generator=self._sr_generator)
            ema_values, model_values = lerp_ema_values, lerp_model_values
        if not ema_values:
            return
        same_dtype = all(ema_v.dtype == model_v.dtype for ema_v, model_v in zip(ema_values, model_values))
        if same_dtype and hasattr(torch, '_foreach_lerp_'):
            torch._foreach_lerp_(ema_values, model_values, weight=weight)
        else:
            # lerp_ requires matching dtypes, add_ promotes in-kernel so no cast copies of the model are needed
            torch._foreach_mul_(ema_values, scalar=1. - weight)
            torch._foreach_add_(ema_values, model_values, alpha=weight)

    def apply_update_(self, model, decay: float):
        # interpolate parameters and buffers
        if self.foreach:
            ema_lerp_values = []
            model_lerp_values = []
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if ema_v.is_floating_point():
                    ema_lerp_values.append(ema_v)
                    model_lerp_values.append(model_v)
                else:
                    ema_v.copy_(model_v)

            self._foreach_lerp_(ema_lerp_values, model_lerp_values, weight=1. - decay)
        else:
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if ema_v.is_floating_point():
                    self._lerp_(ema_v, model_v.to(device=self.device), weight=1. - decay)
                else:
                    ema_v.copy_(model_v.to(device=self.device))

    def apply_update_no_buffers_(self, model, decay: float):
        # interpolate parameters, copy buffers
        ema_params = list(self.module.parameters())
        model_params = list(model.parameters())
        if self.foreach:
            self._foreach_lerp_(ema_params, model_params, weight=1. - decay)
        else:
            for ema_p, model_p in zip(ema_params, model_params):
                self._lerp_(ema_p, model_p.to(device=self.device), weight=1. - decay)

        for ema_b, model_b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(model_b.to(device=self.device))

    @torch.no_grad()
    def set(self, model):
        for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
            ema_v.copy_(model_v.to(device=self.device))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
