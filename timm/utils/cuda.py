""" CUDA / AMP utils

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch

try:
    from apex import amp
    has_apex = True
except ImportError:
    amp = None
    has_apex = False

from .clip_grad import dispatch_clip_grad


class ApexScaler:
    state_dict_key = "amp"

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False, step_and_update=True):
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(create_graph=create_graph)
        if clip_grad is not None:
            dispatch_clip_grad(amp.master_params(optimizer), clip_grad, mode=clip_mode)
        optimizer.step()

    def state_dict(self):
        if 'state_dict' in amp.__dict__:
            return amp.state_dict()

    def load_state_dict(self, state_dict):
        if 'load_state_dict' in amp.__dict__:
            amp.load_state_dict(state_dict)

    def step_and_update(self):
        pass

class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False, step_and_update=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        self._optimizer = optimizer
        if clip_grad is not None:
            assert parameters is not None
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
        if step_and_update:
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

    def step_and_update(self):
        # need to separate this step from "scaler.scale(loss)" since scalar.step() syncs GPU with CPU,
        # which is not allowed for cuda graph capture
        self._scaler.step(self._optimizer)
        self._scaler.update()
