""" Optimizer Factory w/ custom Weight Decay & Layer Decay support

Hacked together by / Copyright 2021 Ross Wightman
"""
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Collection, Dict, List, Optional, Set, Tuple, Type, Union
from fnmatch import fnmatch
import importlib

import torch
import torch.nn as nn
import torch.optim

from ._param_groups import param_groups_layer_decay, param_groups_weight_decay
from ._types import ParamsT, OptimType, OptimizerCallable
from .adabelief import AdaBelief
from .adafactor import Adafactor
from .adafactor_bv import AdafactorBigVision
from .adahessian import Adahessian
from .adamp import AdamP
from .adamw import AdamWLegacy
from .adan import Adan
from .adopt import Adopt
from .kron import Kron
from .lamb import Lamb
from .laprop import LaProp
from .lars import Lars
from .lion import Lion
from .lookahead import Lookahead
from .madgrad import MADGRAD
from .mars import Mars
from .muon import Muon
from .nadam import NAdamLegacy
from .nadamw import NAdamW
from .nvnovograd import NvNovoGrad
from .radam import RAdamLegacy
from .rmsprop_tf import RMSpropTF
from .sgdp import SGDP
from .sgdw import SGDW

_logger = logging.getLogger(__name__)


def _import_class(class_string: str) -> Type:
    """Dynamically import a class from a string."""
    try:
        module_name, class_name = class_string.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import {class_string}: {e}")



@dataclass(frozen=True)
class OptimInfo:
    """Immutable configuration for an optimizer.

    Attributes:
        name: Unique identifier for the optimizer
        opt_class: The optimizer class
        description: Brief description of the optimizer's characteristics and behavior
        has_eps: Whether the optimizer accepts epsilon parameter
        has_momentum: Whether the optimizer accepts momentum parameter
        has_betas: Whether the optimizer accepts a tuple of beta parameters
        num_betas: number of betas in tuple (valid IFF has_betas = True)
        defaults: Optional default parameters for the optimizer
    """
    name: str
    opt_class: Union[str, OptimType]
    description: str = ''
    has_eps: bool = True
    has_momentum: bool = False
    has_betas: bool = False
    num_betas: int = 2
    second_order: bool = False
    defaults: Optional[Dict[str, Any]] = None


class OptimizerRegistry:
    """Registry managing optimizer configurations and instantiation.

    This class provides a central registry for optimizer configurations and handles
    their instantiation with appropriate parameter groups and settings.
    """

    def __init__(self) -> None:
        self._optimizers: Dict[str, OptimInfo] = {}
        self._foreach_defaults: Set[str] = {'lion'}

    def register(self, info: OptimInfo) -> None:
        """Register an optimizer configuration.

        Args:
            info: The OptimInfo configuration containing name, type and description
        """
        name = info.name.lower()
        if name in self._optimizers:
            _logger.warning(f'Optimizer {name} already registered, overwriting')
        self._optimizers[name] = info

    def register_alias(self, alias: str, target: str) -> None:
        """Register an alias for an existing optimizer.

        Args:
            alias: The alias name
            target: The target optimizer name

        Raises:
            KeyError: If target optimizer doesn't exist
        """
        target = target.lower()
        if target not in self._optimizers:
            raise KeyError(f'Cannot create alias for non-existent optimizer {target}')
        self._optimizers[alias.lower()] = self._optimizers[target]

    def register_foreach_default(self, name: str) -> None:
        """Register an optimizer as defaulting to foreach=True."""
        self._foreach_defaults.add(name.lower())

    def list_optimizers(
            self,
            filter: Union[str, List[str]] = '',
            exclude_filters: Optional[List[str]] = None,
            with_description: bool = False
    ) -> List[Union[str, Tuple[str, str]]]:
        """List available optimizer names, optionally filtered.

        Args:
            filter: Wildcard style filter string (e.g., 'adam*')
            exclude_filters: Optional list of wildcard patterns to exclude
            with_description: If True, return tuples of (name, description)

        Returns:
            List of either optimizer names or (name, description) tuples
        """
        names = sorted(self._optimizers.keys())

        if filter:
            if isinstance(filter, str):
                filters = [filter]
            else:
                filters = filter
            filtered_names = set()
            for f in filters:
                filtered_names.update(n for n in names if fnmatch(n, f))
            names = sorted(filtered_names)

        if exclude_filters:
            for exclude_filter in exclude_filters:
                names = [n for n in names if not fnmatch(n, exclude_filter)]

        if with_description:
            return [(name, self._optimizers[name].description) for name in names]

        return names

    def get_optimizer_info(self, name: str) -> OptimInfo:
        """Get the OptimInfo for an optimizer.

        Args:
            name: Name of the optimizer

        Returns:
            OptimInfo configuration

        Raises:
            ValueError: If optimizer is not found
        """
        name = name.lower()
        if name not in self._optimizers:
            raise ValueError(f'Optimizer {name} not found in registry')
        return self._optimizers[name]

    def get_optimizer_class(
            self,
            name_or_info: Union[str, OptimInfo],
            bind_defaults: bool = True,
    ) -> Union[OptimType, OptimizerCallable]:
        """Get the optimizer class with any default arguments applied.

        This allows direct instantiation of optimizers with their default configs
        without going through the full factory.

        Args:
            name_or_info: Name of the optimizer
            bind_defaults: Bind default arguments to optimizer class via `partial` before returning

        Returns:
            Optimizer class or partial with defaults applied

        Raises:
            ValueError: If optimizer not found
        """
        if isinstance(name_or_info, str):
            opt_info = self.get_optimizer_info(name_or_info)
        else:
            assert isinstance(name_or_info, OptimInfo)
            opt_info = name_or_info

        if isinstance(opt_info.opt_class, str):
            # Special handling for APEX and BNB optimizers
            if opt_info.opt_class.startswith('apex.'):
                assert torch.cuda.is_available(), 'CUDA required for APEX optimizers'
                try:
                    opt_class = _import_class(opt_info.opt_class)
                except ImportError as e:
                    raise ImportError('APEX optimizers require apex to be installed') from e
            elif opt_info.opt_class.startswith('bitsandbytes.'):
                assert torch.cuda.is_available(), 'CUDA required for bitsandbytes optimizers'
                try:
                    opt_class = _import_class(opt_info.opt_class)
                except ImportError as e:
                    raise ImportError('bitsandbytes optimizers require bitsandbytes to be installed') from e
            else:
                opt_class = _import_class(opt_info.opt_class)
        else:
            opt_class = opt_info.opt_class

        # Return class or partial with defaults
        if bind_defaults and opt_info.defaults:
            opt_class = partial(opt_class, **opt_info.defaults)

        return opt_class

    def create_optimizer(
            self,
            model_or_params: Union[nn.Module, ParamsT],
            opt: str,
            lr: Optional[float] = None,
            weight_decay: float = 0.,
            momentum: float = 0.9,
            foreach: Optional[bool] = None,
            weight_decay_exclude_1d: bool = True,
            fallback_list: Collection[str] = (),
            fallback_no_weight_decay: bool = False,
            layer_decay: Optional[float] = None,
            layer_decay_min_scale: Optional[float] = None,
            layer_decay_no_opt_scale: Optional[float] = None,
            param_group_fn: Optional[Callable[[nn.Module], ParamsT]] = None,
            **kwargs: Any,
    ) -> torch.optim.Optimizer:
        """Create an optimizer instance.

        Args:
            model_or_params: Model or parameters to optimize
            opt: Name of optimizer to create
            lr: Learning rate
            weight_decay: Weight decay factor
            momentum: Momentum factor for applicable optimizers
            foreach: Enable/disable foreach operation
            weight_decay_exclude_1d: Whether to skip weight decay for 1d params (biases and norm affine)
            fallback_list: Collection of parameter name patterns to use fallback optimizer for hybrid optimizers
            fallback_no_weight_decay: If True, params in no_weight_decay list will use fallback optimizer (e.g., AdamW for Muon)
            layer_decay: Layer-wise learning rate decay
            layer_scale_min_scale: Minimum layer scale factor clamp value
            layer_scale_no_opt_scale: Layer scale below which optimization is disabled
            param_group_fn: Optional custom parameter grouping function
            **kwargs: Additional optimizer-specific arguments

        Returns:
            Configured optimizer instance

        Raises:
            ValueError: If optimizer not found or configuration invalid
        """

        # Get parameters to optimize
        if isinstance(model_or_params, nn.Module):
            # Extract parameters from a nn.Module, build param groups w/ weight-decay and/or layer-decay applied
            no_weight_decay = getattr(model_or_params, 'no_weight_decay', lambda: set())()

            if param_group_fn:
                # run custom fn to generate param groups from nn.Module
                params = param_group_fn(model_or_params)
            elif layer_decay is not None:
                params = param_groups_layer_decay(
                    model_or_params,
                    weight_decay=weight_decay,
                    layer_decay=layer_decay,
                    no_weight_decay_list=no_weight_decay,
                    fallback_list=fallback_list,
                    fallback_no_weight_decay=fallback_no_weight_decay,
                    weight_decay_exclude_1d=weight_decay_exclude_1d,
                    min_scale=layer_decay_min_scale,
                    no_opt_scale=layer_decay_no_opt_scale,
                )
                weight_decay = 0.
            elif weight_decay and weight_decay_exclude_1d:
                params = param_groups_weight_decay(
                    model_or_params,
                    weight_decay=weight_decay,
                    no_weight_decay_list=no_weight_decay,
                    fallback_list=fallback_list,
                    fallback_no_weight_decay=fallback_no_weight_decay,
                )
                weight_decay = 0.
            else:
                params = model_or_params.parameters()
        else:
            # pass parameters / parameter groups through to optimizer
            params = model_or_params

        # Parse optimizer name
        opt_split = opt.lower().split('_')
        opt_name = opt_split[-1]
        use_lookahead = opt_split[0] == 'lookahead' if len(opt_split) > 1 else False

        opt_info = self.get_optimizer_info(opt_name)

        # Build optimizer arguments
        opt_args: Dict[str, Any] = {'weight_decay': weight_decay, **kwargs}

        # Add LR to args, if None optimizer default is used, some optimizers manage LR internally if None.
        if lr is not None:
            opt_args['lr'] = lr

        # Apply optimizer-specific settings
        if opt_info.defaults:
            for k, v in opt_info.defaults.items():
                opt_args.setdefault(k, v)

        # timm has always defaulted momentum to 0.9 if optimizer supports momentum, keep for backward compat.
        if opt_info.has_momentum:
            opt_args.setdefault('momentum', momentum)

        # Remove commonly used kwargs that aren't always supported
        if not opt_info.has_eps:
            opt_args.pop('eps', None)
        if not opt_info.has_betas:
            opt_args.pop('betas', None)

        if foreach is not None:
            # Explicitly activate or deactivate multi-tensor foreach impl.
            # Not all optimizers support this, and those that do usually default to using
            # multi-tensor impl if foreach is left as default 'None' and can be enabled.
            opt_args.setdefault('foreach', foreach)

        # Create optimizer
        opt_class = self.get_optimizer_class(opt_info, bind_defaults=False)
        optimizer = opt_class(params, **opt_args)

        # Apply Lookahead if requested
        if use_lookahead:
            optimizer = Lookahead(optimizer)

        return optimizer


def _register_sgd_variants(registry: OptimizerRegistry) -> None:
    """Register SGD-based optimizers"""
    sgd_optimizers = [
        OptimInfo(
            name='sgd',
            opt_class=torch.optim.SGD,
            description='torch.Optim Stochastic Gradient Descent (SGD) with Nesterov momentum',
            has_eps=False,
            has_momentum=True,
            defaults={'nesterov': True}
        ),
        OptimInfo(
            name='momentum',
            opt_class=torch.optim.SGD,
            description='torch.Optim Stochastic Gradient Descent (SGD) with classical momentum',
            has_eps=False,
            has_momentum=True,
            defaults={'nesterov': False}
        ),
        OptimInfo(
            name='sgdp',
            opt_class=SGDP,
            description='SGD with built-in projection to unit norm sphere',
            has_momentum=True,
            defaults={'nesterov': True}
        ),
        OptimInfo(
            name='sgdw',
            opt_class=SGDW,
            description='SGD with decoupled weight decay and Nesterov momentum',
            has_eps=False,
            has_momentum=True,
            defaults={'nesterov': True}
        ),
    ]
    for opt in sgd_optimizers:
        registry.register(opt)


def _register_adam_variants(registry: OptimizerRegistry) -> None:
    """Register Adam-based optimizers"""
    adam_optimizers = [
        OptimInfo(
            name='adam',
            opt_class=torch.optim.Adam,
            description='torch.optim.Adam, Adaptive Moment Estimation',
            has_betas=True
        ),
        OptimInfo(
            name='adamw',
            opt_class=torch.optim.AdamW,
            description='torch.optim.AdamW, Adam with decoupled weight decay',
            has_betas=True
        ),
        OptimInfo(
            name='adamwlegacy',
            opt_class=AdamWLegacy,
            description='legacy impl of AdamW that pre-dates inclusion to torch.optim',
            has_betas=True
        ),
        OptimInfo(
            name='adamp',
            opt_class=AdamP,
            description='Adam with built-in projection to unit norm sphere',
            has_betas=True,
            defaults={'wd_ratio': 0.01, 'nesterov': True}
        ),
        OptimInfo(
            name='nadam',
            opt_class=torch.optim.NAdam,
            description='torch.optim.NAdam, Adam with Nesterov momentum',
            has_betas=True
        ),
        OptimInfo(
            name='nadamlegacy',
            opt_class=NAdamLegacy,
            description='legacy impl of NAdam that pre-dates inclusion in torch.optim',
            has_betas=True
        ),
        OptimInfo(
            name='nadamw',
            opt_class=NAdamW,
            description='Adam with Nesterov momentum and decoupled weight decay, mlcommons/algorithmic-efficiency impl',
            has_betas=True
        ),
        OptimInfo(
            name='radam',
            opt_class=torch.optim.RAdam,
            description='torch.optim.RAdam, Rectified Adam with variance adaptation',
            has_betas=True
        ),
        OptimInfo(
            name='radamlegacy',
            opt_class=RAdamLegacy,
            description='legacy impl of RAdam that predates inclusion in torch.optim',
            has_betas=True
        ),
        OptimInfo(
            name='radamw',
            opt_class=torch.optim.RAdam,
            description='torch.optim.RAdamW, Rectified Adam with variance adaptation and decoupled weight decay',
            has_betas=True,
            defaults={'decoupled_weight_decay': True}
        ),
        OptimInfo(
            name='adamax',
            opt_class=torch.optim.Adamax,
            description='torch.optim.Adamax, Adam with infinity norm for more stable updates',
            has_betas=True
        ),
        OptimInfo(
            name='adafactor',
            opt_class=Adafactor,
            description='Memory-efficient implementation of Adam with factored gradients',
        ),
        OptimInfo(
            name='adafactorbv',
            opt_class=AdafactorBigVision,
            description='Big Vision variant of Adafactor with factored gradients, half precision momentum',
        ),
        OptimInfo(
            name='adopt',
            opt_class=Adopt,
            description='Modified Adam that can converge with any β2 with the optimal rate',
        ),
        OptimInfo(
            name='adoptw',
            opt_class=Adopt,
            description='Modified AdamW (decoupled decay) that can converge with any β2 with the optimal rate',
            defaults={'decoupled': True}
        ),
    ]
    for opt in adam_optimizers:
        registry.register(opt)


def _register_lamb_lars(registry: OptimizerRegistry) -> None:
    """Register LAMB and LARS variants"""
    lamb_lars_optimizers = [
        OptimInfo(
            name='lamb',
            opt_class=Lamb,
            description='Layer-wise Adaptive Moments for batch optimization',
            has_betas=True
        ),
        OptimInfo(
            name='lambc',
            opt_class=Lamb,
            description='LAMB with trust ratio clipping for stability',
            has_betas=True,
            defaults={'trust_clip': True}
        ),
        OptimInfo(
            name='lambw',
            opt_class=Lamb,
            description='LAMB with decoupled weight decay',
            has_betas=True,
            defaults={'decoupled_decay': True}
        ),
        OptimInfo(
            name='lambcw',
            opt_class=Lamb,
            description='LAMB with trust ratio clipping for stability and decoupled decay',
            has_betas=True,
            defaults={'trust_clip': True, 'decoupled_decay': True}
        ),
        OptimInfo(
            name='lars',
            opt_class=Lars,
            description='Layer-wise Adaptive Rate Scaling',
            has_momentum=True
        ),
        OptimInfo(
            name='larc',
            opt_class=Lars,
            description='LARS with trust ratio clipping for stability',
            has_momentum=True,
            defaults={'trust_clip': True}
        ),
        OptimInfo(
            name='nlars',
            opt_class=Lars,
            description='LARS with Nesterov momentum',
            has_momentum=True,
            defaults={'nesterov': True}
        ),
        OptimInfo(
            name='nlarc',
            opt_class=Lars,
            description='LARS with Nesterov momentum & trust ratio clipping',
            has_momentum=True,
            defaults={'nesterov': True, 'trust_clip': True}
        ),
    ]
    for opt in lamb_lars_optimizers:
        registry.register(opt)


def _register_corrected_decay_optimizers(registry: OptimizerRegistry) -> None:
    """Register corrected weight decay optimizer variants"""
    corrected_optimizers = [
        OptimInfo(
            name='adamc',
            opt_class=AdamWLegacy,
            description='AdamW with corrected weight decay (lr²/max_lr scaling)',
            has_betas=True,
            defaults={'corrected_weight_decay': True}
        ),
        OptimInfo(
            name='nadamc',
            opt_class=NAdamW,
            description='NAdamW with corrected weight decay (lr²/max_lr scaling)',
            has_betas=True,
            defaults={'corrected_weight_decay': True}
        ),
        OptimInfo(
            name='sgdc',
            opt_class=SGDW,
            description='SGD with corrected decoupled weight decay (lr²/max_lr scaling)',
            has_eps=False,
            has_momentum=True,
            defaults={'nesterov': True, 'corrected_weight_decay': True}
        ),
        OptimInfo(
            name='adoptc',
            opt_class=Adopt,
            description='Adopt with corrected decoupled weight decay (lr²/max_lr scaling)',
            defaults={'decoupled': True, 'corrected_weight_decay': True}
        ),
        OptimInfo(
            name='lambcd',
            opt_class=Lamb,
            description='LAMB with corrected decoupled weight decay (lr²/max_lr scaling)',
            has_betas=True,
            defaults={'decoupled_decay': True, 'corrected_weight_decay': True}
        ),
        OptimInfo(
            name='kronc',
            opt_class=Kron,
            description='PSGD Kron with corrected decoupled weight decay (lr²/max_lr scaling)',
            has_momentum=True,
            defaults={'decoupled_decay': True, 'corrected_weight_decay': True}
        ),
        OptimInfo(
            name='lionc',
            opt_class=Lion,
            description='Lion with corrected weight decay (lr²/max_lr scaling)',
            has_eps=False,
            has_betas=True,
            defaults={'corrected_weight_decay': True}
        ),
        OptimInfo(
            name='lapropc',
            opt_class=LaProp,
            description='LaProp with corrected weight decay (lr²/max_lr scaling)',
            has_betas=True,
            defaults={'corrected_weight_decay': True}
        ),
        OptimInfo(
            name='rmsproptfc',
            opt_class=RMSpropTF,
            description='RMSprop TF-style with corrected decoupled weight decay (lr²/max_lr scaling)',
            has_momentum=True,
            defaults={'alpha': 0.9, 'decoupled_decay': True, 'corrected_weight_decay': True}
        ),
        OptimInfo(
            name='adafactorbvc',
            opt_class=AdafactorBigVision,
            description='Adafactor Big Vision with corrected weight decay (lr²/max_lr or lr/max_lr scaling)',
            defaults={'corrected_weight_decay': True}
        ),
    ]
    for opt in corrected_optimizers:
        registry.register(opt)

    # Cautious + corrected variants
    cautious_corrected = [
        OptimInfo(
            name='cadamc',
            opt_class=AdamWLegacy,
            description='Cautious AdamW with corrected weight decay (lr²/max_lr scaling)',
            has_betas=True,
            defaults={'caution': True, 'corrected_weight_decay': True}
        ),
        OptimInfo(
            name='cadoptc',
            opt_class=Adopt,
            description='Cautious Adopt with corrected decoupled weight decay (lr²/max_lr scaling)',
            defaults={'decoupled': True, 'caution': True, 'corrected_weight_decay': True}
        ),
        OptimInfo(
            name='cnadamc',
            opt_class=NAdamW,
            description='Cautious NAdamW with corrected weight decay (lr²/max_lr scaling)',
            has_betas=True,
            defaults={'caution': True, 'corrected_weight_decay': True}
        ),
        OptimInfo(
            name='csgdc',
            opt_class=SGDW,
            description='Cautious SGD with corrected decoupled weight decay (lr²/max_lr scaling)',
            has_eps=False,
            has_momentum=True,
            defaults={'nesterov': True, 'caution': True, 'corrected_weight_decay': True}
        ),
        OptimInfo(
            name='clionc',
            opt_class=Lion,
            description='Cautious Lion with corrected weight decay (lr²/max_lr scaling)',
            has_eps=False,
            has_betas=True,
            defaults={'caution': True, 'corrected_weight_decay': True}
        ),
        OptimInfo(
            name='cadafactorbvc',
            opt_class=AdafactorBigVision,
            description='Cautious Adafactor Big Vision with corrected weight decay',
            defaults={'caution': True, 'corrected_weight_decay': True}
        ),
    ]
    for opt in cautious_corrected:
        registry.register(opt)


def _register_cautious_optimizers(registry: OptimizerRegistry) -> None:
    cautious_optimizers = [
        OptimInfo(
            name='cadafactor',
            opt_class=Adafactor,
            description='Cautious Adafactor',
            defaults={'caution': True}
        ),
        OptimInfo(
            name='cadafactorbv',
            opt_class=AdafactorBigVision,
            description='Cautious Big Vision Adafactor',
            defaults={'caution': True}
        ),
        OptimInfo(
            name='cadamw',
            opt_class=AdamWLegacy,
            description='Cautious AdamW',
            has_betas=True,
            defaults={'caution': True}
        ),
        OptimInfo(
            name='cadopt',
            opt_class=Adopt,
            description='Cautious Adopt',
            defaults={'caution': True}
        ),
        OptimInfo(
            name='cadan',
            opt_class=Adan,
            description='Cautious Adaptive Nesterov Momentum Algorithm',
            defaults={'caution': True, 'no_prox': False},
            has_betas=True,
            num_betas=3
        ),
        OptimInfo(
            name='cadanw',
            opt_class=Adan,
            description='Cautious Adaptive Nesterov Momentum with decoupled weight decay',
            defaults={'caution': True, 'no_prox': True},
            has_betas=True,
            num_betas=3
        ),
        OptimInfo(
            name='cadoptw',
            opt_class=Adopt,
            description='Cautious AdoptW (decoupled decay)',
            defaults={'decoupled': True, 'caution': True}
        ),
        OptimInfo(
            name='clamb',
            opt_class=Lamb,
            description='Cautious LAMB',
            has_betas=True,
            defaults={'caution': True}
        ),
        OptimInfo(
            name='clambw',
            opt_class=Lamb,
            description='Cautious LAMB with decoupled weight decay',
            has_betas=True,
            defaults={'caution': True, 'decoupled_decay': True}
        ),
        OptimInfo(
            name='claprop',
            opt_class=LaProp,
            description='Cautious LaProp',
            has_betas=True,
            defaults={'caution': True}
        ),
        OptimInfo(
            name='clion',
            opt_class=Lion,
            description='Cautious Lion',
            has_eps=False,
            has_betas=True,
            defaults = {'caution': True}
        ),
        OptimInfo(
            name='cmars',
            opt_class=Mars,
            description='Cautious MARS',
            has_betas=True,
            defaults={'caution': True}
        ),
        OptimInfo(
            name='cnadamw',
            opt_class=NAdamW,
            description='Cautious NAdamW',
            has_betas=True,
            defaults={'caution': True}
        ),
        OptimInfo(
            name='crmsproptf',
            opt_class=RMSpropTF,
            description='Cautious TensorFlow-style RMSprop',
            has_momentum=True,
            defaults={'alpha': 0.9, 'caution': True}
        ),
        OptimInfo(
            name='csgdw',
            opt_class=SGDW,
            description='Cautious SGD with decoupled weight decay and Nesterov momentum',
            has_eps=False,
            has_momentum=True,
            defaults={'nesterov': True, 'caution': True}
        ),
    ]
    for opt in cautious_optimizers:
        registry.register(opt)

def _register_other_optimizers(registry: OptimizerRegistry) -> None:
    """Register miscellaneous optimizers"""
    other_optimizers = [
        OptimInfo(
            name='adabelief',
            opt_class=AdaBelief,
            description='Adapts learning rate based on gradient prediction error',
            has_betas=True,
            defaults={'rectify': False}
        ),
        OptimInfo(
            name='radabelief',
            opt_class=AdaBelief,
            description='Rectified AdaBelief with variance adaptation',
            has_betas=True,
            defaults={'rectify': True}
        ),
        OptimInfo(
            name='adadelta',
            opt_class=torch.optim.Adadelta,
            description='torch.optim.Adadelta, Adapts learning rates based on running windows of gradients'
        ),
        OptimInfo(
            name='adagrad',
            opt_class=torch.optim.Adagrad,
            description='torch.optim.Adagrad, Adapts learning rates using cumulative squared gradients',
            defaults={'eps': 1e-8}
        ),
        OptimInfo(
            name='adan',
            opt_class=Adan,
            description='Adaptive Nesterov Momentum Algorithm',
            defaults={'no_prox': False},
            has_betas=True,
            num_betas=3
        ),
        OptimInfo(
            name='adanw',
            opt_class=Adan,
            description='Adaptive Nesterov Momentum with decoupled weight decay',
            defaults={'no_prox': True},
            has_betas=True,
            num_betas=3
        ),
        OptimInfo(
            name='adahessian',
            opt_class=Adahessian,
            description='An Adaptive Second Order Optimizer',
            has_betas=True,
            second_order=True,
        ),
        OptimInfo(
            name='kron',
            opt_class=Kron,
            description='PSGD optimizer with Kronecker-factored preconditioner',
            has_momentum=True,
        ),
        OptimInfo(
            name='kronw',
            opt_class=Kron,
            description='PSGD optimizer with Kronecker-factored preconditioner and decoupled weight decay',
            has_momentum=True,
            defaults={'decoupled_decay': True}
        ),
        OptimInfo(
            name='laprop',
            opt_class=LaProp,
            description='Separating Momentum and Adaptivity in Adam',
            has_betas=True,
        ),
        OptimInfo(
            name='lion',
            opt_class=Lion,
            description='Evolved Sign Momentum optimizer for improved convergence',
            has_eps=False,
            has_betas=True
        ),
        OptimInfo(
            name='madgrad',
            opt_class=MADGRAD,
            description='Momentum-based Adaptive gradient method',
            has_momentum=True
        ),
        OptimInfo(
            name='madgradw',
            opt_class=MADGRAD,
            description='MADGRAD with decoupled weight decay',
            has_momentum=True,
            defaults={'decoupled_decay': True}
        ),
        OptimInfo(
            name='mars',
            opt_class=Mars,
            description='Unleashing the Power of Variance Reduction for Training Large Models',
            has_betas=True,
        ),
        OptimInfo(
            name='muon',
            opt_class=Muon,
            description='MomentUm Orthogonalized by Newton-schulz with AdamW fallback for 1D params',
            has_momentum=True,
            has_eps=True,
            has_betas=True,
        ),
        OptimInfo(
            name='nmuon',
            opt_class=Muon,
            description='MomentUm Orthogonalized by Newton-schulz with Nesterov and NAdamW fallback for 1D params',
            has_momentum=True,
            has_eps=True,
            has_betas=True,
            defaults={'nesterov': True}
        ),
        OptimInfo(
            name='novograd',
            opt_class=NvNovoGrad,
            description='Normalized Adam with L2 norm gradient normalization',
            has_betas=True
        ),
        OptimInfo(
            name='rmsprop',
            opt_class=torch.optim.RMSprop,
            description='torch.optim.RMSprop, Root Mean Square Propagation',
            has_momentum=True,
            defaults={'alpha': 0.9}
        ),
        OptimInfo(
            name='rmsproptf',
            opt_class=RMSpropTF,
            description='TensorFlow-style RMSprop implementation, Root Mean Square Propagation',
            has_momentum=True,
            defaults={'alpha': 0.9}
        ),
    ]
    for opt in other_optimizers:
        registry.register(opt)
    registry.register_foreach_default('lion')


def _register_apex_optimizers(registry: OptimizerRegistry) -> None:
    """Register APEX optimizers (lazy import)"""
    apex_optimizers = [
        OptimInfo(
            name='fusedsgd',
            opt_class='apex.optimizers.FusedSGD',
            description='NVIDIA APEX fused SGD implementation for faster training',
            has_eps=False,
            has_momentum=True,
            defaults={'nesterov': True}
        ),
        OptimInfo(
            name='fusedadam',
            opt_class='apex.optimizers.FusedAdam',
            description='NVIDIA APEX fused Adam implementation',
            has_betas=True,
            defaults={'adam_w_mode': False}
        ),
        OptimInfo(
            name='fusedadamw',
            opt_class='apex.optimizers.FusedAdam',
            description='NVIDIA APEX fused AdamW implementation',
            has_betas=True,
            defaults={'adam_w_mode': True}
        ),
        OptimInfo(
            name='fusedlamb',
            opt_class='apex.optimizers.FusedLAMB',
            description='NVIDIA APEX fused LAMB implementation',
            has_betas=True
        ),
        OptimInfo(
            name='fusednovograd',
            opt_class='apex.optimizers.FusedNovoGrad',
            description='NVIDIA APEX fused NovoGrad implementation',
            has_betas=True,
            defaults={'betas': (0.95, 0.98)}
        ),
    ]
    for opt in apex_optimizers:
        registry.register(opt)


def _register_bnb_optimizers(registry: OptimizerRegistry) -> None:
    """Register bitsandbytes optimizers (lazy import)"""
    bnb_optimizers = [
        OptimInfo(
            name='bnbsgd',
            opt_class='bitsandbytes.optim.SGD',
            description='bitsandbytes SGD',
            has_eps=False,
            has_momentum=True,
            defaults={'nesterov': True}
        ),
        OptimInfo(
            name='bnbsgd8bit',
            opt_class='bitsandbytes.optim.SGD8bit',
            description='bitsandbytes 8-bit SGD with dynamic quantization',
            has_eps=False,
            has_momentum=True,
            defaults={'nesterov': True}
        ),
        OptimInfo(
            name='bnbadam',
            opt_class='bitsandbytes.optim.Adam',
            description='bitsandbytes Adam',
            has_betas=True
        ),
        OptimInfo(
            name='bnbadam8bit',
            opt_class='bitsandbytes.optim.Adam',
            description='bitsandbytes 8-bit Adam with dynamic quantization',
            has_betas=True
        ),
        OptimInfo(
            name='bnbadamw',
            opt_class='bitsandbytes.optim.AdamW',
            description='bitsandbytes AdamW',
            has_betas=True
        ),
        OptimInfo(
            name='bnbadamw8bit',
            opt_class='bitsandbytes.optim.AdamW',
            description='bitsandbytes 8-bit AdamW with dynamic quantization',
            has_betas=True
        ),
        OptimInfo(
            'bnblion',
            'bitsandbytes.optim.Lion',
            description='bitsandbytes Lion',
            has_eps=False,
            has_betas=True
        ),
        OptimInfo(
            'bnblion8bit',
            'bitsandbytes.optim.Lion8bit',
            description='bitsandbytes 8-bit Lion with dynamic quantization',
            has_eps=False,
            has_betas=True
        ),
        OptimInfo(
            'bnbademamix',
            'bitsandbytes.optim.AdEMAMix',
            description='bitsandbytes AdEMAMix',
            has_betas=True,
            num_betas=3,
        ),
        OptimInfo(
            'bnbademamix8bit',
            'bitsandbytes.optim.AdEMAMix8bit',
            description='bitsandbytes 8-bit AdEMAMix with dynamic quantization',
            has_betas=True,
            num_betas=3,
        ),
    ]
    for opt in bnb_optimizers:
        registry.register(opt)


default_registry = OptimizerRegistry()

def _register_default_optimizers() -> None:
    """Register all default optimizers to the global registry."""
    # Register all optimizer groups
    _register_sgd_variants(default_registry)
    _register_adam_variants(default_registry)
    _register_lamb_lars(default_registry)
    _register_other_optimizers(default_registry)
    _register_apex_optimizers(default_registry)
    _register_bnb_optimizers(default_registry)
    _register_cautious_optimizers(default_registry)
    _register_corrected_decay_optimizers(default_registry)

    # Register aliases
    default_registry.register_alias('nesterov', 'sgd')
    default_registry.register_alias('nesterovw', 'sgdw')


# Initialize default registry
_register_default_optimizers()

# Public API

def list_optimizers(
        filter: Union[str, List[str]] = '',
        exclude_filters: Optional[List[str]] = None,
        with_description: bool = False,
) -> List[Union[str, Tuple[str, str]]]:
    """List available optimizer names, optionally filtered.

    List all registered optimizers, with optional filtering using wildcard patterns.
    Optimizers can be filtered using include and exclude patterns, and can optionally
    return descriptions with each optimizer name.

    Args:
        filter: Wildcard style filter string or list of filter strings
            (e.g., 'adam*' for all Adam variants, or ['adam*', '*8bit'] for
            Adam variants and 8-bit optimizers). Empty string means no filtering.
        exclude_filters: Optional list of wildcard patterns to exclude. For example,
            ['*8bit', 'fused*'] would exclude 8-bit and fused implementations.
        with_description: If True, returns tuples of (name, description) instead of
            just names. Descriptions provide brief explanations of optimizer characteristics.

    Returns:
        If with_description is False:
            List of optimizer names as strings (e.g., ['adam', 'adamw', ...])
        If with_description is True:
            List of tuples of (name, description) (e.g., [('adam', 'Adaptive Moment...'), ...])

    Examples:
        >>> list_optimizers()
        ['adam', 'adamw', 'sgd', ...]

        >>> list_optimizers(['la*', 'nla*'])  # List lamb & lars
        ['lamb', 'lambc', 'larc', 'lars', 'nlarc', 'nlars']

        >>> list_optimizers('*adam*', exclude_filters=['bnb*', 'fused*'])  # Exclude bnb & apex adam optimizers
        ['adam', 'adamax', 'adamp', 'adamw', 'nadam', 'nadamw', 'radam']

        >>> list_optimizers(with_description=True)  # Get descriptions
        [('adabelief', 'Adapts learning rate based on gradient prediction error'),
         ('adadelta', 'torch.optim Adadelta, Adapts learning rates based on running windows of gradients'),
         ('adafactor', 'Memory-efficient implementation of Adam with factored gradients'),
        ...]
    """
    return default_registry.list_optimizers(filter, exclude_filters, with_description)


def get_optimizer_info(name: str) -> OptimInfo:
    """Get the OptimInfo for an optimizer.

    Args:
        name: Name of the optimizer

    Returns:
        OptimInfo configuration

    Raises:
        ValueError: If optimizer is not found
    """
    return default_registry.get_optimizer_info(name)


def get_optimizer_class(
        name: str,
        bind_defaults: bool = True,
) -> Union[OptimType, OptimizerCallable]:
    """Get optimizer class by name with option to bind default arguments.

    Retrieves the optimizer class or a partial function with default arguments bound.
    This allows direct instantiation of optimizers with their default configurations
    without going through the full factory.

    Args:
        name: Name of the optimizer to retrieve (e.g., 'adam', 'sgd')
        bind_defaults: If True, returns a partial function with default arguments from OptimInfo bound.
            If False, returns the raw optimizer class.

    Returns:
        If bind_defaults is False:
            The optimizer class (e.g., torch.optim.Adam)
        If bind_defaults is True:
            A partial function with default arguments bound

    Raises:
        ValueError: If optimizer name is not found in registry

    Examples:
        >>> # Get SGD with nesterov momentum default
        >>> SGD = get_optimizer_class('sgd')  # nesterov=True bound
        >>> opt = SGD(model.parameters(), lr=0.1, momentum=0.9)

        >>> # Get raw optimizer class
        >>> SGD = get_optimizer_class('sgd')
        >>> opt = SGD(model.parameters(), lr=1e-3, momentum=0.9)

    """
    return default_registry.get_optimizer_class(name, bind_defaults=bind_defaults)


def create_optimizer_v2(
        model_or_params: Union[nn.Module, ParamsT],
        opt: str = 'sgd',
        lr: Optional[float] = None,
        weight_decay: float = 0.,
        momentum: float = 0.9,
        foreach: Optional[bool] = None,
        filter_bias_and_bn: bool = True,
        fallback_list: Collection[str] = (),
        fallback_no_weight_decay: bool = False,
        layer_decay: Optional[float] = None,
        layer_decay_min_scale: float = 0.0,
        layer_decay_no_opt_scale: Optional[float] = None,
        param_group_fn: Optional[Callable[[nn.Module], ParamsT]] = None,
        **kwargs: Any,
) -> torch.optim.Optimizer:
    """Create an optimizer instance via timm registry.

    Creates and configures an optimizer with appropriate parameter groups and settings.
    Supports automatic parameter group creation for weight decay and layer-wise learning
    rates, as well as custom parameter grouping.

    Args:
        model_or_params: A PyTorch model or an iterable of parameters/parameter groups.
            If a model is provided, parameters will be automatically extracted and grouped
            based on the other arguments.
        opt: Name of the optimizer to create (e.g., 'adam', 'adamw', 'sgd').
            Use list_optimizers() to see available options.
        lr: Learning rate. If None, will use the optimizer's default.
        weight_decay: Weight decay factor. Will be used to create param groups if model_or_params is a model.
        momentum: Momentum factor for optimizers that support it. Only used if the
            chosen optimizer accepts a momentum parameter.
        foreach: Enable/disable foreach (multi-tensor) implementation if available.
            If None, will use optimizer-specific defaults.
        filter_bias_and_bn: If True, bias, norm layer parameters (all 1d params) will not have
            weight decay applied. Only used when model_or_params is a model and
            weight_decay > 0.
        fallback_list: Collection of parameter name patterns to use fallback optimizer for
            hybrid optimizers (e.g., AdamW for Muon). Supports wildcard matching.
        fallback_no_weight_decay: If True, params in model's no_weight_decay() list will use
            fallback optimizer for hybrid optimizers (e.g., AdamW for Muon).
        layer_decay: Optional layer-wise learning rate decay factor. If provided,
            learning rates will be scaled by layer_decay^(max_depth - layer_depth).
            Only used when model_or_params is a model.
        param_group_fn: Optional function to create custom parameter groups.
            If provided, other parameter grouping options will be ignored.
        **kwargs: Additional optimizer-specific arguments (e.g., betas for Adam).

    Returns:
        Configured optimizer instance.

    Examples:
        >>> # Basic usage with a model
        >>> optimizer = create_optimizer_v2(model, 'adamw', lr=1e-3)

        >>> # SGD with momentum and weight decay
        >>> optimizer = create_optimizer_v2(
        ...     model, 'sgd', lr=0.1, momentum=0.9, weight_decay=1e-4
        ... )

        >>> # Adam with layer-wise learning rate decay
        >>> optimizer = create_optimizer_v2(
        ...     model, 'adam', lr=1e-3, layer_decay=0.7
        ... )

        >>> # Custom parameter groups
        >>> def group_fn(model):
        ...     return [
        ...         {'params': model.backbone.parameters(), 'lr': 1e-4},
        ...         {'params': model.head.parameters(), 'lr': 1e-3}
        ...     ]
        >>> optimizer = create_optimizer_v2(
        ...     model, 'sgd', param_group_fn=group_fn
        ... )

    Note:
        Parameter group handling precedence:
        1. If param_group_fn is provided, it will be used exclusively
        2. If layer_decay is provided, layer-wise groups will be created
        3. If weight_decay > 0 and filter_bias_and_bn is True, weight decay groups will be created
        4. Otherwise, all parameters will be in a single group
    """

    return default_registry.create_optimizer(
        model_or_params,
        opt=opt,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        foreach=foreach,
        weight_decay_exclude_1d=filter_bias_and_bn,
        fallback_list=fallback_list,
        fallback_no_weight_decay=fallback_no_weight_decay,
        layer_decay=layer_decay,
        layer_decay_min_scale=layer_decay_min_scale,
        layer_decay_no_opt_scale=layer_decay_no_opt_scale,
        param_group_fn=param_group_fn,
        **kwargs
    )


def optimizer_kwargs(cfg):
    """Convert argparse-style `cfg` object to kwargs for an optimizer factory."""
    kwargs = {
        'opt':           cfg.opt,
        'lr':            cfg.lr,
        'weight_decay':  cfg.weight_decay,
        'momentum':      cfg.momentum,
    }
    if (eps := getattr(cfg, 'opt_eps', None)) is not None:
        kwargs['eps'] = eps
    if (betas := getattr(cfg, 'opt_betas', None)) is not None:
        kwargs['betas'] = betas
    if (layer_decay := getattr(cfg, 'layer_decay', None)) is not None:
        kwargs['layer_decay'] = layer_decay
    if (ld_min := getattr(cfg, 'layer_decay_min_scale', None)) is not None:
        kwargs['layer_decay_min_scale'] = ld_min
    if (ld_no_opt := getattr(cfg, 'layer_decay_no_opt_scale', None)) is not None:
        kwargs['layer_decay_no_opt_scale'] = ld_no_opt
    if (opt_args := getattr(cfg, 'opt_args', None)) is not None:
        kwargs.update(opt_args)
    if (foreach := getattr(cfg, 'opt_foreach', None)) is not None:
        kwargs['foreach'] = foreach

    return kwargs


def create_optimizer(
        args,
        model: Union[nn.Module, ParamsT],
        filter_bias_and_bn: bool = True,
) -> torch.optim.Optimizer:
    """ Legacy optimizer factory for backwards compatibility.
    NOTE: Use create_optimizer_v2 for new code.
    """
    return create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        filter_bias_and_bn=filter_bias_and_bn,
    )

