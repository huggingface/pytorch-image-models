# lots of uses of these functions directly, ala 'import timm.optim.optim_factory as optim_factory', fun :/

from ._optim_factory import create_optimizer, create_optimizer_v2, optimizer_kwargs
from ._param_groups import param_groups_layer_decay, param_groups_weight_decay, group_parameters, _layer_map, _group

import warnings
warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.optim", FutureWarning)
