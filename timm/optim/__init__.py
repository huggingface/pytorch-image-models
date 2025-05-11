from .adabelief import AdaBelief
from .adafactor import Adafactor
from .adafactor_bv import AdafactorBigVision
from .adahessian import Adahessian
from .adamp import AdamP
from .adamw import AdamWLegacy
from .adan import Adan
from .adopt import Adopt
from .lamb import Lamb
from .laprop import LaProp
from .lars import Lars
from .lion import Lion
from .lookahead import Lookahead
from .madgrad import MADGRAD
from .mars import Mars
from .nadam import NAdamLegacy
from .nadamw import NAdamW
from .nvnovograd import NvNovoGrad
from .radam import RAdamLegacy
from .rmsprop_tf import RMSpropTF
from .sgdp import SGDP
from .sgdw import SGDW

# bring common torch.optim Optimizers into timm.optim namespace for consistency
from torch.optim import Adadelta, Adagrad, Adamax, Adam, AdamW, RMSprop, SGD
try:
    # in case any very old torch versions being used
    from torch.optim import NAdam, RAdam
except ImportError:
    pass

from ._optim_factory import list_optimizers, get_optimizer_class, get_optimizer_info, OptimInfo, OptimizerRegistry, \
    create_optimizer_v2, create_optimizer, optimizer_kwargs
from ._param_groups import param_groups_layer_decay, param_groups_weight_decay, auto_group_layers
