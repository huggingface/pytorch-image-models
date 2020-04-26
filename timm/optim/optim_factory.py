import torch
from torch import optim as optim
from timm.optim import Nadam, RMSpropTF, AdamW, RAdam, NovoGrad, NvNovoGrad, Lookahead
try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def create_optimizer(args, model, filter_bias_and_bn=True):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if 'adamw' in opt_lower or 'radam' in opt_lower:
        # Compensate for the way current AdamW and RAdam optimizers apply LR to the weight-decay
        # I don't believe they follow the paper or original Torch7 impl which schedules weight
        # decay based on the ratio of current_lr/initial_lr
        weight_decay /= args.lr
    if weight_decay and filter_bias_and_bn:
        parameters = add_weight_decay(model, weight_decay)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        optimizer = optim.SGD(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay, nesterov=True)
    elif opt_lower == 'momentum':
        optimizer = optim.SGD(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay, nesterov=False)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(
            parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'adamw':
        optimizer = AdamW(
            parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'nadam':
        optimizer = Nadam(
            parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'radam':
        optimizer = RAdam(
            parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(
            parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(
            parameters, lr=args.lr, alpha=0.9, eps=args.opt_eps,
            momentum=args.momentum, weight_decay=weight_decay)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(
            parameters, lr=args.lr, alpha=0.9, eps=args.opt_eps,
            momentum=args.momentum, weight_decay=weight_decay)
    elif opt_lower == 'novograd':
        optimizer = NovoGrad(parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'fusedsgd':
        optimizer = FusedSGD(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay, nesterov=True)
    elif opt_lower == 'fusedmomentum':
        optimizer = FusedSGD(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay, nesterov=False)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(
            parameters, lr=args.lr, adam_w_mode=False, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(
            parameters, lr=args.lr, adam_w_mode=True, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
    elif opt_lower == 'fusednovograd':
        optimizer = FusedNovoGrad(
            parameters, lr=args.lr, betas=(0.95, 0.98), weight_decay=weight_decay, eps=args.opt_eps)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer
