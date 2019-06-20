from torch import optim as optim
from timm.optim import Nadam, RMSpropTF


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
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        parameters = add_weight_decay(model, weight_decay)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if args.opt.lower() == 'sgd':
        optimizer = optim.SGD(
            parameters, lr=args.lr,
            momentum=args.momentum, weight_decay=weight_decay, nesterov=True)
    elif args.opt.lower() == 'adam':
        optimizer = optim.Adam(
            parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
    elif args.opt.lower() == 'nadam':
        optimizer = Nadam(
            parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
    elif args.opt.lower() == 'adadelta':
        optimizer = optim.Adadelta(
            parameters, lr=args.lr, weight_decay=weight_decay, eps=args.opt_eps)
    elif args.opt.lower() == 'rmsprop':
        optimizer = optim.RMSprop(
            parameters, lr=args.lr, alpha=0.9, eps=args.opt_eps,
            momentum=args.momentum, weight_decay=weight_decay)
    elif args.opt.lower() == 'rmsproptf':
        optimizer = RMSpropTF(
            parameters, lr=args.lr, alpha=0.9, eps=args.opt_eps,
            momentum=args.momentum, weight_decay=weight_decay)
    else:
        assert False and "Invalid optimizer"
        raise ValueError
    return optimizer
