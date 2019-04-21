from torch import optim as optim
from optim import Nadam, AdaBound, RMSpropTF


def create_optimizer(args, parameters):
    if args.opt.lower() == 'sgd':
        optimizer = optim.SGD(
            parameters, lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif args.opt.lower() == 'adam':
        optimizer = optim.Adam(
            parameters, lr=args.lr, weight_decay=args.weight_decay, eps=args.opt_eps)
    elif args.opt.lower() == 'nadam':
        optimizer = Nadam(
            parameters, lr=args.lr, weight_decay=args.weight_decay, eps=args.opt_eps)
    elif args.opt.lower() == 'adabound':
        optimizer = AdaBound(
            parameters, lr=args.lr / 100, weight_decay=args.weight_decay, eps=args.opt_eps,
            final_lr=args.lr)
    elif args.opt.lower() == 'adadelta':
        optimizer = optim.Adadelta(
            parameters, lr=args.lr, weight_decay=args.weight_decay, eps=args.opt_eps)
    elif args.opt.lower() == 'rmsprop':
        optimizer = optim.RMSprop(
            parameters, lr=args.lr, alpha=0.9, eps=args.opt_eps,
            momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt.lower() == 'rmsproptf':
        optimizer = RMSpropTF(
            parameters, lr=args.lr, alpha=0.9, eps=args.opt_eps,
            momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        assert False and "Invalid optimizer"
        raise ValueError
    return optimizer
