from .cosine_lr import CosineLRScheduler
from .tanh_lr import TanhLRScheduler
from .step_lr import StepLRScheduler


def create_scheduler(args, optimizer):
    num_epochs = args.epochs
    #FIXME expose cycle parms of the scheduler config to arguments
    if args.sched == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            t_mul=1.0,
            lr_min=1e-5,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            cycle_limit=1,
            t_in_epochs=True,
        )
        num_epochs = lr_scheduler.get_cycle_length() + 10
    elif args.sched == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=num_epochs,
            t_mul=1.0,
            lr_min=1e-5,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            cycle_limit=1,
            t_in_epochs=True,
        )
        num_epochs = lr_scheduler.get_cycle_length() + 10
    else:
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=args.decay_epochs,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
        )
    return lr_scheduler, num_epochs
