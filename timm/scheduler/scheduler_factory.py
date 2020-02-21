from .cosine_lr import CosineLRScheduler
from .tanh_lr import TanhLRScheduler
from .step_lr import StepLRScheduler


def create_scheduler(args, optimizer):
    num_epochs = args.epochs
    lr_scheduler = None
    #FIXME expose cycle parms of the scheduler config to arguments
    if args.sched == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            t_mul=1.0,
            lr_min=args.min_lr,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            cycle_limit=1,
            t_in_epochs=True,
        )
        num_epochs = lr_scheduler.get_cycle_length() + args.cooldown_epochs
    elif args.sched == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=num_epochs,
            t_mul=1.0,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            cycle_limit=1,
            t_in_epochs=True,
        )
        num_epochs = lr_scheduler.get_cycle_length() + args.cooldown_epochs
    elif args.sched == 'step':
        if isinstance(args.lr_noise, (list, tuple)):
            noise_range = [n * num_epochs for n in args.lr_noise]
        else:
            noise_range = args.lr_noise * num_epochs
        print(noise_range)
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=args.decay_epochs,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            noise_range_t=noise_range,
            noise_std=args.lr_noise_std,
        )
    return lr_scheduler, num_epochs
