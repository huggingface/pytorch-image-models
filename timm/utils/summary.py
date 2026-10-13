""" Summary utilities

Hacked together by / Copyright 2020 Ross Wightman
"""
import csv
import logging
import os
from collections import OrderedDict
try: 
    import wandb
except ImportError:
    pass


_logger = logging.getLogger(__name__)


def get_outdir(path, *paths, inc=False):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '-' + str(count)
            assert count < 100
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir


def trim_summary(filename, start_epoch):
    """Drop summary rows at or after ``start_epoch`` so a run resumed into its own output dir doesn't duplicate them.

    A fresh run (``start_epoch == 0``) into a directory with an old summary starts a new file.
    """
    if not os.path.exists(filename):
        return
    with open(filename, mode='r', newline='') as cf:
        rows = list(csv.reader(cf))
    if not rows:
        return
    header, body = rows[0], rows[1:]
    keep = [r for r in body if r and r[0] != 'epoch' and int(float(r[0])) < start_epoch]
    if len(keep) == len(body):
        return
    _logger.info(
        f'Dropping {len(body) - len(keep)} rows at or after epoch {start_epoch} from {filename}'
        + (' (new run into an existing output dir).' if start_epoch == 0 else '.'))
    with open(filename, mode='w', newline='') as cf:
        writer = csv.writer(cf)
        if keep:
            writer.writerow(header)
            writer.writerows(keep)


def update_summary(
        epoch,
        train_metrics,
        eval_metrics,
        filename,
        lr=None,
        write_header=None,
        log_wandb=False,
):
    """Append an epoch's metrics to the summary CSV. By default a header is written when the file is new or empty."""
    rowd = OrderedDict(epoch=epoch)
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    if eval_metrics:
        rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    if lr is not None:
        rowd['lr'] = lr
    if log_wandb:
        wandb.log(rowd)
    if write_header is None:
        write_header = not os.path.exists(filename) or os.path.getsize(filename) == 0
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:
            dw.writeheader()
        dw.writerow(rowd)
