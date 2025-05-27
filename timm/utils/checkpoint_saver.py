""" Checkpoint Saver

Track top-n training checkpoints and maintain recovery checkpoints on specified intervals.

Hacked together by / Copyright 2020 Ross Wightman
"""

import glob
import logging
import operator
import os
import shutil

import torch

from .model import unwrap_model, get_state_dict


_logger = logging.getLogger(__name__)


class CheckpointSaver:
    def __init__(
            self,
            model,
            optimizer,
            args=None,
            model_ema=None,
            amp_scaler=None,
            checkpoint_prefix='checkpoint',
            recovery_prefix='recovery',
            checkpoint_dir='',
            recovery_dir='',
            decreasing=False,
            max_history=10,
            unwrap_fn=unwrap_model
    ):

        # objects to save state_dicts of
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.model_ema = model_ema
        self.amp_scaler = amp_scaler

        # state
        self.checkpoint_files = []  # (filename, metric) tuples in order of decreasing betterness
        self.best_epoch = None
        self.best_metric = None
        self.curr_recovery_file = ''
        self.prev_recovery_file = ''
        self.can_hardlink = True

        # config
        self.checkpoint_dir = checkpoint_dir
        self.recovery_dir = recovery_dir
        self.save_prefix = checkpoint_prefix
        self.recovery_prefix = recovery_prefix
        self.extension = '.pth.tar'
        self.decreasing = decreasing  # a lower metric is better if True
        self.cmp = operator.lt if decreasing else operator.gt  # True if lhs better than rhs
        self.max_history = max_history
        self.unwrap_fn = unwrap_fn
        assert self.max_history >= 1

    def _replace(self, src, dst):
        if self.can_hardlink:
            try:
                if os.path.exists(dst):
                    os.unlink(dst)  # required for Windows support.
            except (OSError, NotImplementedError) as e:
                self.can_hardlink = False
        os.replace(src, dst)

    def _duplicate(self, src, dst):
        if self.can_hardlink:
            try:
                if os.path.exists(dst):
                    # for Windows
                    os.unlink(dst)
                os.link(src, dst)
                return
            except (OSError, NotImplementedError) as e:
                self.can_hardlink = False
        shutil.copy2(src, dst)

    def _save(self, save_path, epoch, metric=None):
        save_state = {
            'epoch': epoch,
            'arch': type(self.model).__name__.lower(),
            'state_dict': get_state_dict(self.model, self.unwrap_fn),
            'optimizer': self.optimizer.state_dict(),
            'version': 2,  # version < 2 increments epoch before save
        }
        if self.args is not None:
            save_state['arch'] = self.args.model
            save_state['args'] = self.args
        if self.amp_scaler is not None:
            save_state[self.amp_scaler.state_dict_key] = self.amp_scaler.state_dict()
        if self.model_ema is not None:
            save_state['state_dict_ema'] = get_state_dict(self.model_ema, self.unwrap_fn)
        if metric is not None:
            save_state['metric'] = metric
        torch.save(save_state, save_path)

    def _cleanup_checkpoints(self, trim=0):
        trim = min(len(self.checkpoint_files), trim)
        delete_index = self.max_history - trim
        if delete_index < 0 or len(self.checkpoint_files) <= delete_index:
            return
        to_delete = self.checkpoint_files[delete_index:]
        for d in to_delete:
            try:
                _logger.debug("Cleaning checkpoint: {}".format(d))
                os.remove(d[0])
            except Exception as e:
                _logger.error("Exception '{}' while deleting checkpoint".format(e))
        self.checkpoint_files = self.checkpoint_files[:delete_index]

    def save_checkpoint(self, epoch, metric=None):
        assert epoch >= 0
        tmp_save_path = os.path.join(self.checkpoint_dir, 'tmp' + self.extension)
        last_save_path = os.path.join(self.checkpoint_dir, 'last' + self.extension)
        self._save(tmp_save_path, epoch, metric)
        self._replace(tmp_save_path, last_save_path)

        worst_file = self.checkpoint_files[-1] if self.checkpoint_files else None
        if (
            len(self.checkpoint_files) < self.max_history
            or metric is None
            or self.cmp(metric, worst_file[1])
        ):
            if len(self.checkpoint_files) >= self.max_history:
                self._cleanup_checkpoints(1)
            filename = '-'.join([self.save_prefix, str(epoch)]) + self.extension
            save_path = os.path.join(self.checkpoint_dir, filename)
            self._duplicate(last_save_path, save_path)

            self.checkpoint_files.append((save_path, metric))
            self.checkpoint_files = sorted(
                self.checkpoint_files,
                key=lambda x: x[1],
                reverse=not self.decreasing  # sort in descending order if a lower metric is not better
            )

            checkpoints_str = "Current checkpoints:\n"
            for c in self.checkpoint_files:
                checkpoints_str += ' {}\n'.format(c)
            _logger.info(checkpoints_str)

            if metric is not None and (self.best_metric is None or self.cmp(metric, self.best_metric)):
                self.best_epoch = epoch
                self.best_metric = metric
                best_save_path = os.path.join(self.checkpoint_dir, 'model_best' + self.extension)
                self._duplicate(last_save_path, best_save_path)

        return (None, None) if self.best_metric is None else (self.best_metric, self.best_epoch)

    def save_recovery(self, epoch, batch_idx=0):
        assert epoch >= 0
        tmp_save_path = os.path.join(self.recovery_dir, 'recovery_tmp' + self.extension)
        self._save(tmp_save_path, epoch)

        filename = '-'.join([self.recovery_prefix, str(epoch), str(batch_idx)]) + self.extension
        save_path = os.path.join(self.recovery_dir, filename)
        self._replace(tmp_save_path, save_path)

        if os.path.exists(self.prev_recovery_file):
            try:
                _logger.debug("Cleaning recovery: {}".format(self.prev_recovery_file))
                os.remove(self.prev_recovery_file)
            except Exception as e:
                _logger.error("Exception '{}' while removing {}".format(e, self.prev_recovery_file))
        self.prev_recovery_file = self.curr_recovery_file
        self.curr_recovery_file = save_path

    def find_recovery(self):
        recovery_path = os.path.join(self.recovery_dir, self.recovery_prefix)
        files = glob.glob(recovery_path + '*' + self.extension)
        files = sorted(files)
        return files[0] if len(files) else ''
