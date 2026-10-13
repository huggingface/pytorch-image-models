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


def load_checkpoint_metadata(checkpoint_path):
    """Load a training checkpoint for its non-tensor entries (epoch, args, metric, saver state).

    Tensors are memory mapped where the torch version allows, so this is cheap even for large checkpoints.
    """
    # local import to avoid a circular import (timm.utils -> timm.models)
    from timm.models._helpers import _torch_load
    return _torch_load(checkpoint_path, map_location='cpu', weights_only=True, mmap=True)


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
            unwrap_fn=unwrap_model,
            task=None,
    ):

        # objects to save state_dicts of
        self.model = model
        self.task = task
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
            'optimizer': self.optimizer.state_dict(),
            'version': 2,  # version < 2 increments epoch before save
        }
        if self.args is not None:
            save_state['arch'] = self.args.model
            save_state['args'] = self.args
        if self.amp_scaler is not None:
            save_state[self.amp_scaler.state_dict_key] = self.amp_scaler.state_dict()
        if self.task is not None:
            save_state.update(self.task.get_checkpoint_state(unwrap_fn=self.unwrap_fn))
            save_state.update(self.task.get_checkpoint_state(ema=True, unwrap_fn=self.unwrap_fn))
        else:
            save_state['state_dict'] = get_state_dict(self.model, self.unwrap_fn)
            if self.model_ema is not None:
                save_state['state_dict_ema'] = get_state_dict(self.model_ema, self.unwrap_fn)
        if metric is not None:
            save_state['metric'] = metric
        save_state['saver'] = self.state_dict(epoch, metric)
        torch.save(save_state, save_path)

    def state_dict(self, epoch=None, metric=None):
        """Bookkeeping needed to continue checkpoint management after a resume into the same directory.

        Args:
            epoch: Epoch of the checkpoint being written, it is not yet in the history when the state is captured.
            metric: Metric of the checkpoint being written.
        """
        return {
            'checkpoint_files': [(os.path.basename(f), m) for f, m in self.checkpoint_files],
            'best_epoch': self.best_epoch,
            'best_metric': self.best_metric,
            'pending': (epoch, metric),
        }

    def load_state_dict(self, state_dict):
        """Restore checkpoint history and best metric tracking from a checkpoint saved into ``checkpoint_dir``.

        History entries whose files no longer exist are dropped. The checkpoint the state was captured from is added
        to the history if it was retained on disk, so a resumed run keeps evicting and ranking exactly as before.
        """
        self.checkpoint_files = [
            (os.path.join(self.checkpoint_dir, f), m) for f, m in state_dict.get('checkpoint_files', [])
            if os.path.exists(os.path.join(self.checkpoint_dir, f))
        ]
        self.best_epoch = state_dict.get('best_epoch')
        self.best_metric = state_dict.get('best_metric')
        epoch, metric = state_dict.get('pending', (None, None))
        if epoch is not None:
            save_path = self._checkpoint_path(epoch)
            if os.path.exists(save_path) and save_path not in [f for f, _ in self.checkpoint_files]:
                self._track_checkpoint(save_path, epoch, metric, update_best_file=False)
        _logger.info(f'Restored checkpoint history ({len(self.checkpoint_files)} files), best metric '
                     f'{self.best_metric} (epoch {self.best_epoch}).')

    def load_state_dict_from_checkpoint(self, checkpoint_path):
        """Restore bookkeeping from a checkpoint file, reading only its metadata where the torch version allows."""
        try:
            checkpoint = load_checkpoint_metadata(checkpoint_path)
        except RuntimeError as e:  # e.g. torch too old to load a training checkpoint with weights_only=True
            _logger.warning(f"Could not read checkpoint history from '{checkpoint_path}' ({e}), starting fresh.")
            return False
        state_dict = checkpoint.get('saver') if isinstance(checkpoint, dict) else None
        if state_dict is None:
            _logger.warning(f"No checkpoint history in '{checkpoint_path}', ranked checkpoints start fresh.")
            return False
        self.load_state_dict(state_dict)
        return True

    def _checkpoint_path(self, epoch):
        return os.path.join(self.checkpoint_dir, '-'.join([self.save_prefix, str(epoch)]) + self.extension)

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
        """Save the current state as ``last`` and, if it earns a place, as a numbered epoch checkpoint.

        Ranked checkpoints (``metric`` given) are kept best to worst, with ties keeping the earlier epoch. Unranked
        checkpoints (``metric`` is None) follow them, newest first, so without a metric the history contains the last
        ``max_history`` epochs. An unranked checkpoint never displaces a ranked one, while a ranked checkpoint always
        displaces an unranked one when the history is full.

        Args:
            epoch: Epoch index of the state being saved.
            metric: Value used to rank the checkpoint, or None to keep it by recency only.

        Returns:
            Tuple of (best_metric, best_epoch), both None until a ranked checkpoint has been saved.
        """
        assert epoch >= 0
        tmp_save_path = os.path.join(self.checkpoint_dir, 'tmp' + self.extension)
        last_save_path = os.path.join(self.checkpoint_dir, 'last' + self.extension)
        self._save(tmp_save_path, epoch, metric)
        self._replace(tmp_save_path, last_save_path)

        worst_file = self.checkpoint_files[-1] if self.checkpoint_files else None
        if (
            len(self.checkpoint_files) < self.max_history
            or worst_file[1] is None
            or (metric is not None and self.cmp(metric, worst_file[1]))
        ):
            if len(self.checkpoint_files) >= self.max_history:
                self._cleanup_checkpoints(1)
            save_path = self._checkpoint_path(epoch)
            self._duplicate(last_save_path, save_path)
            self._track_checkpoint(save_path, epoch, metric)

        return (None, None) if self.best_metric is None else (self.best_metric, self.best_epoch)

    def _track_checkpoint(self, save_path, epoch, metric=None, update_best_file=True):
        """Add a retained checkpoint to the ranked history and update the best metric tracking."""
        self.checkpoint_files.append((save_path, metric))
        ranked_files = sorted(
            [c for c in self.checkpoint_files if c[1] is not None],
            key=lambda c: c[1],
            reverse=not self.decreasing  # sort in descending order if a lower metric is not better
        )
        unranked_files = [c for c in self.checkpoint_files if c[1] is None]
        if metric is None:
            # Existing unranked entries are already newest first; move only the newly appended entry.
            unranked_files.insert(0, unranked_files.pop())
        self.checkpoint_files = ranked_files + unranked_files

        checkpoints_str = "Current checkpoints:\n"
        for c in self.checkpoint_files:
            checkpoints_str += ' {}\n'.format(c)
        _logger.info(checkpoints_str)

        if metric is not None and (self.best_metric is None or self.cmp(metric, self.best_metric)):
            self.best_epoch = epoch
            self.best_metric = metric
            if update_best_file:
                best_save_path = os.path.join(self.checkpoint_dir, 'model_best' + self.extension)
                self._duplicate(os.path.join(self.checkpoint_dir, 'last' + self.extension), best_save_path)

    def save_milestone(self, completed_epochs):
        """Snapshot the ``last`` checkpoint as an immutable ``milestone-N`` file, N the completed epoch count.

        Milestones are outside the ranked history so they are never evicted, and they hardlink ``last`` where the
        filesystem allows so no extra copy of the tensors is made. Call after ``save_checkpoint()`` for the epoch.

        Returns:
            Path of the milestone file.
        """
        last_save_path = os.path.join(self.checkpoint_dir, 'last' + self.extension)
        save_path = os.path.join(self.checkpoint_dir, f'milestone-{completed_epochs}' + self.extension)
        self._duplicate(last_save_path, save_path)
        _logger.info(f'Saved milestone checkpoint {save_path}')
        return save_path

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
