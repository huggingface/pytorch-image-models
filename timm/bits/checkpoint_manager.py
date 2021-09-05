""" Checkpoint Manager

Track top-n training checkpoints and maintain recovery checkpoints on specified intervals.

Hacked together by / Copyright 2021 Ross Wightman
"""
import glob
import logging
import operator
import os
import shutil
from typing import Optional, Dict, Callable, List
from dataclasses import dataclass, replace


from .checkpoint import save_train_state
from .train_state import TrainState

_logger = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    path: str = ''
    metrics: Dict[str, float] = None  # all metrics at time of checkpoint save
    metric_name: str = 'loss'
    metric_decreasing: bool = True
    epoch: int = 0
    global_step: int = 0

    @property
    def valid_key(self):
        return self.metric_name and self.metrics and self.metric_name in self.metrics

    @property
    def sort_key(self):
        return self.metrics[self.metric_name] if self.valid_key else self.epoch

    @property
    def decreasing_key(self):
        return self.metric_decreasing if self.valid_key else False


class CheckpointManager:
    def __init__(
            self,
            hparams=None,
            save_state_fn=None,
            checkpoint_dir='',
            recovery_dir='',
            checkpoint_tmpl=None,
            recovery_tmpl=None,
            metric_name='loss',
            metric_decreasing=True,
            max_history=10):

        # extra items to include in checkpoint
        self.hparams = hparams  # train arguments (config / hparams) # FIXME this will change with new config system

        # state
        self.checkpoint_files: List[CheckpointInfo] = []  # (filename, metric) tuples in order of decreasing betterness
        self.best_checkpoint = None
        self.curr_recovery_file = ''
        self.prev_recovery_file = ''
        self.can_hardlink = True

        # util / helper fn
        self.save_state_fn = save_state_fn or save_train_state

        # file / folder config
        self.extension = '.pth.tar'
        self.checkpoint_dir = checkpoint_dir
        self.recovery_dir = recovery_dir
        self.checkpoint_tmpl = (checkpoint_tmpl or 'checkpoint-{index}') + self.extension
        self.recovery_tmpl = (recovery_tmpl or 'recovery-{index}') + self.extension

        # ordering / history config
        self.metric_name = metric_name
        self.metric_decreasing = metric_decreasing
        self.metric_cmp_fn = operator.lt if metric_decreasing else operator.gt
        self.max_history = max_history
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

    def _save(self, save_path, train_state: TrainState, metrics: Optional[Dict[str, float]] = None):
        extra_state = dict(
            # version < 2 increments epoch before save
            # version < 3, pre timm bits
            # version 3, first timm bits checkpoitns
            version=3,
        )
        if self.hparams is not None:
            extra_state.update(dict(arch=self.hparams['model'], hparams=self.hparams))
        else:
            arch = getattr(train_state.model, 'default_cfg', dict()).get('architecture', None)
            if arch is None:
                arch = type(train_state.model).__name__.lower()
            extra_state.update(dict(arch=arch))
        if metrics is not None:
            # save the metrics and how we originally sorted them in the checkpoint for future comparisons
            extra_state.update(dict(
                metrics=metrics,
                metric_name=self.metric_name,
                metric_decreasing=self.metric_decreasing
            ))

        self.save_state_fn(save_path, train_state, extra_state)

        checkpoint_info = CheckpointInfo(
            path=save_path,
            metrics=metrics,
            metric_name=self.metric_name,
            metric_decreasing=self.metric_decreasing,
            epoch=train_state.epoch,
            global_step=train_state.step_count_global,
        )
        return checkpoint_info

    def _udpate_checkpoints(self, info: CheckpointInfo):
        self.checkpoint_files.append(info)
        self.checkpoint_files = sorted(
            self.checkpoint_files,
            key=lambda x: x.sort_key,
            reverse=not info.decreasing_key,  # sort in descending order if a lower metric is not better
        )

    def _cleanup_checkpoints(self, trim=0):
        trim = min(len(self.checkpoint_files), trim)
        delete_index = self.max_history - trim
        if delete_index < 0 or len(self.checkpoint_files) <= delete_index:
            return
        to_delete = self.checkpoint_files[delete_index:]
        for d in to_delete:
            try:
                _logger.debug("Cleaning checkpoint: {}".format(d))
                os.remove(d.path)
            except OSError as e:
                _logger.error("Exception '{}' while deleting checkpoint".format(e))
        self.checkpoint_files = self.checkpoint_files[:delete_index]

    def _compare_metric(self, lhs: CheckpointInfo, rhs: CheckpointInfo):
        # compare metrics against an existing checkpoint
        if not lhs or not lhs.valid_key or not rhs or not rhs.valid_key:
            # always assume lhs metrics are better if there are no usable metrics to compare
            return True
        return self.metric_cmp_fn(lhs.sort_key, rhs.sort_key)

    def save_checkpoint(self, train_state: TrainState, metrics: Optional[Dict[str, float]] = None):
        assert train_state.epoch >= 0
        tmp_save_path = os.path.join(self.checkpoint_dir, 'tmp' + self.extension)
        last_save_path = os.path.join(self.checkpoint_dir, 'last' + self.extension)
        curr_checkpoint = self._save(tmp_save_path, train_state, metrics)
        self._replace(tmp_save_path, last_save_path)

        worst_checkpoint = self.checkpoint_files[-1] if self.checkpoint_files else None
        if len(self.checkpoint_files) < self.max_history or self._compare_metric(curr_checkpoint, worst_checkpoint):
            if len(self.checkpoint_files) >= self.max_history:
                self._cleanup_checkpoints(1)

            filename = self.checkpoint_tmpl.format(index=train_state.epoch)
            save_path = os.path.join(self.checkpoint_dir, filename)
            curr_checkpoint = replace(curr_checkpoint, path=save_path)
            self._duplicate(last_save_path, save_path)
            self._udpate_checkpoints(curr_checkpoint)

            checkpoints_str = "Current checkpoints:\n"
            for c in self.checkpoint_files:
                checkpoints_str += f' {c.path}, {c.sort_key}\n'.format(c)
            _logger.info(checkpoints_str)

            if curr_checkpoint.valid_key and self._compare_metric(curr_checkpoint, self.best_checkpoint):
                self.best_checkpoint = curr_checkpoint
                best_save_path = os.path.join(self.checkpoint_dir, 'best' + self.extension)
                self._duplicate(last_save_path, best_save_path)

        return curr_checkpoint if self.best_checkpoint is None else self.best_checkpoint

    def save_recovery(self, train_state: TrainState):
        tmp_save_path = os.path.join(self.recovery_dir, 'recovery_tmp' + self.extension)
        self._save(tmp_save_path, train_state)

        filename = self.recovery_tmpl.format(index=train_state.step_count_global)
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
