""" A dataset parser that reads tarfile based datasets

This parser can read and extract image samples from:
* a single tar of image files
* a folder of multiple tarfiles containing imagefiles
* a tar of tars containing image files

Labels are based on the combined folder and/or tar name structure.

Hacked together by / Copyright 2020 Ross Wightman
"""
import logging
import os
import pickle
import tarfile
from glob import glob
from typing import List, Tuple, Dict, Set, Optional, Union

import numpy as np

from timm.utils.misc import natural_key

from .class_map import load_class_map
from .img_extensions import get_img_extensions
from .parser import Parser

_logger = logging.getLogger(__name__)
CACHE_FILENAME_SUFFIX = '_tarinfos.pickle'


class TarState:

    def __init__(self, tf: tarfile.TarFile = None, ti: tarfile.TarInfo = None):
        self.tf: tarfile.TarFile = tf
        self.ti: tarfile.TarInfo = ti
        self.children: Dict[str, TarState] = {}  # child states (tars within tars)

    def reset(self):
        self.tf = None


def _extract_tarinfo(tf: tarfile.TarFile, parent_info: Dict, extensions: Set[str]):
    sample_count = 0
    for i, ti in enumerate(tf):
        if not ti.isfile():
            continue
        dirname, basename = os.path.split(ti.path)
        name, ext = os.path.splitext(basename)
        ext = ext.lower()
        if ext == '.tar':
            with tarfile.open(fileobj=tf.extractfile(ti), mode='r|') as ctf:
                child_info = dict(
                    name=ti.name, path=os.path.join(parent_info['path'], name), ti=ti, children=[], samples=[])
                sample_count += _extract_tarinfo(ctf, child_info, extensions=extensions)
                _logger.debug(f'{i}/?. Extracted child tarinfos from {ti.name}. {len(child_info["samples"])} images.')
                parent_info['children'].append(child_info)
        elif ext in extensions:
            parent_info['samples'].append(ti)
            sample_count += 1
    return sample_count


def extract_tarinfos(
        root,
        class_name_to_idx: Optional[Dict] = None,
        cache_tarinfo: Optional[bool] = None,
        extensions: Optional[Union[List, Tuple, Set]] = None,
        sort: bool = True
):
    extensions = get_img_extensions(as_set=True) if not extensions else set(extensions)
    root_is_tar = False
    if os.path.isfile(root):
        assert os.path.splitext(root)[-1].lower() == '.tar'
        tar_filenames = [root]
        root, root_name = os.path.split(root)
        root_name = os.path.splitext(root_name)[0]
        root_is_tar = True
    else:
        root_name = root.strip(os.path.sep).split(os.path.sep)[-1]
        tar_filenames = glob(os.path.join(root, '*.tar'), recursive=True)
    num_tars = len(tar_filenames)
    tar_bytes = sum([os.path.getsize(f) for f in tar_filenames])
    assert num_tars, f'No .tar files found at specified path ({root}).'

    _logger.info(f'Scanning {tar_bytes/1024**2:.2f}MB of tar files...')
    info = dict(tartrees=[])
    cache_path = ''
    if cache_tarinfo is None:
        cache_tarinfo = True if tar_bytes > 10*1024**3 else False  # FIXME magic number, 10GB
    if cache_tarinfo:
        cache_filename = '_' + root_name + CACHE_FILENAME_SUFFIX
        cache_path = os.path.join(root, cache_filename)
    if os.path.exists(cache_path):
        _logger.info(f'Reading tar info from cache file {cache_path}.')
        with open(cache_path, 'rb') as pf:
            info = pickle.load(pf)
        assert len(info['tartrees']) == num_tars, "Cached tartree len doesn't match number of tarfiles"
    else:
        for i, fn in enumerate(tar_filenames):
            path = '' if root_is_tar else os.path.splitext(os.path.basename(fn))[0]
            with tarfile.open(fn, mode='r|') as tf:  # tarinfo scans done in streaming mode
                parent_info = dict(name=os.path.relpath(fn, root), path=path, ti=None, children=[], samples=[])
                num_samples = _extract_tarinfo(tf, parent_info, extensions=extensions)
                num_children = len(parent_info["children"])
                _logger.debug(
                    f'{i}/{num_tars}. Extracted tarinfos from {fn}. {num_children} children, {num_samples} samples.')
            info['tartrees'].append(parent_info)
        if cache_path:
            _logger.info(f'Writing tar info to cache file {cache_path}.')
            with open(cache_path, 'wb') as pf:
                pickle.dump(info, pf)

    samples = []
    labels = []
    build_class_map = False
    if class_name_to_idx is None:
        build_class_map = True

    # Flatten tartree info into lists of samples and targets w/ targets based on label id via
    # class map arg or from unique paths.
    # NOTE: currently only flattening up to two-levels, filesystem .tars and then one level of sub-tar children
    # this covers my current use cases and keeps things a little easier to test for now.
    tarfiles = []

    def _label_from_paths(*path, leaf_only=True):
        path = os.path.join(*path).strip(os.path.sep)
        return path.split(os.path.sep)[-1] if leaf_only else path.replace(os.path.sep, '_')

    def _add_samples(info, fn):
        added = 0
        for s in info['samples']:
            label = _label_from_paths(info['path'], os.path.dirname(s.path))
            if not build_class_map and label not in class_name_to_idx:
                continue
            samples.append((s, fn, info['ti']))
            labels.append(label)
            added += 1
        return added

    _logger.info(f'Collecting samples and building tar states.')
    for parent_info in info['tartrees']:
        # if tartree has children, we assume all samples are at the child level
        tar_name = None if root_is_tar else parent_info['name']
        tar_state = TarState()
        parent_added = 0
        for child_info in parent_info['children']:
            child_added = _add_samples(child_info, fn=tar_name)
            if child_added:
                tar_state.children[child_info['name']] = TarState(ti=child_info['ti'])
            parent_added += child_added
        parent_added += _add_samples(parent_info, fn=tar_name)
        if parent_added:
            tarfiles.append((tar_name, tar_state))
    del info

    if build_class_map:
        # build class index
        sorted_labels = list(sorted(set(labels), key=natural_key))
        class_name_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}

    _logger.info(f'Mapping targets and sorting samples.')
    samples_and_targets = [(s, class_name_to_idx[l]) for s, l in zip(samples, labels) if l in class_name_to_idx]
    if sort:
        samples_and_targets = sorted(samples_and_targets, key=lambda k: natural_key(k[0][0].path))
    samples, targets = zip(*samples_and_targets)
    samples = np.array(samples)
    targets = np.array(targets)
    _logger.info(f'Finished processing {len(samples)} samples across {len(tarfiles)} tar files.')
    return samples, targets, class_name_to_idx, tarfiles


class ParserImageInTar(Parser):
    """ Multi-tarfile dataset parser where there is one .tar file per class
    """

    def __init__(self, root, class_map='', cache_tarfiles=True, cache_tarinfo=None):
        super().__init__()

        class_name_to_idx = None
        if class_map:
            class_name_to_idx = load_class_map(class_map, root)
        self.root = root
        self.samples, self.targets, self.class_name_to_idx, tarfiles = extract_tarinfos(
            self.root,
            class_name_to_idx=class_name_to_idx,
            cache_tarinfo=cache_tarinfo
        )
        self.class_idx_to_name = {v: k for k, v in self.class_name_to_idx.items()}
        if len(tarfiles) == 1 and tarfiles[0][0] is None:
            self.root_is_tar = True
            self.tar_state = tarfiles[0][1]
        else:
            self.root_is_tar = False
            self.tar_state = dict(tarfiles)
        self.cache_tarfiles = cache_tarfiles

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        target = self.targets[index]
        sample_ti, parent_fn, child_ti = sample
        parent_abs = os.path.join(self.root, parent_fn) if parent_fn else self.root

        tf = None
        cache_state = None
        if self.cache_tarfiles:
            cache_state = self.tar_state if self.root_is_tar else self.tar_state[parent_fn]
            tf = cache_state.tf
        if tf is None:
            tf = tarfile.open(parent_abs)
            if self.cache_tarfiles:
                cache_state.tf = tf
        if child_ti is not None:
            ctf = cache_state.children[child_ti.name].tf if self.cache_tarfiles else None
            if ctf is None:
                ctf = tarfile.open(fileobj=tf.extractfile(child_ti))
                if self.cache_tarfiles:
                    cache_state.children[child_ti.name].tf = ctf
            tf = ctf

        return tf.extractfile(sample_ti), target

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0].name
        if basename:
            filename = os.path.basename(filename)
        return filename
