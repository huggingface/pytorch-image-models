""" A dataset parser that reads single tarfile based datasets

This parser can read datasets consisting if a single tarfile containing images.
I am planning to deprecated it in favour of ParerImageInTar.

Hacked together by / Copyright 2020 Ross Wightman
"""
import os
import tarfile

from timm.utils.misc import natural_key

from .class_map import load_class_map
from .img_extensions import get_img_extensions
from .parser import Parser


def extract_tarinfo(tarfile, class_to_idx=None, sort=True):
    extensions = get_img_extensions(as_set=True)
    files = []
    labels = []
    for ti in tarfile.getmembers():
        if not ti.isfile():
            continue
        dirname, basename = os.path.split(ti.path)
        label = os.path.basename(dirname)
        ext = os.path.splitext(basename)[1]
        if ext.lower() in extensions:
            files.append(ti)
            labels.append(label)
    if class_to_idx is None:
        unique_labels = set(labels)
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    tarinfo_and_targets = [(f, class_to_idx[l]) for f, l in zip(files, labels) if l in class_to_idx]
    if sort:
        tarinfo_and_targets = sorted(tarinfo_and_targets, key=lambda k: natural_key(k[0].path))
    return tarinfo_and_targets, class_to_idx


class ParserImageTar(Parser):
    """ Single tarfile dataset where classes are mapped to folders within tar
    NOTE: This class is being deprecated in favour of the more capable ParserImageInTar that can
    operate on folders of tars or tars in tars.
    """
    def __init__(self, root, class_map=''):
        super().__init__()

        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        assert os.path.isfile(root)
        self.root = root

        with tarfile.open(root) as tf:  # cannot keep this open across processes, reopen later
            self.samples, self.class_to_idx = extract_tarinfo(tf, class_to_idx)
        self.imgs = self.samples
        self.tarfile = None  # lazy init in __getitem__

    def __getitem__(self, index):
        if self.tarfile is None:
            self.tarfile = tarfile.open(self.root)
        tarinfo, target = self.samples[index]
        fileobj = self.tarfile.extractfile(tarinfo)
        return fileobj, target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0].name
        if basename:
            filename = os.path.basename(filename)
        return filename
