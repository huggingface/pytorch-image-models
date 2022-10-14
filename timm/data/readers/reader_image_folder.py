""" A dataset reader that extracts images from folders

Folders are scanned recursively to find image files. Labels are based
on the folder hierarchy, just leaf folders by default.

Hacked together by / Copyright 2020 Ross Wightman
"""
import os
from typing import Dict, List, Optional, Set, Tuple, Union

from timm.utils.misc import natural_key

from .class_map import load_class_map
from .img_extensions import get_img_extensions
from .reader import Reader


def find_images_and_targets(
        folder: str,
        types: Optional[Union[List, Tuple, Set]] = None,
        class_to_idx: Optional[Dict] = None,
        leaf_name_only: bool = True,
        sort: bool = True
):
    """ Walk folder recursively to discover images and map them to classes by folder names.

    Args:
        folder: root of folder to recrusively search
        types: types (file extensions) to search for in path
        class_to_idx: specify mapping for class (folder name) to class index if set
        leaf_name_only: use only leaf-name of folder walk for class names
        sort: re-sort found images by name (for consistent ordering)

    Returns:
        A list of image and target tuples, class_to_idx mapping
    """
    types = get_img_extensions(as_set=True) if not types else set(types)
    labels = []
    filenames = []
    for root, subdirs, files in os.walk(folder, topdown=False, followlinks=True):
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types:
                filenames.append(os.path.join(root, f))
                labels.append(label)
    if class_to_idx is None:
        # building class index
        unique_labels = set(labels)
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    images_and_targets = [(f, class_to_idx[l]) for f, l in zip(filenames, labels) if l in class_to_idx]
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    return images_and_targets, class_to_idx


class ReaderImageFolder(Reader):

    def __init__(
            self,
            root,
            class_map=''):
        super().__init__()

        self.root = root
        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        self.samples, self.class_to_idx = find_images_and_targets(root, class_to_idx=class_to_idx)
        if len(self.samples) == 0:
            raise RuntimeError(
                f'Found 0 images in subfolders of {root}. '
                f'Supported image extensions are {", ".join(get_img_extensions())}')

    def __getitem__(self, index):
        path, target = self.samples[index]
        return open(path, 'rb'), target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename
