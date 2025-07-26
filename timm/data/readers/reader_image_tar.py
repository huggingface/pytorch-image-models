""" A dataset reader that reads single tarfile based datasets

This reader can read datasets consisting if a single tarfile containing images.
I am planning to deprecated it in favour of ParerImageInTar.

Hacked together by / Copyright 2020 Ross Wightman
"""
import os
import tarfile

from timm.utils.misc import natural_key

from .class_map import load_class_map
from .img_extensions import get_img_extensions
from .reader import Reader


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


class ReaderImageTar(Reader):
    """ Single tarfile dataset where classes are mapped to folders within tar
    NOTE: This class is being deprecated in favour of the more capable ReaderImageInTar that can
    operate on folders of tars or tars in tars.
    """
    def __init__(self, root, class_map=''):
        super().__init__()

        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        assert os.path.isfile(root), f'Root file {root} not found'
        self.root = root
        
        # Initialize worker info attributes
        self._worker_info = None
        self._worker_id = 0
        self._num_workers = 1

        # Extract tar info without keeping the file open
        with tarfile.open(root) as tf:
            self.samples, self.class_to_idx = extract_tarinfo(tf, class_to_idx)
        self.imgs = self.samples
        self.tarfile = None  # lazy init in __getitem__
        
    def __del__(self):
        # Clean up the tarfile when the reader is garbage collected
        if hasattr(self, 'tarfile') and self.tarfile is not None:
            try:
                self.tarfile.close()
            except Exception as e:
                import warnings
                warnings.warn(f'Error closing tarfile {self.root}: {str(e)}')

    def __getitem__(self, index):
        if self.tarfile is None:
            # Only keep one tarfile open per worker process to avoid file descriptor leaks
            if not hasattr(self, '_worker_info'):
                import torch.utils.data
                worker_info = torch.utils.data.get_worker_info()
                if worker_info is not None:
                    self._worker_info = worker_info
                    self._worker_id = worker_info.id
                    self._num_workers = worker_info.num_workers
            
            self.tarfile = tarfile.open(self.root)
            
        tarinfo, target = self.samples[index]
        try:
            fileobj = self.tarfile.extractfile(tarinfo)
            if fileobj is None:
                raise RuntimeError(f'Failed to extract file {tarinfo.name} from tar {self.root}')
            # Read the file content immediately and close the file object
            content = fileobj.read()
            fileobj.close()
            return io.BytesIO(content), target
        except Exception as e:
            raise RuntimeError(f'Error reading {tarinfo.name} from {self.root}: {str(e)}')

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0].name
        if basename:
            filename = os.path.basename(filename)
        return filename
