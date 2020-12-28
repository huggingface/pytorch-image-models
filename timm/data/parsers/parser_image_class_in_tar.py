import os
import tarfile
import pickle
from glob import glob
import numpy as np

from timm.utils.misc import natural_key

from .parser import Parser
from .class_map import load_class_map
from .constants import IMG_EXTENSIONS


def extract_tarinfos(root, class_name_to_idx=None, cache_filename=None, extensions=None):
    tar_filenames = glob(os.path.join(root, '*.tar'), recursive=True)
    assert len(tar_filenames)
    num_tars = len(tar_filenames)

    cache_path = ''
    if cache_filename is not None:
        cache_path = os.path.join(root, cache_filename)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as pf:
            tarinfo_map = pickle.load(pf)
    else:
        tarinfo_map = {}
        for fi, fn in enumerate(tar_filenames):
            if fi % 1000 == 0:
                print(f'DEBUG: tar {fi}/{num_tars}')
            # cannot keep this open across processes, reopen later
            name = os.path.splitext(os.path.basename(fn))[0]
            with tarfile.open(fn) as tf:
                if extensions is None:
                    # assume all files are valid samples
                    class_tarinfos = tf.getmembers()
                else:
                    class_tarinfos = [m for m in tf.getmembers() if os.path.splitext(m.name)[1].lower() in extensions]
                tarinfo_map[name] = dict(tarinfos=class_tarinfos)
            print(f'DEBUG: {len(class_tarinfos)} images for class {name}')
        tarinfo_map = {k: v for k, v in sorted(tarinfo_map.items(), key=lambda k: natural_key(k[0]))}
        if cache_path:
            with open(cache_path, 'wb') as pf:
                pickle.dump(tarinfo_map, pf, protocol=pickle.HIGHEST_PROTOCOL)

    tarinfos = []
    targets = []
    build_class_map = False
    if class_name_to_idx is None:
        class_name_to_idx = {}
        build_class_map = True
    for i, (name, metadata) in enumerate(tarinfo_map.items()):
        class_idx = i
        if build_class_map:
            class_name_to_idx[name] = i
        else:
            if name not in class_name_to_idx:
                # only samples with class in class mapping are added
                continue
            class_idx = class_name_to_idx[name]
        num_samples = len(metadata['tarinfos'])
        tarinfos.extend(metadata['tarinfos'])
        targets.extend([class_idx] * num_samples)

    return tarinfos, np.array(targets), class_name_to_idx


class ParserImageClassInTar(Parser):
    """ Multi-tarfile dataset parser where there is one .tar file per class
    """

    CACHE_FILENAME = '_tarinfos.pickle'

    def __init__(self, root, class_map=''):
        super().__init__()

        class_name_to_idx = None
        if class_map:
            class_name_to_idx = load_class_map(class_map, root)
        assert os.path.isdir(root)
        self.root = root
        self.tarinfos, self.targets, self.class_name_to_idx = extract_tarinfos(
            self.root, class_name_to_idx=class_name_to_idx,
            cache_filename=self.CACHE_FILENAME, extensions=IMG_EXTENSIONS)
        self.class_idx_to_name = {v: k for k, v in self.class_name_to_idx.items()}
        self.tarfiles = {}  # to open lazily
        self.cache_tarfiles = False

    def __len__(self):
        return len(self.tarinfos)

    def __getitem__(self, index):
        tarinfo = self.tarinfos[index]
        target = self.targets[index]
        class_name = self.class_idx_to_name[target]
        if self.cache_tarfiles:
            tf = self.tarfiles.setdefault(
                class_name, tarfile.open(os.path.join(self.root, class_name + '.tar')))
        else:
            tf = tarfile.open(os.path.join(self.root, class_name + '.tar'))
        fileobj = tf.extractfile(tarinfo)
        return fileobj, target

    def _filename(self, index, basename=False, absolute=False):
        filename = self.tarinfos[index].name
        if basename:
            filename = os.path.basename(filename)
        return filename
