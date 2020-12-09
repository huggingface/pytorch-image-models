import os
import io
import re
import torch
import tarfile
import pickle
from glob import glob
import numpy as np

import torch.utils.data as data

from timm.utils.misc import natural_key

from .constants import IMG_EXTENSIONS


def load_class_map(filename, root=''):
    class_map_path = filename
    if not os.path.exists(class_map_path):
        class_map_path = os.path.join(root, filename)
        assert os.path.exists(class_map_path), 'Cannot locate specified class map file (%s)' % filename
    class_map_ext = os.path.splitext(filename)[-1].lower()
    if class_map_ext == '.txt':
        with open(class_map_path) as f:
            class_to_idx = {v.strip(): k for k, v in enumerate(f)}
    else:
        assert False, 'Unsupported class map extension'
    return class_to_idx


class ParserIn21kTar(data.Dataset):

    CACHE_FILENAME = 'class_info.pickle'

    def __init__(self, root, class_map=''):

        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        assert os.path.isdir(root)
        self.root = root
        tar_filenames = glob(os.path.join(self.root, '*.tar'), recursive=True)
        assert len(tar_filenames)
        num_tars = len(tar_filenames)

        if os.path.exists(self.CACHE_FILENAME):
            with open(self.CACHE_FILENAME, 'rb') as pf:
                class_info = pickle.load(pf)
        else:
            class_info = {}
            for fi, fn in enumerate(tar_filenames):
                if fi % 1000 == 0:
                    print(f'DEBUG: tar {fi}/{num_tars}')
                # cannot keep this open across processes, reopen later
                name = os.path.splitext(os.path.basename(fn))[0]
                img_tarinfos = []
                with tarfile.open(fn) as tf:
                    img_tarinfos.extend(tf.getmembers())
                    class_info[name] = dict(img_tarinfos=img_tarinfos)
                print(f'DEBUG: {len(img_tarinfos)} images for synset {name}')
            class_info = {k: v for k, v in sorted(class_info.items())}

            with open('class_info.pickle', 'wb') as pf:
                pickle.dump(class_info, pf, protocol=pickle.HIGHEST_PROTOCOL)

        if class_to_idx is not None:
            out_dict = {}
            for k, v in class_info.items():
                if k in class_to_idx:
                    class_idx = class_to_idx[k]
                    v['class_idx'] = class_idx
                    out_dict[k] = v
            class_info = {k: v for k, v in sorted(out_dict.items(), key=lambda x: x[1]['class_idx'])}
        else:
            for i, (k, v) in enumerate(class_info.items()):
                v['class_idx'] = i

        self.img_infos = []
        self.targets = []
        self.tarnames = []
        for k, v in class_info.items():
            num_samples = len(v['img_tarinfos'])
            self.img_infos.extend(v['img_tarinfos'])
            self.targets.extend([v['class_idx']] * num_samples)
            self.tarnames.extend([k] * num_samples)
        self.targets = np.array(self.targets)  # separate, uniform np array are more memory efficient
        self.tarnames = np.array(self.tarnames)

        self.tarfiles = {}  # to open lazily
        del class_info

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, idx):
        img_tarinfo = self.img_infos[idx]
        name = self.tarnames[idx]
        tf = self.tarfiles.setdefault(name, tarfile.open(os.path.join(self.root, name + '.tar')))
        img_bytes = tf.extractfile(img_tarinfo)
        if self.targets:
            target = self.targets[idx]
        else:
            target = None
        return img_bytes, target
