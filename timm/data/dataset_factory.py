import csv
import os

from .dataset import IterableImageDataset, ImageDataset, COAIImageClassDataset


def _search_split(root, split):
    # look for sub-folder with name of split in root and use that if it exists
    split_name = split.split('[')[0]
    try_root = os.path.join(root, split_name)
    if os.path.exists(try_root):
        return try_root
    if split_name == 'validation':
        try_root = os.path.join(root, 'val')
        if os.path.exists(try_root):
            return try_root
    return root


def create_dataset(name, root, split='validation', search_split=True, is_training=False, batch_size=None, **kwargs):
    name = name.lower()
    if name.startswith('tfds'):
        ds = IterableImageDataset(
            root, parser=name, split=split, is_training=is_training, batch_size=batch_size, **kwargs)
    elif name.startswith('coaiclass'):
        # Get Dict from csv(current implementation)/mongodb(needs to be added)
        dict = _get_dict_from_csv(root)
        ds = COAIImageClassDataset(dict=dict)
    else:
        # FIXME support more advance split cfg for ImageFolder/Tar datasets in the future
        kwargs.pop('repeats', 0)  # FIXME currently only Iterable dataset support the repeat multiplier
        if search_split and os.path.isdir(root):
            root = _search_split(root, split)
        ds = ImageDataset(root, parser=name, **kwargs)
    return ds


def _convert(lst):
    res_dct = {lst[i][0]: lst[i][1] for i in range(len(lst))}
    return res_dct


def _get_dict_from_csv(data_folder):
    with open(data_folder + '/train.csv', 'r') as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    return _convert(data)