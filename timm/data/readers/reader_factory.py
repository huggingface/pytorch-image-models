import os

from .reader_image_folder import ReaderImageFolder
from .reader_image_in_tar import ReaderImageInTar


def create_reader(name, root, split='train', **kwargs):
    name = name.lower()
    name = name.split('/', 1)
    prefix = ''
    if len(name) > 1:
        prefix = name[0]
    name = name[-1]

    # FIXME improve the selection right now just tfds prefix or fallback path, will need options to
    # explicitly select other options shortly
    if prefix == 'hfds':
        from .reader_hfds import ReaderHfds  # defer tensorflow import
        reader = ReaderHfds(root, name, split=split, **kwargs)
    elif prefix == 'tfds':
        from .reader_tfds import ReaderTfds  # defer tensorflow import
        reader = ReaderTfds(root, name, split=split, **kwargs)
    elif prefix == 'wds':
        from .reader_wds import ReaderWds
        kwargs.pop('download', False)
        reader = ReaderWds(root, name, split=split, **kwargs)
    else:
        assert os.path.exists(root)
        # default fallback path (backwards compat), use image tar if root is a .tar file, otherwise image folder
        # FIXME support split here or in reader?
        if os.path.isfile(root) and os.path.splitext(root)[1] == '.tar':
            reader = ReaderImageInTar(root, **kwargs)
        else:
            reader = ReaderImageFolder(root, **kwargs)
    return reader
