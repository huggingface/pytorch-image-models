from copy import deepcopy

__all__ = ['get_img_extensions', 'is_img_extension', 'set_img_extensions', 'add_img_extensions', 'del_img_extensions']


IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg')  # singleton, kept public for bwd compat use
_IMG_EXTENSIONS_SET = set(IMG_EXTENSIONS)  # set version, private, kept in sync


def _set_extensions(extensions):
    global IMG_EXTENSIONS
    global _IMG_EXTENSIONS_SET
    dedupe = set()  # NOTE de-duping tuple while keeping original order
    IMG_EXTENSIONS = tuple(x for x in extensions if x not in dedupe and not dedupe.add(x))
    _IMG_EXTENSIONS_SET = set(extensions)


def _valid_extension(x: str):
    return x and isinstance(x, str) and len(x) >= 2 and x.startswith('.')


def is_img_extension(ext):
    return ext in _IMG_EXTENSIONS_SET


def get_img_extensions(as_set=False):
    return deepcopy(_IMG_EXTENSIONS_SET if as_set else IMG_EXTENSIONS)


def set_img_extensions(extensions):
    assert len(extensions)
    for x in extensions:
        assert _valid_extension(x)
    _set_extensions(extensions)


def add_img_extensions(ext):
    if not isinstance(ext, (list, tuple, set)):
        ext = (ext,)
    for x in ext:
        assert _valid_extension(x)
    extensions = IMG_EXTENSIONS + tuple(ext)
    _set_extensions(extensions)


def del_img_extensions(ext):
    if not isinstance(ext, (list, tuple, set)):
        ext = (ext,)
    extensions = tuple(x for x in IMG_EXTENSIONS if x not in ext)
    _set_extensions(extensions)
