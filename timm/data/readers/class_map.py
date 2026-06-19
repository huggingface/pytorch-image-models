import os
import pickle


class _ClassMapUnpickler(pickle.Unpickler):
    """Restricted unpickler for `.pkl` class map files.

    A class map is a plain ``{class_name: index}`` dict of built-in types, which never
    triggers ``find_class``. Disallowing all globals therefore blocks arbitrary code
    execution from a crafted class map file (CWE-502) without affecting valid files.
    """

    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError(f'Global {module}.{name} is not permitted in a class map file.')


def load_class_map(map_or_filename, root=''):
    if isinstance(map_or_filename, dict):
        assert dict, 'class_map dict must be non-empty'
        return map_or_filename
    class_map_path = map_or_filename
    if not os.path.exists(class_map_path):
        class_map_path = os.path.join(root, class_map_path)
        assert os.path.exists(class_map_path), 'Cannot locate specified class map file (%s)' % map_or_filename
    class_map_ext = os.path.splitext(map_or_filename)[-1].lower()
    if class_map_ext == '.txt':
        with open(class_map_path) as f:
            class_to_idx = {v.strip(): k for k, v in enumerate(f)}
    elif class_map_ext == '.pkl':
        with open(class_map_path, 'rb') as f:
            class_to_idx = _ClassMapUnpickler(f).load()
        if not isinstance(class_to_idx, dict):
            raise ValueError(f'Invalid class map file, expected a dict ({class_map_path}).')
    else:
        assert False, f'Unsupported class map file extension ({class_map_ext}).'
    return class_to_idx

