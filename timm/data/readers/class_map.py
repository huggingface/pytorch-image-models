import numbers
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


def _is_sequence(value) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    return isinstance(value, (list, tuple)) or getattr(value, 'ndim', 0) > 0  # lists, tuples, numpy / torch arrays


def remap_labels(label, class_to_idx, source_names=None):
    """Map a label, or a sequence of labels, through class_to_idx.

    An integer label missing from class_to_idx is first converted to its source name when
    source_names (index -> name) is given, so name-keyed class maps work with datasets that
    store class indices.
    """
    if _is_sequence(label):
        return [remap_labels(value, class_to_idx, source_names) for value in label]
    if source_names and isinstance(label, numbers.Integral) and label not in class_to_idx:
        label = source_names[label]
    return class_to_idx[label]


def remap_dense_target(values, class_to_idx):
    """Reorder a dense (multi-hot) target by position through a source index -> class index map.

    Positions absent from the map are dropped. Vector positions carry no names, so the map
    must be keyed by integer source index.
    """
    if not all(isinstance(key, numbers.Integral) for key in class_to_idx):
        raise ValueError(
            'Dense (multihot) targets need a class map keyed by source class index, names are not available.')
    values = list(values)
    if max(class_to_idx) >= len(values):
        raise ValueError(
            f'Class map refers to source index {max(class_to_idx)} but the target has {len(values)} entries.')
    result = [0.] * (max(class_to_idx.values()) + 1)
    for source, index in class_to_idx.items():
        result[index] = float(values[source])
    return result


def remap_target(target, class_to_idx, source_names=None, dense=False):
    """Remap a target through a class map: scalars and index lists by value, dense targets by position."""
    if dense:
        return remap_dense_target(target, class_to_idx)
    return remap_labels(target, class_to_idx, source_names)


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

