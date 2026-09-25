"""Helpers for resolving target fields, including multi-field binary targets (one on/off field per label)."""
from typing import Dict, List, Optional, Sequence, Tuple

TARGET_FORMATS = ('indices', 'multihot')


def check_target_format(target_format: Optional[str], target_keys: Optional[Sequence[str]] = None) -> str:
    """Validate the target format of a single-key target ('indices' or 'multihot', None means 'indices').

    Multi-field (comma separated) target keys define their own format, so any explicit non-default
    format alongside them is a configuration error.
    """
    target_format = target_format or 'indices'
    if target_format not in TARGET_FORMATS:
        raise ValueError(f"Unknown target format '{target_format}', expected one of {TARGET_FORMATS}.")
    if target_keys and target_format != 'indices':
        raise ValueError(
            'Multi-field target keys define their own target format, leave the target format at its default.')
    return target_format


def _unwrap_sequence(value, part: str):
    """Descend through Sequence-style feature wrappers (HF datasets / TFDS) that have no key of their own."""
    while not (hasattr(value, '__contains__') and part in value) and hasattr(value, 'feature'):
        value = value.feature
    return value


def _contains(value, parts) -> bool:
    for i, part in enumerate(parts):
        if isinstance(value, (list, tuple)):
            # a list of dicts, the remaining path must exist in every item
            return all(_contains(item, parts[i:]) for item in value)
        value = _unwrap_sequence(value, part)
        if not hasattr(value, '__contains__') or part not in value:
            return False
        value = value[part]
    return True


def _resolve(value, parts):
    for i, part in enumerate(parts):
        if isinstance(value, (list, tuple)):
            # a list of dicts, map the remaining path over the items
            return [_resolve(item, parts[i:]) for item in value]
        value = _unwrap_sequence(value, part)[part]
    return value


def has_field(container, path: str) -> bool:
    """Whether a field name, or a '/' separated path into nested dict-like containers, exists."""
    if path in container:
        return True
    return _contains(container, path.split('/'))


def get_field(container, path: str):
    """Look up a field by name, or by a '/' separated path into nested dict-like containers.

    A literal key containing '/' (TFDS has top-level names such as 'image/filename') takes precedence.
    Feature schemas are walked through Sequence / List wrappers, so 'objects/category' resolves the
    ClassLabel inside a sequence of dicts. In samples the same path yields the list of values whether
    the sequence is stored as a dict of lists (HF Sequence, TFDS) or a list of dicts (HF List, JSON).
    """
    if path in container:
        return container[path]
    return _resolve(container, path.split('/'))


def parse_target_keys(target_key: Optional[str]) -> Optional[Tuple[str, ...]]:
    """Split a comma separated target key into the names of multiple binary target fields.

    Returns None for a single key (or no key) so callers keep their scalar target path.
    """
    if not target_key or ',' not in target_key:
        return None
    keys = tuple(key.strip() for key in target_key.split(','))
    if not all(keys) or len(set(keys)) != len(keys):
        raise ValueError(f"Multi-field target key '{target_key}' must list unique, non-empty field names.")
    return keys


def multi_field_class_to_idx(keys: Sequence[str], features=None) -> Dict[str, int]:
    """Class index mapping for multi-field targets, one class per field in the order given.

    When the dataset's features are known, every field (or '/' path) must exist in them.
    """
    if features is not None:
        missing = [key for key in keys if not has_field(features, key)]
        if missing:
            raise ValueError(f'Target fields {missing} not found in dataset features {sorted(features.keys())}.')
    return {key: index for index, key in enumerate(keys)}


def multi_field_target(sample, keys: Sequence[str]) -> List[int]:
    """Indices of the binary target fields that are on (> 0) in sample, i.e. a multi-label index list."""
    return [index for index, key in enumerate(keys) if get_field(sample, key) > 0]
