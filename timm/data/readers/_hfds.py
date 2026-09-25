"""Shared helpers for the Hugging Face datasets / TFDS readers."""
import inspect

from .targets import get_field, has_field


def remote_code_kwargs(trust_remote_code: bool) -> dict:
    """Kwargs for datasets.load_dataset* that request loading-script execution, only when asked for.

    datasets 4.0 removed loading scripts and the trust_remote_code parameter; passing it there falls
    through to the builder config and fails with a confusing error, so raise a clear one instead.
    """
    if not trust_remote_code:
        return {}
    import datasets
    if 'trust_remote_code' not in inspect.signature(datasets.load_dataset).parameters:
        raise ValueError(
            'trust_remote_code is not supported by the installed datasets version, loading scripts were removed in '
            'datasets 4.0. Use a parquet / data-file dataset or install datasets<4.')
    return dict(trust_remote_code=True)


def get_class_labels(info, label_key='label'):
    """Class name -> index mapping of a ClassLabel feature (or Sequence of them) in HF datasets / TFDS info."""
    features = getattr(info, 'features', None)
    feature = get_field(features, label_key) if features is not None and has_field(features, label_key) else None
    # Sequence / List of ClassLabel uses the same vocabulary as a scalar ClassLabel.
    while hasattr(feature, 'feature'):
        feature = feature.feature
    names = getattr(feature, 'names', None)
    return {name: index for index, name in enumerate(names)} if names is not None else {}
