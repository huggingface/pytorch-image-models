import argparse

from timm.utils.misc import ParseKwargs


def _parse_model_kwargs(tokens):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-kwargs', nargs='*', default={}, action=ParseKwargs)
    args = parser.parse_args(['--model-kwargs'] + tokens)
    return args.model_kwargs


def test_parse_kwargs_literal_values():
    assert _parse_model_kwargs(['depth=12', 'drop_rate=0.1', 'pretrained=True']) == {
        'depth': 12,
        'drop_rate': 0.1,
        'pretrained': True,
    }


def test_parse_kwargs_value_with_equals():
    # A value that itself contains '=' must split only on the first '=' and be
    # kept verbatim. Before, split('=') raised "too many values to unpack", and
    # the literal_eval fallback only caught ValueError so 'size=large' (invalid
    # syntax) still raised SyntaxError.
    assert _parse_model_kwargs(['note=size=large', 'url=http://h/p?a=1&b=2']) == {
        'note': 'size=large',
        'url': 'http://h/p?a=1&b=2',
    }


def test_parse_kwargs_non_literal_falls_back_to_string():
    assert _parse_model_kwargs(['act=nn.GELU', 'cfg={"a": 1}']) == {
        'act': 'nn.GELU',
        'cfg': {'a': 1},
    }
