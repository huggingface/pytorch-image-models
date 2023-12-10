*This guideline is very much a work-in-progress.*

Contributions to `timm` for code, documentation, tests are more than welcome!

There haven't been any formal guidelines to date so please bear with me, and feel free to add to this guide.

# Coding style

Code linting and auto-format (black) are not currently in place but open to consideration. In the meantime, the style to follow is (mostly) aligned with Google's guide: https://google.github.io/styleguide/pyguide.html. 

A few specific differences from Google style (or black)
1. Line length is 120 char. Going over is okay in some cases (e.g. I prefer not to break URL across lines).
2. Hanging indents are always prefered, please avoid aligning arguments with closing brackets or braces.

Example, from Google guide, but this is a NO here:
```
   # Aligned with opening delimiter.
   foo = long_function_name(var_one, var_two,
                            var_three, var_four)
   meal = (spam,
           beans)

   # Aligned with opening delimiter in a dictionary.
   foo = {
       'long_dictionary_key': value1 +
                              value2,
       ...
   }
```
This is YES:

```
   # 4-space hanging indent; nothing on first line,
   # closing parenthesis on a new line.
   foo = long_function_name(
       var_one, var_two, var_three,
       var_four
   )
   meal = (
       spam,
       beans,
   )

   # 4-space hanging indent in a dictionary.
   foo = {
       'long_dictionary_key':
           long_dictionary_value,
       ...
   }
```

When there is discrepancy in a given source file (there are many origins for various bits of code and not all have been updated to what I consider current goal), please follow the style in a given file.

In general, if you add new code, formatting it with black using the following options should result in a style that is compatible with the rest of the code base:

```
black --skip-string-normalization --line-length 120 <path-to-file>
```

Avoid formatting code that is unrelated to your PR though.

PR with pure formatting / style fixes will be accepted but only in isolation from functional changes, best to ask before starting such a change.

# Documentation

As with code style, docstrings style based on the Google guide: guide: https://google.github.io/styleguide/pyguide.html

The goal for the code is to eventually move to have all major functions and `__init__` methods use PEP484 type annotations.

When type annotations are used for a function, as per the Google pyguide, they should **NOT** be duplicated in the docstrings, please leave annotations as the one source of truth re typing.

There are a LOT of gaps in current documentation relative to the functionality in timm, please, document away!

# Installation

Create a Python virtual environment using Python 3.10. Inside the environment, install torch` and `torchvision` using the instructions matching your system as listed on the [PyTorch website](https://pytorch.org/).

Then install the remaining dependencies:

```
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt  # for testing
python -m pip install -e .
```

## Unit tests

Run the tests using:

```
pytest tests/
```

Since the whole test suite takes a lot of time to run locally (a few hours), you may want to select a subset of tests relating to the changes you made by using the `-k` option of [`pytest`](https://docs.pytest.org/en/7.1.x/example/markers.html#using-k-expr-to-select-tests-based-on-their-name). Moreover, running tests in parallel (in this example 4 processes) with the `-n` option may help:

```
pytest -k "substring-to-match" -n 4 tests/
```

## Building documentation

Please refer to [this document](https://github.com/huggingface/pytorch-image-models/tree/main/hfdocs).

# Questions

If you have any questions about contribution, where / how to contribute, please ask in the [Discussions](https://github.com/huggingface/pytorch-image-models/discussions/categories/contributing) (there is a `Contributing` topic).
