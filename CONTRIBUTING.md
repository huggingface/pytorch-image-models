*This guideline is very much a work-in-progress.*

Contriubtions to `timm` for code, documentation, tests are more than welcome!

There haven't been any formal guidelines to date so please bear with me, and feel free to add to this guide.

# Code

Code linting and auto-format (black) are not currently in place but open to consideration. In the meantime, the style to follow is (mostly) aligned with Google's guide: https://google.github.io/styleguide/pyguide.html

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

When there is descrepancy in a given source file (there are many origins for various bits of code and not all have been updated to what I consider current goal), please follow the style in a given file.

PR with pure formatting / style fixes will be accepted but only in isolation from functional changes, best to ask before starting such a change.


# Documentation

As with code style, docstrings style based on the Google guide: guide: https://google.github.io/styleguide/pyguide.html

The goal for the code is to eventually move to have all major functions and `__init__` methods use PEP484 type annotations.

When type annotations are used for a function, as per the Google pyguide, they should **NOT** be duplicated in the docstrings, please leave annotations as the one source of truth re typing.

There are a LOT of gaps in current documentation relative to the functionality in timm, please, document away!

# Questions

If you have any questions about contribution, where / how to contribute, please ask in the Discussions (there is a `Contributing` topic).


