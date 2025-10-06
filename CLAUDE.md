# CLAUDE.md - PyTorch Image Models (timm)

## Build/Test Commands
- Install: `python -m pip install -e .`
- Run tests: `pytest tests/`
- Run specific test: `pytest tests/test_models.py::test_specific_function -v`
- Run tests in parallel: `pytest -n 4 tests/`
- Filter tests: `pytest -k "substring-to-match" tests/`

## Code Style Guidelines
- Line length: 120 chars
- Indentation: 4-space hanging indents, arguments should have an extra level of indent, use 'sadface' (closing parenthesis and colon on a separate line)
- Typing: Use PEP484 type annotations in function signatures
- Docstrings: Google style (do not duplicate type annotations and defaults)
- Imports: Standard library first, then third-party, then local
- Function naming: snake_case
- Class naming: PascalCase
- Error handling: Use try/except with specific exceptions
- Conditional expressions: Use parentheses for complex expressions