# AGENTS.md - PyTorch Image Models (timm)

## Build/Test Commands
- Install: `python -m pip install -e .`
- Run tests: `pytest tests/` (the full suite is slow as it covers every model, run the tests relevant to your change)
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

## Contributions
- See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.
- Please don't open issues or PRs for edge cases unlikely to be hit in real use. Maintainer time is limited, and small fixes like these add review load without helping users. Focus on bugs that affect real training, inference, or model-loading workflows, and include a realistic scenario that triggers the bug.
- Keep diffs minimal: no reformatting, style changes, or refactors of code unrelated to the change.
- Preserve backwards compatibility: existing models, pretrained weights, and public functions should produce the same outputs unless the change is fixing a real bug.
