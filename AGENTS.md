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
- Agent-found nits are usually closed. The following cases alone are insufficient justification for a PR; show a supported training, inference, or model-loading workflow that encounters the bug:
  - an unusual call order (e.g. freeze, then reset the classifier or resize; a partial `.to()` after another conversion)
  - meta device, integer/complex dtypes, or unbatched inputs to helpers that are only called batched
  - missing validation or a clearer error for invalid arguments
  - an internal helper with no caller that hits the bug
  - a behaviour that is arguably intended and changing it would alter existing training results
- Search open and closed PRs before starting, and read related discussions. Don't resubmit a declined proposal without addressing the reason it was declined.
- One PR per root cause. Check related code for the same mistake and include any affected sites in the same PR.
- Tests: for bug fixes, add the smallest regression test to the existing test module for that area (e.g. `tests/test_task.py`, `tests/test_layers.py`, `tests/test_models.py`). Don't create new test files for a single fix.
- Keep diffs minimal: no reformatting, style changes, or refactors of code unrelated to the change.
- Preserve backwards compatibility: existing models, pretrained weights, and public functions should produce the same outputs unless the change is fixing a real bug.
- Disclose AI assistance in the PR description. A human submitter is expected to have read the diff and to answer review questions.
