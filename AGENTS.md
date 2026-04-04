# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Build, Lint, and Test Commands

```bash
uv sync --dev  # Install dev dependencies into the uv-managed .venv
```

```bash
uv run ruff check .                 # Lint with Ruff
uv run ruff format .                # Format with Ruff
uv run pre-commit install           # Install pre-commit hooks
uv run pre-commit run --all-files
```

```bash
uv run pytest                                # Run all tests
uv run pytest tests/test_pairs.py            # Run single test file
uv run pytest tests/test_pairs.py::test_fn   # Run single test function
uv run pytest tests/test_pairs.py::TestCls   # Run single test class
uv run pytest --cov=geosam                   # Run with coverage
```

## Build Docs

```bash
cd docs && make html              # Build HTML docs
open _build/html/index.html       # View docs (macOS)
```

## Code Style Guidelines

### Language and Naming

- Write all code and comments in English
- Use descriptive English names for variables, functions, and classes
- Follow PEP 8 and PEP 257 standards

## Code Conventions

- **Python 3.11+** required
- **`from __future__ import annotations`** in every file
- **Type hints**: Use Python 3.11+ syntax (`str | None`, `dict[str, int]`). Use `Literal` for fixed value sets. Do not using Union，Optional.
- **Docstrings**: Use NumPy-style docstrings for all public modules, classes, functions, and methods. Include Parameters, Returns, Raises, and Examples where applicable, and use Sphinx reStructuredText markup such as :func:..., :class:..., and directives like .. note::, .. tip::, and .. warning:: when needed.
- **Logging**: Use `from geosam.logging import setup_logger; logger = setup_logger(__name__)` — log before raising exceptions errors
- **Paths**: Use `pathlib.Path` internally; convert to `str` only when passing to ISCE2/ISCE3 APIs
- **Type-checking imports**: Put heavy/circular imports inside `if TYPE_CHECKING:` blocks
- **Commits**: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`)
- **Linter**: ruff only (no black, no flake8). Line length 88. Ruff excludes `tests/`, `docs/`, `examples/` directories.