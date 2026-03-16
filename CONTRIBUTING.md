# Contributing to DREAM-Net

Thank you for considering contributing to DREAM-Net! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Reporting Issues](#reporting-issues)

---

## Code of Conduct

This project adheres to the [Contributor Covenant](https://www.contributor-covenant.org/). By participating, you are expected to uphold this code.

---

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/karl4th/dream-net.git
   cd dream-net
   ```
3. **Set up** the development environment (see below)

---

## Development Setup

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

```bash
# Create virtual environment and install dependencies
uv sync

# Or with pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Verify Setup

```bash
# Run tests
pytest

# Run linter
ruff check src/

# Type checking
mypy src/
```

---

## Pull Request Guidelines

### Before Submitting

1. **Create an issue** describing the change (unless it's a trivial fix)
2. **Write tests** for new functionality
3. **Update documentation** if API changes
4. **Run all checks**:
   ```bash
   pytest
   ruff check src/ tests/
   mypy src/
   ```

### PR Description

Please include:
- **Summary**: Brief description of the change
- **Motivation**: Why is this change needed?
- **Testing**: How was it tested?
- **Breaking Changes**: List any API breaks

### Review Process

1. Maintainers will review within 7 days
2. Address feedback in follow-up commits
3. CI must pass before merging

---

## Coding Standards

### Style Guide

- **Formatter**: Code is auto-formatted with Ruff
- **Line Length**: 100 characters
- **Imports**: Sorted automatically by Ruff
- **Type Hints**: Required for all public functions

### Example

```python
"""Module docstring."""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from dream_net.core.config import DREAMConfig


class DREAMCell(nn.Module):
    """Short class docstring."""

    def __init__(self, config: DREAMConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        state: "DREAMState"
    ) -> Tuple[torch.Tensor, "DREAMState"]:
        """Process one timestep.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch, input_dim)
        state : DREAMState
            Current state

        Returns
        -------
        h_new : torch.Tensor
            New hidden state (batch, hidden_dim)
        state : DREAMState
            Updated state
        """
        ...
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Modules | snake_case | `dream_net/core/cell.py` |
| Classes | PascalCase | `DREAMCell`, `DREAMConfig` |
| Functions | snake_case | `compute_ltc_update()` |
| Variables | snake_case | `hidden_state`, `batch_size` |
| Constants | UPPER_CASE | `DEFAULT_RANK`, `EPS` |
| Private | Leading underscore | `_internal_method()` |

---

## Testing

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src/dream_net

# Specific test file
pytest tests/test_cell.py
```

### Writing Tests

- Place tests in `tests/` directory
- Name files `test_*.py`
- Use descriptive test names: `test_fast_weights_update()`
- Test edge cases and error conditions

```python
"""Tests for DREAMCell."""

import pytest
import torch
from dream_net import DREAMCell, DREAMConfig


def test_cell_forward():
    """Test basic forward pass."""
    config = DREAMConfig(input_dim=80, hidden_dim=256, rank=8)
    cell = DREAMCell(config)
    state = cell.init_state(batch_size=4)

    x = torch.randn(4, 80)
    h_new, state_new = cell(x, state)

    assert h_new.shape == (4, 256)
    assert state_new.h.shape == (4, 256)
```

---

## Documentation

### Docstrings

All public functions and classes must have docstrings in NumPy format:

```python
def function_name(param1: int, param2: str) -> bool:
    """Short one-line description.

    Longer description if needed.

    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str
        Description of param2

    Returns
    -------
    bool
        Description of return value

    Raises
    ------
    ValueError
        When parameter is invalid

    Examples
    --------
    >>> function_name(42, "test")
    True
    """
```

### Building Docs

```bash
# Install docs dependencies
uv sync --extra docs

# Build HTML
cd docs
make html

# Open in browser
open _build/html/index.html
```

---

## Reporting Issues

### Bug Reports

Include:
- **Python version**: `python --version`
- **PyTorch version**: `pip show torch`
- **DREAM-Net version**: `pip show dream-net`
- **Minimal reproducible example**
- **Expected vs actual behavior**
- **Error traceback**

### Feature Requests

Include:
- **Use case**: Why do you need this?
- **Proposed API**: How should it work?
- **Alternatives considered**: What else did you try?

---

## 📧 Contact

- **General questions**: Open a [Discussion](https://github.com/karl4th/dream-net/discussions)
- **Bug reports**: Open an [Issue](https://github.com/karl4th/dream-net/issues)
- **Security issues**: Email bagzhankarl@manifestro.io

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
