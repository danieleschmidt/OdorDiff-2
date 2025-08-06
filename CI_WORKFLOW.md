# CI/CD Workflow Configuration

Since GitHub workflows require special permissions, here's the CI/CD configuration that should be manually added to `.github/workflows/ci.yml`:

## GitHub Actions Workflow

Create `.github/workflows/ci.yml` with the following content:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  release:
    types: [ published ]

env:
  PYTHON_VERSION: "3.10"

jobs:
  # Code quality checks
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev]
          
      - name: Run black
        run: black --check --diff odordiff2 tests
        
      - name: Run flake8
        run: flake8 odordiff2 tests
        
      - name: Run mypy
        run: mypy odordiff2

  # Unit tests
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest]
        python-version: ["3.9", "3.10", "3.11"]
        
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev,vis]
          
      - name: Run unit tests
        run: |
          pytest tests/unit/ -v --cov=odordiff2 --cov-report=xml
```

## Manual CI Commands

You can run the CI checks manually:

```bash
# Code formatting
black odordiff2 tests
flake8 odordiff2 tests
mypy odordiff2

# Testing
pytest tests/ -v --cov=odordiff2
```