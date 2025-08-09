# GitHub Actions CI/CD Workflow Configuration

Due to GitHub App permissions restrictions, the comprehensive testing workflow cannot be automatically created. Below is the complete workflow configuration that should be manually added to `.github/workflows/comprehensive-testing.yml`:

```yaml
name: Comprehensive Testing Pipeline

on:
  push:
    branches: [ main, develop, 'terragon/*' ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    # Run nightly performance regression tests
    - cron: '0 2 * * *'

env:
  PYTHON_VERSION: '3.10'
  NODE_VERSION: '18'

jobs:
  # Unit Testing Job
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt', '**/pyproject.toml') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev,test]
    
    - name: Run unit tests with coverage
      run: |
        python -m pytest tests/unit/ -v --cov=odordiff2 --cov-report=xml --cov-report=html --cov-report=term-missing
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  # Integration Testing Job
  integration-tests:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      postgres:
        image: postgres:15-alpine
        env:
          POSTGRES_PASSWORD: testpass
          POSTGRES_DB: odordiff_test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev,test]
    
    - name: Run integration tests
      env:
        REDIS_URL: redis://localhost:6379
        DATABASE_URL: postgresql://postgres:testpass@localhost:5432/odordiff_test
      run: |
        python -m pytest tests/integration/ -v --tb=short

  # Performance Testing Job
  performance-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule' || contains(github.event.head_commit.message, '[perf-test]')
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev,test]
        pip install pytest-benchmark
    
    - name: Run performance benchmarks
      run: |
        python -m pytest tests/performance/ -v --benchmark-json=benchmark.json
    
    - name: Store benchmark results
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: benchmark.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true

  # Security Testing Job
  security-tests:
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
        pip install -e .[dev,test]
        pip install bandit safety
    
    - name: Run Bandit security scan
      run: |
        bandit -r odordiff2/ -f json -o bandit-report.json || true
    
    - name: Run Safety vulnerability scan
      run: |
        safety check --json --output safety-report.json || true
    
    - name: Run security tests
      run: |
        python -m pytest tests/security/ -v
    
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json

  # Code Quality Job
  code-quality:
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
        pip install -e .[dev,test]
        pip install black isort flake8 mypy
    
    - name: Check code formatting with Black
      run: black --check odordiff2/ tests/
    
    - name: Check import sorting with isort
      run: isort --check-only odordiff2/ tests/
    
    - name: Lint with flake8
      run: flake8 odordiff2/ tests/
    
    - name: Type check with mypy
      run: mypy odordiff2/

  # Load Testing Job (only on main branch)
  load-tests:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev,test]
        pip install locust
    
    - name: Run load tests
      run: |
        python testing/load_testing.py --headless --users 100 --spawn-rate 10 --run-time 60s --host http://localhost:8000 > load-test-results.txt
    
    - name: Upload load test results
      uses: actions/upload-artifact@v3
      with:
        name: load-test-results
        path: load-test-results.txt

  # Build and Deploy Job
  build-deploy:
    runs-on: ubuntu-latest
    needs: [unit-tests, integration-tests, security-tests, code-quality]
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        file: ./deployment/docker/Dockerfile.production
        push: true
        tags: |
          ghcr.io/${{ github.repository }}:latest
          ghcr.io/${{ github.repository }}:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  # Notification Job
  notify:
    runs-on: ubuntu-latest
    needs: [unit-tests, integration-tests, security-tests, code-quality, load-tests, build-deploy]
    if: always()
    
    steps:
    - name: Notify on success
      if: ${{ needs.unit-tests.result == 'success' && needs.integration-tests.result == 'success' }}
      run: echo "✅ All tests passed successfully!"
    
    - name: Notify on failure
      if: ${{ needs.unit-tests.result == 'failure' || needs.integration-tests.result == 'failure' }}
      run: |
        echo "❌ Tests failed. Please check the logs."
        exit 1
```

## Setup Instructions

1. **Create the workflow file manually** by adding the above content to `.github/workflows/comprehensive-testing.yml` in your repository
2. **Configure repository secrets** for any external services (e.g., CODECOV_TOKEN)
3. **Enable GitHub Actions** in your repository settings
4. **Configure branch protection rules** to require status checks before merging

## Features

- **Multi-Python Version Testing**: Tests against Python 3.9, 3.10, and 3.11
- **Comprehensive Coverage**: Unit, integration, performance, and security tests
- **Code Quality Checks**: Black, isort, flake8, and mypy
- **Security Scanning**: Bandit and Safety vulnerability detection
- **Performance Monitoring**: Benchmark tracking and regression detection
- **Docker Build**: Automated container builds and registry publishing
- **Load Testing**: Stress testing with configurable parameters

This workflow ensures production readiness and maintains code quality standards throughout the development lifecycle.