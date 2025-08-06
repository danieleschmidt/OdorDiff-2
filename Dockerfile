# Multi-stage Docker build for OdorDiff-2
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app user
RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/bash --create-home app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libxrender1 \
    libxext6 \
    libfontconfig1 \
    libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Development stage
FROM base as development

# Install development dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY --chown=app:app . .

USER app

# Install package in development mode
RUN pip install -e .[dev,vis,monitoring]

EXPOSE 8000 8050

CMD ["uvicorn", "odordiff2.api.endpoints:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Install only production dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt && \
    pip install gunicorn[gevent]==21.2.0

# Copy source code and install package
COPY --chown=app:app . .
RUN pip install -e . && \
    python -c "import odordiff2; print('Package installed successfully')"

# Create necessary directories
RUN mkdir -p /app/logs /app/cache /app/data && \
    chown -R app:app /app

USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Production command
CMD ["gunicorn", "odordiff2.api.endpoints:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--log-level", "info"]