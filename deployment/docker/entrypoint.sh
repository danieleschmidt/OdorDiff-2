#!/bin/bash
set -euo pipefail

# Production entrypoint script for OdorDiff-2
# Handles initialization, health checks, and graceful shutdown

echo "Starting OdorDiff-2 Production Server..."
echo "=================================================="

# Environment validation
echo "🔍 Validating environment..."
required_vars=("DATABASE_URL" "REDIS_URL" "ENVIRONMENT")
for var in "${required_vars[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        echo "❌ ERROR: Required environment variable $var is not set"
        exit 1
    fi
done

# Wait for dependencies
echo "⏳ Waiting for dependencies..."

# Wait for PostgreSQL
echo "  - Waiting for PostgreSQL..."
until pg_isready -d "$DATABASE_URL" > /dev/null 2>&1; do
    echo "    PostgreSQL is unavailable - sleeping"
    sleep 2
done
echo "  ✅ PostgreSQL is ready"

# Wait for Redis
echo "  - Waiting for Redis..."
redis_host=$(echo "$REDIS_URL" | sed -n 's/.*@\([^:]*\):.*/\1/p')
redis_port=$(echo "$REDIS_URL" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
until nc -z "$redis_host" "$redis_port" > /dev/null 2>&1; do
    echo "    Redis is unavailable - sleeping"
    sleep 2
done
echo "  ✅ Redis is ready"

# Database migrations (if needed)
echo "🗄️ Running database migrations..."
python -c "
import os
import sys
sys.path.append('/app')
from odordiff2.config.settings import get_database_connection
try:
    conn = get_database_connection()
    print('✅ Database connection successful')
    conn.close()
except Exception as e:
    print(f'❌ Database connection failed: {e}')
    sys.exit(1)
"

# Model initialization
echo "🧠 Initializing AI models..."
python -c "
import sys
sys.path.append('/app')
try:
    from odordiff2.core.diffusion import DiffusionModel
    model = DiffusionModel()
    print('✅ AI models initialized successfully')
except Exception as e:
    print(f'❌ Model initialization failed: {e}')
    sys.exit(1)
"

# Performance tuning
echo "⚡ Applying performance optimizations..."
# Set optimal memory settings
export MALLOC_ARENA_MAX=4
export MALLOC_MMAP_THRESHOLD_=1048576
export MALLOC_TRIM_THRESHOLD_=1048576

# Create required directories
mkdir -p /app/logs /app/tmp /app/data
chmod 755 /app/logs /app/tmp /app/data

# Setup logging
echo "📝 Setting up logging..."
export PYTHONUNBUFFERED=1
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Health check endpoint test
echo "🏥 Testing health check endpoint..."
timeout 30 bash -c "
until curl -f http://localhost:8000/health > /dev/null 2>&1; do
    echo '  Health check not ready yet...'
    sleep 2
done
" &

# Signal handling for graceful shutdown
cleanup() {
    echo "🛑 Shutting down gracefully..."
    # Kill all child processes
    jobs -p | xargs -r kill
    exit 0
}
trap cleanup SIGTERM SIGINT

# Start the application
echo "🚀 Starting OdorDiff-2 API server..."
echo "  Environment: $ENVIRONMENT"
echo "  Workers: ${WORKERS:-4}"
echo "  Log Level: $LOG_LEVEL"
echo "=================================================="

# Execute the main command
exec "$@"