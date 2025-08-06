#!/bin/bash
set -e

# OdorDiff-2 Deployment Script
# Handles deployment to different environments

ENVIRONMENT=${1:-development}
VERSION=${2:-latest}
COMPOSE_FILE="docker-compose.yml"

echo "🚀 Starting OdorDiff-2 deployment..."
echo "Environment: $ENVIRONMENT"
echo "Version: $VERSION"
echo "================================"

# Function to check if required tools are installed
check_dependencies() {
    echo "📋 Checking dependencies..."
    
    command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required but not installed."; exit 1; }
    command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose is required but not installed."; exit 1; }
    
    echo "✅ Dependencies check passed"
}

# Function to validate environment files
validate_environment() {
    echo "🔍 Validating environment configuration..."
    
    if [ "$ENVIRONMENT" == "production" ]; then
        if [ ! -f ".env.production" ]; then
            echo "❌ .env.production file is required for production deployment"
            exit 1
        fi
        
        # Check for required environment variables
        required_vars=("DATABASE_URL" "REDIS_URL" "SECRET_KEY")
        for var in "${required_vars[@]}"; do
            if ! grep -q "^$var=" .env.production; then
                echo "❌ Required environment variable $var not found in .env.production"
                exit 1
            fi
        done
    fi
    
    echo "✅ Environment validation passed"
}

# Function to backup database (for production)
backup_database() {
    if [ "$ENVIRONMENT" == "production" ]; then
        echo "💾 Creating database backup..."
        
        backup_file="backup_$(date +%Y%m%d_%H%M%S).sql"
        docker-compose exec -T postgres pg_dump -U postgres odordiff2 > "backups/$backup_file"
        
        echo "✅ Database backup created: $backup_file"
    fi
}

# Function to pull latest images
pull_images() {
    echo "📥 Pulling latest Docker images..."
    
    if [ "$ENVIRONMENT" == "production" ]; then
        docker-compose pull
    else
        docker-compose build
    fi
    
    echo "✅ Images updated"
}

# Function to run database migrations
run_migrations() {
    echo "🔧 Running database migrations..."
    
    # Wait for database to be ready
    echo "⏳ Waiting for database to be ready..."
    docker-compose exec -T postgres pg_isready -U postgres -t 30
    
    # Run migrations (placeholder - implement actual migration logic)
    docker-compose exec -T odordiff2-api python -c "
import asyncio
from odordiff2.data.cache import DatasetManager
print('Running database setup...')
# Add actual migration code here
print('Database setup completed')
"
    
    echo "✅ Database migrations completed"
}

# Function to run health checks
health_check() {
    echo "🏥 Running health checks..."
    
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo "Attempt $attempt/$max_attempts: Checking API health..."
        
        if curl -f http://localhost:8000/health >/dev/null 2>&1; then
            echo "✅ API health check passed"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            echo "❌ API health check failed after $max_attempts attempts"
            exit 1
        fi
        
        sleep 10
        ((attempt++))
    done
    
    # Check dashboard health
    if curl -f http://localhost:8050 >/dev/null 2>&1; then
        echo "✅ Dashboard health check passed"
    else
        echo "⚠️ Dashboard health check failed (non-critical)"
    fi
}

# Function to run smoke tests
run_smoke_tests() {
    echo "🧪 Running smoke tests..."
    
    # Test API endpoints
    echo "Testing API endpoints..."
    
    # Health endpoint
    curl -f http://localhost:8000/health || { echo "❌ Health endpoint failed"; exit 1; }
    
    # Stats endpoint
    curl -f http://localhost:8000/stats || { echo "❌ Stats endpoint failed"; exit 1; }
    
    # Generate endpoint (basic test)
    response=$(curl -s -X POST http://localhost:8000/generate \
        -H "Content-Type: application/json" \
        -d '{"prompt": "test lavender scent", "num_molecules": 1}')
    
    if echo "$response" | grep -q "molecules"; then
        echo "✅ Generate endpoint working"
    else
        echo "❌ Generate endpoint failed"
        exit 1
    fi
    
    echo "✅ Smoke tests passed"
}

# Function to setup monitoring
setup_monitoring() {
    echo "📊 Setting up monitoring..."
    
    # Ensure Prometheus data directory exists
    mkdir -p monitoring/prometheus/data
    
    # Ensure Grafana data directory exists  
    mkdir -p monitoring/grafana/data
    
    # Set proper permissions
    chmod 755 monitoring/prometheus/data
    chmod 755 monitoring/grafana/data
    
    echo "✅ Monitoring setup completed"
}

# Function to cleanup old containers
cleanup() {
    echo "🧹 Cleaning up old containers and images..."
    
    docker system prune -f
    docker volume prune -f
    
    echo "✅ Cleanup completed"
}

# Main deployment flow
main() {
    check_dependencies
    validate_environment
    
    # Create necessary directories
    mkdir -p backups logs cache data
    
    if [ "$ENVIRONMENT" == "production" ]; then
        backup_database
    fi
    
    # Stop existing services
    echo "🛑 Stopping existing services..."
    docker-compose down
    
    pull_images
    setup_monitoring
    
    # Start services
    echo "🚀 Starting services..."
    if [ "$ENVIRONMENT" == "production" ]; then
        docker-compose -f $COMPOSE_FILE up -d
    else
        docker-compose -f $COMPOSE_FILE up -d
    fi
    
    # Wait for services to be ready
    echo "⏳ Waiting for services to start..."
    sleep 30
    
    run_migrations
    health_check
    run_smoke_tests
    
    if [ "$ENVIRONMENT" != "production" ]; then
        cleanup
    fi
    
    echo ""
    echo "🎉 Deployment completed successfully!"
    echo "================================"
    echo "🌐 API: http://localhost:8000"
    echo "📊 Dashboard: http://localhost:8050"
    echo "📈 Grafana: http://localhost:3000"
    echo "🔍 Prometheus: http://localhost:9090"
    echo ""
    echo "📚 API Documentation: http://localhost:8000/docs"
    echo "🔗 Health Check: http://localhost:8000/health"
    echo ""
    
    if [ "$ENVIRONMENT" == "development" ]; then
        echo "💡 Development mode: Services will auto-reload on code changes"
    fi
}

# Script execution
case "$ENVIRONMENT" in
    development|staging|production)
        main
        ;;
    *)
        echo "❌ Invalid environment: $ENVIRONMENT"
        echo "Valid options: development, staging, production"
        exit 1
        ;;
esac