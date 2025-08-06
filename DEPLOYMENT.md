# OdorDiff-2 Deployment Guide

This guide covers deploying OdorDiff-2 in various environments from development to production.

## Quick Start

### Development Deployment

```bash
# Clone repository
git clone https://github.com/danieleschmidt/odordiff-2.git
cd odordiff-2

# Copy environment template
cp .env.example .env.development

# Deploy with Docker Compose
./scripts/deploy.sh development
```

### Production Deployment

```bash
# Prepare production environment
cp .env.example .env.production
# Edit .env.production with production values

# Deploy to production
./scripts/deploy.sh production
```

## Architecture Overview

OdorDiff-2 uses a microservices architecture with the following components:

- **API Service**: FastAPI-based REST API for molecule generation
- **Dashboard**: Interactive Dash-based web interface
- **Database**: PostgreSQL for persistent data storage
- **Cache**: Redis for caching and session management
- **Reverse Proxy**: Nginx for load balancing and SSL termination
- **Monitoring**: Prometheus + Grafana for metrics and dashboards

## Prerequisites

### System Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 50GB+ available space
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2

### Software Requirements

- Docker 20.10+
- Docker Compose 2.0+
- Git
- Bash (for deployment scripts)
- curl (for health checks)

### Optional Requirements

- NVIDIA GPU with CUDA support (for accelerated inference)
- SSL certificates (for HTTPS in production)

## Environment Configuration

### Development Environment

Create `.env.development`:

```env
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=info
DATABASE_URL=postgresql://postgres:password@postgres:5432/odordiff2
REDIS_URL=redis://redis:6379/0
API_RELOAD=true
```

### Staging Environment

Create `.env.staging`:

```env
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=warning
DATABASE_URL=postgresql://postgres:secure_password@postgres:5432/odordiff2
REDIS_URL=redis://redis:6379/0
API_RELOAD=false
SECRET_KEY=staging-secret-key
```

### Production Environment

Create `.env.production`:

```env
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=error
DATABASE_URL=postgresql://postgres:very_secure_password@postgres:5432/odordiff2
REDIS_URL=redis://redis:6379/0
API_RELOAD=false
SECRET_KEY=production-secret-key-very-secure
JWT_SECRET_KEY=jwt-secret-key-very-secure
CORS_ORIGINS=["https://yourdomain.com"]
DATABASE_SSL_REQUIRE=true
```

## Deployment Methods

### Method 1: Docker Compose (Recommended)

This is the easiest method for most deployments:

```bash
# Deploy to development
./scripts/deploy.sh development

# Deploy to production
./scripts/deploy.sh production
```

The deployment script handles:
- Dependency checking
- Environment validation
- Database migrations
- Health checks
- Smoke tests

### Method 2: Manual Docker Deployment

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# Check health
curl http://localhost:8000/health
```

### Method 3: Kubernetes Deployment

For Kubernetes deployment, use the provided manifests:

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=odordiff2
```

### Method 4: Cloud Deployment

#### AWS ECS Deployment

```bash
# Configure AWS CLI
aws configure

# Deploy to ECS
./scripts/deploy-aws.sh
```

#### Google Cloud Run Deployment

```bash
# Configure gcloud CLI
gcloud auth login

# Deploy to Cloud Run
./scripts/deploy-gcp.sh
```

#### Azure Container Instances

```bash
# Configure Azure CLI
az login

# Deploy to Azure
./scripts/deploy-azure.sh
```

## Database Setup

### PostgreSQL Configuration

The system uses PostgreSQL for persistent storage. The database is automatically initialized with:

- Molecule cache tables
- Generation request tracking
- Safety assessment history
- Synthesis route storage
- User session management
- System metrics

### Database Migrations

Migrations are automatically run during deployment. For manual migration:

```bash
# Run database migrations
docker-compose exec odordiff2-api python -m odordiff2.data.migrations
```

### Database Backup

```bash
# Create backup
docker-compose exec postgres pg_dump -U postgres odordiff2 > backup_$(date +%Y%m%d).sql

# Restore backup
docker-compose exec -T postgres psql -U postgres odordiff2 < backup_20250101.sql
```

## SSL/HTTPS Configuration

### Development (Self-Signed)

```bash
# Generate self-signed certificates
mkdir -p ssl
openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes
```

### Production (Let's Encrypt)

```bash
# Install certbot
sudo apt-get install certbot

# Generate certificates
sudo certbot certonly --standalone -d yourdomain.com

# Copy certificates
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ssl/cert.pem
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ssl/key.pem
```

Update `nginx.conf` to enable HTTPS configuration.

## Monitoring Setup

### Prometheus Configuration

Prometheus is automatically configured to scrape metrics from:
- OdorDiff-2 API endpoints
- System metrics
- Database metrics
- Redis metrics

### Grafana Dashboards

Access Grafana at `http://localhost:3000`:
- Username: `admin`
- Password: `admin` (change in production)

Pre-configured dashboards include:
- API Performance
- System Resources
- Cache Hit Rates
- Error Rates

### Custom Metrics

Add custom metrics in your code:

```python
from odordiff2.utils.logging import get_logger

logger = get_logger(__name__)
logger.record_metric("generation_requests_total", 1.0, {"status": "success"})
```

## Performance Tuning

### CPU Optimization

```env
# Increase worker processes
API_WORKERS=8
MODEL_MAX_WORKERS=8

# Enable CPU affinity
WORKER_CPU_AFFINITY=true
```

### Memory Optimization

```env
# Adjust cache sizes
CACHE_MAX_SIZE=2000
MODEL_CACHE_SIZE=500MB

# Configure garbage collection
PYTHONGC=1
```

### GPU Acceleration

```env
# Enable GPU support
MODEL_DEVICE=cuda
CUDA_VISIBLE_DEVICES=0,1
```

Update Docker Compose to use GPU runtime:

```yaml
services:
  odordiff2-api:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

## Security Considerations

### Network Security

- Use firewalls to restrict access to internal services
- Enable SSL/TLS for all external communications
- Use VPN for administrative access

### Application Security

- Change default passwords and secrets
- Enable CORS only for trusted domains
- Implement rate limiting
- Regular security updates

### Data Protection

```env
# Enable encryption
DATABASE_ENCRYPT=true
REDIS_ENCRYPT=true
LOG_ENCRYPT=true

# Configure backup encryption
BACKUP_ENCRYPT=true
BACKUP_KEY=your-backup-encryption-key
```

## Scaling

### Horizontal Scaling

```yaml
# Scale API service
services:
  odordiff2-api:
    deploy:
      replicas: 4
```

### Load Balancing

Configure Nginx for load balancing:

```nginx
upstream api_backend {
    server odordiff2-api-1:8000;
    server odordiff2-api-2:8000;
    server odordiff2-api-3:8000;
    server odordiff2-api-4:8000;
}
```

### Database Scaling

- Configure read replicas for PostgreSQL
- Use Redis Cluster for cache scaling
- Consider database sharding for very large datasets

## Troubleshooting

### Common Issues

#### Service Won't Start

```bash
# Check logs
docker-compose logs odordiff2-api

# Check service status
docker-compose ps
```

#### Database Connection Issues

```bash
# Test database connection
docker-compose exec odordiff2-api python -c "
import asyncio
import asyncpg
asyncio.run(asyncpg.connect('postgresql://postgres:password@postgres:5432/odordiff2'))
"
```

#### Performance Issues

```bash
# Monitor resource usage
docker stats

# Check API metrics
curl http://localhost:8000/stats
```

### Debug Mode

Enable debug mode for troubleshooting:

```env
DEBUG=true
LOG_LEVEL=debug
API_RELOAD=true
```

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Database health
docker-compose exec postgres pg_isready -U postgres

# Redis health
docker-compose exec redis redis-cli ping
```

## Backup and Recovery

### Automated Backups

```bash
# Schedule daily backups
crontab -e
0 2 * * * /path/to/scripts/backup.sh
```

### Disaster Recovery

```bash
# Full system restore
./scripts/restore.sh backup_20250101.tar.gz
```

### Data Migration

```bash
# Migrate to new environment
./scripts/migrate.sh source_env target_env
```

## Updates and Maintenance

### Regular Updates

```bash
# Update to latest version
git pull origin main
./scripts/deploy.sh production

# Update dependencies
docker-compose pull
docker-compose up -d
```

### Maintenance Windows

```bash
# Enable maintenance mode
./scripts/maintenance.sh enable

# Perform maintenance
./scripts/update.sh

# Disable maintenance mode
./scripts/maintenance.sh disable
```

### Zero-Downtime Deployment

```bash
# Blue-green deployment
./scripts/deploy-blue-green.sh production
```

## Support

For deployment issues:

1. Check the logs: `docker-compose logs`
2. Review health checks: `curl http://localhost:8000/health`
3. Consult troubleshooting section above
4. Open an issue: https://github.com/danieleschmidt/odordiff-2/issues

For production support, contact: support@terragonlabs.ai