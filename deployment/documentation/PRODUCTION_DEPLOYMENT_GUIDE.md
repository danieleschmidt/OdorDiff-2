# OdorDiff-2 Production Deployment Guide

## 🚀 Quick Start

This guide provides comprehensive instructions for deploying OdorDiff-2 to production environments. The system is designed to achieve 99.99% uptime with enterprise-grade security, monitoring, and disaster recovery capabilities.

## 📋 Prerequisites

### Infrastructure Requirements
- **Kubernetes Cluster**: v1.24+ (EKS, GKE, or AKS recommended)
- **Node Resources**: Minimum 3 nodes with 4 CPU cores and 16GB RAM each
- **Storage**: 1TB+ SSD storage for databases and persistent volumes
- **Network**: Load balancer with SSL termination capabilities

### Tool Requirements
- `kubectl` v1.24+
- `helm` v3.10+
- `terraform` v1.5+ (for infrastructure provisioning)
- `velero` v1.12+ (for backup management)
- `argocd` CLI v2.8+ (for GitOps deployment)

### Access Requirements
- Kubernetes cluster admin access
- Cloud provider credentials (AWS/GCP/Azure)
- DNS management access for domain configuration
- SSL certificate management capabilities

## 🏗️ Architecture Overview

```
                                    ┌─────────────────┐
                                    │   Global CDN    │
                                    │  (CloudFlare)   │
                                    └─────────┬───────┘
                                              │
                                    ┌─────────▼───────┐
                                    │  Load Balancer  │
                                    │   (ALB/GLB)     │
                                    └─────────┬───────┘
                                              │
                            ┌─────────────────┼─────────────────┐
                            │                 │                 │
                    ┌───────▼────────┐ ┌─────▼─────┐ ┌────────▼────────┐
                    │  Istio Gateway │ │   API     │ │  Static Assets  │
                    │   (Ingress)    │ │ Services  │ │     (S3/GCS)    │
                    └───────┬────────┘ └─────┬─────┘ └─────────────────┘
                            │                 │
                    ┌───────▼────────┐       │
                    │ Service Mesh   │       │
                    │    (Istio)     │       │
                    └───────┬────────┘       │
                            │                 │
              ┌─────────────┼─────────────────┼─────────────────┐
              │             │                 │                 │
    ┌─────────▼────┐ ┌─────▼─────┐ ┌────────▼────┐ ┌──────────▼───┐
    │ OdorDiff API │ │  Workers  │ │ PostgreSQL  │ │    Redis     │
    │  (3 replicas)│ │(2 replicas)│ │   (HA)     │ │   (Cluster)  │
    └─────────┬────┘ └───────────┘ └─────────────┘ └──────────────┘
              │
    ┌─────────▼────┐
    │  Monitoring  │
    │ (Prometheus, │
    │  Grafana,    │
    │   Jaeger)    │
    └──────────────┘
```

## 🚦 Deployment Steps

### 1. Infrastructure Provisioning

#### 1.1 Using Terraform (Recommended)

```bash
# Clone the repository
git clone https://github.com/danieleschmidt/odordiff-2.git
cd odordiff-2

# Navigate to Terraform directory
cd deployment/terraform

# Initialize Terraform
terraform init

# Review and customize variables
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your specific values

# Plan deployment
terraform plan -var-file=terraform.tfvars

# Apply infrastructure
terraform apply -var-file=terraform.tfvars
```

#### 1.2 Manual Kubernetes Setup

If not using Terraform, set up your Kubernetes cluster manually:

```bash
# Create namespaces
kubectl apply -f deployment/kubernetes/namespace.yaml

# Apply RBAC and security policies
kubectl apply -f deployment/security/pod-security.yaml
kubectl apply -f deployment/security/network-policies.yaml

# Set up storage classes
kubectl apply -f deployment/kubernetes/storage-class.yaml
```

### 2. Security Configuration

#### 2.1 HashiCorp Vault Setup

```bash
# Deploy Vault
kubectl apply -f deployment/security/vault-config.yaml

# Initialize Vault (one-time setup)
kubectl exec -n security vault-0 -- vault operator init -key-shares=5 -key-threshold=3

# Unseal Vault (repeat with different keys)
kubectl exec -n security vault-0 -- vault operator unseal <unseal-key-1>
kubectl exec -n security vault-0 -- vault operator unseal <unseal-key-2>
kubectl exec -n security vault-0 -- vault operator unseal <unseal-key-3>

# Configure Vault policies
kubectl exec -n security vault-0 -- vault auth enable kubernetes
kubectl exec -n security vault-0 -- vault secrets enable -path=kv kv-v2
```

#### 2.2 SSL/TLS Configuration

```bash
# Install cert-manager
helm repo add jetstack https://charts.jetstack.io
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --set installCRDs=true

# Apply SSL certificates
kubectl apply -f deployment/security/ssl-certificates.yaml
```

### 3. Core Application Deployment

#### 3.1 Using Helm (Recommended)

```bash
# Add OdorDiff-2 Helm repository
helm repo add odordiff https://danieleschmidt.github.io/odordiff-2-helm
helm repo update

# Install with production values
helm install odordiff-2 odordiff/odordiff-2 \
  --namespace odordiff \
  --create-namespace \
  --values deployment/helm/values-production.yaml \
  --wait --timeout=600s
```

#### 3.2 Using GitOps (ArgoCD)

```bash
# Install ArgoCD
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Deploy OdorDiff-2 application
kubectl apply -f deployment/argocd/application.yaml

# Monitor deployment
argocd app sync odordiff-2
argocd app wait odordiff-2 --timeout 600
```

### 4. Monitoring Stack Deployment

```bash
# Deploy Prometheus and Grafana
kubectl apply -f deployment/monitoring/prometheus-config.yaml
kubectl apply -f deployment/monitoring/alerting-rules.yaml

# Import Grafana dashboards
kubectl apply -f deployment/monitoring/grafana-dashboards.json

# Deploy Jaeger for distributed tracing
kubectl apply -f deployment/monitoring/jaeger-config.yaml
```

### 5. Backup and Disaster Recovery

```bash
# Install Velero
velero install \
  --provider aws \
  --plugins velero/velero-plugin-for-aws:v1.8.0 \
  --bucket odordiff-velero-backup \
  --secret-file ./credentials-velero \
  --backup-location-config region=us-west-2

# Deploy backup schedules
kubectl apply -f deployment/backup/backup-strategy.yaml

# Test backup and restore
velero backup create test-backup --include-namespaces odordiff
velero restore create --from-backup test-backup
```

### 6. Performance Optimization

```bash
# Deploy CDN and edge configurations
kubectl apply -f deployment/performance/cdn-config.yaml

# Set up database optimization
kubectl apply -f deployment/performance/database-optimization.yaml

# Configure auto-scaling
kubectl apply -f deployment/kubernetes/hpa.yaml
```

## ✅ Post-Deployment Verification

### 1. Health Checks

```bash
# Check all pods are running
kubectl get pods -n odordiff

# Verify services are accessible
kubectl get services -n odordiff

# Test API health
curl https://api.odordiff.ai/health

# Check monitoring endpoints
curl https://api.odordiff.ai/metrics
```

### 2. Integration Tests

```bash
# Run comprehensive test suite
kubectl run test-runner --rm -i \
  --image=odordiff2:1.0.0 \
  --restart=Never \
  -- python -m pytest tests/integration/ -v

# Test API functionality
curl -X POST https://api.odordiff.ai/api/v1/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"description": "vanilla scent", "num_molecules": 1}'
```

### 3. Performance Validation

```bash
# Run load tests
kubectl apply -f deployment/testing/load-test.yaml

# Check response times
kubectl logs -n odordiff -l app=load-test

# Verify auto-scaling
kubectl get hpa -n odordiff
```

## 🔧 Configuration Management

### Environment Variables

Key environment variables for production:

```bash
# Core Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# Database Configuration
DATABASE_URL=postgresql://user:pass@postgres:5432/odordiff
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Cache Configuration
REDIS_URL=redis://redis:6379/0
CACHE_TTL=3600

# Security Configuration
SECRET_KEY=<strong-random-key>
JWT_SECRET=<jwt-signing-key>
ENCRYPTION_KEY=<encryption-key>

# Monitoring Configuration
PROMETHEUS_ENABLED=true
JAEGER_ENABLED=true
METRICS_PORT=8000

# Performance Configuration
WORKERS=4
WORKER_CLASS=gevent
MAX_REQUESTS=1000
TIMEOUT=30
```

### Resource Limits

Recommended resource allocation:

```yaml
# API Pods
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"

# Worker Pods
resources:
  requests:
    memory: "1.5Gi"
    cpu: "1000m"
  limits:
    memory: "3Gi"
    cpu: "2000m"

# Database
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

## 🚨 Troubleshooting Guide

### Common Issues

#### 1. Pod Startup Failures

```bash
# Check pod status
kubectl describe pod -n odordiff <pod-name>

# View logs
kubectl logs -n odordiff <pod-name> --previous

# Check resource constraints
kubectl top pods -n odordiff
```

#### 2. Database Connection Issues

```bash
# Test database connectivity
kubectl exec -n odordiff deployment/odordiff-api -- \
  python -c "from odordiff2.config.settings import get_database_connection; get_database_connection()"

# Check database logs
kubectl logs -n odordiff -l app=postgresql

# Verify credentials
kubectl get secret postgres-credentials -o yaml
```

#### 3. High Response Times

```bash
# Check API metrics
curl https://api.odordiff.ai/metrics | grep http_request_duration

# Review slow query logs
kubectl exec -n odordiff deployment/postgres -- \
  psql -U odordiff -d odordiff -c "SELECT * FROM slow_queries LIMIT 10;"

# Check cache hit rates
kubectl logs -n odordiff -l app=redis | grep "hit rate"
```

### Emergency Procedures

#### Rolling Back Deployment

```bash
# Using Helm
helm rollback odordiff-2 <previous-revision>

# Using ArgoCD
argocd app rollback odordiff-2 <previous-revision>

# Manual rollback
kubectl rollout undo deployment/odordiff-api -n odordiff
```

#### Database Recovery

```bash
# List available backups
velero backup get

# Restore from backup
velero restore create emergency-restore --from-backup <backup-name>

# Or restore from database backup
kubectl exec -n odordiff deployment/postgres -- \
  pg_restore -U odordiff -d odordiff -c /backup/latest.dump
```

## 📊 Monitoring and Alerting

### Key Metrics to Monitor

1. **Application Metrics**:
   - Request rate and response time
   - Error rate and status codes
   - Active connections and queue length

2. **Infrastructure Metrics**:
   - CPU and memory utilization
   - Disk space and I/O performance
   - Network throughput and latency

3. **Business Metrics**:
   - Daily active users
   - Generation success rate
   - API usage patterns

### Alert Thresholds

```yaml
# Critical Alerts
- API error rate > 1%
- Response time P95 > 1000ms
- Database connections > 90%
- Disk space < 10%

# Warning Alerts  
- API error rate > 0.1%
- Response time P95 > 500ms
- Memory usage > 80%
- Cache hit rate < 90%
```

### Grafana Dashboards

Access Grafana at `https://grafana.odordiff.ai` with the following dashboards:

1. **OdorDiff-2 Overview**: High-level system metrics
2. **API Performance**: Request metrics and error rates
3. **Infrastructure**: Server and database metrics
4. **Business Metrics**: Usage and conversion analytics

## 🔄 Maintenance Procedures

### Regular Maintenance Tasks

#### Daily
- Review monitoring alerts and dashboards
- Check backup completion status
- Monitor resource utilization trends
- Review application logs for errors

#### Weekly
- Update security patches
- Review and optimize database queries
- Test disaster recovery procedures
- Update documentation as needed

#### Monthly
- Review and update resource allocations
- Analyze performance trends and optimization opportunities
- Update SSL certificates (if not automated)
- Conduct security reviews and audits

### Scaling Operations

#### Horizontal Scaling

```bash
# Scale API pods
kubectl scale deployment odordiff-api -n odordiff --replicas=5

# Scale workers
kubectl scale deployment odordiff-worker -n odordiff --replicas=4

# Update HPA limits
kubectl patch hpa odordiff-api-hpa -n odordiff -p '{"spec":{"maxReplicas":10}}'
```

#### Vertical Scaling

```bash
# Update resource requests/limits
kubectl patch deployment odordiff-api -n odordiff -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"odordiff-api","resources":{"requests":{"memory":"2Gi","cpu":"1000m"},"limits":{"memory":"4Gi","cpu":"2000m"}}}]}}}}'
```

## 🔒 Security Best Practices

### Production Security Checklist

- [ ] All secrets stored in HashiCorp Vault
- [ ] SSL/TLS enabled for all communications
- [ ] Network policies restricting pod-to-pod communication
- [ ] Pod security policies enforced
- [ ] RBAC configured with principle of least privilege
- [ ] Regular security scanning enabled
- [ ] Audit logging configured
- [ ] WAF rules configured and active
- [ ] Rate limiting implemented
- [ ] Input validation and sanitization active

### Compliance Requirements

#### SOC 2 Compliance
- Continuous monitoring and logging
- Access controls and authentication
- Data encryption at rest and in transit
- Regular security assessments
- Incident response procedures

#### GDPR Compliance
- Data processing records
- User consent management
- Data portability features
- Right to be forgotten implementation
- Data breach notification procedures

## 📞 Support and Contact Information

### Emergency Contacts
- **On-Call Engineer**: +1-XXX-XXX-XXXX
- **Platform Team Lead**: platform-lead@odordiff.ai
- **Security Team**: security@odordiff.ai

### Support Channels
- **Slack**: #odordiff-production
- **Email**: support@odordiff.ai
- **Documentation**: https://docs.odordiff.ai
- **Status Page**: https://status.odordiff.ai

### Escalation Procedures

1. **Level 1**: Infrastructure issues, basic troubleshooting
2. **Level 2**: Application issues, complex debugging  
3. **Level 3**: Security incidents, data corruption
4. **Emergency**: Complete service outage, data breach

---

## 📚 Additional Resources

- [API Documentation](https://api.odordiff.ai/docs)
- [Architecture Decision Records](./ADRs/)
- [Security Guidelines](./SECURITY.md)
- [Performance Tuning Guide](./PERFORMANCE.md)
- [Disaster Recovery Runbook](./DISASTER_RECOVERY.md)
- [Monitoring Playbook](./MONITORING.md)

---

**Document Version**: 1.0.0  
**Last Updated**: August 9, 2025  
**Next Review Date**: November 9, 2025

*This document is maintained by the OdorDiff-2 Platform Team. For updates or corrections, please submit a pull request or contact the team directly.*