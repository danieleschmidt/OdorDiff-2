# OdorDiff-2 Operations Runbook

## 🎯 Purpose

This runbook provides step-by-step procedures for operating, monitoring, and troubleshooting the OdorDiff-2 production system. It serves as the primary reference for on-call engineers and operations teams.

## 📞 Emergency Response

### Immediate Response Protocol

```
1. ACKNOWLEDGE the alert within 5 minutes
2. ASSESS the severity and impact
3. ESCALATE if needed (see escalation matrix)
4. COMMUNICATE status updates every 15 minutes
5. DOCUMENT all actions taken
```

### Escalation Matrix

| Severity | Response Time | Escalation |
|----------|---------------|------------|
| Critical | 5 minutes | Immediate escalation to on-call engineer |
| High | 15 minutes | Escalate after 30 minutes if unresolved |
| Medium | 1 hour | Escalate after 2 hours if unresolved |
| Low | 4 hours | Escalate after 1 business day |

### Emergency Contacts

```
Primary On-Call: +1-XXX-XXX-XXXX (Slack: @oncall-primary)
Secondary On-Call: +1-XXX-XXX-YYYY (Slack: @oncall-secondary)
Platform Lead: platform-lead@odordiff.ai (Slack: @platform-lead)
Security Team: security@odordiff.ai (Slack: @security-team)
```

## 🚨 Alert Response Procedures

### Critical Alerts

#### ALERT: API Service Down

**Symptoms**: All API endpoints returning errors, health checks failing

**Investigation Steps**:
```bash
# 1. Check pod status
kubectl get pods -n odordiff -l app=odordiff-api

# 2. Describe failing pods
kubectl describe pod -n odordiff <failing-pod-name>

# 3. Check logs
kubectl logs -n odordiff -l app=odordiff-api --tail=100

# 4. Check service endpoints
kubectl get endpoints -n odordiff odordiff-api

# 5. Test database connectivity
kubectl exec -n odordiff deployment/odordiff-api -- \
  python -c "import psycopg2; psycopg2.connect('host=postgres port=5432 dbname=odordiff user=odordiff')"
```

**Common Causes & Solutions**:
- **Database connection failure**: Restart database, check credentials
- **Memory/CPU exhaustion**: Scale up pods, check resource limits
- **Configuration errors**: Rollback recent changes, verify ConfigMaps
- **Network issues**: Check network policies, service mesh configuration

**Recovery Actions**:
```bash
# Restart failing pods
kubectl rollout restart deployment/odordiff-api -n odordiff

# Scale up if needed
kubectl scale deployment odordiff-api -n odordiff --replicas=5

# Emergency rollback
helm rollback odordiff-2
```

#### ALERT: Database Connection Pool Exhausted

**Symptoms**: High database connection errors, slow response times

**Investigation Steps**:
```bash
# 1. Check connection pool status
kubectl exec -n odordiff deployment/postgres -- \
  psql -U odordiff -d odordiff -c "SELECT * FROM pg_stat_activity WHERE state = 'active';"

# 2. Check PgBouncer metrics
kubectl exec -n odordiff deployment/pgbouncer -- \
  psql -p 5432 -U pgbouncer pgbouncer -c "SHOW POOLS;"

# 3. Review slow queries
kubectl exec -n odordiff deployment/postgres -- \
  psql -U odordiff -d odordiff -c "SELECT * FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;"
```

**Recovery Actions**:
```bash
# Restart PgBouncer to reset connections
kubectl rollout restart deployment/pgbouncer -n odordiff

# Terminate long-running queries
kubectl exec -n odordiff deployment/postgres -- \
  psql -U odordiff -d odordiff -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE query_start < NOW() - INTERVAL '5 minutes';"

# Scale up database resources
kubectl patch deployment postgres -n odordiff -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"postgres","resources":{"limits":{"memory":"4Gi","cpu":"2000m"}}}]}}}}'
```

#### ALERT: High Error Rate (>1%)

**Investigation Steps**:
```bash
# 1. Check error breakdown by endpoint
curl -s https://api.odordiff.ai/metrics | grep http_requests_total | grep 5..

# 2. Review application logs
kubectl logs -n odordiff -l app=odordiff-api --since=10m | grep ERROR

# 3. Check downstream services
kubectl get pods -n odordiff -l app=redis
kubectl get pods -n odordiff -l app=postgresql

# 4. Review recent deployments
kubectl rollout history deployment/odordiff-api -n odordiff
```

**Common Causes**:
- Recent deployment introducing bugs
- Downstream service failures (Redis, PostgreSQL)
- Resource exhaustion (memory, CPU)
- External API dependencies failing

### High Priority Alerts

#### ALERT: High Response Time (P95 > 1000ms)

**Investigation Steps**:
```bash
# 1. Check current response times
curl -s https://api.odordiff.ai/metrics | grep http_request_duration_seconds

# 2. Identify slow endpoints
kubectl logs -n odordiff -l app=odordiff-api --since=5m | grep "duration.*[5-9][0-9][0-9][0-9]"

# 3. Check database performance
kubectl exec -n odordiff deployment/postgres -- \
  psql -U odordiff -d odordiff -c "SELECT * FROM slow_queries LIMIT 5;"

# 4. Verify cache performance
kubectl exec -n odordiff deployment/redis -- redis-cli info stats | grep hit_rate
```

**Optimization Actions**:
```bash
# Scale horizontally
kubectl patch hpa odordiff-api-hpa -n odordiff -p '{"spec":{"minReplicas":5}}'

# Warm cache
kubectl create job cache-warming-emergency --from=cronjob/cdn-cache-warming -n odordiff

# Optimize database
kubectl create job emergency-vacuum --from=cronjob/database-performance-monitoring -n odordiff
```

#### ALERT: Disk Space Low (<10%)

**Investigation Steps**:
```bash
# 1. Check disk usage across pods
kubectl exec -n odordiff deployment/postgres -- df -h
kubectl exec -n odordiff deployment/redis -- df -h

# 2. Identify large files
kubectl exec -n odordiff deployment/postgres -- \
  find /var/lib/postgresql/data -type f -size +100M -exec ls -lh {} \;

# 3. Check log file sizes
kubectl exec -n odordiff -l app=odordiff-api -- \
  du -sh /app/logs/*
```

**Recovery Actions**:
```bash
# Clean up old log files
kubectl exec -n odordiff -l app=odordiff-api -- \
  find /app/logs -name "*.log" -mtime +7 -delete

# Clean up old database logs
kubectl exec -n odordiff deployment/postgres -- \
  find /var/lib/postgresql/data/log -name "*.log" -mtime +3 -delete

# Expand persistent volumes (if possible)
kubectl patch pvc postgres-data-postgres-0 -n odordiff -p \
  '{"spec":{"resources":{"requests":{"storage":"200Gi"}}}}'
```

### Warning Alerts

#### ALERT: Cache Hit Rate Low (<90%)

**Investigation Steps**:
```bash
# 1. Check Redis statistics
kubectl exec -n odordiff deployment/redis -- redis-cli info stats

# 2. Check cache configuration
kubectl get configmap -n odordiff odordiff-config -o yaml | grep -A5 cache

# 3. Review cache key patterns
kubectl exec -n odordiff deployment/redis -- redis-cli --scan --pattern "*" | head -20
```

**Optimization Actions**:
```bash
# Adjust cache TTL
kubectl patch configmap odordiff-config -n odordiff --type merge -p \
  '{"data":{"cache_ttl":"7200"}}'

# Warm cache for popular content
kubectl create job manual-cache-warming --from=cronjob/cdn-cache-warming -n odordiff

# Restart application to pick up config changes
kubectl rollout restart deployment/odordiff-api -n odordiff
```

## 🔧 Operational Procedures

### Deployment Procedures

#### Standard Deployment

```bash
# 1. Pre-deployment checks
kubectl get pods -n odordiff
kubectl get pdb -n odordiff

# 2. Create deployment record
echo "Deployment started at $(date)" >> /tmp/deployment-log.txt

# 3. Deploy using Helm
helm upgrade odordiff-2 odordiff/odordiff-2 \
  --namespace odordiff \
  --values deployment/helm/values-production.yaml \
  --wait --timeout=600s

# 4. Verify deployment
kubectl rollout status deployment/odordiff-api -n odordiff
curl https://api.odordiff.ai/health

# 5. Run smoke tests
kubectl run smoke-test --rm -i \
  --image=odordiff2:latest \
  --restart=Never \
  -- python -m pytest tests/smoke/ -v
```

#### Emergency Rollback

```bash
# 1. Immediate rollback
helm rollback odordiff-2

# 2. Verify rollback
kubectl get pods -n odordiff
kubectl rollout status deployment/odordiff-api -n odordiff

# 3. Check application health
curl https://api.odordiff.ai/health
curl https://api.odordiff.ai/ready

# 4. Monitor for 15 minutes
watch -n 30 'kubectl get pods -n odordiff | grep -v Running'
```

### Scaling Procedures

#### Manual Scaling

```bash
# Scale API pods
kubectl scale deployment odordiff-api -n odordiff --replicas=<desired-count>

# Scale worker pods  
kubectl scale deployment odordiff-worker -n odordiff --replicas=<desired-count>

# Verify scaling
kubectl get pods -n odordiff
kubectl top pods -n odordiff
```

#### Auto-scaling Configuration

```bash
# Update HPA settings
kubectl patch hpa odordiff-api-hpa -n odordiff -p \
  '{"spec":{"minReplicas":3,"maxReplicas":20,"targetCPUUtilizationPercentage":70}}'

# Check HPA status
kubectl get hpa -n odordiff
kubectl describe hpa odordiff-api-hpa -n odordiff
```

### Database Operations

#### Database Maintenance

```bash
# 1. Create maintenance window
kubectl annotate deployment odordiff-api -n odordiff \
  maintenance="true" --overwrite

# 2. Scale down to minimum
kubectl scale deployment odordiff-api -n odordiff --replicas=1

# 3. Run VACUUM and ANALYZE
kubectl exec -n odordiff deployment/postgres -- \
  psql -U odordiff -d odordiff -c "VACUUM ANALYZE;"

# 4. Update statistics
kubectl exec -n odordiff deployment/postgres -- \
  psql -U odordiff -d odordiff -c "ANALYZE;"

# 5. Scale back up
kubectl scale deployment odordiff-api -n odordiff --replicas=3

# 6. Remove maintenance annotation
kubectl annotate deployment odordiff-api -n odordiff maintenance-
```

#### Database Backup Verification

```bash
# 1. List recent backups
velero backup get | head -5

# 2. Test latest backup
LATEST_BACKUP=$(velero backup get -o json | jq -r '.items[0].metadata.name')
velero restore create test-restore-$(date +%s) --from-backup $LATEST_BACKUP

# 3. Verify restore success
velero restore get | head -3

# 4. Clean up test restore
velero restore delete test-restore-$(date +%s)
```

### Certificate Management

#### SSL Certificate Renewal

```bash
# 1. Check certificate expiry
kubectl get certificate -n odordiff

# 2. Force renewal if needed
kubectl delete secret odordiff-tls -n odordiff
kubectl annotate certificate odordiff-tls -n odordiff \
  cert-manager.io/issue-temporary-certificate=true --overwrite

# 3. Verify new certificate
kubectl get certificate -n odordiff
openssl s_client -connect api.odordiff.ai:443 -servername api.odordiff.ai </dev/null 2>/dev/null | \
  openssl x509 -noout -dates
```

## 🔍 Monitoring and Observability

### Key Metrics to Monitor

#### Application Metrics
```bash
# Request rate
curl -s https://api.odordiff.ai/metrics | grep http_requests_total

# Response time percentiles  
curl -s https://api.odordiff.ai/metrics | grep http_request_duration_seconds

# Error rates
curl -s https://api.odordiff.ai/metrics | grep http_requests_total | grep "5.."

# Active connections
curl -s https://api.odordiff.ai/metrics | grep http_connections_active
```

#### Infrastructure Metrics
```bash
# Pod resource usage
kubectl top pods -n odordiff

# Node resource usage
kubectl top nodes

# Persistent volume usage
kubectl exec -n odordiff deployment/postgres -- df -h /var/lib/postgresql/data
```

#### Business Metrics
```bash
# Daily active users
curl -s https://api.odordiff.ai/metrics | grep odordiff_daily_active_users

# Generation requests
curl -s https://api.odordiff.ai/metrics | grep odordiff_generation_requests_total

# Success rate
curl -s https://api.odordiff.ai/metrics | grep odordiff_generation_success_total
```

### Log Analysis

#### Application Logs
```bash
# Recent errors
kubectl logs -n odordiff -l app=odordiff-api --since=1h | grep ERROR

# Slow requests
kubectl logs -n odordiff -l app=odordiff-api --since=1h | grep "duration.*[0-9][0-9][0-9][0-9]"

# Authentication failures
kubectl logs -n odordiff -l app=odordiff-api --since=1h | grep "401\|403"
```

#### Database Logs
```bash
# Connection errors
kubectl logs -n odordiff -l app=postgresql --since=1h | grep "connection\|error"

# Slow queries
kubectl exec -n odordiff deployment/postgres -- \
  tail -n 100 /var/lib/postgresql/data/log/postgresql-$(date +%Y-%m-%d).log | grep "duration.*ms"

# Deadlocks
kubectl logs -n odordiff -l app=postgresql --since=1h | grep deadlock
```

### Distributed Tracing

#### Jaeger Queries
```bash
# Access Jaeger UI
kubectl port-forward -n monitoring svc/jaeger-query 16686:16686

# Then visit: http://localhost:16686

# Common trace queries:
# - Service: odordiff-api
# - Operation: POST /api/v1/generate  
# - Tags: error=true (for error traces)
# - Lookback: Last hour
```

## 🛠️ Maintenance Tasks

### Daily Tasks
- [ ] Review monitoring dashboards for anomalies
- [ ] Check backup completion status
- [ ] Review error logs and alerts
- [ ] Monitor resource utilization trends
- [ ] Verify SSL certificate status

### Weekly Tasks
- [ ] Run database maintenance procedures
- [ ] Review and update resource quotas
- [ ] Check for security updates
- [ ] Test disaster recovery procedures
- [ ] Review performance metrics and trends

### Monthly Tasks
- [ ] Update SSL certificates (if not automated)
- [ ] Review and optimize database indexes
- [ ] Conduct security reviews
- [ ] Update operational documentation
- [ ] Review capacity planning metrics

### Quarterly Tasks
- [ ] Conduct full disaster recovery test
- [ ] Review and update monitoring thresholds
- [ ] Security audit and penetration testing
- [ ] Performance baseline review
- [ ] Infrastructure cost optimization review

## 🔐 Security Operations

### Security Incident Response

#### Suspected Security Breach

```bash
# 1. Immediate containment
kubectl scale deployment odordiff-api -n odordiff --replicas=0

# 2. Collect evidence
kubectl logs -n odordiff -l app=odordiff-api --since=24h > incident-logs-$(date +%s).txt
kubectl get events -n odordiff --sort-by=.metadata.creationTimestamp > incident-events-$(date +%s).txt

# 3. Isolate affected pods
kubectl label pod <suspicious-pod> quarantine=true -n odordiff
kubectl patch networkpolicy default-deny-all -n odordiff -p \
  '{"spec":{"podSelector":{"matchLabels":{"quarantine":"true"}}}}'

# 4. Notify security team
# Send alert to security@odordiff.ai with collected evidence

# 5. Begin forensic analysis
kubectl cp <suspicious-pod>:/app/logs ./forensic-analysis/ -n odordiff
```

### Access Control Management

#### User Access Review
```bash
# List all service accounts
kubectl get serviceaccount -A

# Review RBAC permissions
kubectl get clusterrolebindings -o yaml | grep -A5 -B5 odordiff

# Check Vault access policies
kubectl exec -n security vault-0 -- vault list auth/kubernetes/role
```

#### Rotate Secrets
```bash
# Generate new API keys
kubectl create secret generic new-api-keys \
  --from-literal=api-key="$(openssl rand -hex 32)" \
  -n odordiff

# Update application to use new keys
kubectl patch deployment odordiff-api -n odordiff -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"odordiff-api","env":[{"name":"API_KEY","valueFrom":{"secretKeyRef":{"name":"new-api-keys","key":"api-key"}}}]}]}}}}'

# Remove old secrets
kubectl delete secret old-api-keys -n odordiff
```

## 📋 Checklists

### Pre-Deployment Checklist
- [ ] Backup current state
- [ ] Review changes in staging environment
- [ ] Check resource availability
- [ ] Verify all dependencies are healthy
- [ ] Confirm rollback procedure
- [ ] Schedule deployment window
- [ ] Notify relevant teams

### Post-Incident Checklist
- [ ] Document incident timeline
- [ ] Identify root cause
- [ ] Implement corrective measures
- [ ] Update monitoring/alerting
- [ ] Conduct post-mortem meeting
- [ ] Update runbooks with lessons learned
- [ ] Communicate resolution to stakeholders

### Disaster Recovery Checklist
- [ ] Assess extent of outage
- [ ] Activate disaster recovery team
- [ ] Communicate with stakeholders
- [ ] Execute recovery procedures
- [ ] Verify system functionality
- [ ] Document recovery process
- [ ] Conduct lessons learned session

## 📞 Contact Information

### Team Contacts
```
Primary On-Call: +1-XXX-XXX-XXXX (Slack: @oncall-primary)
Secondary On-Call: +1-XXX-XXX-YYYY (Slack: @oncall-secondary)
Platform Team Lead: platform-lead@odordiff.ai
Database Administrator: dba@odordiff.ai
Security Team: security@odordiff.ai
Network Operations: netops@odordiff.ai
```

### Vendor Contacts
```
Cloud Provider Support: Available in cloud console
CDN Provider (Cloudflare): Available in Cloudflare dashboard
Monitoring Vendor: support@monitoring-vendor.com
SSL Certificate Provider: support@ssl-provider.com
```

### External Resources
```
Status Page: https://status.odordiff.ai
Documentation: https://docs.odordiff.ai
Monitoring Dashboard: https://grafana.odordiff.ai
Log Aggregation: https://logs.odordiff.ai
```

---

## 📚 Additional Resources

- [Architecture Documentation](./ARCHITECTURE.md)
- [Security Procedures](./SECURITY_PROCEDURES.md)
- [Performance Tuning Guide](./PERFORMANCE_TUNING.md)
- [Monitoring Playbook](./MONITORING_PLAYBOOK.md)
- [Disaster Recovery Plan](./DISASTER_RECOVERY.md)

---

**Document Version**: 1.0.0  
**Last Updated**: August 9, 2025  
**Next Review Date**: November 9, 2025  
**Owner**: Platform Engineering Team

*This runbook is a living document. Please update it with new procedures, lessons learned, and process improvements as they are discovered.*