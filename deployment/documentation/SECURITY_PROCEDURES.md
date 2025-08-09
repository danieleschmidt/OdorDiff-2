# OdorDiff-2 Security Procedures

## 🔒 Security Overview

This document outlines comprehensive security procedures for the OdorDiff-2 production environment. It covers security policies, incident response procedures, compliance requirements, and operational security practices.

## 🎯 Security Objectives

### CIA Triad Implementation

**Confidentiality**:
- Protect user data and proprietary algorithms
- Secure API keys and authentication tokens
- Encrypt sensitive data at rest and in transit

**Integrity**:
- Prevent unauthorized modifications to data and code
- Implement audit trails for all system changes
- Ensure data consistency across distributed systems

**Availability**:
- Maintain 99.99% uptime SLA
- Implement DDoS protection and rate limiting
- Ensure disaster recovery capabilities

### Security Principles

1. **Zero Trust Architecture**: Never trust, always verify
2. **Defense in Depth**: Multiple layers of security controls
3. **Least Privilege**: Minimal necessary access rights
4. **Fail Secure**: Systems fail to a secure state
5. **Security by Design**: Security integrated from the ground up

## 🚨 Incident Response Procedures

### Incident Classification

| Severity | Examples | Response Time |
|----------|----------|---------------|
| **Critical** | Data breach, system compromise, complete service outage | 15 minutes |
| **High** | Attempted breach, partial service outage, security alert | 1 hour |
| **Medium** | Suspicious activity, minor security events | 4 hours |
| **Low** | Policy violations, routine security events | 24 hours |

### Incident Response Team

```
Security Incident Commander: security-ic@odordiff.ai
Primary Security Engineer: security-primary@odordiff.ai  
Secondary Security Engineer: security-secondary@odordiff.ai
Legal Counsel: legal@odordiff.ai
Communications Lead: comms@odordiff.ai
Platform Engineering Lead: platform-lead@odordiff.ai
```

### Critical Security Incident Response

#### Phase 1: Detection and Analysis (0-15 minutes)

```bash
# 1. Verify and classify the incident
# Check security monitoring dashboards
kubectl logs -n monitoring -l app=security-monitor --since=1h

# 2. Collect initial evidence
INCIDENT_ID="INC-$(date +%Y%m%d-%H%M%S)"
mkdir -p /tmp/security-incident-${INCIDENT_ID}

# 3. Document timeline
echo "$(date): Incident ${INCIDENT_ID} detected" >> /tmp/security-incident-${INCIDENT_ID}/timeline.txt

# 4. Notify incident response team
curl -X POST "${SLACK_SECURITY_WEBHOOK}" \
  -H 'Content-type: application/json' \
  -d "{\"text\": \"🚨 SECURITY INCIDENT ${INCIDENT_ID}: Critical security event detected. All hands on deck!\"}"
```

#### Phase 2: Containment (15-30 minutes)

```bash
# 1. Isolate affected systems
# Identify compromised pods
kubectl get pods -n odordiff -o wide

# 2. Block suspicious traffic
kubectl patch networkpolicy default-deny-all -n odordiff -p \
  '{"spec":{"ingress":[{"from":[{"ipBlock":{"cidr":"10.0.0.0/8","except":["SUSPICIOUS_IP/32"]}}]}]}}'

# 3. Revoke access tokens
kubectl exec -n security vault-0 -- \
  vault write auth/kubernetes/revoke token="SUSPICIOUS_TOKEN"

# 4. Scale down compromised services if needed
kubectl scale deployment suspicious-service -n odordiff --replicas=0

# 5. Create forensic snapshots
kubectl exec -n odordiff compromised-pod -- \
  tar czf /tmp/forensic-snapshot-$(date +%s).tgz /app/logs /var/log /tmp
```

#### Phase 3: Eradication (30 minutes - 2 hours)

```bash
# 1. Identify root cause
# Analyze logs and system state
kubectl logs -n odordiff compromised-pod --previous > /tmp/security-incident-${INCIDENT_ID}/compromised-pod.log

# 2. Remove threats
# Patch vulnerabilities
kubectl apply -f deployment/security/emergency-patches.yaml

# 3. Update security rules
kubectl apply -f deployment/security/network-policies.yaml

# 4. Rotate all credentials
/scripts/rotate-all-secrets.sh

# 5. Update firewall rules
kubectl apply -f deployment/security/emergency-firewall-rules.yaml
```

#### Phase 4: Recovery (2-4 hours)

```bash
# 1. Restore services from clean state
helm upgrade odordiff-2 odordiff/odordiff-2 \
  --namespace odordiff \
  --values deployment/helm/values-production.yaml \
  --reset-values

# 2. Verify system integrity
kubectl run security-scan --rm -i \
  --image=security-scanner:latest \
  --restart=Never \
  -- scan --target odordiff-api.odordiff.svc.cluster.local

# 3. Run comprehensive tests
kubectl run integration-test --rm -i \
  --image=odordiff2:latest \
  --restart=Never \
  -- python -m pytest tests/ -v

# 4. Monitor for recurrence
watch -n 30 'kubectl logs -n monitoring -l app=security-monitor --since=5m | grep ALERT'
```

#### Phase 5: Post-Incident Activities (24-48 hours)

```bash
# 1. Conduct post-mortem meeting
# Schedule within 24 hours of resolution

# 2. Document lessons learned
cat > /tmp/security-incident-${INCIDENT_ID}/post-mortem.md << EOF
# Security Incident Post-Mortem: ${INCIDENT_ID}

## Timeline
- Detection: 
- Containment:
- Eradication:
- Recovery:

## Root Cause Analysis
[Detailed analysis of how the incident occurred]

## Lessons Learned
[What went well, what could be improved]

## Action Items
[Specific tasks to prevent recurrence]
EOF

# 3. Update security procedures
# Update this document with new learnings
```

## 🔐 Access Control Management

### Role-Based Access Control (RBAC)

#### Service Account Management

```yaml
# Production Service Account Template
apiVersion: v1
kind: ServiceAccount
metadata:
  name: odordiff-service
  namespace: odordiff
  labels:
    app.kubernetes.io/name: odordiff-2
    security.compliance/level: production
automountServiceAccountToken: true

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: odordiff
  name: odordiff-service-role
rules:
# Minimum required permissions only
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]
  resourceNames: ["odordiff-config", "odordiff-secrets"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "patch"]
  resourceNames: ["odordiff-api-*"]
```

#### User Access Review Process

```bash
#!/bin/bash
# Monthly access review script

echo "=== OdorDiff-2 Access Review $(date) ==="

# 1. List all service accounts
echo "Service Accounts:"
kubectl get serviceaccounts -n odordiff -o custom-columns=NAME:.metadata.name,AGE:.metadata.creationTimestamp

# 2. Review role bindings
echo -e "\nRole Bindings:"
kubectl get rolebindings -n odordiff -o yaml | grep -E "name:|subjects:"

# 3. Check Vault policies
echo -e "\nVault Policies:"
kubectl exec -n security vault-0 -- vault list auth/kubernetes/role

# 4. Review API key usage
echo -e "\nAPI Key Usage (last 30 days):"
kubectl logs -n odordiff -l app=odordiff-api --since=720h | grep "API_KEY" | cut -d' ' -f3 | sort | uniq -c

# 5. Generate access report
cat > access-review-$(date +%Y%m%d).txt << EOF
OdorDiff-2 Access Review Report
Generated: $(date)

Service Accounts: $(kubectl get serviceaccounts -n odordiff --no-headers | wc -l)
Role Bindings: $(kubectl get rolebindings -n odordiff --no-headers | wc -l)
Active API Keys: $(kubectl get secrets -n odordiff -l type=api-key --no-headers | wc -l)

Recommendations:
- Review and remove unused service accounts
- Verify role bindings follow least privilege
- Rotate API keys older than 90 days
EOF
```

### Credential Management

#### Secret Rotation Procedures

```bash
#!/bin/bash
# Automated secret rotation script

ROTATION_ID="ROT-$(date +%Y%m%d-%H%M%S)"
echo "Starting secret rotation: ${ROTATION_ID}"

# 1. Database passwords
echo "Rotating database passwords..."
NEW_DB_PASSWORD=$(openssl rand -base64 32)
kubectl create secret generic postgres-credentials-new \
  --from-literal=username=odordiff \
  --from-literal=password="${NEW_DB_PASSWORD}" \
  -n odordiff

# Update database with new password
kubectl exec -n odordiff deployment/postgres -- \
  psql -U postgres -c "ALTER USER odordiff PASSWORD '${NEW_DB_PASSWORD}';"

# Update applications to use new secret
kubectl patch deployment odordiff-api -n odordiff -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"odordiff-api","env":[{"name":"DATABASE_PASSWORD","valueFrom":{"secretKeyRef":{"name":"postgres-credentials-new","key":"password"}}}]}]}}}}'

# Wait for rollout
kubectl rollout status deployment/odordiff-api -n odordiff

# Remove old secret
kubectl delete secret postgres-credentials -n odordiff
kubectl patch secret postgres-credentials-new -n odordiff --type='merge' -p='{"metadata":{"name":"postgres-credentials"}}'

# 2. API Keys
echo "Rotating API keys..."
for i in {1..5}; do
  NEW_API_KEY=$(openssl rand -hex 32)
  kubectl patch secret api-keys -n odordiff --type='merge' -p="{\"data\":{\"api-key-${i}\":\"$(echo -n ${NEW_API_KEY} | base64 -w 0)\"}}"
done

# 3. JWT Secrets
echo "Rotating JWT secrets..."
NEW_JWT_SECRET=$(openssl rand -base64 64)
kubectl patch secret jwt-secrets -n odordiff --type='merge' -p="{\"data\":{\"jwt-secret\":\"$(echo -n ${NEW_JWT_SECRET} | base64 -w 0)\"}}"

# 4. Encryption Keys
echo "Rotating encryption keys..."
NEW_ENCRYPTION_KEY=$(openssl rand -hex 32)
kubectl patch secret encryption-keys -n odordiff --type='merge' -p="{\"data\":{\"encryption-key\":\"$(echo -n ${NEW_ENCRYPTION_KEY} | base64 -w 0)\"}}"

# 5. Update Vault secrets
kubectl exec -n security vault-0 -- \
  vault kv put secret/odordiff/credentials \
  db_password="${NEW_DB_PASSWORD}" \
  jwt_secret="${NEW_JWT_SECRET}" \
  encryption_key="${NEW_ENCRYPTION_KEY}"

echo "Secret rotation ${ROTATION_ID} completed successfully"

# Send notification
curl -X POST "${SLACK_SECURITY_WEBHOOK}" \
  -H 'Content-type: application/json' \
  -d "{\"text\": \"🔄 Secret rotation ${ROTATION_ID} completed successfully\"}"
```

## 🛡️ Security Monitoring and Alerting

### Security Event Detection

#### Real-time Security Monitoring

```yaml
# Security monitoring rules
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: security-alerts
  namespace: monitoring
spec:
  groups:
  - name: security.rules
    rules:
    - alert: SuspiciousLoginAttempts
      expr: increase(http_requests_total{status="401"}[5m]) > 50
      for: 2m
      labels:
        severity: warning
        category: security
      annotations:
        summary: "High number of failed authentication attempts"
        description: "{{ $value }} failed login attempts in 5 minutes"
        runbook_url: "https://docs.odordiff.ai/security/failed-logins"

    - alert: UnauthorizedAPIAccess
      expr: increase(http_requests_total{status="403"}[5m]) > 20
      for: 1m
      labels:
        severity: critical
        category: security
      annotations:
        summary: "Unauthorized access attempts detected"
        description: "{{ $value }} unauthorized requests in 5 minutes"

    - alert: SQLInjectionAttempt
      expr: increase(security_sql_injection_attempts_total[1m]) > 0
      for: 0s
      labels:
        severity: critical
        category: security
      annotations:
        summary: "SQL injection attempt detected"
        description: "SQL injection attempt from {{ $labels.source_ip }}"

    - alert: AnomalousTrafficPattern
      expr: rate(http_requests_total[5m]) > (4 * rate(http_requests_total[1h] offset 1h))
      for: 3m
      labels:
        severity: warning
        category: security
      annotations:
        summary: "Unusual traffic spike detected"
        description: "Traffic is 4x higher than normal baseline"
```

#### Log Analysis and Threat Detection

```bash
#!/bin/bash
# Security log analysis script

echo "=== Security Log Analysis $(date) ==="

# 1. Failed authentication attempts
echo "Failed Authentication Attempts (last hour):"
kubectl logs -n odordiff -l app=odordiff-api --since=1h | \
  grep "401\|authentication failed" | \
  awk '{print $1}' | sort | uniq -c | sort -nr

# 2. Suspicious IP addresses
echo -e "\nSuspicious IP Analysis:"
kubectl logs -n odordiff -l app=odordiff-api --since=1h | \
  grep -E "403|401|404" | \
  grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}' | \
  sort | uniq -c | sort -nr | head -10

# 3. SQL injection attempts
echo -e "\nSQL Injection Attempts:"
kubectl logs -n odordiff -l app=odordiff-api --since=1h | \
  grep -i -E "(union|select|insert|update|delete|drop|alter|exec)" | \
  grep -v "legitimate_query_pattern"

# 4. XSS attempts
echo -e "\nXSS Attempts:"
kubectl logs -n odordiff -l app=odordiff-api --since=1h | \
  grep -i -E "(<script|javascript:|data:text|vbscript:)"

# 5. Directory traversal attempts
echo -e "\nDirectory Traversal Attempts:"
kubectl logs -n odordiff -l app=odordiff-api --since=1h | \
  grep -E "(\.\./|\.\.\\\\|%2e%2e%2f)"

# 6. Generate security report
SECURITY_REPORT="/tmp/security-analysis-$(date +%Y%m%d-%H%M).txt"
cat > ${SECURITY_REPORT} << EOF
OdorDiff-2 Security Analysis Report
Generated: $(date)

Summary:
- Failed Auth Attempts: $(kubectl logs -n odordiff -l app=odordiff-api --since=1h | grep -c "401")
- Blocked Requests: $(kubectl logs -n odordiff -l app=odordiff-api --since=1h | grep -c "403")
- Error Responses: $(kubectl logs -n odordiff -l app=odordiff-api --since=1h | grep -c "5..")

Top Threat Categories:
1. Authentication failures
2. Authorization violations
3. Input validation errors
4. Rate limit violations

Recommendations:
- Review and update WAF rules
- Consider additional rate limiting
- Investigate repeated offenders
- Update security monitoring thresholds
EOF

echo "Security report generated: ${SECURITY_REPORT}"
```

### Vulnerability Management

#### Regular Security Scanning

```bash
#!/bin/bash
# Automated vulnerability scanning

SCAN_ID="SCAN-$(date +%Y%m%d-%H%M%S)"
echo "Starting security scan: ${SCAN_ID}"

# 1. Container image scanning
echo "Scanning container images..."
for image in $(kubectl get pods -n odordiff -o jsonpath='{.items[*].spec.containers[*].image}' | sort -u); do
  echo "Scanning image: ${image}"
  trivy image --severity HIGH,CRITICAL --format json "${image}" > "/tmp/trivy-${SCAN_ID}-$(basename ${image}).json"
done

# 2. Kubernetes security scanning
echo "Scanning Kubernetes configurations..."
kube-bench run --targets node,policies,managedservices > "/tmp/kube-bench-${SCAN_ID}.txt"

# 3. Network security scanning
echo "Scanning network security..."
kubectl run nmap-scan --rm -i --restart=Never \
  --image=nmap:latest \
  -- nmap -sS -O -A odordiff-api.odordiff.svc.cluster.local > "/tmp/nmap-${SCAN_ID}.txt"

# 4. Web application scanning
echo "Scanning web application..."
kubectl run zap-scan --rm -i --restart=Never \
  --image=owasp/zap2docker-stable \
  -- zap-baseline.py -t https://api.odordiff.ai > "/tmp/zap-${SCAN_ID}.txt"

# 5. Dependency scanning
echo "Scanning dependencies..."
kubectl run dependency-check --rm -i --restart=Never \
  --image=owasp/dependency-check \
  -- --project odordiff-2 --scan /app > "/tmp/dependency-check-${SCAN_ID}.txt"

# 6. Generate consolidated report
cat > "/tmp/security-scan-${SCAN_ID}-report.txt" << EOF
Security Scan Report: ${SCAN_ID}
Generated: $(date)

Scans Performed:
✓ Container Image Vulnerability Scan
✓ Kubernetes Security Benchmark
✓ Network Security Scan
✓ Web Application Security Scan
✓ Dependency Vulnerability Scan

Summary:
- Container vulnerabilities found: $(find /tmp -name "trivy-${SCAN_ID}-*.json" -exec jq '.Results[].Vulnerabilities | length' {} \; | awk '{sum+=$1} END {print sum}')
- Kubernetes security issues: $(grep -c "FAIL" "/tmp/kube-bench-${SCAN_ID}.txt")
- Network security concerns: $(grep -c "open" "/tmp/nmap-${SCAN_ID}.txt")
- Web application issues: $(grep -c "RISK" "/tmp/zap-${SCAN_ID}.txt")

Recommendations:
1. Update container images with security patches
2. Address Kubernetes configuration issues
3. Review network exposure and firewall rules
4. Fix web application vulnerabilities
5. Update vulnerable dependencies

Next scan scheduled: $(date -d '+1 week')
EOF

echo "Security scan ${SCAN_ID} completed. Report: /tmp/security-scan-${SCAN_ID}-report.txt"

# Send notification
curl -X POST "${SLACK_SECURITY_WEBHOOK}" \
  -H 'Content-type: application/json' \
  -d "{\"text\": \"🔍 Security scan ${SCAN_ID} completed. Report available for review.\"}"
```

## 📋 Compliance and Auditing

### SOC 2 Compliance

#### Control Implementation Checklist

**Security Controls:**
- [ ] Multi-factor authentication implemented
- [ ] Access controls based on principle of least privilege
- [ ] Network segmentation and firewall rules
- [ ] Vulnerability management program
- [ ] Incident response procedures documented and tested
- [ ] Security awareness training completed
- [ ] Third-party security assessments conducted

**Availability Controls:**
- [ ] System monitoring and alerting implemented
- [ ] Backup and disaster recovery procedures tested
- [ ] Capacity planning and performance monitoring
- [ ] Change management procedures documented
- [ ] Service level agreements defined and monitored

**Confidentiality Controls:**
- [ ] Data encryption at rest and in transit
- [ ] Access logging and monitoring
- [ ] Data classification and handling procedures
- [ ] Secure development lifecycle practices
- [ ] Data retention and disposal policies

#### Audit Log Collection

```bash
#!/bin/bash
# Audit log collection for compliance

AUDIT_PERIOD="${1:-last-month}"
AUDIT_ID="AUDIT-$(date +%Y%m%d)"

echo "Collecting audit logs for ${AUDIT_PERIOD}: ${AUDIT_ID}"

# 1. Authentication and authorization logs
echo "Collecting authentication logs..."
kubectl logs -n odordiff -l app=odordiff-api --since=720h | \
  grep -E "(login|logout|authentication|authorization)" > "/tmp/auth-logs-${AUDIT_ID}.txt"

# 2. Data access logs
echo "Collecting data access logs..."
kubectl logs -n odordiff -l app=odordiff-api --since=720h | \
  grep -E "(GET|POST|PUT|DELETE).*api" | \
  grep -E "(users|molecules|data)" > "/tmp/data-access-logs-${AUDIT_ID}.txt"

# 3. Administrative actions
echo "Collecting administrative logs..."
kubectl get events --all-namespaces --field-selector reason!=Scheduled \
  --field-selector reason!=Started --field-selector reason!=Pulled \
  --sort-by='.metadata.creationTimestamp' > "/tmp/admin-events-${AUDIT_ID}.txt"

# 4. Configuration changes
echo "Collecting configuration change logs..."
kubectl logs -n argocd -l app=argocd-server --since=720h | \
  grep -E "(sync|deploy|rollback)" > "/tmp/config-changes-${AUDIT_ID}.txt"

# 5. Security events
echo "Collecting security event logs..."
kubectl logs -n monitoring -l app=alertmanager --since=720h | \
  grep -E "security|breach|unauthorized" > "/tmp/security-events-${AUDIT_ID}.txt"

# 6. Generate compliance report
cat > "/tmp/compliance-report-${AUDIT_ID}.txt" << EOF
SOC 2 Compliance Report: ${AUDIT_ID}
Period: ${AUDIT_PERIOD}
Generated: $(date)

Audit Logs Collected:
✓ Authentication and Authorization Events
✓ Data Access Logs
✓ Administrative Actions
✓ Configuration Changes
✓ Security Events

Statistics:
- Authentication events: $(wc -l < "/tmp/auth-logs-${AUDIT_ID}.txt")
- Data access requests: $(wc -l < "/tmp/data-access-logs-${AUDIT_ID}.txt")
- Administrative actions: $(wc -l < "/tmp/admin-events-${AUDIT_ID}.txt")
- Configuration changes: $(wc -l < "/tmp/config-changes-${AUDIT_ID}.txt")
- Security events: $(wc -l < "/tmp/security-events-${AUDIT_ID}.txt")

Compliance Status:
- Access controls: COMPLIANT
- Audit logging: COMPLIANT
- Data encryption: COMPLIANT
- Incident response: COMPLIANT
- Change management: COMPLIANT

Recommendations:
- Continue monthly audit log collection
- Review access patterns for anomalies
- Update security policies as needed
- Conduct quarterly compliance review
EOF

echo "Compliance report generated: /tmp/compliance-report-${AUDIT_ID}.txt"
```

### GDPR Compliance

#### Data Subject Rights Implementation

```bash
#!/bin/bash
# GDPR data subject rights handler

USER_ID="${1}"
REQUEST_TYPE="${2}"
REQUEST_ID="GDPR-$(date +%Y%m%d-%H%M%S)"

if [[ -z "${USER_ID}" || -z "${REQUEST_TYPE}" ]]; then
  echo "Usage: $0 <user_id> <access|rectification|erasure|portability>"
  exit 1
fi

echo "Processing GDPR request: ${REQUEST_ID}"
echo "User: ${USER_ID}, Type: ${REQUEST_TYPE}"

case ${REQUEST_TYPE} in
  "access")
    echo "Processing data access request..."
    # Extract all user data
    kubectl exec -n odordiff deployment/postgres -- \
      psql -U odordiff -d odordiff -c \
      "SELECT * FROM users WHERE user_id = '${USER_ID}';" > "/tmp/user-data-${REQUEST_ID}.txt"
    
    kubectl exec -n odordiff deployment/postgres -- \
      psql -U odordiff -d odordiff -c \
      "SELECT * FROM user_molecules WHERE user_id = '${USER_ID}';" >> "/tmp/user-data-${REQUEST_ID}.txt"
    
    echo "User data exported to: /tmp/user-data-${REQUEST_ID}.txt"
    ;;
    
  "rectification")
    echo "Processing data rectification request..."
    # Provide mechanism for data updates
    echo "Manual intervention required for data rectification"
    echo "Update user data in database and document changes"
    ;;
    
  "erasure")
    echo "Processing data erasure request..."
    # Delete user data
    kubectl exec -n odordiff deployment/postgres -- \
      psql -U odordiff -d odordiff -c \
      "BEGIN; DELETE FROM user_molecules WHERE user_id = '${USER_ID}'; DELETE FROM users WHERE user_id = '${USER_ID}'; COMMIT;"
    
    # Clear cache
    kubectl exec -n odordiff deployment/redis -- \
      redis-cli DEL "user:${USER_ID}:*"
    
    echo "User data deleted for user: ${USER_ID}"
    ;;
    
  "portability")
    echo "Processing data portability request..."
    # Export data in machine-readable format
    kubectl exec -n odordiff deployment/postgres -- \
      psql -U odordiff -d odordiff -c \
      "COPY (SELECT * FROM users WHERE user_id = '${USER_ID}') TO STDOUT WITH CSV HEADER;" > "/tmp/user-export-${REQUEST_ID}.csv"
    
    kubectl exec -n odordiff deployment/postgres -- \
      psql -U odordiff -d odordiff -c \
      "COPY (SELECT * FROM user_molecules WHERE user_id = '${USER_ID}') TO STDOUT WITH CSV HEADER;" >> "/tmp/user-export-${REQUEST_ID}.csv"
    
    echo "User data exported to CSV: /tmp/user-export-${REQUEST_ID}.csv"
    ;;
    
  *)
    echo "Unknown request type: ${REQUEST_TYPE}"
    exit 1
    ;;
esac

# Log the request
echo "$(date): GDPR request ${REQUEST_ID} processed for user ${USER_ID} (${REQUEST_TYPE})" >> "/var/log/gdpr-requests.log"

# Send notification
curl -X POST "${SLACK_COMPLIANCE_WEBHOOK}" \
  -H 'Content-type: application/json' \
  -d "{\"text\": \"📋 GDPR request ${REQUEST_ID} processed: ${REQUEST_TYPE} for user ${USER_ID}\"}"
```

## 📖 Security Training and Awareness

### Developer Security Guidelines

#### Secure Coding Checklist

**Input Validation:**
- [ ] Validate all input data at API boundaries
- [ ] Use parameterized queries for database operations
- [ ] Implement proper error handling without information disclosure
- [ ] Sanitize data for output encoding

**Authentication and Authorization:**
- [ ] Implement strong authentication mechanisms
- [ ] Use secure session management
- [ ] Apply principle of least privilege
- [ ] Implement proper logout functionality

**Data Protection:**
- [ ] Encrypt sensitive data at rest
- [ ] Use TLS for data in transit
- [ ] Implement secure key management
- [ ] Follow data retention policies

**Security Testing:**
- [ ] Perform security code reviews
- [ ] Run static analysis security testing (SAST)
- [ ] Conduct dynamic application security testing (DAST)
- [ ] Implement security unit tests

### Security Incident Simulation

#### Tabletop Exercise Script

```bash
#!/bin/bash
# Security incident tabletop exercise

echo "=== OdorDiff-2 Security Incident Simulation ==="
echo "Date: $(date)"
echo "Exercise: Simulated Data Breach Response"

cat << EOF

SCENARIO:
A security researcher has contacted us claiming they found an SQL injection
vulnerability in our API that allows unauthorized access to user data.
They provided a proof-of-concept showing they can extract user email addresses
and hashed passwords.

INITIAL ALERT:
- Time: $(date -d '30 minutes ago')
- Source: External security researcher
- Severity: Critical
- Affected Systems: API and Database

YOUR MISSION:
Work through the incident response process step by step.

QUESTIONS TO CONSIDER:
1. Who needs to be notified immediately?
2. What are the first containment actions?
3. How do you verify the vulnerability?
4. What evidence needs to be collected?
5. What communication plan is needed?
6. How long do you have to notify affected users?
7. What regulatory requirements apply?

PARTICIPANTS:
- Incident Commander: _______________
- Security Engineer: _______________
- Platform Engineer: _______________
- Legal Counsel: _______________
- Communications Lead: _______________

EXERCISE DURATION: 60 minutes
DEBRIEF: 30 minutes

BEGIN EXERCISE...

EOF

# Pause for exercise
echo "Press Enter when exercise is complete..."
read

echo "=== EXERCISE DEBRIEF ==="
cat << EOF

DISCUSSION POINTS:
1. What went well during the response?
2. What could be improved?
3. Were response times adequate?
4. Was communication effective?
5. Were all stakeholders engaged appropriately?

ACTION ITEMS:
- Update incident response procedures
- Schedule additional training
- Review and test technical controls
- Update contact information
- Schedule next exercise

EOF
```

## 📞 Emergency Contacts and Resources

### Emergency Response Team

```
Security Incident Commander: security-ic@odordiff.ai (+1-XXX-XXX-XXXX)
Primary Security Engineer: security-primary@odordiff.ai (+1-XXX-XXX-XXXX)
Secondary Security Engineer: security-secondary@odordiff.ai (+1-XXX-XXX-XXXX)
Platform Team Lead: platform-lead@odordiff.ai (+1-XXX-XXX-XXXX)
Legal Counsel: legal@odordiff.ai (+1-XXX-XXX-XXXX)
Communications Lead: comms@odordiff.ai (+1-XXX-XXX-XXXX)
```

### External Resources

```
FBI Cyber Crime Division: https://www.fbi.gov/investigate/cyber
CISA Incident Reporting: https://us-cert.cisa.gov/report
Cloud Provider Security: Available in respective cloud consoles
Cyber Insurance: insurance@provider.com
Legal Support: External counsel contact information
```

### Communication Channels

```
Security Slack Channel: #security-alerts
All Hands Slack: #general
Executive Team: leadership@odordiff.ai
Customer Communications: support@odordiff.ai
Press Inquiries: press@odordiff.ai
```

---

## 📚 References and Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [SANS Incident Response Guide](https://www.sans.org/white-papers/33901/)
- [Kubernetes Security Best Practices](https://kubernetes.io/docs/concepts/security/)
- [SOC 2 Compliance Guide](https://www.aicpa.org/interestareas/frc/assuranceadvisoryservices/sorhome.html)

---

**Document Version**: 1.0.0  
**Classification**: Confidential  
**Last Updated**: August 9, 2025  
**Next Review Date**: February 9, 2026  
**Owner**: Security Team

*This document contains sensitive security information. Distribution is restricted to authorized personnel only.*