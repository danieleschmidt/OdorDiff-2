# Terraform variables for OdorDiff-2 infrastructure
# Production-ready configuration options

# Core configuration
variable "environment" {
  description = "Environment name (development, staging, production)"
  type        = string
  default     = "production"
  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Environment must be one of: development, staging, production."
  }
}

variable "cloud_provider" {
  description = "Cloud provider to use (aws, gcp, azure)"
  type        = string
  default     = "aws"
  validation {
    condition     = contains(["aws", "gcp", "azure"], var.cloud_provider)
    error_message = "Cloud provider must be one of: aws, gcp, azure."
  }
}

variable "region" {
  description = "Cloud provider region"
  type        = string
  default     = "us-west-2"
}

# Network configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR must be a valid CIDR block."
  }
}

variable "private_subnets" {
  description = "List of private subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnets" {
  description = "List of public subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

# Kubernetes configuration
variable "kubernetes_version" {
  description = "Kubernetes cluster version"
  type        = string
  default     = "1.28"
}

variable "node_groups" {
  description = "Configuration for Kubernetes node groups"
  type = map(object({
    instance_type     = string
    min_size         = number
    max_size         = number
    desired_capacity = number
    disk_size        = number
    labels           = map(string)
    taints = list(object({
      key    = string
      value  = string
      effect = string
    }))
  }))
  default = {
    general = {
      instance_type     = "m5.xlarge"
      min_size         = 3
      max_size         = 10
      desired_capacity = 3
      disk_size        = 100
      labels = {
        role = "general"
      }
      taints = []
    }
    compute = {
      instance_type     = "c5.2xlarge"
      min_size         = 2
      max_size         = 20
      desired_capacity = 2
      disk_size        = 200
      labels = {
        role = "compute"
      }
      taints = [{
        key    = "compute"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
    memory = {
      instance_type     = "r5.2xlarge"
      min_size         = 1
      max_size         = 10
      desired_capacity = 1
      disk_size        = 200
      labels = {
        role = "memory"
      }
      taints = [{
        key    = "memory"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
  }
}

# Database configuration
variable "db_instance_class" {
  description = "Database instance class"
  type        = string
  default     = "db.r5.xlarge"
}

variable "db_allocated_storage" {
  description = "Database allocated storage in GB"
  type        = number
  default     = 500
  validation {
    condition     = var.db_allocated_storage >= 20
    error_message = "Database storage must be at least 20 GB."
  }
}

variable "db_backup_retention_period" {
  description = "Database backup retention period in days"
  type        = number
  default     = 30
  validation {
    condition     = var.db_backup_retention_period >= 7
    error_message = "Backup retention period must be at least 7 days."
  }
}

variable "db_multi_az" {
  description = "Enable Multi-AZ deployment for database"
  type        = bool
  default     = true
}

# Redis configuration
variable "redis_node_type" {
  description = "Redis node type"
  type        = string
  default     = "cache.r6g.xlarge"
}

variable "redis_memory_size" {
  description = "Redis memory size in GB (for GCP)"
  type        = number
  default     = 4
}

variable "redis_capacity" {
  description = "Redis capacity (for Azure)"
  type        = number
  default     = 2
}

variable "redis_num_cache_nodes" {
  description = "Number of Redis cache nodes"
  type        = number
  default     = 2
}

# Application configuration
variable "image_tag" {
  description = "OdorDiff-2 Docker image tag"
  type        = string
  default     = "1.0.0"
}

variable "helm_chart_version" {
  description = "OdorDiff-2 Helm chart version"
  type        = string
  default     = "1.0.0"
}

variable "replica_count" {
  description = "Number of application replicas"
  type        = number
  default     = 3
  validation {
    condition     = var.replica_count >= 1
    error_message = "Replica count must be at least 1."
  }
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "api.odordiff.ai"
}

# Monitoring configuration
variable "enable_monitoring" {
  description = "Enable monitoring stack (Prometheus, Grafana)"
  type        = bool
  default     = true
}

variable "enable_logging" {
  description = "Enable centralized logging (ELK/Loki)"
  type        = bool
  default     = true
}

variable "enable_jaeger" {
  description = "Enable distributed tracing with Jaeger"
  type        = bool
  default     = true
}

variable "monitoring_retention_days" {
  description = "Monitoring data retention period in days"
  type        = number
  default     = 30
}

# Security configuration
variable "enable_waf" {
  description = "Enable Web Application Firewall"
  type        = bool
  default     = true
}

variable "enable_guard_duty" {
  description = "Enable AWS GuardDuty (AWS only)"
  type        = bool
  default     = true
}

variable "enable_network_policies" {
  description = "Enable Kubernetes network policies"
  type        = bool
  default     = true
}

variable "enable_pod_security_policies" {
  description = "Enable Kubernetes pod security policies"
  type        = bool
  default     = true
}

# Service mesh configuration
variable "enable_istio" {
  description = "Enable Istio service mesh"
  type        = bool
  default     = true
}

variable "istio_version" {
  description = "Istio version to deploy"
  type        = string
  default     = "1.19.0"
}

# Backup configuration
variable "enable_backup" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Backup retention period in days"
  type        = number
  default     = 30
}

variable "enable_cross_region_backup" {
  description = "Enable cross-region backup replication"
  type        = bool
  default     = true
}

# SSL/TLS configuration
variable "ssl_certificate_arn" {
  description = "SSL certificate ARN (AWS only)"
  type        = string
  default     = ""
}

variable "enable_ssl_redirect" {
  description = "Enable automatic SSL redirect"
  type        = bool
  default     = true
}

# Cost optimization
variable "enable_spot_instances" {
  description = "Enable spot instances for cost optimization"
  type        = bool
  default     = false
}

variable "spot_instance_percentage" {
  description = "Percentage of spot instances in node groups"
  type        = number
  default     = 50
  validation {
    condition     = var.spot_instance_percentage >= 0 && var.spot_instance_percentage <= 100
    error_message = "Spot instance percentage must be between 0 and 100."
  }
}

# Scaling configuration
variable "enable_cluster_autoscaler" {
  description = "Enable cluster autoscaler"
  type        = bool
  default     = true
}

variable "enable_vertical_pod_autoscaler" {
  description = "Enable vertical pod autoscaler"
  type        = bool
  default     = true
}

variable "enable_horizontal_pod_autoscaler" {
  description = "Enable horizontal pod autoscaler"
  type        = bool
  default     = true
}

# Performance configuration
variable "enable_gpu_nodes" {
  description = "Enable GPU-enabled node groups"
  type        = bool
  default     = false
}

variable "gpu_node_type" {
  description = "GPU node instance type"
  type        = string
  default     = "p3.2xlarge"
}

# Development/testing configuration
variable "enable_dev_tools" {
  description = "Enable development tools (only for dev/staging)"
  type        = bool
  default     = false
}

variable "enable_debug_logging" {
  description = "Enable debug logging"
  type        = bool
  default     = false
}

# Compliance and governance
variable "enable_compliance_scanning" {
  description = "Enable compliance and security scanning"
  type        = bool
  default     = true
}

variable "enable_resource_tagging" {
  description = "Enable comprehensive resource tagging"
  type        = bool
  default     = true
}

variable "compliance_framework" {
  description = "Compliance framework to adhere to (SOC2, HIPAA, PCI-DSS)"
  type        = string
  default     = "SOC2"
  validation {
    condition     = contains(["SOC2", "HIPAA", "PCI-DSS", "GDPR"], var.compliance_framework)
    error_message = "Compliance framework must be one of: SOC2, HIPAA, PCI-DSS, GDPR."
  }
}

# Additional tags
variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}