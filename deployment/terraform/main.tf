# Terraform configuration for OdorDiff-2 multi-cloud deployment
# Supports AWS, GCP, and Azure with production-grade infrastructure

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }

  # Backend configuration - customize based on your setup
  backend "s3" {
    # bucket  = "odordiff-terraform-state"
    # key     = "production/terraform.tfstate"
    # region  = "us-west-2"
    # encrypt = true
  }
}

# Local values for configuration
locals {
  project_name = "odordiff-2"
  environment  = var.environment
  region       = var.region
  
  # Common tags
  common_tags = {
    Project     = local.project_name
    Environment = local.environment
    ManagedBy   = "Terraform"
    Team        = "AI-Platform"
    CreatedBy   = "terraform"
  }
  
  # Kubernetes cluster name
  cluster_name = "${local.project_name}-${local.environment}-cluster"
}

# Random password generation
resource "random_password" "db_password" {
  length  = 16
  special = true
}

resource "random_password" "redis_password" {
  length  = 16
  special = false
}

# Data sources
data "aws_availability_zones" "available" {
  count = var.cloud_provider == "aws" ? 1 : 0
  state = "available"
}

data "google_compute_zones" "available" {
  count   = var.cloud_provider == "gcp" ? 1 : 0
  region  = var.region
  status  = "UP"
}

data "azurerm_client_config" "current" {
  count = var.cloud_provider == "azure" ? 1 : 0
}

# Module imports based on cloud provider
module "aws_infrastructure" {
  count  = var.cloud_provider == "aws" ? 1 : 0
  source = "./modules/aws"
  
  project_name         = local.project_name
  environment         = local.environment
  region              = var.region
  availability_zones  = data.aws_availability_zones.available[0].names
  
  # VPC configuration
  vpc_cidr            = var.vpc_cidr
  private_subnets     = var.private_subnets
  public_subnets      = var.public_subnets
  
  # EKS configuration
  cluster_name        = local.cluster_name
  cluster_version     = var.kubernetes_version
  node_groups         = var.node_groups
  
  # Database configuration
  db_instance_class   = var.db_instance_class
  db_allocated_storage = var.db_allocated_storage
  db_password         = random_password.db_password.result
  
  # Redis configuration
  redis_node_type     = var.redis_node_type
  redis_password      = random_password.redis_password.result
  
  # Monitoring
  enable_monitoring   = var.enable_monitoring
  
  # Security
  enable_waf          = var.enable_waf
  enable_guard_duty   = var.enable_guard_duty
  
  tags = local.common_tags
}

module "gcp_infrastructure" {
  count  = var.cloud_provider == "gcp" ? 1 : 0
  source = "./modules/gcp"
  
  project_name    = local.project_name
  environment     = local.environment
  region          = var.region
  zones           = data.google_compute_zones.available[0].names
  
  # Network configuration
  vpc_cidr            = var.vpc_cidr
  private_subnets     = var.private_subnets
  public_subnets      = var.public_subnets
  
  # GKE configuration
  cluster_name        = local.cluster_name
  cluster_version     = var.kubernetes_version
  node_pools          = var.node_groups
  
  # Database configuration
  db_instance_tier    = var.db_instance_class
  db_disk_size        = var.db_allocated_storage
  db_password         = random_password.db_password.result
  
  # Redis configuration
  redis_memory_size   = var.redis_memory_size
  redis_auth_password = random_password.redis_password.result
  
  # Monitoring
  enable_monitoring   = var.enable_monitoring
  
  labels = local.common_tags
}

module "azure_infrastructure" {
  count  = var.cloud_provider == "azure" ? 1 : 0
  source = "./modules/azure"
  
  project_name     = local.project_name
  environment      = local.environment
  location         = var.region
  
  # Network configuration
  vnet_cidr           = var.vpc_cidr
  private_subnets     = var.private_subnets
  public_subnets      = var.public_subnets
  
  # AKS configuration
  cluster_name        = local.cluster_name
  cluster_version     = var.kubernetes_version
  node_pools          = var.node_groups
  
  # Database configuration
  db_sku_name         = var.db_instance_class
  db_storage_mb       = var.db_allocated_storage * 1024
  db_password         = random_password.db_password.result
  
  # Redis configuration
  redis_capacity      = var.redis_capacity
  redis_password      = random_password.redis_password.result
  
  # Monitoring
  enable_monitoring   = var.enable_monitoring
  
  tags = local.common_tags
}

# Kubernetes provider configuration
data "aws_eks_cluster" "cluster" {
  count = var.cloud_provider == "aws" ? 1 : 0
  name  = module.aws_infrastructure[0].cluster_name
}

data "aws_eks_cluster_auth" "cluster" {
  count = var.cloud_provider == "aws" ? 1 : 0
  name  = module.aws_infrastructure[0].cluster_name
}

data "google_container_cluster" "cluster" {
  count    = var.cloud_provider == "gcp" ? 1 : 0
  name     = module.gcp_infrastructure[0].cluster_name
  location = var.region
}

data "azurerm_kubernetes_cluster" "cluster" {
  count               = var.cloud_provider == "azure" ? 1 : 0
  name                = module.azure_infrastructure[0].cluster_name
  resource_group_name = module.azure_infrastructure[0].resource_group_name
}

provider "kubernetes" {
  host                   = var.cloud_provider == "aws" ? data.aws_eks_cluster.cluster[0].endpoint : var.cloud_provider == "gcp" ? "https://${data.google_container_cluster.cluster[0].endpoint}" : data.azurerm_kubernetes_cluster.cluster[0].kube_config.0.host
  cluster_ca_certificate = var.cloud_provider == "aws" ? base64decode(data.aws_eks_cluster.cluster[0].certificate_authority.0.data) : var.cloud_provider == "gcp" ? base64decode(data.google_container_cluster.cluster[0].master_auth.0.cluster_ca_certificate) : base64decode(data.azurerm_kubernetes_cluster.cluster[0].kube_config.0.cluster_ca_certificate)
  token                  = var.cloud_provider == "aws" ? data.aws_eks_cluster_auth.cluster[0].token : var.cloud_provider == "gcp" ? null : data.azurerm_kubernetes_cluster.cluster[0].kube_config.0.password
  
  dynamic "exec" {
    for_each = var.cloud_provider == "gcp" ? [1] : []
    content {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "gke-gcloud-auth-plugin"
    }
  }
}

provider "helm" {
  kubernetes {
    host                   = var.cloud_provider == "aws" ? data.aws_eks_cluster.cluster[0].endpoint : var.cloud_provider == "gcp" ? "https://${data.google_container_cluster.cluster[0].endpoint}" : data.azurerm_kubernetes_cluster.cluster[0].kube_config.0.host
    cluster_ca_certificate = var.cloud_provider == "aws" ? base64decode(data.aws_eks_cluster.cluster[0].certificate_authority.0.data) : var.cloud_provider == "gcp" ? base64decode(data.google_container_cluster.cluster[0].master_auth.0.cluster_ca_certificate) : base64decode(data.azurerm_kubernetes_cluster.cluster[0].kube_config.0.cluster_ca_certificate)
    token                  = var.cloud_provider == "aws" ? data.aws_eks_cluster_auth.cluster[0].token : var.cloud_provider == "gcp" ? null : data.azurerm_kubernetes_cluster.cluster[0].kube_config.0.password
    
    dynamic "exec" {
      for_each = var.cloud_provider == "gcp" ? [1] : []
      content {
        api_version = "client.authentication.k8s.io/v1beta1"
        command     = "gke-gcloud-auth-plugin"
      }
    }
  }
}

# Deploy OdorDiff-2 application using Helm
resource "helm_release" "odordiff2" {
  name       = "odordiff-2"
  repository = "https://danieleschmidt.github.io/odordiff-2-helm"
  chart      = "odordiff-2"
  version    = var.helm_chart_version
  namespace  = "odordiff"

  create_namespace = true
  wait            = true
  timeout         = 600

  values = [
    templatefile("${path.module}/helm-values.yaml.tpl", {
      environment      = local.environment
      image_tag        = var.image_tag
      replica_count    = var.replica_count
      db_host         = var.cloud_provider == "aws" ? module.aws_infrastructure[0].db_endpoint : var.cloud_provider == "gcp" ? module.gcp_infrastructure[0].db_connection_name : module.azure_infrastructure[0].db_fqdn
      db_password     = random_password.db_password.result
      redis_host      = var.cloud_provider == "aws" ? module.aws_infrastructure[0].redis_endpoint : var.cloud_provider == "gcp" ? module.gcp_infrastructure[0].redis_host : module.azure_infrastructure[0].redis_hostname
      redis_password  = random_password.redis_password.result
      monitoring_enabled = var.enable_monitoring
      domain_name     = var.domain_name
    })
  ]

  depends_on = [
    module.aws_infrastructure,
    module.gcp_infrastructure,
    module.azure_infrastructure
  ]
}

# Install monitoring stack if enabled
resource "helm_release" "prometheus" {
  count = var.enable_monitoring ? 1 : 0
  
  name       = "prometheus"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  version    = "54.0.1"
  namespace  = "monitoring"

  create_namespace = true
  wait            = true
  timeout         = 600

  values = [
    file("${path.module}/monitoring-values.yaml")
  ]

  depends_on = [helm_release.odordiff2]
}

# Install Istio service mesh if enabled
resource "helm_release" "istio_base" {
  count = var.enable_istio ? 1 : 0
  
  name       = "istio-base"
  repository = "https://istio-release.storage.googleapis.com/charts"
  chart      = "base"
  version    = "1.19.0"
  namespace  = "istio-system"

  create_namespace = true
  wait            = true
}

resource "helm_release" "istiod" {
  count = var.enable_istio ? 1 : 0
  
  name       = "istiod"
  repository = "https://istio-release.storage.googleapis.com/charts"
  chart      = "istiod"
  version    = "1.19.0"
  namespace  = "istio-system"

  wait    = true
  timeout = 600

  depends_on = [helm_release.istio_base]
}