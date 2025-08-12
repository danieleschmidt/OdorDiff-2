"""
Performance optimization package for OdorDiff-2.

This package provides comprehensive performance optimization capabilities including:
- Model inference optimization and caching
- Dynamic load balancing and request routing
- Auto-scaling based on metrics and predictions
- Performance monitoring and alerting
- Advanced batching and resource management

Key components:
- ModelOptimizer: Core performance optimization with caching
- LoadBalancer: Intelligent request distribution
- AutoScaler: Dynamic scaling based on metrics
- PerformanceMonitor: Comprehensive monitoring and alerting
- ModelOptimizationEngine: Advanced inference optimization
"""

from .optimization import (
    ModelOptimizer,
    ResourcePool,
    BatchProcessor,
    OptimizationConfig,
    PerformanceMetrics,
    get_model_optimizer,
    get_batch_processor,
    monitor_performance
)

from .load_balancing import (
    LoadBalancer,
    ServerInstance,
    ServerHealth,
    LoadBalancingStrategy,
    LoadBalancerConfig,
    ServiceRegistry,
    get_load_balancer,
    get_service_registry
)

from .auto_scaling import (
    AutoScaler,
    AutoScalingConfig,
    ScalingMetrics,
    ScalingAction,
    MetricsCollector,
    PredictiveScaler,
    get_auto_scaler
)

from .monitoring import (
    PerformanceMonitor,
    MonitoringConfig,
    MetricsRegistry,
    AlertManager,
    Alert,
    AlertSeverity,
    MetricType,
    get_performance_monitor,
    monitor_function,
    monitor_async_function
)

from .model_optimization import (
    ModelOptimizationEngine,
    OptimizationProfile,
    OptimizationTechnique,
    PrecisionMode,
    ModelCache,
    DynamicBatcher,
    InferenceRequest,
    InferenceResult,
    OPTIMIZATION_PROFILES,
    get_optimization_engine,
    create_optimization_engine
)

# Version
__version__ = "1.0.0"

# All exports
__all__ = [
    # Core optimization
    "ModelOptimizer",
    "ResourcePool", 
    "BatchProcessor",
    "OptimizationConfig",
    "PerformanceMetrics",
    
    # Load balancing
    "LoadBalancer",
    "ServerInstance", 
    "ServerHealth",
    "LoadBalancingStrategy",
    "LoadBalancerConfig",
    "ServiceRegistry",
    
    # Auto-scaling
    "AutoScaler",
    "AutoScalingConfig", 
    "ScalingMetrics",
    "ScalingAction",
    "MetricsCollector",
    "PredictiveScaler",
    
    # Monitoring
    "PerformanceMonitor",
    "MonitoringConfig",
    "MetricsRegistry",
    "AlertManager",
    "Alert",
    "AlertSeverity", 
    "MetricType",
    
    # Model optimization
    "ModelOptimizationEngine",
    "OptimizationProfile",
    "OptimizationTechnique",
    "PrecisionMode",
    "ModelCache",
    "DynamicBatcher",
    "InferenceRequest", 
    "InferenceResult",
    "OPTIMIZATION_PROFILES",
    
    # Factory functions
    "get_model_optimizer",
    "get_batch_processor", 
    "get_load_balancer",
    "get_service_registry",
    "get_auto_scaler",
    "get_performance_monitor",
    "get_optimization_engine",
    "create_optimization_engine",
    
    # Decorators
    "monitor_performance",
    "monitor_function",
    "monitor_async_function"
]


def initialize_performance_system(
    optimization_config: OptimizationConfig = None,
    load_balancer_config: LoadBalancerConfig = None,
    scaling_config: AutoScalingConfig = None,
    monitoring_config: MonitoringConfig = None,
    optimization_profile: str = "balanced"
) -> dict:
    """
    Initialize the complete performance optimization system.
    
    Args:
        optimization_config: Configuration for model optimizer
        load_balancer_config: Configuration for load balancer  
        scaling_config: Configuration for auto-scaler
        monitoring_config: Configuration for performance monitor
        optimization_profile: Profile name for model optimization engine
        
    Returns:
        Dictionary containing all initialized components
    """
    
    # Initialize core components
    model_optimizer = get_model_optimizer(optimization_config)
    load_balancer = get_load_balancer(load_balancer_config)
    auto_scaler = get_auto_scaler(scaling_config)
    performance_monitor = get_performance_monitor(monitoring_config)
    optimization_engine = get_optimization_engine(optimization_profile)
    
    # Setup integrations
    performance_monitor.integrate_model_optimizer(model_optimizer)
    performance_monitor.integrate_load_balancer(load_balancer)
    performance_monitor.integrate_auto_scaler(auto_scaler)
    
    components = {
        'model_optimizer': model_optimizer,
        'load_balancer': load_balancer,
        'auto_scaler': auto_scaler,
        'performance_monitor': performance_monitor,
        'optimization_engine': optimization_engine
    }
    
    return components


async def start_performance_system(components: dict = None):
    """
    Start all performance system components.
    
    Args:
        components: Dictionary of components from initialize_performance_system()
                   If None, will initialize with defaults
    """
    if components is None:
        components = initialize_performance_system()
    
    # Start all async components
    await components['performance_monitor'].start()
    await components['auto_scaler'].start()
    await components['optimization_engine'].start()
    
    return components


async def stop_performance_system(components: dict):
    """
    Stop all performance system components.
    
    Args:
        components: Dictionary of components to stop
    """
    # Stop all async components
    await components['performance_monitor'].stop()
    await components['auto_scaler'].stop() 
    await components['optimization_engine'].stop()


def get_system_performance_stats(components: dict = None) -> dict:
    """
    Get comprehensive performance statistics from all components.
    
    Args:
        components: Dictionary of components. If None, uses global instances.
        
    Returns:
        Dictionary containing performance statistics from all components
    """
    if components is None:
        components = {
            'model_optimizer': get_model_optimizer(),
            'load_balancer': get_load_balancer(),
            'auto_scaler': get_auto_scaler(),
            'performance_monitor': get_performance_monitor(),
            'optimization_engine': get_optimization_engine()
        }
    
    stats = {}
    
    # Collect stats from each component
    try:
        stats['model_optimizer'] = components['model_optimizer'].get_performance_metrics()
    except:
        stats['model_optimizer'] = None
        
    try:
        stats['load_balancer'] = components['load_balancer'].get_stats()
    except:
        stats['load_balancer'] = None
        
    try:
        stats['auto_scaler'] = components['auto_scaler'].get_scaling_stats()
    except:
        stats['auto_scaler'] = None
        
    try:
        stats['performance_monitor'] = components['performance_monitor'].get_monitoring_stats()
    except:
        stats['performance_monitor'] = None
        
    try:
        stats['optimization_engine'] = components['optimization_engine'].get_optimization_stats()
    except:
        stats['optimization_engine'] = None
    
    return stats