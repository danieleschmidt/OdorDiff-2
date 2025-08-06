"""
Comprehensive logging and monitoring system.
"""

import logging
import sys
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import threading
from functools import wraps


class OdorDiffLogger:
    """Centralized logging system for OdorDiff-2."""
    
    def __init__(self, name: str = "odordiff2", log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
            
        self._metrics = {}
        self._metrics_lock = threading.Lock()
        
    def _setup_handlers(self):
        """Setup logging handlers."""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        
        # File handler for general logs
        file_handler = logging.FileHandler(
            self.log_dir / f"{self.name}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        
        # Error file handler
        error_handler = logging.FileHandler(
            self.log_dir / f"{self.name}_errors.log"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_format)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        self.logger.info(self._format_message(message, **kwargs))
        
    def error(self, message: str, **kwargs):
        """Log error message with optional structured data."""
        self.logger.error(self._format_message(message, **kwargs))
        
    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data."""
        self.logger.warning(self._format_message(message, **kwargs))
        
    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data."""
        self.logger.debug(self._format_message(message, **kwargs))
        
    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with structured data."""
        if kwargs:
            structured_data = json.dumps(kwargs, default=str)
            return f"{message} | Data: {structured_data}"
        return message
    
    def log_generation_attempt(self, prompt: str, num_molecules: int, **kwargs):
        """Log molecule generation attempt."""
        self.info(
            "Generation attempt",
            prompt=prompt,
            num_molecules=num_molecules,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )
        
    def log_generation_result(self, prompt: str, generated_count: int, filtered_count: int, **kwargs):
        """Log molecule generation result."""
        self.info(
            "Generation completed",
            prompt=prompt,
            generated=generated_count,
            filtered=filtered_count,
            success_rate=filtered_count/max(1, generated_count),
            **kwargs
        )
        
    def log_safety_assessment(self, molecule_smiles: str, safety_report: Dict[str, Any]):
        """Log safety assessment results."""
        self.info(
            "Safety assessment",
            smiles=molecule_smiles,
            toxicity=safety_report.get('toxicity', 0),
            ifra_compliant=safety_report.get('ifra_compliant', False),
            regulatory_flags=len(safety_report.get('regulatory_flags', []))
        )
        
    def log_error_with_context(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with full context."""
        self.error(
            f"Error occurred: {str(error)}",
            error_type=type(error).__name__,
            context=context or {},
            timestamp=datetime.utcnow().isoformat()
        )
        
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a metric value."""
        with self._metrics_lock:
            if name not in self._metrics:
                self._metrics[name] = []
            self._metrics[name].append({
                'value': value,
                'timestamp': time.time(),
                'tags': tags or {}
            })
            
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of recorded metrics."""
        with self._metrics_lock:
            summary = {}
            for name, values in self._metrics.items():
                if values:
                    latest_values = [v['value'] for v in values[-100:]]  # Last 100
                    summary[name] = {
                        'count': len(values),
                        'latest': values[-1]['value'],
                        'avg': sum(latest_values) / len(latest_values),
                        'min': min(latest_values),
                        'max': max(latest_values)
                    }
            return summary


def log_function_call(logger: OdorDiffLogger):
    """Decorator to log function calls."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = f"{func.__module__}.{func.__name__}"
            
            logger.debug(f"Calling {func_name}", args_count=len(args), kwargs_keys=list(kwargs.keys()))
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.debug(f"Completed {func_name}", execution_time=execution_time)
                logger.record_metric(f"{func_name}_execution_time", execution_time)
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.log_error_with_context(
                    e, 
                    {
                        'function': func_name,
                        'execution_time': execution_time,
                        'args_count': len(args),
                        'kwargs': list(kwargs.keys())
                    }
                )
                raise
                
        return wrapper
    return decorator


class PerformanceMonitor:
    """Monitor system performance and resource usage."""
    
    def __init__(self, logger: OdorDiffLogger):
        self.logger = logger
        self._monitoring = False
        
    def start_monitoring(self, interval: int = 60):
        """Start background monitoring."""
        if self._monitoring:
            return
            
        self._monitoring = True
        
        def monitor_loop():
            while self._monitoring:
                try:
                    # Monitor memory usage
                    import psutil
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    cpu_percent = process.cpu_percent()
                    
                    self.logger.record_metric("memory_usage_mb", memory_mb)
                    self.logger.record_metric("cpu_usage_percent", cpu_percent)
                    
                    # Log if usage is high
                    if memory_mb > 1000:  # >1GB
                        self.logger.warning(f"High memory usage: {memory_mb:.1f}MB")
                    if cpu_percent > 80:
                        self.logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
                        
                except ImportError:
                    # psutil not available
                    break
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
                    
                time.sleep(interval)
                
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring = False


# Global logger instance
_global_logger: Optional[OdorDiffLogger] = None

def get_logger(name: str = "odordiff2") -> OdorDiffLogger:
    """Get global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = OdorDiffLogger(name)
    return _global_logger