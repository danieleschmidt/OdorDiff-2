"""
Enhanced logging system with structured logging, correlation IDs, and distributed tracing.
"""

import logging
import sys
import json
import time
import uuid
import contextvars
import asyncio
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
from datetime import datetime
import threading
from functools import wraps
import traceback
import inspect
from collections import defaultdict, deque

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# Correlation ID context variable for tracking requests
correlation_id_context: contextvars.ContextVar[str] = contextvars.ContextVar('correlation_id', default=None)

# Trace ID context variable for distributed tracing
trace_id_context: contextvars.ContextVar[str] = contextvars.ContextVar('trace_id', default=None)

# Span context for nested operations
span_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar('span', default=None)


class TraceSpan:
    """Represents a trace span for distributed tracing."""
    
    def __init__(self, name: str, parent_span: Optional['TraceSpan'] = None, 
                 trace_id: str = None, metadata: Dict[str, Any] = None):
        self.span_id = str(uuid.uuid4())
        self.name = name
        self.trace_id = trace_id or str(uuid.uuid4())
        self.parent_span_id = parent_span.span_id if parent_span else None
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.metadata = metadata or {}
        self.tags = {}
        self.logs = []
        self.status = "OK"
        self.error: Optional[Exception] = None
        
    def add_tag(self, key: str, value: Any):
        """Add a tag to the span."""
        self.tags[key] = str(value)
    
    def add_log(self, message: str, level: str = "INFO", **fields):
        """Add a log entry to the span."""
        self.logs.append({
            'timestamp': time.time(),
            'message': message,
            'level': level,
            'fields': fields
        })
    
    def set_error(self, error: Exception):
        """Mark span as having an error."""
        self.status = "ERROR"
        self.error = error
        self.add_tag("error", True)
        self.add_tag("error.type", type(error).__name__)
        self.add_tag("error.message", str(error))
    
    def finish(self):
        """Finish the span."""
        self.end_time = time.time()
    
    def duration(self) -> float:
        """Get span duration."""
        end = self.end_time or time.time()
        return end - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary."""
        return {
            'span_id': self.span_id,
            'trace_id': self.trace_id,
            'parent_span_id': self.parent_span_id,
            'name': self.name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration(),
            'status': self.status,
            'tags': self.tags,
            'logs': self.logs,
            'metadata': self.metadata,
            'error': str(self.error) if self.error else None
        }


class CorrelationContext:
    """Context manager for correlation IDs and tracing."""
    
    def __init__(self, correlation_id: str = None, trace_id: str = None):
        self.correlation_id = correlation_id or self.generate_correlation_id()
        self.trace_id = trace_id or str(uuid.uuid4())
        self.previous_correlation_id = None
        self.previous_trace_id = None
    
    @staticmethod
    def generate_correlation_id() -> str:
        """Generate a unique correlation ID."""
        return f"req_{uuid.uuid4().hex[:16]}"
    
    def __enter__(self):
        self.previous_correlation_id = correlation_id_context.get()
        self.previous_trace_id = trace_id_context.get()
        
        correlation_id_context.set(self.correlation_id)
        trace_id_context.set(self.trace_id)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        correlation_id_context.set(self.previous_correlation_id)
        trace_id_context.set(self.previous_trace_id)


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add correlation ID if available
        correlation_id = correlation_id_context.get()
        if correlation_id:
            log_data['correlation_id'] = correlation_id
        
        # Add trace ID if available
        trace_id = trace_id_context.get()
        if trace_id:
            log_data['trace_id'] = trace_id
        
        # Add span information if available
        current_span = span_context.get()
        if current_span:
            log_data['span_id'] = current_span.get('span_id')
            log_data['span_name'] = current_span.get('name')
        
        # Add exception information
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from the log record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 'msecs',
                          'relativeCreated', 'thread', 'threadName', 'processName', 'process',
                          'message', 'exc_info', 'exc_text', 'stack_info']:
                log_data['extra'] = log_data.get('extra', {})
                log_data['extra'][key] = value
        
        return json.dumps(log_data, default=str, separators=(',', ':'))


class DistributedTracer:
    """Distributed tracing system."""
    
    def __init__(self, service_name: str = "odordiff2"):
        self.service_name = service_name
        self.active_spans: Dict[str, TraceSpan] = {}
        self.completed_traces: deque = deque(maxlen=1000)
        self.lock = threading.RLock()
    
    def start_span(self, name: str, parent_span: TraceSpan = None, 
                   metadata: Dict[str, Any] = None) -> TraceSpan:
        """Start a new trace span."""
        # Get current trace ID or create new one
        current_trace_id = trace_id_context.get()
        if not current_trace_id:
            current_trace_id = str(uuid.uuid4())
            trace_id_context.set(current_trace_id)
        
        # Create span
        span = TraceSpan(
            name=name,
            parent_span=parent_span,
            trace_id=current_trace_id,
            metadata=metadata
        )
        
        # Add service information
        span.add_tag("service.name", self.service_name)
        span.add_tag("service.version", "1.0.0")
        
        with self.lock:
            self.active_spans[span.span_id] = span
        
        return span
    
    def finish_span(self, span: TraceSpan):
        """Finish a trace span."""
        span.finish()
        
        with self.lock:
            self.active_spans.pop(span.span_id, None)
            
            # Add to completed traces
            self.completed_traces.append(span.to_dict())
    
    def get_active_spans(self) -> List[Dict[str, Any]]:
        """Get all active spans."""
        with self.lock:
            return [span.to_dict() for span in self.active_spans.values()]
    
    def get_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """Get all spans for a specific trace."""
        spans = []
        
        # Check active spans
        with self.lock:
            for span in self.active_spans.values():
                if span.trace_id == trace_id:
                    spans.append(span.to_dict())
        
        # Check completed traces
        for span_data in self.completed_traces:
            if span_data.get('trace_id') == trace_id:
                spans.append(span_data)
        
        # Sort by start time
        spans.sort(key=lambda x: x.get('start_time', 0))
        return spans
    
    def get_trace_stats(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        with self.lock:
            total_completed = len(self.completed_traces)
            active_count = len(self.active_spans)
            
            # Calculate average span duration
            durations = [span.get('duration', 0) for span in self.completed_traces if span.get('duration')]
            avg_duration = sum(durations) / len(durations) if durations else 0
            
            # Count spans by status
            status_counts = defaultdict(int)
            for span in self.completed_traces:
                status_counts[span.get('status', 'UNKNOWN')] += 1
            
            return {
                'total_completed_spans': total_completed,
                'active_spans': active_count,
                'average_span_duration': avg_duration,
                'status_distribution': dict(status_counts)
            }


class TraceContextManager:
    """Context manager for trace spans."""
    
    def __init__(self, tracer: DistributedTracer, name: str, 
                 metadata: Dict[str, Any] = None):
        self.tracer = tracer
        self.name = name
        self.metadata = metadata
        self.span: Optional[TraceSpan] = None
        self.previous_span = None
    
    def __enter__(self) -> TraceSpan:
        # Get current span context
        self.previous_span = span_context.get()
        
        # Create new span
        parent_span = None
        if self.previous_span:
            # Find the actual span object
            parent_span_id = self.previous_span.get('span_id')
            if parent_span_id in self.tracer.active_spans:
                parent_span = self.tracer.active_spans[parent_span_id]
        
        self.span = self.tracer.start_span(self.name, parent_span, self.metadata)
        
        # Set span context
        span_context.set({
            'span_id': self.span.span_id,
            'name': self.span.name,
            'trace_id': self.span.trace_id
        })
        
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            if exc_val:
                self.span.set_error(exc_val)
            
            self.tracer.finish_span(self.span)
        
        # Restore previous span context
        span_context.set(self.previous_span)


class OdorDiffLogger:
    """Centralized logging system for OdorDiff-2 with enhanced observability."""
    
    def __init__(self, name: str = "odordiff2", log_dir: str = "logs", structured_logging: bool = True):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.structured_logging = structured_logging
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Initialize distributed tracer
        self.tracer = DistributedTracer(service_name=name)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
            
        self._metrics = {}
        self._metrics_lock = threading.Lock()
        
    def _setup_handlers(self):
        """Setup logging handlers with structured logging support."""
        # Choose formatter based on configuration
        if self.structured_logging:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # File handler for general logs
        file_handler = logging.FileHandler(
            self.log_dir / f"{self.name}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Structured JSON logs file handler
        json_handler = logging.FileHandler(
            self.log_dir / f"{self.name}_structured.jsonl"
        )
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(StructuredFormatter())
        
        # Error file handler
        error_handler = logging.FileHandler(
            self.log_dir / f"{self.name}_errors.log"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(json_handler)
        self.logger.addHandler(error_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        self._log_with_context('info', message, **kwargs)
        
    def error(self, message: str, **kwargs):
        """Log error message with optional structured data."""
        self._log_with_context('error', message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data."""
        self._log_with_context('warning', message, **kwargs)
        
    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data."""
        self._log_with_context('debug', message, **kwargs)
    
    def _log_with_context(self, level: str, message: str, **kwargs):
        """Log message with correlation and tracing context."""
        # Add current span information to log if available
        current_span = span_context.get()
        if current_span:
            kwargs.update({
                'span_id': current_span.get('span_id'),
                'span_name': current_span.get('name'),
                'trace_id': current_span.get('trace_id')
            })
        
        # Add correlation ID if available
        correlation_id = correlation_id_context.get()
        if correlation_id:
            kwargs['correlation_id'] = correlation_id
        
        # Add trace ID if available
        trace_id = trace_id_context.get()
        if trace_id:
            kwargs['trace_id'] = trace_id
        
        # Create log record with extra fields
        if self.structured_logging:
            # For structured logging, add kwargs as extra fields
            log_func = getattr(self.logger, level)
            log_func(message, extra=kwargs)
        else:
            # For traditional logging, format message with data
            formatted_message = self._format_message(message, **kwargs)
            log_func = getattr(self.logger, level)
            log_func(formatted_message)
        
    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with structured data for traditional logging."""
        if kwargs:
            structured_data = json.dumps(kwargs, default=str)
            return f"{message} | Data: {structured_data}"
        return message
    
    # Tracing integration methods
    def start_span(self, name: str, metadata: Dict[str, Any] = None) -> TraceContextManager:
        """Start a distributed trace span."""
        return TraceContextManager(self.tracer, name, metadata)
    
    def log_span_event(self, message: str, level: str = "INFO", **fields):
        """Log an event within the current span."""
        current_span = span_context.get()
        if current_span:
            # Add to span logs
            span_id = current_span.get('span_id')
            if span_id in self.tracer.active_spans:
                span_obj = self.tracer.active_spans[span_id]
                span_obj.add_log(message, level, **fields)
        
        # Also log normally
        log_method = getattr(self, level.lower(), self.info)
        log_method(f"[SPAN] {message}", **fields)
    
    def add_span_tag(self, key: str, value: Any):
        """Add a tag to the current span."""
        current_span = span_context.get()
        if current_span:
            span_id = current_span.get('span_id')
            if span_id in self.tracer.active_spans:
                span_obj = self.tracer.active_spans[span_id]
                span_obj.add_tag(key, value)
    
    def log_trace_summary(self, trace_id: str):
        """Log a summary of a completed trace."""
        trace_spans = self.tracer.get_trace(trace_id)
        if trace_spans:
            total_duration = sum(span.get('duration', 0) for span in trace_spans)
            span_count = len(trace_spans)
            error_count = sum(1 for span in trace_spans if span.get('status') == 'ERROR')
            
            self.info(
                "Trace completed",
                trace_id=trace_id,
                total_duration=total_duration,
                span_count=span_count,
                error_count=error_count,
                spans=[{
                    'name': span.get('name'),
                    'duration': span.get('duration'),
                    'status': span.get('status')
                } for span in trace_spans]
            )
    
    def get_tracing_stats(self) -> Dict[str, Any]:
        """Get distributed tracing statistics."""
        return self.tracer.get_trace_stats()
    
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


def log_function_call(logger: OdorDiffLogger, enable_tracing: bool = True):
    """Decorator to log function calls with optional distributed tracing."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = f"{func.__module__}.{func.__name__}"
            
            # Create span if tracing is enabled
            span_ctx = None
            if enable_tracing:
                span_metadata = {
                    'function': func_name,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }
                span_ctx = logger.start_span(f"function:{func.__name__}", span_metadata)
            
            try:
                if span_ctx:
                    with span_ctx as span:
                        logger.debug(f"Calling {func_name}", args_count=len(args), kwargs_keys=list(kwargs.keys()))
                        
                        result = func(*args, **kwargs)
                        execution_time = time.time() - start_time
                        
                        # Add tags to span
                        span.add_tag("function.module", func.__module__)
                        span.add_tag("function.name", func.__name__)
                        span.add_tag("execution_time", execution_time)
                        span.add_tag("success", True)
                        
                        logger.debug(f"Completed {func_name}", execution_time=execution_time)
                        logger.record_metric(f"{func_name}_execution_time", execution_time)
                        return result
                else:
                    # No tracing, just regular logging
                    logger.debug(f"Calling {func_name}", args_count=len(args), kwargs_keys=list(kwargs.keys()))
                    
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    logger.debug(f"Completed {func_name}", execution_time=execution_time)
                    logger.record_metric(f"{func_name}_execution_time", execution_time)
                    return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Mark span as error if tracing
                if span_ctx and 'span' in locals():
                    span.set_error(e)
                    span.add_tag("execution_time", execution_time)
                
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


def trace_function(logger: OdorDiffLogger, operation_name: str = None):
    """Decorator to add distributed tracing to functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            span_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with logger.start_span(span_name) as span:
                # Add function metadata
                span.add_tag("function.module", func.__module__)
                span.add_tag("function.name", func.__name__)
                span.add_tag("function.args_count", len(args))
                
                # Add specific argument information if available
                if args and hasattr(args[0], '__dict__'):
                    span.add_tag("function.self_class", args[0].__class__.__name__)
                
                try:
                    result = func(*args, **kwargs)
                    span.add_tag("success", True)
                    return result
                except Exception as e:
                    span.set_error(e)
                    raise
        
        return wrapper
    return decorator


def with_correlation_context(correlation_id: str = None):
    """Context manager to set correlation ID for a block of code."""
    return CorrelationContext(correlation_id)


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

def get_logger(name: str = "odordiff2", structured_logging: bool = True) -> OdorDiffLogger:
    """Get global logger instance with enhanced observability features."""
    global _global_logger
    if _global_logger is None:
        _global_logger = OdorDiffLogger(name, structured_logging=structured_logging)
    return _global_logger


def configure_observability(
    service_name: str = "odordiff2",
    log_level: str = "INFO",
    structured_logging: bool = True,
    enable_tracing: bool = True,
    log_directory: str = "logs"
) -> OdorDiffLogger:
    """Configure comprehensive observability for the application."""
    global _global_logger
    
    # Create enhanced logger
    _global_logger = OdorDiffLogger(
        name=service_name,
        log_dir=log_directory,
        structured_logging=structured_logging
    )
    
    # Set log level
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    if log_level.upper() in log_level_map:
        _global_logger.logger.setLevel(log_level_map[log_level.upper()])
    
    _global_logger.info(
        "Observability configured",
        service_name=service_name,
        log_level=log_level,
        structured_logging=structured_logging,
        enable_tracing=enable_tracing,
        log_directory=log_directory
    )
    
    return _global_logger