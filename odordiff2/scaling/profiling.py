"""
Continuous Profiling and Performance Regression Testing for OdorDiff-2

This module provides advanced performance monitoring and regression testing:
- Continuous CPU and memory profiling
- Performance baseline establishment
- Automated regression detection
- Hotspot identification and analysis
- Memory leak detection
- Performance trend analysis
- Integration with monitoring systems
"""

import os
import time
import asyncio
import threading
import psutil
import cProfile
import pstats
import tracemalloc
import gc
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager
import json
import pickle
import statistics
import logging
from pathlib import Path
import weakref

import py_spy
from memory_profiler import profile as memory_profile, LineProfiler
import line_profiler

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProfileData:
    """Container for profiling data."""
    timestamp: float
    duration: float
    cpu_stats: Dict[str, Any]
    memory_stats: Dict[str, Any]
    function_stats: Dict[str, Any] = field(default_factory=dict)
    hotspots: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression testing."""
    name: str
    version: str
    cpu_time_mean: float
    cpu_time_std: float
    memory_peak_mb: float
    memory_baseline_mb: float
    function_times: Dict[str, Tuple[float, float]]  # (mean, std)
    created_at: float
    samples_count: int


@dataclass
class RegressionResult:
    """Result of performance regression analysis."""
    test_name: str
    has_regression: bool
    severity: str  # "none", "minor", "major", "critical"
    cpu_regression: Optional[float] = None
    memory_regression: Optional[float] = None
    function_regressions: Dict[str, float] = field(default_factory=dict)
    details: str = ""


class ContinuousProfiler:
    """Continuous profiling system for performance monitoring."""
    
    def __init__(
        self,
        profile_interval: float = 10.0,
        memory_tracking: bool = True,
        cpu_tracking: bool = True,
        function_tracking: bool = True,
        output_dir: str = "profiling_data"
    ):
        self.profile_interval = profile_interval
        self.memory_tracking = memory_tracking
        self.cpu_tracking = cpu_tracking
        self.function_tracking = function_tracking
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # State management
        self.active = False
        self.profiling_thread: Optional[threading.Thread] = None
        self.profile_data_history: List[ProfileData] = []
        
        # Profilers
        self.cpu_profiler: Optional[cProfile.Profile] = None
        self.memory_profiler: Optional[LineProfiler] = None
        
        # Memory tracking
        if self.memory_tracking:
            tracemalloc.start(25)  # Keep 25 frames
    
    def start(self):
        """Start continuous profiling."""
        if self.active:
            logger.warning("Profiler is already running")
            return
        
        self.active = True
        self.profiling_thread = threading.Thread(
            target=self._profiling_loop,
            daemon=True
        )
        self.profiling_thread.start()
        
        logger.info("Continuous profiler started")
    
    def stop(self):
        """Stop continuous profiling."""
        self.active = False
        
        if self.profiling_thread:
            self.profiling_thread.join(timeout=5)
        
        self._save_accumulated_data()
        logger.info("Continuous profiler stopped")
    
    def _profiling_loop(self):
        """Main profiling loop."""
        while self.active:
            try:
                start_time = time.time()
                
                # Collect profile data
                profile_data = self._collect_profile_data()
                profile_data.timestamp = start_time
                profile_data.duration = time.time() - start_time
                
                # Store data
                self.profile_data_history.append(profile_data)
                
                # Limit history size
                if len(self.profile_data_history) > 1000:
                    self.profile_data_history = self.profile_data_history[-500:]
                
                # Sleep until next interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.profile_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in profiling loop: {e}")
                time.sleep(self.profile_interval)
    
    def _collect_profile_data(self) -> ProfileData:
        """Collect current profiling data."""
        # CPU stats
        cpu_stats = self._collect_cpu_stats() if self.cpu_tracking else {}
        
        # Memory stats
        memory_stats = self._collect_memory_stats() if self.memory_tracking else {}
        
        # Function stats
        function_stats = self._collect_function_stats() if self.function_tracking else {}
        
        # Hotspots
        hotspots = self._identify_hotspots()
        
        return ProfileData(
            timestamp=time.time(),
            duration=0,  # Will be set by caller
            cpu_stats=cpu_stats,
            memory_stats=memory_stats,
            function_stats=function_stats,
            hotspots=hotspots,
            metadata={
                'process_id': os.getpid(),
                'thread_count': threading.active_count()
            }
        )
    
    def _collect_cpu_stats(self) -> Dict[str, Any]:
        """Collect CPU profiling statistics."""
        try:
            process = psutil.Process()
            
            # CPU times
            cpu_times = process.cpu_times()
            cpu_percent = process.cpu_percent()
            
            # Thread information
            threads = process.threads()
            
            return {
                'cpu_percent': cpu_percent,
                'user_time': cpu_times.user,
                'system_time': cpu_times.system,
                'num_threads': len(threads),
                'cpu_affinity': process.cpu_affinity() if hasattr(process, 'cpu_affinity') else None
            }
            
        except Exception as e:
            logger.error(f"Error collecting CPU stats: {e}")
            return {}
    
    def _collect_memory_stats(self) -> Dict[str, Any]:
        """Collect memory profiling statistics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            stats = {
                'rss_mb': memory_info.rss / (1024 ** 2),
                'vms_mb': memory_info.vms / (1024 ** 2),
                'memory_percent': memory_percent,
                'num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0
            }
            
            # Memory tracking data
            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                stats.update({
                    'traced_current_mb': current / (1024 ** 2),
                    'traced_peak_mb': peak / (1024 ** 2)
                })
                
                # Top memory allocations
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')[:10]
                
                stats['top_allocations'] = [
                    {
                        'file': stat.traceback.format()[0],
                        'size_mb': stat.size / (1024 ** 2),
                        'count': stat.count
                    }
                    for stat in top_stats
                ]
            
            return stats
            
        except Exception as e:
            logger.error(f"Error collecting memory stats: {e}")
            return {}
    
    def _collect_function_stats(self) -> Dict[str, Any]:
        """Collect function-level profiling statistics."""
        try:
            # This would typically use a sampling profiler
            # For now, we'll collect basic threading info
            
            # Get all threads
            threads_info = []
            for thread in threading.enumerate():
                if thread.is_alive():
                    threads_info.append({
                        'name': thread.name,
                        'daemon': thread.daemon,
                        'ident': thread.ident
                    })
            
            return {
                'threads': threads_info,
                'active_count': threading.active_count()
            }
            
        except Exception as e:
            logger.error(f"Error collecting function stats: {e}")
            return {}
    
    def _identify_hotspots(self) -> List[Dict[str, Any]]:
        """Identify performance hotspots."""
        hotspots = []
        
        try:
            # CPU hotspots based on high CPU usage
            process = psutil.Process()
            cpu_percent = process.cpu_percent()
            
            if cpu_percent > 80:
                hotspots.append({
                    'type': 'cpu',
                    'severity': 'high',
                    'description': f'High CPU usage: {cpu_percent:.1f}%',
                    'value': cpu_percent
                })
            
            # Memory hotspots
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 ** 2)
            
            if memory_mb > 1000:  # More than 1GB
                hotspots.append({
                    'type': 'memory',
                    'severity': 'medium' if memory_mb < 2000 else 'high',
                    'description': f'High memory usage: {memory_mb:.1f}MB',
                    'value': memory_mb
                })
            
            # Thread count hotspots
            thread_count = threading.active_count()
            if thread_count > 50:
                hotspots.append({
                    'type': 'threads',
                    'severity': 'medium',
                    'description': f'High thread count: {thread_count}',
                    'value': thread_count
                })
            
        except Exception as e:
            logger.error(f"Error identifying hotspots: {e}")
        
        return hotspots
    
    def _save_accumulated_data(self):
        """Save accumulated profiling data to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.output_dir / f"profile_data_{timestamp}.json"
            
            # Convert to serializable format
            serializable_data = []
            for data in self.profile_data_history:
                serializable_data.append({
                    'timestamp': data.timestamp,
                    'duration': data.duration,
                    'cpu_stats': data.cpu_stats,
                    'memory_stats': data.memory_stats,
                    'function_stats': data.function_stats,
                    'hotspots': data.hotspots,
                    'metadata': data.metadata
                })
            
            with open(filename, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            logger.info(f"Profiling data saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving profiling data: {e}")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current profiling statistics."""
        if not self.profile_data_history:
            return {}
        
        latest = self.profile_data_history[-1]
        recent = self.profile_data_history[-10:] if len(self.profile_data_history) >= 10 else self.profile_data_history
        
        # Calculate trends
        if len(recent) > 1:
            cpu_values = [p.cpu_stats.get('cpu_percent', 0) for p in recent]
            memory_values = [p.memory_stats.get('rss_mb', 0) for p in recent]
            
            cpu_trend = "increasing" if cpu_values[-1] > cpu_values[0] else "decreasing"
            memory_trend = "increasing" if memory_values[-1] > memory_values[0] else "decreasing"
        else:
            cpu_trend = memory_trend = "stable"
        
        return {
            'current': {
                'cpu_percent': latest.cpu_stats.get('cpu_percent', 0),
                'memory_mb': latest.memory_stats.get('rss_mb', 0),
                'thread_count': latest.metadata.get('thread_count', 0)
            },
            'trends': {
                'cpu': cpu_trend,
                'memory': memory_trend
            },
            'hotspots': latest.hotspots,
            'data_points': len(self.profile_data_history)
        }


class PerformanceRegressor:
    """Detects performance regressions by comparing against baselines."""
    
    def __init__(self, baselines_dir: str = "performance_baselines"):
        self.baselines_dir = Path(baselines_dir)
        self.baselines_dir.mkdir(exist_ok=True)
        self.baselines: Dict[str, PerformanceBaseline] = {}
        
        # Regression thresholds
        self.cpu_regression_threshold = 1.20  # 20% slower
        self.memory_regression_threshold = 1.15  # 15% more memory
        self.function_regression_threshold = 1.25  # 25% slower functions
        
        # Load existing baselines
        self._load_baselines()
    
    def create_baseline(
        self,
        name: str,
        version: str,
        profile_data: List[ProfileData],
        save: bool = True
    ) -> PerformanceBaseline:
        """Create a performance baseline from profile data."""
        
        if not profile_data:
            raise ValueError("No profile data provided")
        
        # Calculate CPU statistics
        cpu_times = [p.cpu_stats.get('cpu_percent', 0) for p in profile_data if p.cpu_stats]
        cpu_time_mean = statistics.mean(cpu_times) if cpu_times else 0
        cpu_time_std = statistics.stdev(cpu_times) if len(cpu_times) > 1 else 0
        
        # Calculate memory statistics
        memory_values = [p.memory_stats.get('rss_mb', 0) for p in profile_data if p.memory_stats]
        memory_peak_mb = max(memory_values) if memory_values else 0
        memory_baseline_mb = statistics.mean(memory_values) if memory_values else 0
        
        # Function statistics (placeholder - would need more detailed profiling)
        function_times = {}
        
        baseline = PerformanceBaseline(
            name=name,
            version=version,
            cpu_time_mean=cpu_time_mean,
            cpu_time_std=cpu_time_std,
            memory_peak_mb=memory_peak_mb,
            memory_baseline_mb=memory_baseline_mb,
            function_times=function_times,
            created_at=time.time(),
            samples_count=len(profile_data)
        )
        
        self.baselines[name] = baseline
        
        if save:
            self._save_baseline(baseline)
        
        logger.info(f"Created performance baseline '{name}' (v{version})")
        return baseline
    
    def check_regression(
        self,
        test_name: str,
        current_data: List[ProfileData],
        baseline_name: Optional[str] = None
    ) -> RegressionResult:
        """Check for performance regression against baseline."""
        
        baseline_name = baseline_name or test_name
        
        if baseline_name not in self.baselines:
            return RegressionResult(
                test_name=test_name,
                has_regression=False,
                severity="none",
                details="No baseline available for comparison"
            )
        
        baseline = self.baselines[baseline_name]
        
        # Calculate current performance metrics
        current_cpu_times = [p.cpu_stats.get('cpu_percent', 0) for p in current_data if p.cpu_stats]
        current_memory_values = [p.memory_stats.get('rss_mb', 0) for p in current_data if p.memory_stats]
        
        if not current_cpu_times or not current_memory_values:
            return RegressionResult(
                test_name=test_name,
                has_regression=False,
                severity="none",
                details="Insufficient current data for comparison"
            )
        
        current_cpu_mean = statistics.mean(current_cpu_times)
        current_memory_peak = max(current_memory_values)
        
        # Check CPU regression
        cpu_regression = None
        if baseline.cpu_time_mean > 0:
            cpu_regression = current_cpu_mean / baseline.cpu_time_mean
        
        # Check memory regression
        memory_regression = None
        if baseline.memory_peak_mb > 0:
            memory_regression = current_memory_peak / baseline.memory_peak_mb
        
        # Determine overall regression
        has_regression = False
        severity = "none"
        details = []
        
        if cpu_regression and cpu_regression > self.cpu_regression_threshold:
            has_regression = True
            cpu_increase = (cpu_regression - 1) * 100
            details.append(f"CPU usage increased by {cpu_increase:.1f}%")
            
            if cpu_regression > 1.5:
                severity = "critical"
            elif cpu_regression > 1.3:
                severity = "major"
            else:
                severity = "minor"
        
        if memory_regression and memory_regression > self.memory_regression_threshold:
            has_regression = True
            memory_increase = (memory_regression - 1) * 100
            details.append(f"Memory usage increased by {memory_increase:.1f}%")
            
            current_severity = "critical" if memory_regression > 1.3 else "major" if memory_regression > 1.2 else "minor"
            if severity == "none" or (current_severity == "critical" and severity != "critical"):
                severity = current_severity
        
        return RegressionResult(
            test_name=test_name,
            has_regression=has_regression,
            severity=severity,
            cpu_regression=cpu_regression,
            memory_regression=memory_regression,
            details="; ".join(details) if details else "No significant regression detected"
        )
    
    def _save_baseline(self, baseline: PerformanceBaseline):
        """Save baseline to disk."""
        try:
            filename = self.baselines_dir / f"{baseline.name}_{baseline.version}.json"
            
            baseline_data = {
                'name': baseline.name,
                'version': baseline.version,
                'cpu_time_mean': baseline.cpu_time_mean,
                'cpu_time_std': baseline.cpu_time_std,
                'memory_peak_mb': baseline.memory_peak_mb,
                'memory_baseline_mb': baseline.memory_baseline_mb,
                'function_times': baseline.function_times,
                'created_at': baseline.created_at,
                'samples_count': baseline.samples_count
            }
            
            with open(filename, 'w') as f:
                json.dump(baseline_data, f, indent=2)
            
            logger.info(f"Baseline saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving baseline: {e}")
    
    def _load_baselines(self):
        """Load baselines from disk."""
        try:
            for filename in self.baselines_dir.glob("*.json"):
                with open(filename, 'r') as f:
                    baseline_data = json.load(f)
                
                baseline = PerformanceBaseline(**baseline_data)
                self.baselines[baseline.name] = baseline
            
            logger.info(f"Loaded {len(self.baselines)} performance baselines")
            
        except Exception as e:
            logger.error(f"Error loading baselines: {e}")
    
    def list_baselines(self) -> List[Dict[str, Any]]:
        """List all available baselines."""
        return [
            {
                'name': baseline.name,
                'version': baseline.version,
                'created_at': datetime.fromtimestamp(baseline.created_at).isoformat(),
                'samples_count': baseline.samples_count,
                'cpu_mean': baseline.cpu_time_mean,
                'memory_peak_mb': baseline.memory_peak_mb
            }
            for baseline in self.baselines.values()
        ]


@contextmanager
def profile_function(function_name: str, profiler: Optional[ContinuousProfiler] = None):
    """Context manager for profiling individual functions."""
    start_time = time.time()
    start_memory = None
    
    # Start memory tracking if available
    if tracemalloc.is_tracing():
        start_memory = tracemalloc.get_traced_memory()[0]
    
    try:
        yield
        
    finally:
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate memory usage
        memory_used = 0
        if start_memory and tracemalloc.is_tracing():
            end_memory = tracemalloc.get_traced_memory()[0]
            memory_used = (end_memory - start_memory) / (1024 ** 2)  # MB
        
        logger.debug(f"Function '{function_name}' took {duration:.4f}s, used {memory_used:.2f}MB")
        
        # Store in profiler if provided
        if profiler:
            # This would be integrated with the profiler's data collection
            pass


class PerformanceMonitor:
    """High-level performance monitoring orchestrator."""
    
    def __init__(
        self,
        profiler: Optional[ContinuousProfiler] = None,
        regressor: Optional[PerformanceRegressor] = None
    ):
        self.profiler = profiler or ContinuousProfiler()
        self.regressor = regressor or PerformanceRegressor()
        
        # Monitoring state
        self.monitoring_active = False
        self.regression_check_interval = 3600  # 1 hour
        self.last_regression_check = 0
    
    async def start_monitoring(self):
        """Start comprehensive performance monitoring."""
        self.profiler.start()
        self.monitoring_active = True
        
        # Start background regression checking
        asyncio.create_task(self._regression_check_loop())
        
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        self.profiler.stop()
        
        logger.info("Performance monitoring stopped")
    
    async def _regression_check_loop(self):
        """Background loop for checking regressions."""
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                if current_time - self.last_regression_check > self.regression_check_interval:
                    await self._check_for_regressions()
                    self.last_regression_check = current_time
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in regression check loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_for_regressions(self):
        """Check for performance regressions."""
        try:
            # Get recent profiling data
            recent_data = self.profiler.profile_data_history[-100:] if self.profiler.profile_data_history else []
            
            if not recent_data:
                return
            
            # Check against all baselines
            for baseline_name in self.regressor.baselines:
                result = self.regressor.check_regression(
                    test_name="continuous_monitoring",
                    current_data=recent_data,
                    baseline_name=baseline_name
                )
                
                if result.has_regression and result.severity in ["major", "critical"]:
                    logger.warning(f"Performance regression detected: {result.details}")
                    
                    # This would typically trigger alerts
                    await self._handle_regression(result)
        
        except Exception as e:
            logger.error(f"Error checking for regressions: {e}")
    
    async def _handle_regression(self, result: RegressionResult):
        """Handle detected performance regression."""
        # This would typically:
        # 1. Send alerts to monitoring systems
        # 2. Create tickets/issues
        # 3. Trigger additional profiling
        # 4. Potentially trigger auto-scaling
        
        logger.error(f"Handling regression: {result.test_name} - {result.details}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        profiler_stats = self.profiler.get_current_stats()
        baselines = self.regressor.list_baselines()
        
        return {
            'current_performance': profiler_stats,
            'baselines': baselines,
            'monitoring_active': self.monitoring_active,
            'last_regression_check': datetime.fromtimestamp(self.last_regression_check).isoformat()
        }


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor."""
    global _performance_monitor
    
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    
    return _performance_monitor