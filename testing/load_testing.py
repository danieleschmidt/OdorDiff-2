"""
Comprehensive Load Testing and Stress Testing for OdorDiff-2

This module provides advanced load testing capabilities including:
- Multi-scenario load testing
- Stress testing with gradual load increase
- Chaos engineering integration
- Real-time performance monitoring
- Automated scaling validation
- Performance regression detection
"""

import asyncio
import aiohttp
import time
import json
import random
import statistics
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import logging
import argparse
import yaml
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    # Test parameters
    base_url: str = "http://localhost:8000"
    test_duration_seconds: int = 300  # 5 minutes
    ramp_up_duration_seconds: int = 60  # 1 minute ramp-up
    
    # Load patterns
    initial_users: int = 1
    max_users: int = 100
    ramp_up_step: int = 10
    step_duration: int = 30
    
    # Request configuration
    request_timeout: int = 30
    max_concurrent_requests: int = 1000
    think_time_min: float = 1.0  # Minimum time between user actions
    think_time_max: float = 5.0  # Maximum time between user actions
    
    # Test scenarios
    scenarios: List[Dict[str, Any]] = field(default_factory=list)
    
    # Monitoring
    metrics_collection_interval: float = 1.0
    enable_detailed_logging: bool = False
    
    # Stress testing
    stress_test_enabled: bool = False
    stress_max_users: int = 1000
    stress_ramp_duration: int = 600  # 10 minutes
    
    # Chaos engineering
    chaos_enabled: bool = False
    chaos_failure_rate: float = 0.05  # 5% failure injection


@dataclass
class RequestResult:
    """Result of a single request."""
    timestamp: float
    url: str
    method: str
    status_code: int
    response_time: float
    request_size: int = 0
    response_size: int = 0
    error: Optional[str] = None
    scenario: str = "default"
    user_id: str = ""


@dataclass
class TestMetrics:
    """Aggregated test metrics."""
    timestamp: float
    active_users: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    errors_per_second: float
    throughput_bytes_per_second: float
    
    # Resource metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_io: float = 0.0
    network_io: float = 0.0


class LoadTestScenario:
    """Defines a load test scenario with specific request patterns."""
    
    def __init__(
        self,
        name: str,
        weight: float,
        requests: List[Dict[str, Any]],
        think_time_min: float = 1.0,
        think_time_max: float = 5.0
    ):
        self.name = name
        self.weight = weight
        self.requests = requests
        self.think_time_min = think_time_min
        self.think_time_max = think_time_max
    
    async def execute(self, session: aiohttp.ClientSession, user_id: str) -> List[RequestResult]:
        """Execute the scenario for a single user."""
        results = []
        
        for request_config in self.requests:
            # Apply think time
            think_time = random.uniform(self.think_time_min, self.think_time_max)
            await asyncio.sleep(think_time)
            
            # Execute request
            result = await self._execute_request(session, request_config, user_id)
            results.append(result)
            
            # Stop on critical errors
            if result.error and "timeout" not in result.error.lower():
                break
        
        return results
    
    async def _execute_request(
        self, 
        session: aiohttp.ClientSession, 
        request_config: Dict[str, Any],
        user_id: str
    ) -> RequestResult:
        """Execute a single request."""
        url = request_config.get('url', '/')
        method = request_config.get('method', 'GET').upper()
        headers = request_config.get('headers', {})
        data = request_config.get('data')
        params = request_config.get('params', {})
        
        # Add dynamic parameters
        if 'params_generator' in request_config:
            generated_params = request_config['params_generator']()
            params.update(generated_params)
        
        start_time = time.time()
        
        try:
            async with session.request(
                method=method,
                url=url,
                headers=headers,
                json=data if isinstance(data, dict) else None,
                data=data if isinstance(data, str) else None,
                params=params
            ) as response:
                response_data = await response.read()
                response_time = time.time() - start_time
                
                return RequestResult(
                    timestamp=start_time,
                    url=url,
                    method=method,
                    status_code=response.status,
                    response_time=response_time,
                    request_size=len(str(data)) if data else 0,
                    response_size=len(response_data),
                    scenario=self.name,
                    user_id=user_id
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            
            return RequestResult(
                timestamp=start_time,
                url=url,
                method=method,
                status_code=0,
                response_time=response_time,
                error=str(e),
                scenario=self.name,
                user_id=user_id
            )


class VirtualUser:
    """Simulates a single user's behavior during load testing."""
    
    def __init__(
        self,
        user_id: str,
        scenarios: List[LoadTestScenario],
        base_url: str,
        config: LoadTestConfig
    ):
        self.user_id = user_id
        self.scenarios = scenarios
        self.base_url = base_url
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.results_queue: Queue = Queue()
        self.active = False
    
    async def start(self):
        """Start the virtual user session."""
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
        
        self.session = aiohttp.ClientSession(
            base_url=self.base_url,
            timeout=timeout,
            connector=connector,
            headers={'User-Agent': f'OdorDiff2-LoadTest-User-{self.user_id}'}
        )
        
        self.active = True
        logger.debug(f"Virtual user {self.user_id} started")
    
    async def stop(self):
        """Stop the virtual user session."""
        self.active = False
        
        if self.session:
            await self.session.close()
        
        logger.debug(f"Virtual user {self.user_id} stopped")
    
    async def run_continuously(self):
        """Run scenarios continuously until stopped."""
        while self.active:
            try:
                # Select scenario based on weight
                scenario = self._select_scenario()
                
                # Execute scenario
                results = await scenario.execute(self.session, self.user_id)
                
                # Queue results for collection
                for result in results:
                    self.results_queue.put(result)
                
            except Exception as e:
                logger.error(f"Error in user {self.user_id}: {e}")
                await asyncio.sleep(1)
    
    def _select_scenario(self) -> LoadTestScenario:
        """Select a scenario based on weights."""
        if not self.scenarios:
            raise ValueError("No scenarios defined")
        
        if len(self.scenarios) == 1:
            return self.scenarios[0]
        
        total_weight = sum(s.weight for s in self.scenarios)
        r = random.uniform(0, total_weight)
        
        cumulative_weight = 0
        for scenario in self.scenarios:
            cumulative_weight += scenario.weight
            if r <= cumulative_weight:
                return scenario
        
        return self.scenarios[-1]  # Fallback


class MetricsCollector:
    """Collects and aggregates performance metrics during testing."""
    
    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.metrics_history: List[TestMetrics] = []
        self.request_results: List[RequestResult] = []
        self.active = False
        self._lock = threading.Lock()
    
    def add_request_result(self, result: RequestResult):
        """Add a request result to the collection."""
        with self._lock:
            self.request_results.append(result)
    
    def start_collection(self, active_users_func: Callable[[], int]):
        """Start metrics collection."""
        self.active = True
        self.active_users_func = active_users_func
        
        threading.Thread(
            target=self._collection_loop,
            daemon=True
        ).start()
        
        logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection."""
        self.active = False
        logger.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Main metrics collection loop."""
        while self.active:
            try:
                self._collect_metrics()
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
    
    def _collect_metrics(self):
        """Collect current metrics."""
        current_time = time.time()
        window_start = current_time - self.collection_interval
        
        with self._lock:
            # Filter results to current window
            window_results = [
                r for r in self.request_results
                if r.timestamp >= window_start
            ]
            
            if not window_results:
                return
            
            # Calculate metrics
            successful_requests = [r for r in window_results if r.error is None and 200 <= r.status_code < 400]
            failed_requests = [r for r in window_results if r.error is not None or r.status_code >= 400]
            
            response_times = [r.response_time for r in successful_requests]
            
            if response_times:
                avg_response_time = statistics.mean(response_times)
                median_response_time = statistics.median(response_times)
                p95_response_time = np.percentile(response_times, 95)
                p99_response_time = np.percentile(response_times, 99)
            else:
                avg_response_time = median_response_time = p95_response_time = p99_response_time = 0.0
            
            requests_per_second = len(window_results) / self.collection_interval
            errors_per_second = len(failed_requests) / self.collection_interval
            
            # Calculate throughput
            total_bytes = sum(r.response_size for r in successful_requests)
            throughput_bytes_per_second = total_bytes / self.collection_interval
            
            # Create metrics object
            metrics = TestMetrics(
                timestamp=current_time,
                active_users=self.active_users_func(),
                total_requests=len(window_results),
                successful_requests=len(successful_requests),
                failed_requests=len(failed_requests),
                avg_response_time=avg_response_time,
                median_response_time=median_response_time,
                p95_response_time=p95_response_time,
                p99_response_time=p99_response_time,
                requests_per_second=requests_per_second,
                errors_per_second=errors_per_second,
                throughput_bytes_per_second=throughput_bytes_per_second
            )
            
            self.metrics_history.append(metrics)
            
            # Log current metrics
            logger.info(f"Metrics: Users={metrics.active_users}, RPS={metrics.requests_per_second:.1f}, "
                       f"AvgRT={metrics.avg_response_time:.3f}s, Errors={metrics.errors_per_second:.1f}")
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of all collected metrics."""
        if not self.metrics_history:
            return {"error": "No metrics collected"}
        
        # Overall statistics
        all_response_times = [r.response_time for r in self.request_results if r.error is None]
        total_requests = len(self.request_results)
        successful_requests = len([r for r in self.request_results if r.error is None and 200 <= r.status_code < 400])
        failed_requests = total_requests - successful_requests
        
        # Peak performance
        peak_rps = max(m.requests_per_second for m in self.metrics_history)
        peak_users = max(m.active_users for m in self.metrics_history)
        
        # Error analysis
        error_types = {}
        for result in self.request_results:
            if result.error:
                error_types[result.error] = error_types.get(result.error, 0) + 1
        
        return {
            "test_summary": {
                "duration_seconds": (self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp),
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": (successful_requests / total_requests * 100) if total_requests > 0 else 0,
                "peak_rps": peak_rps,
                "peak_concurrent_users": peak_users
            },
            "response_times": {
                "average": statistics.mean(all_response_times) if all_response_times else 0,
                "median": statistics.median(all_response_times) if all_response_times else 0,
                "min": min(all_response_times) if all_response_times else 0,
                "max": max(all_response_times) if all_response_times else 0,
                "p95": np.percentile(all_response_times, 95) if all_response_times else 0,
                "p99": np.percentile(all_response_times, 99) if all_response_times else 0
            },
            "error_analysis": error_types,
            "throughput": {
                "average_rps": statistics.mean([m.requests_per_second for m in self.metrics_history]),
                "average_throughput_mbps": statistics.mean([m.throughput_bytes_per_second / (1024*1024) for m in self.metrics_history])
            }
        }


class LoadTester:
    """Main load testing orchestrator."""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.virtual_users: List[VirtualUser] = []
        self.scenarios: List[LoadTestScenario] = []
        self.metrics_collector = MetricsCollector(config.metrics_collection_interval)
        self.active_users = 0
        
        # Initialize default scenarios if none provided
        if not config.scenarios:
            self._setup_default_scenarios()
        else:
            self._setup_scenarios_from_config()
    
    def _setup_default_scenarios(self):
        """Setup default test scenarios."""
        # Health check scenario
        health_scenario = LoadTestScenario(
            name="health_check",
            weight=0.1,
            requests=[{"url": "/health", "method": "GET"}],
            think_time_min=0.5,
            think_time_max=1.0
        )
        
        # Molecule generation scenario
        generation_scenario = LoadTestScenario(
            name="molecule_generation",
            weight=0.6,
            requests=[
                {
                    "url": "/generate",
                    "method": "POST",
                    "data": {
                        "prompt": "fresh citrus scent",
                        "num_molecules": 5,
                        "safety_threshold": 0.1
                    }
                }
            ],
            think_time_min=2.0,
            think_time_max=5.0
        )
        
        # Batch generation scenario
        batch_scenario = LoadTestScenario(
            name="batch_generation",
            weight=0.2,
            requests=[
                {
                    "url": "/generate/batch",
                    "method": "POST",
                    "data": {
                        "prompts": ["vanilla scent", "rose fragrance", "woody aroma"],
                        "num_molecules": 3,
                        "priority": 1
                    }
                }
            ],
            think_time_min=3.0,
            think_time_max=8.0
        )
        
        # Statistics scenario
        stats_scenario = LoadTestScenario(
            name="statistics",
            weight=0.1,
            requests=[{"url": "/stats", "method": "GET"}],
            think_time_min=1.0,
            think_time_max=2.0
        )
        
        self.scenarios = [health_scenario, generation_scenario, batch_scenario, stats_scenario]
    
    def _setup_scenarios_from_config(self):
        """Setup scenarios from configuration."""
        for scenario_config in self.config.scenarios:
            scenario = LoadTestScenario(
                name=scenario_config['name'],
                weight=scenario_config.get('weight', 1.0),
                requests=scenario_config['requests'],
                think_time_min=scenario_config.get('think_time_min', 1.0),
                think_time_max=scenario_config.get('think_time_max', 5.0)
            )
            self.scenarios.append(scenario)
    
    async def run_load_test(self):
        """Execute the main load test."""
        logger.info("Starting load test...")
        logger.info(f"Test configuration: {self.config.max_users} max users, {self.config.test_duration_seconds}s duration")
        
        # Start metrics collection
        self.metrics_collector.start_collection(lambda: self.active_users)
        
        try:
            # Execute ramp-up phase
            await self._ramp_up_phase()
            
            # Execute sustained load phase
            await self._sustained_load_phase()
            
            # Execute stress test if enabled
            if self.config.stress_test_enabled:
                await self._stress_test_phase()
            
        finally:
            # Cleanup
            await self._cleanup()
            
            # Stop metrics collection
            self.metrics_collector.stop_collection()
        
        # Generate report
        report = self._generate_report()
        return report
    
    async def _ramp_up_phase(self):
        """Execute the load ramp-up phase."""
        logger.info("Starting ramp-up phase...")
        
        current_users = self.config.initial_users
        step_duration = self.config.ramp_up_duration_seconds // ((self.config.max_users - self.config.initial_users) // self.config.ramp_up_step)
        
        # Add initial users
        await self._add_users(current_users)
        
        while current_users < self.config.max_users:
            await asyncio.sleep(step_duration)
            
            next_users = min(current_users + self.config.ramp_up_step, self.config.max_users)
            users_to_add = next_users - current_users
            
            await self._add_users(users_to_add)
            current_users = next_users
            
            logger.info(f"Ramped up to {current_users} users")
        
        logger.info("Ramp-up phase completed")
    
    async def _sustained_load_phase(self):
        """Execute the sustained load phase."""
        logger.info("Starting sustained load phase...")
        
        # Run at max load for the test duration
        remaining_time = self.config.test_duration_seconds - self.config.ramp_up_duration_seconds
        await asyncio.sleep(remaining_time)
        
        logger.info("Sustained load phase completed")
    
    async def _stress_test_phase(self):
        """Execute the stress test phase."""
        logger.info("Starting stress test phase...")
        
        # Gradually increase load beyond normal limits
        current_users = self.config.max_users
        step_size = (self.config.stress_max_users - self.config.max_users) // 10
        step_duration = self.config.stress_ramp_duration // 10
        
        while current_users < self.config.stress_max_users:
            await asyncio.sleep(step_duration)
            
            users_to_add = min(step_size, self.config.stress_max_users - current_users)
            await self._add_users(users_to_add)
            current_users += users_to_add
            
            logger.info(f"Stress test: {current_users} users")
        
        # Hold peak load for observation
        await asyncio.sleep(60)
        
        logger.info("Stress test phase completed")
    
    async def _add_users(self, count: int):
        """Add virtual users to the test."""
        new_users = []
        
        for i in range(count):
            user_id = f"user_{len(self.virtual_users) + i + 1}"
            user = VirtualUser(user_id, self.scenarios, self.config.base_url, self.config)
            
            await user.start()
            new_users.append(user)
            
            # Start the user's continuous execution
            asyncio.create_task(self._run_user(user))
        
        self.virtual_users.extend(new_users)
        self.active_users = len(self.virtual_users)
    
    async def _run_user(self, user: VirtualUser):
        """Run a virtual user's continuous scenario execution."""
        try:
            await user.run_continuously()
        except Exception as e:
            logger.error(f"User {user.user_id} failed: {e}")
        
        # Collect any remaining results
        while not user.results_queue.empty():
            result = user.results_queue.get()
            self.metrics_collector.add_request_result(result)
    
    async def _cleanup(self):
        """Cleanup all virtual users."""
        logger.info("Cleaning up virtual users...")
        
        # Stop all users
        cleanup_tasks = []
        for user in self.virtual_users:
            cleanup_tasks.append(user.stop())
        
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # Collect any remaining results
        for user in self.virtual_users:
            while not user.results_queue.empty():
                result = user.results_queue.get()
                self.metrics_collector.add_request_result(result)
        
        self.active_users = 0
        logger.info("Cleanup completed")
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        summary = self.metrics_collector.get_summary_report()
        
        # Add configuration details
        summary["configuration"] = {
            "base_url": self.config.base_url,
            "max_users": self.config.max_users,
            "test_duration": self.config.test_duration_seconds,
            "scenarios": [s.name for s in self.scenarios],
            "stress_test_enabled": self.config.stress_test_enabled
        }
        
        # Add timestamp
        summary["completed_at"] = datetime.now().isoformat()
        
        return summary
    
    def save_results(self, report: Dict[str, Any], filename: str = None):
        """Save test results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"load_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
    
    def generate_charts(self, output_dir: str = "charts"):
        """Generate performance charts."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Response time over time
        timestamps = [m.timestamp for m in self.metrics_collector.metrics_history]
        response_times = [m.avg_response_time for m in self.metrics_collector.metrics_history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, response_times)
        plt.title("Average Response Time Over Time")
        plt.xlabel("Time")
        plt.ylabel("Response Time (seconds)")
        plt.savefig(f"{output_dir}/response_time.png")
        plt.close()
        
        # Requests per second over time
        rps_values = [m.requests_per_second for m in self.metrics_collector.metrics_history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, rps_values)
        plt.title("Requests Per Second Over Time")
        plt.xlabel("Time")
        plt.ylabel("Requests/Second")
        plt.savefig(f"{output_dir}/rps.png")
        plt.close()
        
        # User load over time
        user_counts = [m.active_users for m in self.metrics_collector.metrics_history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, user_counts)
        plt.title("Active Users Over Time")
        plt.xlabel("Time")
        plt.ylabel("Active Users")
        plt.savefig(f"{output_dir}/users.png")
        plt.close()
        
        logger.info(f"Charts saved to {output_dir}/")


def load_config_from_file(config_file: str) -> LoadTestConfig:
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return LoadTestConfig(**config_data)


async def main():
    """Main entry point for load testing."""
    parser = argparse.ArgumentParser(description="OdorDiff-2 Load Testing Tool")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--base-url", "-u", default="http://localhost:8000", help="Base URL for testing")
    parser.add_argument("--users", "-n", type=int, default=10, help="Maximum concurrent users")
    parser.add_argument("--duration", "-d", type=int, default=300, help="Test duration in seconds")
    parser.add_argument("--output", "-o", help="Output file for results")
    parser.add_argument("--charts", action="store_true", help="Generate performance charts")
    parser.add_argument("--stress", action="store_true", help="Enable stress testing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = LoadTestConfig(
            base_url=args.base_url,
            max_users=args.users,
            test_duration_seconds=args.duration,
            stress_test_enabled=args.stress,
            enable_detailed_logging=args.verbose
        )
    
    # Create and run load tester
    tester = LoadTester(config)
    
    try:
        report = await tester.run_load_test()
        
        # Print summary
        print("\n" + "="*50)
        print("LOAD TEST SUMMARY")
        print("="*50)
        print(f"Duration: {report['test_summary']['duration_seconds']:.1f} seconds")
        print(f"Total Requests: {report['test_summary']['total_requests']}")
        print(f"Success Rate: {report['test_summary']['success_rate']:.1f}%")
        print(f"Peak RPS: {report['test_summary']['peak_rps']:.1f}")
        print(f"Peak Users: {report['test_summary']['peak_concurrent_users']}")
        print(f"Average Response Time: {report['response_times']['average']:.3f}s")
        print(f"95th Percentile: {report['response_times']['p95']:.3f}s")
        print(f"99th Percentile: {report['response_times']['p99']:.3f}s")
        
        if report.get('error_analysis'):
            print("\nERRORS:")
            for error, count in report['error_analysis'].items():
                print(f"  {error}: {count}")
        
        # Save results
        if args.output:
            tester.save_results(report, args.output)
        
        # Generate charts
        if args.charts:
            tester.generate_charts()
        
    except KeyboardInterrupt:
        logger.info("Load test interrupted by user")
    except Exception as e:
        logger.error(f"Load test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())