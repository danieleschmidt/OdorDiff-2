"""
Load balancing and auto-scaling components for OdorDiff-2.
Provides intelligent request distribution and dynamic scaling capabilities.
"""

import asyncio
import time
import random
import threading
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import psutil
import socket
from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path

from ..utils.logging import get_logger
from .optimization import PerformanceMetrics, ResourcePool

logger = get_logger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    CONSISTENT_HASH = "consistent_hash"
    ADAPTIVE = "adaptive"


class ServerHealth(Enum):
    """Server health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    OFFLINE = "offline"


@dataclass
class ServerInstance:
    """Represents a server instance in the load balancer."""
    id: str
    host: str
    port: int
    weight: float = 1.0
    health: ServerHealth = ServerHealth.HEALTHY
    
    # Performance metrics
    active_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_response_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    
    # Health check
    last_health_check: float = 0.0
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    
    # Load balancing
    last_selected: float = 0.0
    selection_count: int = 0
    
    def __post_init__(self):
        self.response_times = deque(maxlen=100)
        self.request_timestamps = deque(maxlen=1000)
    
    @property
    def url(self) -> str:
        """Get server URL."""
        return f"http://{self.host}:{self.port}"
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    @property
    def requests_per_second(self) -> float:
        """Calculate current RPS."""
        now = time.time()
        recent_requests = [
            ts for ts in self.request_timestamps 
            if now - ts < 60  # Last minute
        ]
        return len(recent_requests) / 60
    
    def update_response_time(self, response_time: float):
        """Update response time metrics."""
        self.last_response_time = response_time
        self.response_times.append(response_time)
        
        if self.response_times:
            self.average_response_time = sum(self.response_times) / len(self.response_times)
    
    def record_request(self, success: bool = True, response_time: float = None):
        """Record a request."""
        now = time.time()
        self.request_timestamps.append(now)
        self.total_requests += 1
        self.last_selected = now
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        if response_time is not None:
            self.update_response_time(response_time)
    
    def calculate_load_score(self) -> float:
        """Calculate current load score (lower is better)."""
        base_score = self.active_connections / max(self.weight, 0.1)
        
        # Adjust for response time
        if self.average_response_time > 0:
            base_score *= (1 + self.average_response_time / 1000)  # Convert ms to factor
        
        # Adjust for success rate
        if self.success_rate < 1.0:
            base_score *= (2 - self.success_rate)  # Penalize failed requests
        
        # Adjust for system resources
        resource_factor = (self.cpu_usage + self.memory_usage) / 200
        base_score *= (1 + resource_factor)
        
        return base_score


@dataclass
class LoadBalancerConfig:
    """Configuration for load balancer."""
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE
    health_check_interval: int = 30
    health_check_timeout: float = 5.0
    max_consecutive_failures: int = 3
    circuit_breaker_threshold: float = 0.5
    recovery_time: int = 60
    
    # Consistent hashing
    virtual_nodes: int = 150
    
    # Session affinity
    session_affinity: bool = False
    session_timeout: int = 3600
    
    # Auto-scaling triggers
    cpu_scale_up_threshold: float = 80.0
    cpu_scale_down_threshold: float = 20.0
    memory_scale_up_threshold: float = 80.0
    memory_scale_down_threshold: float = 20.0
    response_time_threshold: float = 1000.0  # ms
    
    # Rate limiting
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 1000


class LoadBalancer:
    """Advanced load balancer with multiple strategies and health checking."""
    
    def __init__(self, config: LoadBalancerConfig = None):
        self.config = config or LoadBalancerConfig()
        self.servers: Dict[str, ServerInstance] = {}
        self.healthy_servers: List[str] = []
        self.current_index = 0
        self.session_map: Dict[str, str] = {}  # session_id -> server_id
        self.request_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Consistent hashing ring
        self.hash_ring: Dict[int, str] = {}
        self.ring_keys: List[int] = []
        
        # Statistics
        self.total_requests = 0
        self.failed_requests = 0
        self.load_balancer_lock = asyncio.Lock()
        
        # Start background tasks
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info(f"LoadBalancer initialized with strategy: {self.config.strategy.value}")
    
    async def add_server(self, server: ServerInstance):
        """Add a server to the load balancer."""
        async with self.load_balancer_lock:
            self.servers[server.id] = server
            
            if server.health == ServerHealth.HEALTHY:
                if server.id not in self.healthy_servers:
                    self.healthy_servers.append(server.id)
            
            # Update hash ring for consistent hashing
            if self.config.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                self._add_to_hash_ring(server.id)
        
        logger.info(f"Added server: {server.id} ({server.url})")
    
    async def remove_server(self, server_id: str):
        """Remove a server from the load balancer."""
        async with self.load_balancer_lock:
            if server_id in self.servers:
                # Mark as offline first
                self.servers[server_id].health = ServerHealth.OFFLINE
                
                if server_id in self.healthy_servers:
                    self.healthy_servers.remove(server_id)
                
                # Remove from hash ring
                if self.config.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                    self._remove_from_hash_ring(server_id)
                
                # Clean up sessions
                sessions_to_remove = [
                    session_id for session_id, srv_id in self.session_map.items()
                    if srv_id == server_id
                ]
                for session_id in sessions_to_remove:
                    del self.session_map[session_id]
                
                # Remove server
                del self.servers[server_id]
        
        logger.info(f"Removed server: {server_id}")
    
    async def select_server(self, session_id: str = None, request_data: Any = None) -> Optional[ServerInstance]:
        """Select the best server for a request."""
        if not self.healthy_servers:
            logger.warning("No healthy servers available")
            return None
        
        self.total_requests += 1
        
        # Check session affinity
        if self.config.session_affinity and session_id and session_id in self.session_map:
            server_id = self.session_map[session_id]
            if server_id in self.healthy_servers:
                server = self.servers[server_id]
                server.selection_count += 1
                return server
            else:
                # Session server is unhealthy, remove session
                del self.session_map[session_id]
        
        # Apply rate limiting
        if self.config.enable_rate_limiting:
            client_id = self._get_client_id(request_data)
            if not await self._check_rate_limit(client_id):
                return None
        
        # Select server based on strategy
        server = await self._select_by_strategy(request_data)
        
        if server:
            server.selection_count += 1
            
            # Create session if affinity enabled
            if self.config.session_affinity and session_id:
                self.session_map[session_id] = server.id
        
        return server
    
    async def _select_by_strategy(self, request_data: Any = None) -> Optional[ServerInstance]:
        """Select server based on configured strategy."""
        strategy = self.config.strategy
        
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return await self._round_robin_selection()
        
        elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return await self._least_connections_selection()
        
        elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return await self._weighted_round_robin_selection()
        
        elif strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return await self._least_response_time_selection()
        
        elif strategy == LoadBalancingStrategy.RESOURCE_BASED:
            return await self._resource_based_selection()
        
        elif strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            return await self._consistent_hash_selection(request_data)
        
        elif strategy == LoadBalancingStrategy.ADAPTIVE:
            return await self._adaptive_selection()
        
        else:
            # Default to round robin
            return await self._round_robin_selection()
    
    async def _round_robin_selection(self) -> Optional[ServerInstance]:
        """Round-robin server selection."""
        if not self.healthy_servers:
            return None
        
        server_id = self.healthy_servers[self.current_index % len(self.healthy_servers)]
        self.current_index += 1
        return self.servers[server_id]
    
    async def _least_connections_selection(self) -> Optional[ServerInstance]:
        """Select server with least active connections."""
        if not self.healthy_servers:
            return None
        
        best_server = None
        min_connections = float('inf')
        
        for server_id in self.healthy_servers:
            server = self.servers[server_id]
            if server.active_connections < min_connections:
                min_connections = server.active_connections
                best_server = server
        
        return best_server
    
    async def _weighted_round_robin_selection(self) -> Optional[ServerInstance]:
        """Weighted round-robin selection."""
        if not self.healthy_servers:
            return None
        
        # Calculate total weight
        total_weight = sum(self.servers[sid].weight for sid in self.healthy_servers)
        
        if total_weight <= 0:
            return await self._round_robin_selection()
        
        # Generate random number and find server
        target = random.uniform(0, total_weight)
        current_weight = 0
        
        for server_id in self.healthy_servers:
            server = self.servers[server_id]
            current_weight += server.weight
            if current_weight >= target:
                return server
        
        # Fallback to last server
        return self.servers[self.healthy_servers[-1]]
    
    async def _least_response_time_selection(self) -> Optional[ServerInstance]:
        """Select server with lowest response time."""
        if not self.healthy_servers:
            return None
        
        best_server = None
        min_response_time = float('inf')
        
        for server_id in self.healthy_servers:
            server = self.servers[server_id]
            response_time = server.average_response_time or 0.0
            
            if response_time < min_response_time:
                min_response_time = response_time
                best_server = server
        
        return best_server or self.servers[self.healthy_servers[0]]
    
    async def _resource_based_selection(self) -> Optional[ServerInstance]:
        """Select server based on resource utilization."""
        if not self.healthy_servers:
            return None
        
        best_server = None
        min_load = float('inf')
        
        for server_id in self.healthy_servers:
            server = self.servers[server_id]
            load = server.calculate_load_score()
            
            if load < min_load:
                min_load = load
                best_server = server
        
        return best_server or self.servers[self.healthy_servers[0]]
    
    async def _consistent_hash_selection(self, request_data: Any = None) -> Optional[ServerInstance]:
        """Consistent hash-based selection."""
        if not self.healthy_servers or not self.ring_keys:
            return await self._round_robin_selection()
        
        # Generate hash key from request data
        if request_data:
            hash_key = hash(str(request_data)) % (2**32)
        else:
            hash_key = random.randint(0, 2**32 - 1)
        
        # Find appropriate server in hash ring
        for ring_hash in self.ring_keys:
            if hash_key <= ring_hash:
                server_id = self.hash_ring[ring_hash]
                if server_id in self.healthy_servers:
                    return self.servers[server_id]
        
        # Wrap around to first server
        if self.ring_keys:
            server_id = self.hash_ring[self.ring_keys[0]]
            if server_id in self.healthy_servers:
                return self.servers[server_id]
        
        # Fallback
        return await self._round_robin_selection()
    
    async def _adaptive_selection(self) -> Optional[ServerInstance]:
        """Adaptive selection based on current conditions."""
        if not self.healthy_servers:
            return None
        
        # Score each server based on multiple factors
        best_server = None
        best_score = float('inf')
        
        for server_id in self.healthy_servers:
            server = self.servers[server_id]
            
            # Calculate composite score
            connection_score = server.active_connections / max(server.weight, 0.1)
            response_time_score = server.average_response_time / 1000.0
            success_rate_score = (1.0 - server.success_rate) * 10
            resource_score = (server.cpu_usage + server.memory_usage) / 200
            
            total_score = (
                connection_score * 0.3 +
                response_time_score * 0.3 +
                success_rate_score * 0.2 +
                resource_score * 0.2
            )
            
            if total_score < best_score:
                best_score = total_score
                best_server = server
        
        return best_server
    
    def _add_to_hash_ring(self, server_id: str):
        """Add server to consistent hash ring."""
        for i in range(self.config.virtual_nodes):
            virtual_key = f"{server_id}:{i}"
            hash_value = hash(virtual_key) % (2**32)
            self.hash_ring[hash_value] = server_id
        
        self.ring_keys = sorted(self.hash_ring.keys())
    
    def _remove_from_hash_ring(self, server_id: str):
        """Remove server from consistent hash ring."""
        keys_to_remove = [
            k for k, v in self.hash_ring.items() 
            if v == server_id
        ]
        
        for key in keys_to_remove:
            del self.hash_ring[key]
        
        self.ring_keys = sorted(self.hash_ring.keys())
    
    def _get_client_id(self, request_data: Any) -> str:
        """Extract client ID from request data."""
        if hasattr(request_data, 'remote_addr'):
            return request_data.remote_addr
        elif hasattr(request_data, 'client_ip'):
            return request_data.client_ip
        else:
            return "unknown"
    
    async def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limits."""
        now = time.time()
        client_requests = self.request_counts[client_id]
        
        # Clean old requests
        while client_requests and now - client_requests[0] > 60:
            client_requests.popleft()
        
        # Check limit
        if len(client_requests) >= self.config.max_requests_per_minute:
            return False
        
        client_requests.append(now)
        return True
    
    async def _health_check_loop(self):
        """Background health checking loop."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_health_checks()
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
    
    async def _perform_health_checks(self):
        """Perform health checks on all servers."""
        if not self.servers:
            return
        
        tasks = []
        for server_id, server in self.servers.items():
            if server.health != ServerHealth.OFFLINE:
                task = asyncio.create_task(self._check_server_health(server))
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_server_health(self, server: ServerInstance):
        """Check health of a single server."""
        try:
            start_time = time.time()
            
            # Perform health check (implement actual HTTP check)
            is_healthy = await self._perform_http_health_check(server)
            
            response_time = (time.time() - start_time) * 1000
            server.last_health_check = time.time()
            
            if is_healthy:
                server.consecutive_failures = 0
                server.last_error = None
                
                # Update server health
                if server.health == ServerHealth.UNHEALTHY:
                    server.health = ServerHealth.HEALTHY
                    async with self.load_balancer_lock:
                        if server.id not in self.healthy_servers:
                            self.healthy_servers.append(server.id)
                    logger.info(f"Server {server.id} recovered")
            
            else:
                server.consecutive_failures += 1
                
                if server.consecutive_failures >= self.config.max_consecutive_failures:
                    if server.health != ServerHealth.UNHEALTHY:
                        server.health = ServerHealth.UNHEALTHY
                        async with self.load_balancer_lock:
                            if server.id in self.healthy_servers:
                                self.healthy_servers.remove(server.id)
                        logger.warning(f"Server {server.id} marked unhealthy")
        
        except Exception as e:
            server.consecutive_failures += 1
            server.last_error = str(e)
            logger.error(f"Health check failed for {server.id}: {e}")
    
    async def _perform_http_health_check(self, server: ServerInstance) -> bool:
        """Perform HTTP health check (stub implementation)."""
        try:
            # In a real implementation, you would make an HTTP request
            # For now, simulate based on server statistics
            if server.success_rate < self.config.circuit_breaker_threshold:
                return False
            
            # Simulate occasional failures
            return random.random() > 0.01  # 99% success rate
            
        except Exception:
            return False
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_old_sessions()
                await self._cleanup_request_counts()
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _cleanup_old_sessions(self):
        """Clean up expired sessions."""
        if not self.config.session_affinity:
            return
        
        now = time.time()
        expired_sessions = []
        
        for session_id, server_id in self.session_map.items():
            # Simple timeout-based cleanup (in practice, you'd track last access)
            if random.random() < 0.01:  # Randomly expire 1% of sessions
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.session_map[session_id]
        
        if expired_sessions:
            logger.debug(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    async def _cleanup_request_counts(self):
        """Clean up old request counts."""
        now = time.time()
        clients_to_remove = []
        
        for client_id, requests in self.request_counts.items():
            # Remove old requests
            while requests and now - requests[0] > 3600:  # 1 hour
                requests.popleft()
            
            # Remove clients with no recent requests
            if not requests:
                clients_to_remove.append(client_id)
        
        for client_id in clients_to_remove:
            del self.request_counts[client_id]
        
        if clients_to_remove:
            logger.debug(f"Cleaned up {len(clients_to_remove)} inactive clients")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        healthy_count = len(self.healthy_servers)
        total_count = len(self.servers)
        
        stats = {
            'strategy': self.config.strategy.value,
            'total_servers': total_count,
            'healthy_servers': healthy_count,
            'unhealthy_servers': total_count - healthy_count,
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'success_rate': (
                (self.total_requests - self.failed_requests) / max(self.total_requests, 1)
            ) * 100,
            'active_sessions': len(self.session_map),
            'tracked_clients': len(self.request_counts)
        }
        
        # Server details
        stats['servers'] = {}
        for server_id, server in self.servers.items():
            stats['servers'][server_id] = {
                'health': server.health.value,
                'active_connections': server.active_connections,
                'total_requests': server.total_requests,
                'success_rate': server.success_rate * 100,
                'average_response_time': server.average_response_time,
                'selection_count': server.selection_count,
                'requests_per_second': server.requests_per_second
            }
        
        return stats


class ServiceRegistry:
    """Service registry for dynamic service discovery."""
    
    def __init__(self):
        self.services: Dict[str, Dict[str, ServerInstance]] = defaultdict(dict)
        self.service_metadata: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.watchers: Dict[str, List[Callable]] = defaultdict(list)
        self.registry_lock = asyncio.Lock()
        
        logger.info("ServiceRegistry initialized")
    
    async def register_service(self, 
                              service_name: str,
                              instance: ServerInstance,
                              metadata: Dict[str, Any] = None):
        """Register a service instance."""
        async with self.registry_lock:
            self.services[service_name][instance.id] = instance
            
            if metadata:
                self.service_metadata[service_name][instance.id] = metadata
            
            # Notify watchers
            await self._notify_watchers(service_name, 'register', instance)
        
        logger.info(f"Registered service: {service_name}/{instance.id}")
    
    async def deregister_service(self, service_name: str, instance_id: str):
        """Deregister a service instance."""
        async with self.registry_lock:
            if service_name in self.services and instance_id in self.services[service_name]:
                instance = self.services[service_name][instance_id]
                del self.services[service_name][instance_id]
                
                if instance_id in self.service_metadata[service_name]:
                    del self.service_metadata[service_name][instance_id]
                
                # Notify watchers
                await self._notify_watchers(service_name, 'deregister', instance)
        
        logger.info(f"Deregistered service: {service_name}/{instance_id}")
    
    async def discover_services(self, service_name: str) -> List[ServerInstance]:
        """Discover all instances of a service."""
        async with self.registry_lock:
            if service_name in self.services:
                return list(self.services[service_name].values())
            return []
    
    async def watch_service(self, service_name: str, callback: Callable):
        """Watch for changes to a service."""
        async with self.registry_lock:
            self.watchers[service_name].append(callback)
    
    async def _notify_watchers(self, service_name: str, event_type: str, instance: ServerInstance):
        """Notify watchers of service changes."""
        if service_name in self.watchers:
            for callback in self.watchers[service_name]:
                try:
                    await callback(event_type, instance)
                except Exception as e:
                    logger.error(f"Watcher notification failed: {e}")
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service registry statistics."""
        stats = {
            'total_services': len(self.services),
            'total_instances': sum(len(instances) for instances in self.services.values()),
            'services': {}
        }
        
        for service_name, instances in self.services.items():
            healthy_count = sum(
                1 for instance in instances.values()
                if instance.health == ServerHealth.HEALTHY
            )
            
            stats['services'][service_name] = {
                'total_instances': len(instances),
                'healthy_instances': healthy_count,
                'instance_ids': list(instances.keys())
            }
        
        return stats


# Global instances
_load_balancer: Optional[LoadBalancer] = None
_service_registry: Optional[ServiceRegistry] = None

def get_load_balancer(config: LoadBalancerConfig = None) -> LoadBalancer:
    """Get global load balancer instance."""
    global _load_balancer
    if _load_balancer is None:
        _load_balancer = LoadBalancer(config)
    return _load_balancer

def get_service_registry() -> ServiceRegistry:
    """Get global service registry instance."""
    global _service_registry
    if _service_registry is None:
        _service_registry = ServiceRegistry()
    return _service_registry