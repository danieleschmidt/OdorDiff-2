"""
Intelligent Load Balancing for OdorDiff-2 API Endpoints

Features:
- Multiple load balancing algorithms (round-robin, least-connections, weighted)
- Health check monitoring with circuit breaker pattern
- Sticky sessions for stateful operations
- Automatic failover and recovery
- Real-time metrics and monitoring
- Geographic routing support
"""

import os
import time
import asyncio
import random
import hashlib
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import aiohttp
from urllib.parse import urljoin

from ..utils.logging import get_logger
from ..utils.circuit_breaker import CircuitBreaker

logger = get_logger(__name__)


class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    IP_HASH = "ip_hash"
    GEOGRAPHIC = "geographic"


class ServerStatus(Enum):
    """Server health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class BackendServer:
    """Represents a backend server instance."""
    id: str
    host: str
    port: int
    weight: float = 1.0
    max_connections: int = 100
    region: str = "default"
    
    # Health and performance metrics
    status: ServerStatus = ServerStatus.UNKNOWN
    current_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_health_check: float = 0.0
    consecutive_failures: int = 0
    
    # Circuit breaker
    circuit_breaker: Optional[CircuitBreaker] = None
    
    @property
    def url(self) -> str:
        """Get server base URL."""
        return f"http://{self.host}:{self.port}"
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.total_requests
        return (self.successful_requests / total) if total > 0 else 0.0
    
    @property
    def is_available(self) -> bool:
        """Check if server is available for requests."""
        return (
            self.status in [ServerStatus.HEALTHY, ServerStatus.DEGRADED] and
            self.current_connections < self.max_connections and
            (not self.circuit_breaker or not self.circuit_breaker.is_open)
        )
    
    def update_metrics(self, success: bool, response_time: float):
        """Update server metrics after request."""
        self.total_requests += 1
        
        if success:
            self.successful_requests += 1
            self.consecutive_failures = 0
        else:
            self.failed_requests += 1
            self.consecutive_failures += 1
        
        # Update average response time (exponential moving average)
        alpha = 0.1
        self.avg_response_time = (
            (1 - alpha) * self.avg_response_time + 
            alpha * response_time
        )


@dataclass 
class HealthCheckConfig:
    """Health check configuration."""
    endpoint: str = "/health"
    interval: int = 30  # seconds
    timeout: int = 5    # seconds
    healthy_threshold: int = 2
    unhealthy_threshold: int = 3
    expected_status: int = 200
    expected_response_time: float = 2.0  # seconds


@dataclass
class StickySession:
    """Sticky session configuration."""
    enabled: bool = False
    cookie_name: str = "lb_session"
    header_name: str = "X-Session-ID"
    ttl: int = 3600  # seconds


class LoadBalancer:
    """Intelligent load balancer with health monitoring."""
    
    def __init__(
        self,
        algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ROUND_ROBIN,
        health_check_config: Optional[HealthCheckConfig] = None,
        sticky_sessions: Optional[StickySession] = None
    ):
        self.algorithm = algorithm
        self.health_check_config = health_check_config or HealthCheckConfig()
        self.sticky_sessions = sticky_sessions or StickySession()
        
        # Server management
        self.servers: Dict[str, BackendServer] = {}
        self.server_list: List[BackendServer] = []
        
        # Load balancing state
        self._round_robin_index = 0
        self._session_map: Dict[str, str] = {}  # session_id -> server_id
        
        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._session = None
        
        # Metrics
        self._total_requests = 0
        self._failed_requests = 0
        self._load_distribution = {}
        
    async def initialize(self):
        """Initialize load balancer."""
        # Create HTTP session for health checks
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.health_check_config.timeout)
        )
        
        # Start health monitoring
        self._health_check_task = asyncio.create_task(self._health_monitor())
        
        logger.info(f"Load balancer initialized with {self.algorithm.value} algorithm")
    
    async def close(self):
        """Close load balancer and cleanup resources."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._session:
            await self._session.close()
        
        logger.info("Load balancer closed")
    
    def add_server(
        self,
        server_id: str,
        host: str,
        port: int,
        weight: float = 1.0,
        max_connections: int = 100,
        region: str = "default"
    ):
        """Add backend server to pool."""
        server = BackendServer(
            id=server_id,
            host=host,
            port=port,
            weight=weight,
            max_connections=max_connections,
            region=region
        )
        
        # Initialize circuit breaker
        server.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=(aiohttp.ClientError, asyncio.TimeoutError)
        )
        
        self.servers[server_id] = server
        self.server_list = list(self.servers.values())
        self._load_distribution[server_id] = 0
        
        logger.info(f"Added server {server_id} ({host}:{port}) to pool")
    
    def remove_server(self, server_id: str):
        """Remove server from pool."""
        if server_id in self.servers:
            del self.servers[server_id]
            self.server_list = list(self.servers.values())
            self._load_distribution.pop(server_id, None)
            
            # Remove from session map
            sessions_to_remove = [
                session_id for session_id, sid in self._session_map.items()
                if sid == server_id
            ]
            for session_id in sessions_to_remove:
                del self._session_map[session_id]
            
            logger.info(f"Removed server {server_id} from pool")
    
    async def get_server(
        self,
        client_ip: Optional[str] = None,
        session_id: Optional[str] = None,
        region: Optional[str] = None
    ) -> Optional[BackendServer]:
        """Select server based on load balancing algorithm."""
        available_servers = [s for s in self.server_list if s.is_available]
        
        if not available_servers:
            logger.error("No available servers in pool")
            return None
        
        # Handle sticky sessions
        if self.sticky_sessions.enabled and session_id:
            if session_id in self._session_map:
                server_id = self._session_map[session_id]
                if server_id in self.servers and self.servers[server_id].is_available:
                    return self.servers[server_id]
                else:
                    # Remove stale session
                    del self._session_map[session_id]
        
        # Filter by region if specified
        if region:
            regional_servers = [s for s in available_servers if s.region == region]
            if regional_servers:
                available_servers = regional_servers
        
        # Apply load balancing algorithm
        server = None
        
        if self.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            server = self._round_robin_select(available_servers)
            
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            server = min(available_servers, key=lambda s: s.current_connections)
            
        elif self.algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            server = self._weighted_round_robin_select(available_servers)
            
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
            server = min(available_servers, key=lambda s: s.avg_response_time)
            
        elif self.algorithm == LoadBalancingAlgorithm.IP_HASH:
            if client_ip:
                server = self._ip_hash_select(available_servers, client_ip)
            else:
                server = self._round_robin_select(available_servers)
                
        elif self.algorithm == LoadBalancingAlgorithm.GEOGRAPHIC:
            server = self._geographic_select(available_servers, region)
        
        # Create sticky session if enabled
        if server and self.sticky_sessions.enabled and session_id:
            self._session_map[session_id] = server.id
        
        return server
    
    def _round_robin_select(self, servers: List[BackendServer]) -> BackendServer:
        """Round-robin server selection."""
        if not servers:
            return None
        
        server = servers[self._round_robin_index % len(servers)]
        self._round_robin_index += 1
        return server
    
    def _weighted_round_robin_select(self, servers: List[BackendServer]) -> BackendServer:
        """Weighted round-robin server selection."""
        if not servers:
            return None
        
        # Create weighted list
        weighted_servers = []
        for server in servers:
            weight = max(1, int(server.weight * 10))  # Scale weights
            weighted_servers.extend([server] * weight)
        
        if not weighted_servers:
            return servers[0]
        
        server = weighted_servers[self._round_robin_index % len(weighted_servers)]
        self._round_robin_index += 1
        return server
    
    def _ip_hash_select(self, servers: List[BackendServer], client_ip: str) -> BackendServer:
        """IP hash-based server selection for session affinity."""
        if not servers:
            return None
        
        hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        index = hash_value % len(servers)
        return servers[index]
    
    def _geographic_select(self, servers: List[BackendServer], preferred_region: Optional[str]) -> BackendServer:
        """Geographic server selection."""
        if not servers:
            return None
        
        if preferred_region:
            regional_servers = [s for s in servers if s.region == preferred_region]
            if regional_servers:
                return self._round_robin_select(regional_servers)
        
        # Fallback to round-robin
        return self._round_robin_select(servers)
    
    @asynccontextmanager
    async def request_context(
        self,
        client_ip: Optional[str] = None,
        session_id: Optional[str] = None,
        region: Optional[str] = None
    ):
        """Context manager for handling requests with load balancing."""
        server = await self.get_server(client_ip, session_id, region)
        
        if not server:
            raise RuntimeError("No available servers")
        
        # Update connection count
        server.current_connections += 1
        self._total_requests += 1
        self._load_distribution[server.id] += 1
        
        start_time = time.time()
        success = False
        
        try:
            yield server
            success = True
            
        except Exception as e:
            logger.error(f"Request failed on server {server.id}: {e}")
            self._failed_requests += 1
            
            # Update circuit breaker
            if server.circuit_breaker:
                server.circuit_breaker.record_failure()
            
            raise
            
        finally:
            # Update metrics
            response_time = time.time() - start_time
            server.update_metrics(success, response_time)
            server.current_connections = max(0, server.current_connections - 1)
            
            if success and server.circuit_breaker:
                server.circuit_breaker.record_success()
    
    async def _health_monitor(self):
        """Background task for health monitoring."""
        while True:
            try:
                await asyncio.sleep(self.health_check_config.interval)
                
                # Check all servers concurrently
                tasks = [
                    self._check_server_health(server)
                    for server in self.servers.values()
                ]
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    async def _check_server_health(self, server: BackendServer):
        """Check health of individual server."""
        url = urljoin(server.url, self.health_check_config.endpoint)
        
        try:
            start_time = time.time()
            
            async with self._session.get(url) as response:
                response_time = time.time() - start_time
                
                # Check status code
                if response.status == self.health_check_config.expected_status:
                    # Check response time
                    if response_time <= self.health_check_config.expected_response_time:
                        self._update_server_status(server, ServerStatus.HEALTHY)
                    else:
                        self._update_server_status(server, ServerStatus.DEGRADED)
                else:
                    self._update_server_status(server, ServerStatus.UNHEALTHY)
                    
        except Exception as e:
            logger.warning(f"Health check failed for server {server.id}: {e}")
            self._update_server_status(server, ServerStatus.UNHEALTHY)
        
        server.last_health_check = time.time()
    
    def _update_server_status(self, server: BackendServer, status: ServerStatus):
        """Update server health status."""
        if server.status != status:
            logger.info(f"Server {server.id} status changed: {server.status.value} -> {status.value}")
            server.status = status
            
            # Reset circuit breaker if server becomes healthy
            if status == ServerStatus.HEALTHY and server.circuit_breaker:
                server.circuit_breaker.reset()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        healthy_servers = sum(1 for s in self.server_list if s.status == ServerStatus.HEALTHY)
        total_servers = len(self.server_list)
        
        server_stats = {}
        for server in self.server_list:
            server_stats[server.id] = {
                'status': server.status.value,
                'current_connections': server.current_connections,
                'total_requests': server.total_requests,
                'success_rate': server.success_rate,
                'avg_response_time': server.avg_response_time,
                'weight': server.weight,
                'region': server.region
            }
        
        return {
            'algorithm': self.algorithm.value,
            'total_servers': total_servers,
            'healthy_servers': healthy_servers,
            'total_requests': self._total_requests,
            'failed_requests': self._failed_requests,
            'success_rate': (
                (self._total_requests - self._failed_requests) / self._total_requests
                if self._total_requests > 0 else 0.0
            ),
            'load_distribution': self._load_distribution,
            'servers': server_stats,
            'sticky_sessions_count': len(self._session_map)
        }


class LoadBalancerMiddleware:
    """FastAPI middleware for load balancing."""
    
    def __init__(self, load_balancer: LoadBalancer):
        self.load_balancer = load_balancer
    
    async def __call__(self, request, call_next):
        """Process request through load balancer."""
        client_ip = request.client.host if request.client else None
        session_id = request.cookies.get(
            self.load_balancer.sticky_sessions.cookie_name
        ) or request.headers.get(
            self.load_balancer.sticky_sessions.header_name
        )
        region = request.headers.get('X-Client-Region')
        
        async with self.load_balancer.request_context(
            client_ip=client_ip,
            session_id=session_id,
            region=region
        ) as server:
            # Add server info to request state
            request.state.backend_server = server
            
            # Process request
            response = await call_next(request)
            
            # Add session cookie if sticky sessions enabled
            if (self.load_balancer.sticky_sessions.enabled and 
                not session_id):
                import uuid
                new_session_id = str(uuid.uuid4())
                response.set_cookie(
                    self.load_balancer.sticky_sessions.cookie_name,
                    new_session_id,
                    max_age=self.load_balancer.sticky_sessions.ttl
                )
            
            # Add server info to response headers
            response.headers['X-Served-By'] = server.id
            
            return response


def create_load_balancer_from_config() -> LoadBalancer:
    """Create load balancer from environment configuration."""
    # Algorithm
    algorithm_str = os.getenv('LB_ALGORITHM', 'round_robin').upper()
    algorithm = LoadBalancingAlgorithm[algorithm_str]
    
    # Health check config
    health_config = HealthCheckConfig(
        endpoint=os.getenv('LB_HEALTH_ENDPOINT', '/health'),
        interval=int(os.getenv('LB_HEALTH_INTERVAL', '30')),
        timeout=int(os.getenv('LB_HEALTH_TIMEOUT', '5')),
        healthy_threshold=int(os.getenv('LB_HEALTHY_THRESHOLD', '2')),
        unhealthy_threshold=int(os.getenv('LB_UNHEALTHY_THRESHOLD', '3'))
    )
    
    # Sticky sessions
    sticky_sessions = StickySession(
        enabled=os.getenv('LB_STICKY_SESSIONS', 'false').lower() == 'true',
        cookie_name=os.getenv('LB_SESSION_COOKIE', 'lb_session'),
        header_name=os.getenv('LB_SESSION_HEADER', 'X-Session-ID'),
        ttl=int(os.getenv('LB_SESSION_TTL', '3600'))
    )
    
    # Create load balancer
    lb = LoadBalancer(
        algorithm=algorithm,
        health_check_config=health_config,
        sticky_sessions=sticky_sessions
    )
    
    # Add servers from environment
    server_config = os.getenv('LB_SERVERS', '')
    if server_config:
        for server_def in server_config.split(','):
            parts = server_def.strip().split(':')
            if len(parts) >= 3:
                server_id, host, port = parts[:3]
                weight = float(parts[3]) if len(parts) > 3 else 1.0
                region = parts[4] if len(parts) > 4 else "default"
                
                lb.add_server(
                    server_id=server_id,
                    host=host,
                    port=int(port),
                    weight=weight,
                    region=region
                )
    
    return lb