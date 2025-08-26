"""
Redis Cluster Configuration and Management
Production-ready Redis cluster setup with failover and monitoring
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import hashlib

import redis
import redis.asyncio as aioredis
from redis.asyncio.cluster import RedisCluster
from redis.exceptions import ClusterError, ConnectionError, TimeoutError
from redis.asyncio.sentinel import Sentinel

from ..core.logger import get_logger
from ..core.exceptions import ClaudeTUIException

logger = get_logger(__name__)


@dataclass
class RedisNodeConfig:
    """Redis node configuration."""
    host: str
    port: int
    password: Optional[str] = None
    ssl: bool = False
    ssl_cert_reqs: Optional[str] = None
    ssl_ca_certs: Optional[str] = None


@dataclass
class RedisClusterMetrics:
    """Redis cluster performance metrics."""
    total_nodes: int
    master_nodes: int
    slave_nodes: int
    total_keys: int
    used_memory: int
    max_memory: int
    memory_usage_percent: float
    ops_per_second: float
    hit_rate: float
    miss_rate: float
    connected_clients: int
    blocked_clients: int
    keyspace_hits: int
    keyspace_misses: int
    expired_keys: int
    evicted_keys: int
    last_updated: datetime


class RedisClusterManager:
    """
    Advanced Redis Cluster Manager with high availability features.
    
    Features:
    - Automatic cluster discovery and monitoring
    - Failover and recovery mechanisms
    - Performance monitoring and metrics
    - Connection pooling and optimization
    - Data sharding and replication management
    """
    
    def __init__(
        self,
        cluster_nodes: List[RedisNodeConfig],
        password: Optional[str] = None,
        max_connections_per_node: int = 50,
        retry_on_timeout: bool = True,
        health_check_interval: int = 30,
        sentinel_enabled: bool = False,
        sentinel_service_name: str = "claude-tui",
        **kwargs
    ):
        """
        Initialize Redis cluster manager.
        
        Args:
            cluster_nodes: List of Redis node configurations
            password: Cluster password
            max_connections_per_node: Maximum connections per node
            retry_on_timeout: Retry operations on timeout
            health_check_interval: Health check interval in seconds
            sentinel_enabled: Enable Redis Sentinel for HA
            sentinel_service_name: Sentinel service name
        """
        self.cluster_nodes = cluster_nodes
        self.password = password
        self.max_connections_per_node = max_connections_per_node
        self.retry_on_timeout = retry_on_timeout
        self.health_check_interval = health_check_interval
        self.sentinel_enabled = sentinel_enabled
        self.sentinel_service_name = sentinel_service_name
        self.kwargs = kwargs
        
        # Cluster connections
        self.cluster: Optional[RedisCluster] = None
        self.sentinel: Optional[Sentinel] = None
        self.fallback_redis: Optional[aioredis.Redis] = None
        
        # Monitoring
        self.metrics: Optional[RedisClusterMetrics] = None
        self.node_health: Dict[str, bool] = {}
        self.last_health_check: Optional[datetime] = None
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        self._monitoring_enabled = True
        
        logger.info(f"Redis cluster manager initialized with {len(cluster_nodes)} nodes")
    
    async def connect(self):
        """Establish cluster connections."""
        try:
            if self.sentinel_enabled:
                await self._connect_with_sentinel()
            else:
                await self._connect_cluster()
            
            # Set up fallback connection
            await self._setup_fallback()
            
            # Start monitoring
            await self.start_monitoring()
            
            logger.info("Redis cluster connected successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis cluster: {e}")
            # Try fallback connection
            await self._setup_fallback()
            if not self.fallback_redis:
                raise ClaudeTUIException(f"Redis cluster connection failed: {e}")
    
    async def _connect_cluster(self):
        """Connect to Redis cluster directly."""
        startup_nodes = [
            {"host": node.host, "port": node.port}
            for node in self.cluster_nodes
        ]
        
        self.cluster = RedisCluster(
            startup_nodes=startup_nodes,
            password=self.password,
            decode_responses=True,
            skip_full_coverage_check=True,
            max_connections_per_node=self.max_connections_per_node,
            retry_on_timeout=self.retry_on_timeout,
            socket_timeout=5,
            socket_connect_timeout=5,
            **self.kwargs
        )
        
        # Test connection
        await self.cluster.ping()
        logger.info("Redis cluster connection established")
    
    async def _connect_with_sentinel(self):
        """Connect using Redis Sentinel for high availability."""
        sentinel_nodes = [
            (node.host, node.port) for node in self.cluster_nodes
        ]
        
        self.sentinel = Sentinel(
            sentinel_nodes,
            password=self.password,
            socket_timeout=5
        )
        
        # Get master connection
        master = self.sentinel.master_for(
            self.sentinel_service_name,
            decode_responses=True,
            password=self.password
        )
        
        # Test connection
        await master.ping()
        self.cluster = master
        
        logger.info("Redis Sentinel connection established")
    
    async def _setup_fallback(self):
        """Set up fallback Redis connection."""
        if self.cluster_nodes:
            node = self.cluster_nodes[0]
            try:
                self.fallback_redis = aioredis.Redis(
                    host=node.host,
                    port=node.port,
                    password=self.password,
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5
                )
                
                await self.fallback_redis.ping()
                logger.info("Fallback Redis connection established")
                
            except Exception as e:
                logger.warning(f"Failed to establish fallback connection: {e}")
                self.fallback_redis = None
    
    async def start_monitoring(self):
        """Start background monitoring tasks."""
        if not self._monitoring_enabled:
            return
        
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._metrics_task = asyncio.create_task(self._metrics_collection_loop())
        
        logger.info("Redis cluster monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring tasks."""
        self._monitoring_enabled = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
        
        if self._metrics_task:
            self._metrics_task.cancel()
        
        logger.info("Redis cluster monitoring stopped")
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while self._monitoring_enabled:
            try:
                await self.check_cluster_health()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _metrics_collection_loop(self):
        """Background metrics collection loop."""
        while self._monitoring_enabled:
            try:
                await self.collect_metrics()
                await asyncio.sleep(60)  # Collect metrics every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(60)
    
    async def check_cluster_health(self):
        """Check cluster health and update node status."""
        healthy_nodes = 0
        total_nodes = len(self.cluster_nodes)
        
        for node in self.cluster_nodes:
            node_key = f"{node.host}:{node.port}"
            
            try:
                # Create temporary connection to test node
                temp_redis = aioredis.Redis(
                    host=node.host,
                    port=node.port,
                    password=self.password,
                    socket_timeout=2,
                    socket_connect_timeout=2
                )
                
                await temp_redis.ping()
                await temp_redis.aclose()
                
                self.node_health[node_key] = True
                healthy_nodes += 1
                
            except Exception:
                self.node_health[node_key] = False
        
        self.last_health_check = datetime.utcnow()
        
        health_ratio = healthy_nodes / total_nodes
        if health_ratio < 0.5:
            logger.warning(f"Cluster health critical: {healthy_nodes}/{total_nodes} nodes healthy")
        elif health_ratio < 0.8:
            logger.warning(f"Cluster health degraded: {healthy_nodes}/{total_nodes} nodes healthy")
        
        logger.debug(f"Cluster health check: {healthy_nodes}/{total_nodes} nodes healthy")
    
    async def collect_metrics(self):
        """Collect comprehensive cluster metrics."""
        try:
            redis_conn = self.get_connection()
            
            if not redis_conn:
                logger.warning("No Redis connection available for metrics collection")
                return
            
            # Get cluster info
            if isinstance(redis_conn, RedisCluster):
                info = await redis_conn.info()
                # Aggregate metrics from all nodes
                total_keys = sum(info[node].get('db0', {}).get('keys', 0) for node in info)
                used_memory = sum(info[node].get('used_memory', 0) for node in info)
                max_memory = sum(info[node].get('maxmemory', 0) for node in info)
                connected_clients = sum(info[node].get('connected_clients', 0) for node in info)
                keyspace_hits = sum(info[node].get('keyspace_hits', 0) for node in info)
                keyspace_misses = sum(info[node].get('keyspace_misses', 0) for node in info)
                
                master_nodes = len([n for n, i in info.items() if i.get('role') == 'master'])
                slave_nodes = len([n for n, i in info.items() if i.get('role') == 'slave'])
                
            else:
                # Single node metrics
                info = await redis_conn.info()
                db_info = await redis_conn.info('keyspace')
                
                total_keys = db_info.get('db0', {}).get('keys', 0)
                used_memory = info.get('used_memory', 0)
                max_memory = info.get('maxmemory', 0)
                connected_clients = info.get('connected_clients', 0)
                keyspace_hits = info.get('keyspace_hits', 0)
                keyspace_misses = info.get('keyspace_misses', 0)
                
                master_nodes = 1
                slave_nodes = 0
            
            # Calculate derived metrics
            memory_usage_percent = (used_memory / max_memory * 100) if max_memory > 0 else 0
            hit_rate = keyspace_hits / max(keyspace_hits + keyspace_misses, 1) * 100
            miss_rate = 100 - hit_rate
            
            self.metrics = RedisClusterMetrics(
                total_nodes=len(self.cluster_nodes),
                master_nodes=master_nodes,
                slave_nodes=slave_nodes,
                total_keys=total_keys,
                used_memory=used_memory,
                max_memory=max_memory,
                memory_usage_percent=memory_usage_percent,
                ops_per_second=0.0,  # Would need time-series data
                hit_rate=hit_rate,
                miss_rate=miss_rate,
                connected_clients=connected_clients,
                blocked_clients=info.get('blocked_clients', 0),
                keyspace_hits=keyspace_hits,
                keyspace_misses=keyspace_misses,
                expired_keys=info.get('expired_keys', 0),
                evicted_keys=info.get('evicted_keys', 0),
                last_updated=datetime.utcnow()
            )
            
            logger.debug("Redis metrics collected successfully")
            
        except Exception as e:
            logger.error(f"Failed to collect Redis metrics: {e}")
    
    def get_connection(self) -> Optional[Union[RedisCluster, aioredis.Redis]]:
        """Get active Redis connection with fallback."""
        if self.cluster:
            return self.cluster
        elif self.fallback_redis:
            return self.fallback_redis
        else:
            return None
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with error handling."""
        try:
            conn = self.get_connection()
            if not conn:
                return default
            
            value = await conn.get(key)
            if value is None:
                return default
            
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            logger.error(f"Redis GET error for key {key}: {e}")
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """Set value in cache with error handling."""
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            # Serialize value if needed
            if not isinstance(value, (str, bytes, int, float)):
                value = json.dumps(value)
            
            result = await conn.set(key, value, ex=expire, nx=nx, xx=xx)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Redis SET error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            result = await conn.delete(key)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Redis DELETE error for key {key}: {e}")
            return False
    
    async def mget(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        try:
            conn = self.get_connection()
            if not conn:
                return {}
            
            values = await conn.mget(keys)
            result = {}
            
            for i, key in enumerate(keys):
                if i < len(values) and values[i] is not None:
                    try:
                        result[key] = json.loads(values[i])
                    except (json.JSONDecodeError, TypeError):
                        result[key] = values[i]
            
            return result
            
        except Exception as e:
            logger.error(f"Redis MGET error: {e}")
            return {}
    
    async def mset(self, mapping: Dict[str, Any], expire: Optional[int] = None) -> bool:
        """Set multiple values in cache."""
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            # Serialize values
            serialized_mapping = {}
            for key, value in mapping.items():
                if not isinstance(value, (str, bytes, int, float)):
                    serialized_mapping[key] = json.dumps(value)
                else:
                    serialized_mapping[key] = value
            
            await conn.mset(serialized_mapping)
            
            # Set expiration if specified
            if expire:
                pipeline = conn.pipeline()
                for key in mapping.keys():
                    pipeline.expire(key, expire)
                await pipeline.execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Redis MSET error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            result = await conn.exists(key)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Redis EXISTS error for key {key}: {e}")
            return False
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set key expiration."""
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            result = await conn.expire(key, seconds)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Redis EXPIRE error for key {key}: {e}")
            return False
    
    async def ttl(self, key: str) -> int:
        """Get key time-to-live."""
        try:
            conn = self.get_connection()
            if not conn:
                return -1
            
            result = await conn.ttl(key)
            return result
            
        except Exception as e:
            logger.error(f"Redis TTL error for key {key}: {e}")
            return -1
    
    async def flush_db(self, database: int = 0) -> bool:
        """Flush database (use with caution)."""
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            if isinstance(conn, RedisCluster):
                # Flush all nodes in cluster
                await conn.flushdb()
            else:
                await conn.flushdb()
            
            logger.warning(f"Redis database {database} flushed")
            return True
            
        except Exception as e:
            logger.error(f"Redis FLUSHDB error: {e}")
            return False
    
    async def get_cluster_info(self) -> Dict[str, Any]:
        """Get comprehensive cluster information."""
        try:
            conn = self.get_connection()
            if not conn:
                return {}
            
            info = {
                'cluster_enabled': isinstance(conn, RedisCluster),
                'node_health': dict(self.node_health),
                'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
                'monitoring_enabled': self._monitoring_enabled,
                'metrics': None
            }
            
            if self.metrics:
                info['metrics'] = {
                    'total_nodes': self.metrics.total_nodes,
                    'master_nodes': self.metrics.master_nodes,
                    'slave_nodes': self.metrics.slave_nodes,
                    'total_keys': self.metrics.total_keys,
                    'used_memory': self.metrics.used_memory,
                    'max_memory': self.metrics.max_memory,
                    'memory_usage_percent': self.metrics.memory_usage_percent,
                    'hit_rate': self.metrics.hit_rate,
                    'miss_rate': self.metrics.miss_rate,
                    'connected_clients': self.metrics.connected_clients,
                    'last_updated': self.metrics.last_updated.isoformat()
                }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get cluster info: {e}")
            return {'error': str(e)}
    
    async def close(self):
        """Clean up connections and stop monitoring."""
        await self.stop_monitoring()
        
        if self.cluster:
            await self.cluster.aclose()
        
        if self.fallback_redis:
            await self.fallback_redis.aclose()
        
        logger.info("Redis cluster connections closed")


# Global cluster manager instance
_cluster_manager: Optional[RedisClusterManager] = None


def get_redis_cluster_manager() -> Optional[RedisClusterManager]:
    """Get global Redis cluster manager."""
    return _cluster_manager


async def setup_redis_cluster(
    cluster_nodes: List[str],
    password: Optional[str] = None,
    **kwargs
) -> RedisClusterManager:
    """Set up Redis cluster management."""
    global _cluster_manager
    
    # Parse node configurations
    node_configs = []
    for node_str in cluster_nodes:
        if ':' in node_str:
            host, port = node_str.split(':', 1)
            node_configs.append(RedisNodeConfig(
                host=host.strip(),
                port=int(port.strip())
            ))
        else:
            node_configs.append(RedisNodeConfig(
                host=node_str.strip(),
                port=6379
            ))
    
    _cluster_manager = RedisClusterManager(
        cluster_nodes=node_configs,
        password=password,
        **kwargs
    )
    
    await _cluster_manager.connect()
    
    logger.info("Redis cluster management enabled")
    return _cluster_manager