"""
Read Replica Manager - Database Load Balancing and Scaling

Advanced read replica management system providing:
- Automatic read/write query routing
- Load balancing across multiple read replicas
- Health monitoring and failover
- Connection pooling optimization for replicas
- Real-time lag monitoring
- Dynamic replica scaling
"""

import asyncio
import time
import logging
import random
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from collections import deque, defaultdict
import statistics
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import text, event
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError

from ..core.logger import get_logger
from ..core.exceptions import ClaudeTUIException
from .session import DatabaseConfig

logger = get_logger(__name__)


class QueryType(Enum):
    """Query type classification."""
    READ = "read"
    WRITE = "write"
    UNKNOWN = "unknown"


@dataclass
class ReplicaHealth:
    """Health metrics for a read replica."""
    replica_id: str
    is_healthy: bool = True
    last_health_check: datetime = field(default_factory=datetime.utcnow)
    consecutive_failures: int = 0
    replication_lag_seconds: float = 0.0
    avg_response_time: float = 0.0
    connection_count: int = 0
    query_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    
    def update_health_check(self, is_healthy: bool, response_time: float = 0.0, error: Optional[str] = None):
        """Update health check results."""
        self.last_health_check = datetime.utcnow()
        
        if is_healthy:
            self.is_healthy = True
            self.consecutive_failures = 0
            self.avg_response_time = (self.avg_response_time + response_time) / 2 if self.avg_response_time > 0 else response_time
        else:
            self.consecutive_failures += 1
            self.error_count += 1
            self.last_error = error
            
            if self.consecutive_failures >= 3:
                self.is_healthy = False


@dataclass
class ReplicaConfig:
    """Configuration for a read replica."""
    replica_id: str
    database_url: str
    weight: float = 1.0  # Load balancing weight
    max_connections: int = 10
    priority: int = 1  # Lower number = higher priority
    read_only: bool = True
    enable_health_checks: bool = True
    health_check_interval: int = 30  # seconds
    max_lag_seconds: float = 5.0  # Maximum acceptable replication lag


class QueryClassifier:
    """Classify queries as read or write operations."""
    
    def __init__(self):
        self.read_keywords = {
            'SELECT', 'SHOW', 'DESCRIBE', 'DESC', 'EXPLAIN', 'WITH'
        }
        
        self.write_keywords = {
            'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER', 
            'TRUNCATE', 'REPLACE', 'MERGE', 'UPSERT'
        }
    
    def classify_query(self, query: str) -> QueryType:
        """Classify query as read, write, or unknown."""
        if not query:
            return QueryType.UNKNOWN
        
        # Normalize query
        normalized = query.strip().upper()
        
        # Remove comments
        if '--' in normalized:
            normalized = normalized.split('--')[0]
        
        if '/*' in normalized:
            parts = normalized.split('/*')
            if len(parts) > 1 and '*/' in parts[1]:
                normalized = parts[0] + parts[1].split('*/', 1)[1]
            else:
                normalized = parts[0]
        
        # Get first meaningful word
        words = normalized.split()
        if not words:
            return QueryType.UNKNOWN
        
        first_word = words[0]
        
        # Check for explicit read operations
        if first_word in self.read_keywords:
            return QueryType.READ
        
        # Check for explicit write operations
        if first_word in self.write_keywords:
            return QueryType.WRITE
        
        # Special cases
        if first_word == 'BEGIN' or first_word == 'START':
            return QueryType.WRITE  # Transactions usually involve writes
        
        if first_word in ('COMMIT', 'ROLLBACK'):
            return QueryType.WRITE
        
        return QueryType.UNKNOWN


class LoadBalancer:
    """Load balancer for read replicas."""
    
    def __init__(self, strategy: str = "weighted_round_robin"):
        """
        Initialize load balancer.
        
        Args:
            strategy: Load balancing strategy
                - "round_robin": Simple round robin
                - "weighted_round_robin": Weighted round robin based on replica weights
                - "least_connections": Route to replica with fewest connections
                - "response_time": Route to replica with best response time
                - "random": Random selection
        """
        self.strategy = strategy
        self.round_robin_index = 0
        self.replica_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
    
    def select_replica(self, replicas: List[ReplicaHealth]) -> Optional[str]:
        """Select optimal replica based on strategy."""
        healthy_replicas = [r for r in replicas if r.is_healthy]
        
        if not healthy_replicas:
            return None
        
        if self.strategy == "round_robin":
            return self._round_robin_selection(healthy_replicas)
        elif self.strategy == "weighted_round_robin":
            return self._weighted_round_robin_selection(healthy_replicas)
        elif self.strategy == "least_connections":
            return self._least_connections_selection(healthy_replicas)
        elif self.strategy == "response_time":
            return self._response_time_selection(healthy_replicas)
        elif self.strategy == "random":
            return random.choice(healthy_replicas).replica_id
        else:
            # Default to round robin
            return self._round_robin_selection(healthy_replicas)
    
    def _round_robin_selection(self, replicas: List[ReplicaHealth]) -> str:
        """Simple round robin selection."""
        if not replicas:
            return None
        
        replica = replicas[self.round_robin_index % len(replicas)]
        self.round_robin_index += 1
        return replica.replica_id
    
    def _weighted_round_robin_selection(self, replicas: List[ReplicaHealth]) -> str:
        """Weighted round robin selection."""
        if not replicas:
            return None
        
        # For simplicity, use round robin with preference for lower lag
        sorted_replicas = sorted(replicas, key=lambda r: r.replication_lag_seconds)
        return sorted_replicas[self.round_robin_index % len(sorted_replicas)].replica_id
    
    def _least_connections_selection(self, replicas: List[ReplicaHealth]) -> str:
        """Select replica with least connections."""
        if not replicas:
            return None
        
        return min(replicas, key=lambda r: r.connection_count).replica_id
    
    def _response_time_selection(self, replicas: List[ReplicaHealth]) -> str:
        """Select replica with best response time."""
        if not replicas:
            return None
        
        return min(replicas, key=lambda r: r.avg_response_time or float('inf')).replica_id


class ReadReplicaManager:
    """
    Advanced read replica manager for database scaling and performance.
    
    Features:
    - Automatic read/write query routing
    - Load balancing across multiple read replicas
    - Health monitoring and automatic failover
    - Connection pool optimization per replica
    - Real-time replication lag monitoring
    - Dynamic scaling and replica management
    """
    
    def __init__(
        self,
        primary_config: DatabaseConfig,
        replica_configs: Optional[List[ReplicaConfig]] = None,
        load_balancing_strategy: str = "weighted_round_robin",
        enable_automatic_failover: bool = True,
        health_check_interval: int = 30
    ):
        """
        Initialize read replica manager.
        
        Args:
            primary_config: Primary database configuration
            replica_configs: List of read replica configurations
            load_balancing_strategy: Load balancing strategy
            enable_automatic_failover: Enable automatic failover to primary
            health_check_interval: Health check interval in seconds
        """
        self.primary_config = primary_config
        self.replica_configs = replica_configs or []
        self.enable_automatic_failover = enable_automatic_failover
        self.health_check_interval = health_check_interval
        
        # Core components
        self.query_classifier = QueryClassifier()
        self.load_balancer = LoadBalancer(load_balancing_strategy)
        
        # Connection management
        self.primary_engine: Optional[AsyncEngine] = None
        self.replica_engines: Dict[str, AsyncEngine] = {}
        self.primary_session_maker: Optional[async_sessionmaker] = None
        self.replica_session_makers: Dict[str, async_sessionmaker] = {}
        
        # Health monitoring
        self.replica_health: Dict[str, ReplicaHealth] = {}
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        
        # Performance metrics
        self.routing_stats = {
            'total_queries': 0,
            'read_queries': 0,
            'write_queries': 0,
            'replica_queries': 0,
            'primary_queries': 0,
            'failover_count': 0
        }
        
        self.query_history: deque = deque(maxlen=10000)
        
        self._initialized = False
        
        logger.info(f"Read replica manager initialized with {len(self.replica_configs)} replicas")
    
    async def initialize(self):
        """Initialize read replica manager and connections."""
        if self._initialized:
            return
        
        try:
            # Initialize primary database connection
            await self._initialize_primary()
            
            # Initialize read replicas
            await self._initialize_replicas()
            
            # Start health monitoring
            await self._start_health_monitoring()
            
            self._initialized = True
            logger.info("Read replica manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize read replica manager: {e}")
            await self.close()
            raise
    
    async def _initialize_primary(self):
        """Initialize primary database connection."""
        engine_kwargs = self.primary_config.get_engine_kwargs()
        self.primary_engine = create_async_engine(
            self.primary_config.database_url,
            **engine_kwargs
        )
        
        self.primary_session_maker = async_sessionmaker(
            bind=self.primary_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        logger.info("Primary database connection initialized")
    
    async def _initialize_replicas(self):
        """Initialize read replica connections."""
        for config in self.replica_configs:
            try:
                # Create engine for replica
                replica_engine_kwargs = {
                    'pool_size': min(config.max_connections, 10),
                    'max_overflow': 5,
                    'pool_timeout': 30,
                    'pool_recycle': 3600,
                    'pool_pre_ping': True,
                    'echo': False
                }
                
                engine = create_async_engine(
                    config.database_url,
                    **replica_engine_kwargs
                )
                
                # Create session maker
                session_maker = async_sessionmaker(
                    bind=engine,
                    class_=AsyncSession,
                    expire_on_commit=False
                )
                
                # Store connections
                self.replica_engines[config.replica_id] = engine
                self.replica_session_makers[config.replica_id] = session_maker
                
                # Initialize health tracking
                self.replica_health[config.replica_id] = ReplicaHealth(
                    replica_id=config.replica_id
                )
                
                logger.info(f"Initialized read replica: {config.replica_id}")
                
            except Exception as e:
                logger.error(f"Failed to initialize replica {config.replica_id}: {e}")
                # Continue with other replicas
                continue
    
    async def _start_health_monitoring(self):
        """Start health monitoring for all replicas."""
        for config in self.replica_configs:
            if config.enable_health_checks and config.replica_id in self.replica_engines:
                task = asyncio.create_task(
                    self._health_check_loop(config.replica_id, config.health_check_interval)
                )
                self.health_check_tasks[config.replica_id] = task
        
        logger.info(f"Started health monitoring for {len(self.health_check_tasks)} replicas")
    
    async def _health_check_loop(self, replica_id: str, interval: int):
        """Health check loop for a specific replica."""
        while True:
            try:
                await self._perform_health_check(replica_id)
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error for replica {replica_id}: {e}")
                await asyncio.sleep(interval)
    
    async def _perform_health_check(self, replica_id: str):
        """Perform health check on specific replica."""
        if replica_id not in self.replica_engines:
            return
        
        start_time = time.time()
        health = self.replica_health.get(replica_id)
        
        if not health:
            return
        
        try:
            # Test connection with simple query
            engine = self.replica_engines[replica_id]
            async with engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            response_time = time.time() - start_time
            
            # Check replication lag (PostgreSQL specific)
            lag_seconds = await self._check_replication_lag(replica_id)
            health.replication_lag_seconds = lag_seconds
            
            # Update health status
            is_healthy = lag_seconds < 10.0  # Consider unhealthy if lag > 10 seconds
            health.update_health_check(is_healthy, response_time)
            
            if is_healthy:
                logger.debug(f"Replica {replica_id} health check passed (lag: {lag_seconds:.2f}s)")
            else:
                logger.warning(f"Replica {replica_id} has high replication lag: {lag_seconds:.2f}s")
            
        except Exception as e:
            health.update_health_check(False, error=str(e))
            logger.error(f"Health check failed for replica {replica_id}: {e}")
    
    async def _check_replication_lag(self, replica_id: str) -> float:
        """Check replication lag for specific replica."""
        try:
            engine = self.replica_engines[replica_id]
            
            # PostgreSQL specific lag check
            if 'postgresql' in str(engine.url):
                async with engine.begin() as conn:
                    result = await conn.execute(text(
                        "SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) as lag"
                    ))
                    row = result.fetchone()
                    return float(row[0]) if row and row[0] is not None else 0.0
            
            # For other databases, return 0 (no lag check implemented)
            return 0.0
            
        except Exception as e:
            logger.debug(f"Could not check replication lag for {replica_id}: {e}")
            return 0.0
    
    @asynccontextmanager
    async def get_session(self, query: Optional[str] = None, force_primary: bool = False):
        """
        Get database session with automatic routing.
        
        Args:
            query: SQL query to classify for routing
            force_primary: Force use of primary database
        """
        if not self._initialized:
            await self.initialize()
        
        # Track query stats
        self.routing_stats['total_queries'] += 1
        
        # Determine routing
        if force_primary:
            session_maker = self.primary_session_maker
            selected_replica = "primary"
        else:
            # Classify query type
            query_type = self.query_classifier.classify_query(query) if query else QueryType.UNKNOWN
            
            if query_type == QueryType.WRITE or query_type == QueryType.UNKNOWN:
                # Route writes and unknown queries to primary
                session_maker = self.primary_session_maker
                selected_replica = "primary"
                self.routing_stats['write_queries'] += 1
                self.routing_stats['primary_queries'] += 1
            else:
                # Route reads to replica
                selected_replica = self._select_read_replica()
                
                if selected_replica and selected_replica != "primary":
                    session_maker = self.replica_session_makers[selected_replica]
                    self.routing_stats['read_queries'] += 1
                    self.routing_stats['replica_queries'] += 1
                else:
                    # Fallback to primary
                    session_maker = self.primary_session_maker
                    selected_replica = "primary"
                    self.routing_stats['read_queries'] += 1
                    self.routing_stats['primary_queries'] += 1
                    
                    if self.enable_automatic_failover:
                        self.routing_stats['failover_count'] += 1
        
        # Record query routing
        self.query_history.append({
            'timestamp': datetime.utcnow(),
            'query_type': query_type.value if 'query_type' in locals() else 'unknown',
            'selected_replica': selected_replica,
            'query_preview': query[:100] if query else None
        })
        
        # Get session
        try:
            async with session_maker() as session:
                # Update connection count for replica
                if selected_replica in self.replica_health:
                    self.replica_health[selected_replica].connection_count += 1
                
                yield session
        finally:
            # Update connection count
            if selected_replica in self.replica_health:
                self.replica_health[selected_replica].connection_count -= 1
                self.replica_health[selected_replica].query_count += 1
    
    def _select_read_replica(self) -> Optional[str]:
        """Select optimal read replica for query."""
        healthy_replicas = [h for h in self.replica_health.values() if h.is_healthy]
        
        if not healthy_replicas:
            logger.warning("No healthy read replicas available, routing to primary")
            return "primary"
        
        selected_replica_id = self.load_balancer.select_replica(healthy_replicas)
        return selected_replica_id or "primary"
    
    async def add_replica(self, config: ReplicaConfig) -> bool:
        """Add new read replica dynamically."""
        try:
            # Create engine and session maker
            engine_kwargs = {
                'pool_size': min(config.max_connections, 10),
                'max_overflow': 5,
                'pool_timeout': 30,
                'pool_recycle': 3600,
                'pool_pre_ping': True,
                'echo': False
            }
            
            engine = create_async_engine(config.database_url, **engine_kwargs)
            session_maker = async_sessionmaker(
                bind=engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connection
            async with engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            # Add to active replicas
            self.replica_engines[config.replica_id] = engine
            self.replica_session_makers[config.replica_id] = session_maker
            self.replica_health[config.replica_id] = ReplicaHealth(replica_id=config.replica_id)
            self.replica_configs.append(config)
            
            # Start health monitoring
            if config.enable_health_checks:
                task = asyncio.create_task(
                    self._health_check_loop(config.replica_id, config.health_check_interval)
                )
                self.health_check_tasks[config.replica_id] = task
            
            logger.info(f"Successfully added read replica: {config.replica_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add replica {config.replica_id}: {e}")
            return False
    
    async def remove_replica(self, replica_id: str) -> bool:
        """Remove read replica dynamically."""
        try:
            # Stop health monitoring
            if replica_id in self.health_check_tasks:
                self.health_check_tasks[replica_id].cancel()
                del self.health_check_tasks[replica_id]
            
            # Close engine
            if replica_id in self.replica_engines:
                await self.replica_engines[replica_id].dispose()
                del self.replica_engines[replica_id]
            
            # Clean up
            self.replica_session_makers.pop(replica_id, None)
            self.replica_health.pop(replica_id, None)
            
            # Remove from config
            self.replica_configs = [c for c in self.replica_configs if c.replica_id != replica_id]
            
            logger.info(f"Successfully removed read replica: {replica_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove replica {replica_id}: {e}")
            return False
    
    def get_replica_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all replicas."""
        status = {
            'primary': {
                'status': 'healthy' if self.primary_engine else 'not_initialized',
                'connection_url': str(self.primary_config.database_url).split('@')[-1]
            },
            'replicas': {},
            'routing_stats': self.routing_stats.copy(),
            'load_balancing_strategy': self.load_balancer.strategy,
            'total_replicas': len(self.replica_configs),
            'healthy_replicas': len([h for h in self.replica_health.values() if h.is_healthy])
        }
        
        # Add replica details
        for replica_id, health in self.replica_health.items():
            config = next((c for c in self.replica_configs if c.replica_id == replica_id), None)
            
            status['replicas'][replica_id] = {
                'is_healthy': health.is_healthy,
                'last_health_check': health.last_health_check.isoformat(),
                'consecutive_failures': health.consecutive_failures,
                'replication_lag_seconds': health.replication_lag_seconds,
                'avg_response_time': health.avg_response_time,
                'connection_count': health.connection_count,
                'query_count': health.query_count,
                'error_count': health.error_count,
                'last_error': health.last_error,
                'weight': config.weight if config else 1.0,
                'priority': config.priority if config else 1,
                'max_connections': config.max_connections if config else 'unknown'
            }
        
        return status
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        # Calculate query distribution over time
        recent_queries = [q for q in self.query_history 
                         if datetime.utcnow() - q['timestamp'] < timedelta(minutes=5)]
        
        query_distribution = defaultdict(int)
        replica_distribution = defaultdict(int)
        
        for query in recent_queries:
            query_distribution[query['query_type']] += 1
            replica_distribution[query['selected_replica']] += 1
        
        # Calculate replica utilization
        replica_utilization = {}
        for replica_id, health in self.replica_health.items():
            config = next((c for c in self.replica_configs if c.replica_id == replica_id), None)
            max_conn = config.max_connections if config else 10
            utilization = health.connection_count / max_conn if max_conn > 0 else 0
            replica_utilization[replica_id] = {
                'utilization_ratio': utilization,
                'active_connections': health.connection_count,
                'max_connections': max_conn,
                'queries_per_minute': len([q for q in recent_queries if q['selected_replica'] == replica_id])
            }
        
        return {
            'routing_efficiency': {
                'total_queries': self.routing_stats['total_queries'],
                'read_replica_ratio': self.routing_stats['replica_queries'] / max(self.routing_stats['read_queries'], 1),
                'primary_fallback_ratio': self.routing_stats['failover_count'] / max(self.routing_stats['total_queries'], 1),
                'query_type_distribution': dict(query_distribution),
                'replica_query_distribution': dict(replica_distribution)
            },
            'replica_performance': {
                replica_id: {
                    'avg_response_time': health.avg_response_time,
                    'replication_lag': health.replication_lag_seconds,
                    'error_rate': health.error_count / max(health.query_count, 1),
                    'uptime_ratio': 1.0 - (health.consecutive_failures / max(health.query_count, 1))
                }
                for replica_id, health in self.replica_health.items()
            },
            'utilization': replica_utilization,
            'recent_activity': {
                'queries_last_5min': len(recent_queries),
                'avg_queries_per_minute': len(recent_queries) / 5,
                'peak_concurrent_connections': max(
                    sum(h.connection_count for h in self.replica_health.values()),
                    self.routing_stats.get('peak_connections', 0)
                )
            }
        }
    
    async def close(self):
        """Clean up all connections and resources."""
        # Stop health monitoring
        for task in self.health_check_tasks.values():
            task.cancel()
        
        # Close replica engines
        for engine in self.replica_engines.values():
            await engine.dispose()
        
        # Close primary engine
        if self.primary_engine:
            await self.primary_engine.dispose()
        
        # Clear all data structures
        self.replica_engines.clear()
        self.replica_session_makers.clear()
        self.replica_health.clear()
        self.health_check_tasks.clear()
        
        self._initialized = False
        logger.info("Read replica manager closed")


# Global read replica manager
_read_replica_manager: Optional[ReadReplicaManager] = None


def get_read_replica_manager() -> Optional[ReadReplicaManager]:
    """Get global read replica manager."""
    return _read_replica_manager


async def setup_read_replica_manager(
    primary_config: DatabaseConfig,
    replica_configs: Optional[List[ReplicaConfig]] = None,
    load_balancing_strategy: str = "weighted_round_robin"
) -> ReadReplicaManager:
    """Set up read replica management system."""
    global _read_replica_manager
    
    _read_replica_manager = ReadReplicaManager(
        primary_config=primary_config,
        replica_configs=replica_configs or [],
        load_balancing_strategy=load_balancing_strategy
    )
    
    await _read_replica_manager.initialize()
    
    logger.info("Read replica management system enabled")
    return _read_replica_manager