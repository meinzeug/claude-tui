#!/usr/bin/env python3
"""
Optimized Connection Pool - High-Performance Database Connection Management

Provides ultra-fast database connection pooling with:
- Target: Database queries <100ms (from current slower performance)
- Aggressive connection reuse and optimization
- Smart connection warming and preloading
- Query result caching integration
- Connection health monitoring
- Automatic failover and recovery
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, AsyncContextManager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from collections import deque
import statistics
import threading
import weakref
import json

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    asyncpg = None

try:
    import aiosqlite
    AIOSQLITE_AVAILABLE = True
except ImportError:
    AIOSQLITE_AVAILABLE = False
    aiosqlite = None

logger = logging.getLogger(__name__)


@dataclass
class ConnectionMetrics:
    """Performance metrics for database connections"""
    connection_id: str
    created_at: datetime
    total_queries: int = 0
    total_query_time: float = 0.0
    avg_query_time: float = 0.0
    last_used: datetime = field(default_factory=datetime.utcnow)
    error_count: int = 0
    is_healthy: bool = True
    
    def record_query(self, execution_time: float):
        """Record query execution metrics"""
        self.total_queries += 1
        self.total_query_time += execution_time
        self.avg_query_time = self.total_query_time / self.total_queries
        self.last_used = datetime.utcnow()
    
    def record_error(self):
        """Record connection error"""
        self.error_count += 1
        if self.error_count > 3:
            self.is_healthy = False


@dataclass
class QueryCacheEntry:
    """Cached query result"""
    query_hash: str
    result: Any
    created_at: datetime
    access_count: int = 0
    ttl_seconds: int = 300  # 5 minutes default
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return datetime.utcnow() - self.created_at > timedelta(seconds=self.ttl_seconds)
    
    def touch(self):
        """Update access time and count"""
        self.access_count += 1


class OptimizedConnectionPool:
    """
    Ultra-fast optimized connection pool with aggressive performance optimization
    
    Features:
    - Target: <100ms query response time
    - Intelligent connection reuse
    - Built-in query result caching
    - Connection warming and preloading
    - Health monitoring and auto-recovery
    - Performance metrics and optimization
    """
    
    def __init__(
        self,
        database_url: str,
        min_connections: int = 5,
        max_connections: int = 20,
        query_cache_size: int = 1000,
        enable_query_cache: bool = True,
        connection_timeout: float = 30.0
    ):
        self.database_url = database_url
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.query_cache_size = query_cache_size
        self.enable_query_cache = enable_query_cache
        self.connection_timeout = connection_timeout
        
        # Connection pool
        self.connections: deque = deque()
        self.active_connections: Dict[str, Any] = {}
        self.connection_metrics: Dict[str, ConnectionMetrics] = {}
        
        # Query caching
        self.query_cache: Dict[str, QueryCacheEntry] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Performance tracking
        self.query_times: deque = deque(maxlen=1000)
        self.total_queries = 0
        self.pool_initialized = False
        
        # Background tasks
        self._maintenance_task: Optional[asyncio.Task] = None
        self._warming_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize the connection pool"""
        logger.info(f"Initializing optimized connection pool (size: {self.min_connections}-{self.max_connections})")
        
        # Create initial connections
        for i in range(self.min_connections):
            try:
                conn = await self._create_connection()
                if conn:
                    self.connections.append(conn)
                    conn_id = str(id(conn))
                    self.connection_metrics[conn_id] = ConnectionMetrics(
                        connection_id=conn_id,
                        created_at=datetime.utcnow()
                    )
            except Exception as e:
                logger.error(f"Failed to create initial connection {i}: {e}")
        
        self.pool_initialized = True
        
        # Start background maintenance
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        self._warming_task = asyncio.create_task(self._warming_loop())
        
        logger.info(f"Connection pool initialized with {len(self.connections)} connections")
    
    async def _create_connection(self):
        """Create a new database connection"""
        try:
            if self.database_url.startswith('postgresql://') and ASYNCPG_AVAILABLE:
                return await asyncpg.connect(self.database_url)
            elif self.database_url.startswith('sqlite://') and AIOSQLITE_AVAILABLE:
                db_path = self.database_url.replace('sqlite://', '')
                return await aiosqlite.connect(db_path)
            else:
                # Fallback - simulate connection for testing
                logger.warning("Using simulated database connection")
                return SimulatedConnection()
        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            return None
    
    @asynccontextmanager
    async def get_connection(self):
        """Get optimized database connection from pool"""
        if not self.pool_initialized:
            await self.initialize()
        
        connection = None
        conn_id = None
        start_time = time.time()
        
        try:
            # Get connection from pool
            if self.connections:
                connection = self.connections.popleft()
                conn_id = str(id(connection))
                
                # Verify connection is healthy
                if conn_id in self.connection_metrics:
                    metrics = self.connection_metrics[conn_id]
                    if not metrics.is_healthy:
                        # Connection is unhealthy, create new one
                        await self._close_connection(connection)
                        connection = await self._create_connection()
                        if connection:
                            conn_id = str(id(connection))
                            self.connection_metrics[conn_id] = ConnectionMetrics(
                                connection_id=conn_id,
                                created_at=datetime.utcnow()
                            )
            else:
                # Pool is empty, create new connection
                if len(self.active_connections) < self.max_connections:
                    connection = await self._create_connection()
                    if connection:
                        conn_id = str(id(connection))
                        self.connection_metrics[conn_id] = ConnectionMetrics(
                            connection_id=conn_id,
                            created_at=datetime.utcnow()
                        )
            
            if connection:
                self.active_connections[conn_id] = connection
                yield OptimizedConnectionWrapper(self, connection, conn_id)
            else:
                raise Exception("Could not obtain database connection")
                
        finally:
            # Return connection to pool
            if connection and conn_id:
                self.active_connections.pop(conn_id, None)
                
                # Check if connection is still healthy
                if conn_id in self.connection_metrics and self.connection_metrics[conn_id].is_healthy:
                    self.connections.append(connection)
                else:
                    await self._close_connection(connection)
    
    async def execute_cached_query(
        self, 
        query: str, 
        params: tuple = (), 
        cache_ttl: int = 300,
        bypass_cache: bool = False
    ):
        """Execute query with intelligent caching"""
        
        # Generate cache key
        cache_key = self._generate_cache_key(query, params)
        
        # Check cache first (if enabled and not bypassing)
        if self.enable_query_cache and not bypass_cache and cache_key in self.query_cache:
            cache_entry = self.query_cache[cache_key]
            if not cache_entry.is_expired:
                cache_entry.touch()
                self.cache_hits += 1
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cache_entry.result
            else:
                # Remove expired entry
                del self.query_cache[cache_key]
        
        # Execute query
        start_time = time.time()
        self.cache_misses += 1
        
        async with self.get_connection() as conn:
            try:
                result = await conn.execute_query(query, params)
                execution_time = time.time() - start_time
                
                # Record performance metrics
                self.query_times.append(execution_time)
                self.total_queries += 1
                
                # Cache result (if enabled and appropriate)
                if self.enable_query_cache and self._should_cache_query(query):
                    # Limit cache size
                    if len(self.query_cache) >= self.query_cache_size:
                        # Remove oldest entries
                        oldest_keys = sorted(
                            self.query_cache.keys(),
                            key=lambda k: self.query_cache[k].created_at
                        )[:10]  # Remove 10 oldest
                        
                        for old_key in oldest_keys:
                            del self.query_cache[old_key]
                    
                    # Add to cache
                    self.query_cache[cache_key] = QueryCacheEntry(
                        query_hash=cache_key,
                        result=result,
                        created_at=datetime.utcnow(),
                        ttl_seconds=cache_ttl
                    )
                
                logger.debug(f"Query executed in {execution_time*1000:.1f}ms: {query[:50]}...")
                return result
                
            except Exception as e:
                # Record error in connection metrics
                if hasattr(conn, 'connection_id') and conn.connection_id in self.connection_metrics:
                    self.connection_metrics[conn.connection_id].record_error()
                
                logger.error(f"Query execution failed: {e}")
                raise
    
    def _generate_cache_key(self, query: str, params: tuple) -> str:
        """Generate cache key for query and parameters"""
        import hashlib
        query_data = f"{query}:{str(params)}"
        return hashlib.md5(query_data.encode()).hexdigest()
    
    def _should_cache_query(self, query: str) -> bool:
        """Determine if query result should be cached"""
        query_lower = query.lower().strip()
        
        # Cache SELECT queries, but not INSERT/UPDATE/DELETE
        if query_lower.startswith('select'):
            return True
        
        # Don't cache data modification queries
        if any(query_lower.startswith(cmd) for cmd in ['insert', 'update', 'delete', 'create', 'drop', 'alter']):
            return False
        
        return False
    
    async def _close_connection(self, connection):
        """Safely close a database connection"""
        try:
            if hasattr(connection, 'close'):
                await connection.close()
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
    
    async def _maintenance_loop(self):
        """Background maintenance for connection pool health"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Check connection health
                unhealthy_connections = []
                for conn_id, metrics in self.connection_metrics.items():
                    # Check if connection hasn't been used recently
                    if datetime.utcnow() - metrics.last_used > timedelta(minutes=30):
                        # Mark as potentially unhealthy if it has errors
                        if metrics.error_count > 0:
                            metrics.is_healthy = False
                            unhealthy_connections.append(conn_id)
                
                # Clean up unhealthy connections
                if unhealthy_connections:
                    logger.info(f"Marking {len(unhealthy_connections)} connections as unhealthy")
                
                # Clean expired cache entries
                if self.enable_query_cache:
                    expired_keys = [
                        key for key, entry in self.query_cache.items()
                        if entry.is_expired
                    ]
                    
                    for key in expired_keys:
                        del self.query_cache[key]
                    
                    if expired_keys:
                        logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")
                
            except Exception as e:
                logger.error(f"Maintenance loop error: {e}")
    
    async def _warming_loop(self):
        """Background connection warming to maintain pool size"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Ensure minimum connections are available
                current_available = len(self.connections)
                current_active = len(self.active_connections)
                total_connections = current_available + current_active
                
                if total_connections < self.min_connections:
                    connections_needed = self.min_connections - total_connections
                    
                    logger.info(f"Warming pool: creating {connections_needed} connections")
                    
                    for _ in range(connections_needed):
                        try:
                            conn = await self._create_connection()
                            if conn:
                                self.connections.append(conn)
                                conn_id = str(id(conn))
                                self.connection_metrics[conn_id] = ConnectionMetrics(
                                    connection_id=conn_id,
                                    created_at=datetime.utcnow()
                                )
                        except Exception as e:
                            logger.error(f"Failed to create warming connection: {e}")
                
            except Exception as e:
                logger.error(f"Warming loop error: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        if not self.query_times:
            return {"status": "no_queries_executed"}
        
        # Calculate performance metrics
        recent_queries = list(self.query_times)[-100:]  # Last 100 queries
        
        avg_query_time = statistics.mean(recent_queries) * 1000  # Convert to ms
        median_query_time = statistics.median(recent_queries) * 1000
        p95_query_time = statistics.quantiles(recent_queries, n=20)[18] * 1000 if len(recent_queries) >= 20 else avg_query_time
        max_query_time = max(recent_queries) * 1000
        
        # Cache performance
        total_cache_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0
        
        return {
            "pool_status": {
                "available_connections": len(self.connections),
                "active_connections": len(self.active_connections),
                "total_connections": len(self.connections) + len(self.active_connections),
                "min_connections": self.min_connections,
                "max_connections": self.max_connections
            },
            "query_performance": {
                "total_queries": self.total_queries,
                "avg_query_time_ms": avg_query_time,
                "median_query_time_ms": median_query_time,
                "p95_query_time_ms": p95_query_time,
                "max_query_time_ms": max_query_time,
                "queries_under_100ms": sum(1 for t in recent_queries if t * 1000 < 100),
                "performance_target_met": avg_query_time < 100  # Target: <100ms
            },
            "cache_performance": {
                "enabled": self.enable_query_cache,
                "cache_size": len(self.query_cache),
                "max_cache_size": self.query_cache_size,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate_percent": cache_hit_rate
            },
            "connection_health": {
                "healthy_connections": sum(1 for m in self.connection_metrics.values() if m.is_healthy),
                "unhealthy_connections": sum(1 for m in self.connection_metrics.values() if not m.is_healthy),
                "total_errors": sum(m.error_count for m in self.connection_metrics.values())
            }
        }
    
    async def close(self):
        """Close connection pool and cleanup resources"""
        logger.info("Closing optimized connection pool")
        
        # Cancel background tasks
        if self._maintenance_task:
            self._maintenance_task.cancel()
        if self._warming_task:
            self._warming_task.cancel()
        
        # Close all connections
        all_connections = list(self.connections) + list(self.active_connections.values())
        
        for connection in all_connections:
            await self._close_connection(connection)
        
        self.connections.clear()
        self.active_connections.clear()
        self.connection_metrics.clear()
        self.query_cache.clear()
        
        logger.info("Connection pool closed")


class OptimizedConnectionWrapper:
    """Wrapper for database connections with performance tracking"""
    
    def __init__(self, pool: OptimizedConnectionPool, connection: Any, connection_id: str):
        self.pool = pool
        self.connection = connection
        self.connection_id = connection_id
    
    async def execute_query(self, query: str, params: tuple = ()):
        """Execute query with performance tracking"""
        start_time = time.time()
        
        try:
            # Execute based on connection type
            if hasattr(self.connection, 'fetch'):
                # asyncpg connection
                result = await self.connection.fetch(query, *params)
            elif hasattr(self.connection, 'execute'):
                # aiosqlite or similar
                async with self.connection.execute(query, params) as cursor:
                    result = await cursor.fetchall()
            else:
                # Simulated connection
                result = await self.connection.execute_query(query, params)
            
            # Record performance metrics
            execution_time = time.time() - start_time
            if self.connection_id in self.pool.connection_metrics:
                self.pool.connection_metrics[self.connection_id].record_query(execution_time)
            
            return result
            
        except Exception as e:
            # Record error
            if self.connection_id in self.pool.connection_metrics:
                self.pool.connection_metrics[self.connection_id].record_error()
            raise


class SimulatedConnection:
    """Simulated database connection for testing"""
    
    def __init__(self):
        self.query_count = 0
    
    async def execute_query(self, query: str, params: tuple = ()):
        """Simulate query execution"""
        self.query_count += 1
        
        # Simulate realistic query time based on query type
        if query.lower().strip().startswith('select'):
            # Simulate faster SELECT queries
            await asyncio.sleep(0.01 + (self.query_count % 5) * 0.005)  # 10-35ms
        else:
            # Simulate slower modification queries
            await asyncio.sleep(0.02 + (self.query_count % 3) * 0.01)   # 20-40ms
        
        # Return simulated result
        if 'count' in query.lower():
            return [{'count': 42}]
        elif 'select' in query.lower():
            return [
                {'id': i, 'name': f'item_{i}', 'created_at': datetime.utcnow().isoformat()}
                for i in range(min(10, self.query_count % 20 + 1))
            ]
        else:
            return [{'affected_rows': 1}]
    
    async def close(self):
        """Simulate connection close"""
        pass


# Global optimized connection pool
_global_pool: Optional[OptimizedConnectionPool] = None


async def get_optimized_pool(database_url: str = "sqlite:///:memory:") -> OptimizedConnectionPool:
    """Get global optimized connection pool"""
    global _global_pool
    
    if _global_pool is None:
        _global_pool = OptimizedConnectionPool(
            database_url=database_url,
            min_connections=3,
            max_connections=15,
            query_cache_size=500,
            enable_query_cache=True
        )
        await _global_pool.initialize()
    
    return _global_pool


async def execute_optimized_query(
    query: str, 
    params: tuple = (), 
    cache_ttl: int = 300,
    database_url: str = "sqlite:///:memory:"
):
    """Execute query using optimized connection pool"""
    pool = await get_optimized_pool(database_url)
    return await pool.execute_cached_query(query, params, cache_ttl)


if __name__ == "__main__":
    async def test_optimized_pool():
        print("🚀 OPTIMIZED CONNECTION POOL - Performance Testing")
        print("=" * 60)
        
        # Create pool
        pool = OptimizedConnectionPool(
            database_url="sqlite:///:memory:",
            min_connections=3,
            max_connections=10,
            enable_query_cache=True
        )
        
        await pool.initialize()
        
        print(f"📊 Pool initialized: {pool.get_performance_stats()}")
        
        # Test query performance
        test_queries = [
            "SELECT * FROM users WHERE id = ?",
            "SELECT COUNT(*) FROM orders",
            "SELECT * FROM products WHERE category = ?",
            "SELECT name FROM categories"
        ]
        
        print(f"\n⚡ Running {len(test_queries) * 20} test queries...")
        
        start_time = time.time()
        
        # Run multiple queries to test performance
        for i in range(20):
            for query in test_queries:
                params = (i % 10,) if "?" in query else ()
                try:
                    result = await pool.execute_cached_query(query, params)
                    # Verify we got results
                    assert result is not None
                except Exception as e:
                    print(f"❌ Query failed: {e}")
        
        total_time = time.time() - start_time
        
        # Get final stats
        stats = pool.get_performance_stats()
        
        print(f"\n📈 Performance Results:")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Queries Executed: {stats['query_performance']['total_queries']}")
        print(f"   Avg Query Time: {stats['query_performance']['avg_query_time_ms']:.1f}ms")
        print(f"   P95 Query Time: {stats['query_performance']['p95_query_time_ms']:.1f}ms")
        print(f"   Target Met (<100ms): {'✅' if stats['query_performance']['performance_target_met'] else '❌'}")
        print(f"   Cache Hit Rate: {stats['cache_performance']['cache_hit_rate_percent']:.1f}%")
        
        await pool.close()
        print(f"\n✅ Optimized connection pool test completed!")
    
    # Run test
    asyncio.run(test_optimized_pool())