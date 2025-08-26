"""
Database Query Optimizer - Production Performance Enhancement

Advanced query optimization system providing:
- Query performance analysis and monitoring
- Index recommendation engine
- Slow query detection and optimization
- Connection pool optimization
- Query plan caching and analysis
- Real-time performance metrics
"""

import time
import logging
import statistics
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from collections import defaultdict, deque
import asyncio
import json
import hashlib

from sqlalchemy import text, event
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.engine import Engine
from sqlalchemy.sql import ClauseElement
from sqlalchemy.dialects import postgresql
import redis.asyncio as redis

from ..core.logger import get_logger
from ..core.exceptions import ClaudeTUIException

logger = get_logger(__name__)


@dataclass
class QueryMetrics:
    """Query performance metrics."""
    query_hash: str
    query_text: str
    execution_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    last_executed: Optional[datetime] = None
    execution_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add_execution(self, execution_time: float):
        """Add execution time to metrics."""
        self.execution_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.avg_time = self.total_time / self.execution_count
        self.last_executed = datetime.utcnow()
        self.execution_times.append(execution_time)
    
    def get_percentiles(self) -> Dict[str, float]:
        """Get performance percentiles."""
        if not self.execution_times:
            return {}
        
        times = sorted(self.execution_times)
        return {
            'p50': statistics.median(times),
            'p95': times[int(len(times) * 0.95)] if len(times) > 20 else self.max_time,
            'p99': times[int(len(times) * 0.99)] if len(times) > 100 else self.max_time
        }


@dataclass
class IndexRecommendation:
    """Database index recommendation."""
    table_name: str
    columns: List[str]
    index_type: str  # 'btree', 'hash', 'gin', etc.
    reason: str
    estimated_improvement: float  # percentage
    priority: str  # 'high', 'medium', 'low'
    query_patterns: List[str] = field(default_factory=list)


class QueryOptimizer:
    """
    Advanced database query optimizer for production performance.
    
    Features:
    - Real-time query performance monitoring
    - Automatic slow query detection
    - Index recommendation engine
    - Query plan caching and analysis
    - Connection pool optimization
    - Performance metrics and reporting
    """
    
    def __init__(
        self,
        slow_query_threshold: float = 1.0,  # seconds
        redis_url: Optional[str] = None,
        enable_query_caching: bool = True,
        max_cached_queries: int = 10000
    ):
        """
        Initialize query optimizer.
        
        Args:
            slow_query_threshold: Threshold for slow query detection (seconds)
            redis_url: Redis URL for caching (optional)
            enable_query_caching: Enable query plan caching
            max_cached_queries: Maximum number of cached query metrics
        """
        self.slow_query_threshold = slow_query_threshold
        self.redis_url = redis_url
        self.enable_query_caching = enable_query_caching
        self.max_cached_queries = max_cached_queries
        
        # Performance tracking
        self.query_metrics: Dict[str, QueryMetrics] = {}
        self.slow_queries: List[QueryMetrics] = []
        self.index_recommendations: List[IndexRecommendation] = []
        
        # Redis cache for query plans
        self.redis_client: Optional[redis.Redis] = None
        self.query_plan_cache: Dict[str, Any] = {}
        
        # Performance monitoring
        self.monitoring_enabled = True
        self.total_queries_executed = 0
        self.total_query_time = 0.0
        
        logger.info(f"Query optimizer initialized (threshold: {slow_query_threshold}s)")
    
    async def initialize(self):
        """Initialize optimizer components."""
        if self.redis_url and self.enable_query_caching:
            try:
                self.redis_client = redis.from_url(self.redis_url)
                await self.redis_client.ping()
                logger.info("Redis cache connected for query optimization")
            except Exception as e:
                logger.warning(f"Redis connection failed, using in-memory cache: {e}")
                self.redis_client = None
    
    def setup_query_monitoring(self, engine: Engine):
        """Set up SQLAlchemy event listeners for query monitoring."""
        
        @event.listens_for(engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Record query start time."""
            context._query_start_time = time.time()
            context._query_statement = statement
        
        @event.listens_for(engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Record query execution time and analyze performance."""
            if not self.monitoring_enabled:
                return
            
            execution_time = time.time() - getattr(context, '_query_start_time', time.time())
            
            # Update global metrics
            self.total_queries_executed += 1
            self.total_query_time += execution_time
            
            # Process query metrics
            asyncio.create_task(self._process_query_execution(statement, execution_time))
    
    async def _process_query_execution(self, query: str, execution_time: float):
        """Process query execution for performance analysis."""
        try:
            # Generate query hash
            query_hash = hashlib.md5(self._normalize_query(query).encode()).hexdigest()
            
            # Update or create query metrics
            if query_hash not in self.query_metrics:
                self.query_metrics[query_hash] = QueryMetrics(
                    query_hash=query_hash,
                    query_text=self._normalize_query(query)
                )
            
            self.query_metrics[query_hash].add_execution(execution_time)
            
            # Check for slow queries
            if execution_time > self.slow_query_threshold:
                await self._handle_slow_query(self.query_metrics[query_hash], execution_time)
            
            # Cleanup old metrics if needed
            if len(self.query_metrics) > self.max_cached_queries:
                await self._cleanup_old_metrics()
            
            # Generate index recommendations periodically
            if self.total_queries_executed % 1000 == 0:
                await self._generate_index_recommendations()
            
        except Exception as e:
            logger.error(f"Error processing query execution: {e}")
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent hashing."""
        # Remove extra whitespace and normalize
        normalized = ' '.join(query.split())
        
        # Replace parameter placeholders for better grouping
        import re
        normalized = re.sub(r'\$\d+', '$?', normalized)  # PostgreSQL parameters
        normalized = re.sub(r'\?', '$?', normalized)      # Generic parameters
        normalized = re.sub(r"'[^']*'", "'?'", normalized)  # String literals
        normalized = re.sub(r'\b\d+\b', '?', normalized)    # Number literals
        
        return normalized
    
    async def _handle_slow_query(self, metrics: QueryMetrics, execution_time: float):
        """Handle slow query detection."""
        logger.warning(
            f"Slow query detected: {execution_time:.3f}s "
            f"(avg: {metrics.avg_time:.3f}s, count: {metrics.execution_count})"
        )
        
        # Add to slow queries list
        if metrics not in self.slow_queries:
            self.slow_queries.append(metrics)
            
        # Analyze query for optimization opportunities
        await self._analyze_slow_query(metrics)
    
    async def _analyze_slow_query(self, metrics: QueryMetrics):
        """Analyze slow query for optimization opportunities."""
        query = metrics.query_text.lower()
        
        # Check for missing indexes
        if 'where' in query and 'index' not in query:
            await self._suggest_index_for_where_clause(query)
        
        # Check for inefficient joins
        if 'join' in query:
            await self._analyze_join_performance(query)
        
        # Check for large result sets
        if 'limit' not in query and 'select *' in query:
            logger.warning(f"Query may return large result set: {query[:100]}...")
    
    async def _suggest_index_for_where_clause(self, query: str):
        """Suggest indexes for WHERE clauses."""
        # Simple pattern matching for common cases
        import re
        
        # Extract table and column from WHERE clauses
        where_patterns = [
            r'from\s+(\w+).*where\s+(\w+)\s*[=<>]',
            r'join\s+(\w+).*on\s+\w+\.(\w+)\s*=',
        ]
        
        for pattern in where_patterns:
            matches = re.findall(pattern, query)
            for table, column in matches:
                recommendation = IndexRecommendation(
                    table_name=table,
                    columns=[column],
                    index_type='btree',
                    reason=f'Frequent WHERE clause on {table}.{column}',
                    estimated_improvement=25.0,
                    priority='medium'
                )
                
                if recommendation not in self.index_recommendations:
                    self.index_recommendations.append(recommendation)
                    logger.info(f"Index recommendation: CREATE INDEX ON {table}({column})")
    
    async def _analyze_join_performance(self, query: str):
        """Analyze JOIN performance patterns."""
        if 'left join' in query or 'inner join' in query:
            # Look for potential N+1 query patterns
            join_count = query.count('join')
            if join_count > 3:
                logger.warning(f"Complex query with {join_count} joins may benefit from optimization")
    
    async def _cleanup_old_metrics(self):
        """Clean up old query metrics to prevent memory bloat."""
        # Remove least recently used queries
        sorted_metrics = sorted(
            self.query_metrics.items(),
            key=lambda x: x[1].last_executed or datetime.min
        )
        
        # Keep only the most recent queries
        keep_count = int(self.max_cached_queries * 0.8)
        to_remove = sorted_metrics[:-keep_count]
        
        for query_hash, _ in to_remove:
            del self.query_metrics[query_hash]
        
        logger.debug(f"Cleaned up {len(to_remove)} old query metrics")
    
    async def _generate_index_recommendations(self):
        """Generate index recommendations based on query patterns."""
        # Analyze most frequently executed queries
        frequent_queries = sorted(
            self.query_metrics.values(),
            key=lambda x: x.execution_count,
            reverse=True
        )[:20]  # Top 20 most frequent
        
        for metrics in frequent_queries:
            if metrics.avg_time > self.slow_query_threshold / 2:
                # This query could benefit from optimization
                await self._analyze_slow_query(metrics)
    
    @asynccontextmanager
    async def optimized_session(self, session: AsyncSession):
        """Context manager for optimized database session."""
        session._query_optimizer = self
        
        try:
            yield session
        finally:
            # Clean up any session-specific optimizations
            if hasattr(session, '_query_optimizer'):
                delattr(session, '_query_optimizer')
    
    async def get_query_plan(self, session: AsyncSession, query: str) -> Dict[str, Any]:
        """Get and cache query execution plan."""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        # Check cache first
        if self.redis_client:
            cached_plan = await self.redis_client.get(f"query_plan:{query_hash}")
            if cached_plan:
                return json.loads(cached_plan)
        elif query_hash in self.query_plan_cache:
            return self.query_plan_cache[query_hash]
        
        # Get execution plan
        try:
            explain_query = f"EXPLAIN (FORMAT JSON, ANALYZE, BUFFERS) {query}"
            result = await session.execute(text(explain_query))
            plan_data = result.fetchone()[0]
            
            # Cache the plan
            if self.redis_client:
                await self.redis_client.setex(
                    f"query_plan:{query_hash}",
                    3600,  # 1 hour TTL
                    json.dumps(plan_data)
                )
            else:
                self.query_plan_cache[query_hash] = plan_data
                
                # Limit in-memory cache size
                if len(self.query_plan_cache) > 1000:
                    # Remove oldest entries
                    oldest_keys = list(self.query_plan_cache.keys())[:100]
                    for key in oldest_keys:
                        del self.query_plan_cache[key]
            
            return plan_data
            
        except Exception as e:
            logger.error(f"Failed to get query plan: {e}")
            return {}
    
    async def analyze_query_performance(self, query: str, session: AsyncSession) -> Dict[str, Any]:
        """Analyze query performance and provide optimization suggestions."""
        # Get execution plan
        plan = await self.get_query_plan(session, query)
        
        # Analyze plan for optimization opportunities
        analysis = {
            'query_hash': hashlib.md5(query.encode()).hexdigest(),
            'estimated_cost': 0,
            'seq_scans': 0,
            'index_scans': 0,
            'suggestions': []
        }
        
        if plan:
            # Extract key metrics from plan
            def analyze_node(node):
                if isinstance(node, dict):
                    node_type = node.get('Node Type', '')
                    
                    if 'Seq Scan' in node_type:
                        analysis['seq_scans'] += 1
                        analysis['suggestions'].append({
                            'type': 'index_needed',
                            'message': f"Sequential scan on {node.get('Relation Name', 'unknown table')}",
                            'priority': 'high'
                        })
                    
                    elif 'Index Scan' in node_type:
                        analysis['index_scans'] += 1
                    
                    analysis['estimated_cost'] += node.get('Total Cost', 0)
                    
                    # Recursively analyze child nodes
                    for child in node.get('Plans', []):
                        analyze_node(child)
            
            if isinstance(plan, list) and plan:
                analyze_node(plan[0])
        
        return analysis
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        if not self.query_metrics:
            return {'status': 'no_data'}
        
        # Calculate aggregate metrics
        total_avg_time = self.total_query_time / max(self.total_queries_executed, 1)
        
        # Get top slow queries
        slow_queries_data = sorted(
            self.query_metrics.values(),
            key=lambda x: x.avg_time,
            reverse=True
        )[:10]
        
        # Get most frequent queries
        frequent_queries_data = sorted(
            self.query_metrics.values(),
            key=lambda x: x.execution_count,
            reverse=True
        )[:10]
        
        return {
            'summary': {
                'total_queries': self.total_queries_executed,
                'total_query_time': self.total_query_time,
                'average_query_time': total_avg_time,
                'slow_queries_count': len(self.slow_queries),
                'monitored_queries': len(self.query_metrics),
                'slow_query_threshold': self.slow_query_threshold
            },
            'slow_queries': [
                {
                    'query_hash': q.query_hash,
                    'query_preview': q.query_text[:100] + '...' if len(q.query_text) > 100 else q.query_text,
                    'execution_count': q.execution_count,
                    'avg_time': q.avg_time,
                    'max_time': q.max_time,
                    'percentiles': q.get_percentiles()
                }
                for q in slow_queries_data
            ],
            'frequent_queries': [
                {
                    'query_hash': q.query_hash,
                    'query_preview': q.query_text[:100] + '...' if len(q.query_text) > 100 else q.query_text,
                    'execution_count': q.execution_count,
                    'avg_time': q.avg_time,
                    'total_time': q.total_time
                }
                for q in frequent_queries_data
            ],
            'index_recommendations': [
                {
                    'table': rec.table_name,
                    'columns': rec.columns,
                    'type': rec.index_type,
                    'reason': rec.reason,
                    'priority': rec.priority,
                    'estimated_improvement': rec.estimated_improvement
                }
                for rec in self.index_recommendations
            ]
        }
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        metrics = self.get_performance_metrics()
        
        if metrics.get('status') == 'no_data':
            return {'status': 'insufficient_data', 'message': 'Not enough query data for analysis'}
        
        # Performance assessment
        avg_time = metrics['summary']['average_query_time']
        slow_ratio = metrics['summary']['slow_queries_count'] / max(metrics['summary']['total_queries'], 1)
        
        performance_score = 100
        if avg_time > 0.1:
            performance_score -= min(50, avg_time * 100)
        if slow_ratio > 0.05:
            performance_score -= min(30, slow_ratio * 100)
        
        # Recommendations
        recommendations = []
        
        if slow_ratio > 0.1:
            recommendations.append({
                'type': 'performance',
                'priority': 'high',
                'message': f'High slow query ratio ({slow_ratio:.2%}). Consider optimizing frequent queries.'
            })
        
        if avg_time > 0.5:
            recommendations.append({
                'type': 'performance',
                'priority': 'high',
                'message': f'High average query time ({avg_time:.3f}s). Database optimization needed.'
            })
        
        if len(metrics['index_recommendations']) > 0:
            recommendations.append({
                'type': 'indexes',
                'priority': 'medium',
                'message': f'{len(metrics["index_recommendations"])} index recommendations available.'
            })
        
        return {
            'performance_score': max(0, performance_score),
            'performance_grade': self._get_performance_grade(performance_score),
            'summary': metrics['summary'],
            'recommendations': recommendations,
            'index_recommendations': metrics['index_recommendations'],
            'top_optimization_targets': metrics['slow_queries'][:5]
        }
    
    def _get_performance_grade(self, score: float) -> str:
        """Convert performance score to grade."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    async def close(self):
        """Clean up optimizer resources."""
        if self.redis_client:
            await self.redis_client.close()
        
        self.monitoring_enabled = False
        logger.info("Query optimizer closed")


# Global optimizer instance
_query_optimizer: Optional[QueryOptimizer] = None


def get_query_optimizer() -> QueryOptimizer:
    """Get global query optimizer instance."""
    global _query_optimizer
    if _query_optimizer is None:
        _query_optimizer = QueryOptimizer()
    return _query_optimizer


async def setup_query_optimization(engine: Engine, redis_url: Optional[str] = None):
    """Set up query optimization for database engine."""
    optimizer = get_query_optimizer()
    
    if redis_url:
        optimizer.redis_url = redis_url
        
    await optimizer.initialize()
    optimizer.setup_query_monitoring(engine)
    
    logger.info("Database query optimization enabled")
    return optimizer