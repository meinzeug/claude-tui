"""
Database Performance Benchmark Suite - Production Performance Analysis

Comprehensive performance testing and benchmarking system providing:
- Automated performance testing with configurable load patterns
- Database operation benchmarks (CRUD, complex queries, transactions)
- Connection pool performance analysis
- Cache effectiveness measurement
- Read replica performance testing
- Performance regression detection
- Detailed reporting with recommendations
"""

import asyncio
import time
import statistics
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import json
import random
import string
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.logger import get_logger
from ..core.exceptions import ClaudeTUIException
from .service import DatabaseService
from .query_optimizer import get_query_optimizer
from .caching import get_database_cache
from .connection_pool import get_connection_pool_manager
from .read_replica_manager import get_read_replica_manager

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Individual benchmark test result."""
    test_name: str
    operations_per_second: float
    average_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    success_rate: float
    total_operations: int
    duration_seconds: float
    error_count: int = 0
    errors: List[str] = field(default_factory=list)
    
    def get_performance_grade(self) -> str:
        """Calculate performance grade based on metrics."""
        score = 100
        
        # Penalize high latency
        if self.average_latency_ms > 100:
            score -= min(50, (self.average_latency_ms - 100) / 10)
        
        # Penalize low throughput
        if self.operations_per_second < 100:
            score -= min(30, (100 - self.operations_per_second) / 5)
        
        # Penalize low success rate
        if self.success_rate < 0.99:
            score -= (1 - self.success_rate) * 100
        
        # Penalize high p99 latency
        if self.p99_latency_ms > 1000:
            score -= min(20, (self.p99_latency_ms - 1000) / 100)
        
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


@dataclass
class LoadPattern:
    """Load pattern configuration for benchmark tests."""
    concurrent_users: int
    operations_per_user: int
    ramp_up_seconds: int = 5
    steady_state_seconds: int = 30
    ramp_down_seconds: int = 5
    think_time_seconds: float = 0.1


class BenchmarkTest:
    """Base class for benchmark tests."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    async def setup(self, session: AsyncSession) -> bool:
        """Set up test prerequisites."""
        return True
    
    async def execute_operation(self, session: AsyncSession, user_id: int, operation_id: int) -> bool:
        """Execute single benchmark operation."""
        try:
            # Generate random test data
            test_data = {
                'user_id': user_id,
                'operation_id': operation_id,
                'data': ''.join(random.choices(string.ascii_letters + string.digits, k=100)),
                'timestamp': datetime.utcnow(),
                'value': random.uniform(0, 1000)
            }
            
            # Execute different types of operations based on operation_id
            operation_type = operation_id % 4
            
            if operation_type == 0:  # INSERT
                await session.execute(
                    text("INSERT INTO test_performance (user_id, data, timestamp, value) VALUES (:user_id, :data, :timestamp, :value) ON CONFLICT DO NOTHING"),
                    test_data
                )
            elif operation_type == 1:  # SELECT
                result = await session.execute(
                    text("SELECT * FROM test_performance WHERE user_id = :user_id LIMIT 10"),
                    {'user_id': user_id}
                )
                _ = result.fetchall()
            elif operation_type == 2:  # UPDATE
                await session.execute(
                    text("UPDATE test_performance SET value = :value WHERE user_id = :user_id AND id = (SELECT id FROM test_performance WHERE user_id = :user_id LIMIT 1)"),
                    {'user_id': user_id, 'value': test_data['value']}
                )
            else:  # DELETE
                await session.execute(
                    text("DELETE FROM test_performance WHERE user_id = :user_id AND id = (SELECT id FROM test_performance WHERE user_id = :user_id ORDER BY timestamp DESC LIMIT 1)"),
                    {'user_id': user_id}
                )
            
            await session.commit()
            return True
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Benchmark operation {operation_id} failed for user {user_id}: {e}")
            return False
    
    async def cleanup(self, session: AsyncSession) -> bool:
        """Clean up test artifacts."""
        return True


class CRUDBenchmarkTest(BenchmarkTest):
    """CRUD operations benchmark test."""
    
    def __init__(self):
        super().__init__(
            "CRUD Operations",
            "Basic Create, Read, Update, Delete operations performance"
        )
        self.test_data_ids: List[str] = []
    
    async def setup(self, session: AsyncSession) -> bool:
        """Set up test data."""
        try:
            # Create test table if not exists (simplified)
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS benchmark_test_data (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100),
                    data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            await session.commit()
            return True
        except Exception as e:
            logger.error(f"CRUD benchmark setup failed: {e}")
            return False
    
    async def execute_operation(self, session: AsyncSession, user_id: int, operation_id: int) -> bool:
        """Execute CRUD operation cycle."""
        try:
            # Generate random test data
            name = f"test_user_{user_id}_{operation_id}"
            data = ''.join(random.choices(string.ascii_letters + string.digits, k=100))
            
            # CREATE
            result = await session.execute(text("""
                INSERT INTO benchmark_test_data (name, data) 
                VALUES (:name, :data) RETURNING id
            """), {"name": name, "data": data})
            
            row = result.fetchone()
            if not row:
                return False
            
            record_id = row[0]
            
            # READ
            result = await session.execute(text("""
                SELECT id, name, data FROM benchmark_test_data WHERE id = :id
            """), {"id": record_id})
            
            if not result.fetchone():
                return False
            
            # UPDATE
            new_data = data + "_updated"
            await session.execute(text("""
                UPDATE benchmark_test_data 
                SET data = :data, updated_at = CURRENT_TIMESTAMP 
                WHERE id = :id
            """), {"id": record_id, "data": new_data})
            
            # DELETE
            await session.execute(text("""
                DELETE FROM benchmark_test_data WHERE id = :id
            """), {"id": record_id})
            
            await session.commit()
            return True
            
        except Exception as e:
            logger.debug(f"CRUD operation failed: {e}")
            return False
    
    async def cleanup(self, session: AsyncSession) -> bool:
        """Clean up test data."""
        try:
            await session.execute(text("DROP TABLE IF EXISTS benchmark_test_data"))
            await session.commit()
            return True
        except Exception as e:
            logger.error(f"CRUD benchmark cleanup failed: {e}")
            return False


class ComplexQueryBenchmarkTest(BenchmarkTest):
    """Complex query performance benchmark."""
    
    def __init__(self):
        super().__init__(
            "Complex Queries",
            "Complex JOIN and aggregation query performance"
        )
    
    async def setup(self, session: AsyncSession) -> bool:
        """Set up test data for complex queries."""
        try:
            # Create test tables
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS bench_users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50),
                    email VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS bench_orders (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER,
                    amount DECIMAL(10,2),
                    status VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Insert sample data
            for i in range(1000):
                await session.execute(text("""
                    INSERT INTO bench_users (username, email) 
                    VALUES (:username, :email)
                """), {
                    "username": f"user_{i}",
                    "email": f"user_{i}@test.com"
                })
            
            for i in range(5000):
                await session.execute(text("""
                    INSERT INTO bench_orders (user_id, amount, status) 
                    VALUES (:user_id, :amount, :status)
                """), {
                    "user_id": random.randint(1, 1000),
                    "amount": round(random.uniform(10.0, 1000.0), 2),
                    "status": random.choice(['pending', 'completed', 'cancelled'])
                })
            
            await session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Complex query benchmark setup failed: {e}")
            return False
    
    async def execute_operation(self, session: AsyncSession, user_id: int, operation_id: int) -> bool:
        """Execute complex query."""
        try:
            # Complex aggregation query with JOIN
            result = await session.execute(text("""
                SELECT 
                    u.username,
                    COUNT(o.id) as order_count,
                    SUM(o.amount) as total_amount,
                    AVG(o.amount) as avg_amount,
                    MAX(o.created_at) as last_order
                FROM bench_users u
                LEFT JOIN bench_orders o ON u.id = o.user_id
                WHERE u.created_at > CURRENT_TIMESTAMP - INTERVAL '30 days'
                GROUP BY u.id, u.username
                HAVING COUNT(o.id) > 0
                ORDER BY total_amount DESC
                LIMIT 100
            """))
            
            rows = result.fetchall()
            return len(rows) > 0
            
        except Exception as e:
            logger.debug(f"Complex query operation failed: {e}")
            return False
    
    async def cleanup(self, session: AsyncSession) -> bool:
        """Clean up test tables."""
        try:
            await session.execute(text("DROP TABLE IF EXISTS bench_orders"))
            await session.execute(text("DROP TABLE IF EXISTS bench_users"))
            await session.commit()
            return True
        except Exception as e:
            logger.error(f"Complex query benchmark cleanup failed: {e}")
            return False


class TransactionBenchmarkTest(BenchmarkTest):
    """Transaction performance benchmark."""
    
    def __init__(self):
        super().__init__(
            "Transactions",
            "Transaction isolation and performance testing"
        )
    
    async def setup(self, session: AsyncSession) -> bool:
        """Set up transaction test data."""
        try:
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS bench_accounts (
                    id SERIAL PRIMARY KEY,
                    account_number VARCHAR(20),
                    balance DECIMAL(15,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Create test accounts
            for i in range(100):
                await session.execute(text("""
                    INSERT INTO bench_accounts (account_number, balance) 
                    VALUES (:account_number, :balance)
                """), {
                    "account_number": f"ACC{i:06d}",
                    "balance": 1000.00
                })
            
            await session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Transaction benchmark setup failed: {e}")
            return False
    
    async def execute_operation(self, session: AsyncSession, user_id: int, operation_id: int) -> bool:
        """Execute transaction operations."""
        try:
            # Simulate money transfer between random accounts
            from_account = random.randint(1, 100)
            to_account = random.randint(1, 100)
            
            if from_account == to_account:
                to_account = (from_account % 100) + 1
            
            amount = round(random.uniform(1.0, 100.0), 2)
            
            async with session.begin():
                # Debit from account
                result = await session.execute(text("""
                    UPDATE bench_accounts 
                    SET balance = balance - :amount 
                    WHERE id = :id AND balance >= :amount
                    RETURNING balance
                """), {"id": from_account, "amount": amount})
                
                if not result.fetchone():
                    await session.rollback()
                    return False
                
                # Credit to account
                await session.execute(text("""
                    UPDATE bench_accounts 
                    SET balance = balance + :amount 
                    WHERE id = :id
                """), {"id": to_account, "amount": amount})
                
                # Simulate some processing time
                await asyncio.sleep(0.001)
            
            return True
            
        except Exception as e:
            logger.debug(f"Transaction operation failed: {e}")
            return False
    
    async def cleanup(self, session: AsyncSession) -> bool:
        """Clean up transaction test data."""
        try:
            await session.execute(text("DROP TABLE IF EXISTS bench_accounts"))
            await session.commit()
            return True
        except Exception as e:
            logger.error(f"Transaction benchmark cleanup failed: {e}")
            return False


class DatabasePerformanceBenchmark:
    """
    Comprehensive database performance benchmark suite.
    
    Features:
    - Multiple benchmark test patterns
    - Configurable load patterns and scenarios
    - Performance regression detection
    - Detailed metrics and reporting
    - Integration with optimization components
    """
    
    def __init__(self, db_service: DatabaseService):
        """
        Initialize performance benchmark suite.
        
        Args:
            db_service: Database service instance
        """
        self.db_service = db_service
        
        # Available benchmark tests
        self.benchmark_tests: Dict[str, BenchmarkTest] = {
            'crud': CRUDBenchmarkTest(),
            'complex_queries': ComplexQueryBenchmarkTest(),
            'transactions': TransactionBenchmarkTest()
        }
        
        # Load patterns
        self.load_patterns: Dict[str, LoadPattern] = {
            'light': LoadPattern(concurrent_users=5, operations_per_user=20),
            'medium': LoadPattern(concurrent_users=20, operations_per_user=50),
            'heavy': LoadPattern(concurrent_users=50, operations_per_user=100),
            'stress': LoadPattern(concurrent_users=100, operations_per_user=200)
        }
        
        # Benchmark history for regression detection
        self.benchmark_history: List[Dict[str, Any]] = []
        
        logger.info("Database performance benchmark suite initialized")
    
    async def run_benchmark_suite(
        self,
        tests: Optional[List[str]] = None,
        load_pattern: str = 'medium',
        iterations: int = 1
    ) -> Dict[str, Any]:
        """
        Run comprehensive benchmark suite.
        
        Args:
            tests: List of test names to run (all if None)
            load_pattern: Load pattern to use
            iterations: Number of iterations to run
            
        Returns:
            Comprehensive benchmark results
        """
        test_names = tests or list(self.benchmark_tests.keys())
        pattern = self.load_patterns.get(load_pattern, self.load_patterns['medium'])
        
        suite_results = {
            'suite_start_time': datetime.utcnow().isoformat(),
            'load_pattern': load_pattern,
            'pattern_config': {
                'concurrent_users': pattern.concurrent_users,
                'operations_per_user': pattern.operations_per_user,
                'total_operations': pattern.concurrent_users * pattern.operations_per_user
            },
            'iterations': iterations,
            'test_results': {},
            'system_metrics': {},
            'performance_summary': {},
            'recommendations': []
        }
        
        logger.info(f"Starting benchmark suite with {len(test_names)} tests, {load_pattern} load pattern")
        
        # Collect baseline system metrics
        suite_results['system_metrics']['baseline'] = await self._collect_system_metrics()
        
        # Run each benchmark test
        for test_name in test_names:
            if test_name not in self.benchmark_tests:
                logger.warning(f"Unknown benchmark test: {test_name}")
                continue
            
            logger.info(f"Running benchmark test: {test_name}")
            
            test_results = []
            for iteration in range(iterations):
                logger.info(f"Iteration {iteration + 1}/{iterations} for {test_name}")
                
                result = await self._run_single_benchmark(
                    self.benchmark_tests[test_name],
                    pattern
                )
                test_results.append(result)
            
            # Calculate aggregate results
            suite_results['test_results'][test_name] = self._aggregate_test_results(test_results)
        
        # Collect final system metrics
        suite_results['system_metrics']['final'] = await self._collect_system_metrics()
        
        # Generate performance summary and recommendations
        suite_results['performance_summary'] = self._generate_performance_summary(suite_results)
        suite_results['recommendations'] = await self._generate_recommendations(suite_results)
        
        # Store results for regression detection
        self.benchmark_history.append({
            'timestamp': datetime.utcnow(),
            'results': suite_results,
            'load_pattern': load_pattern
        })
        
        suite_results['suite_end_time'] = datetime.utcnow().isoformat()
        logger.info("Benchmark suite completed")
        
        return suite_results
    
    async def _run_single_benchmark(self, test: BenchmarkTest, pattern: LoadPattern) -> BenchmarkResult:
        """Run single benchmark test with specified load pattern."""
        start_time = time.time()
        latencies: List[float] = []
        errors: List[str] = []
        successful_operations = 0
        total_operations = pattern.concurrent_users * pattern.operations_per_user
        
        # Test setup
        async with self.db_service.get_session() as setup_session:
            setup_success = await test.setup(setup_session)
            if not setup_success:
                return BenchmarkResult(
                    test_name=test.name,
                    operations_per_second=0.0,
                    average_latency_ms=0.0,
                    min_latency_ms=0.0,
                    max_latency_ms=0.0,
                    p95_latency_ms=0.0,
                    p99_latency_ms=0.0,
                    success_rate=0.0,
                    total_operations=0,
                    duration_seconds=0.0,
                    error_count=1,
                    errors=["Test setup failed"]
                )
        
        try:
            # Create semaphore to control concurrency
            semaphore = asyncio.Semaphore(pattern.concurrent_users)
            
            # Execute benchmark operations
            tasks = []
            for user_id in range(pattern.concurrent_users):
                task = asyncio.create_task(
                    self._execute_user_operations(test, user_id, pattern, semaphore, latencies, errors)
                )
                tasks.append(task)
            
            # Wait for all operations to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successful operations
            for result in results:
                if isinstance(result, int):
                    successful_operations += result
                elif isinstance(result, Exception):
                    errors.append(str(result))
        
        finally:
            # Test cleanup
            try:
                async with self.db_service.get_session() as cleanup_session:
                    await test.cleanup(cleanup_session)
            except Exception as e:
                logger.error(f"Test cleanup failed: {e}")
        
        # Calculate metrics
        duration = time.time() - start_time
        ops_per_second = successful_operations / max(duration, 0.001)
        success_rate = successful_operations / max(total_operations, 1)
        
        # Calculate latency statistics
        if latencies:
            latencies.sort()
            avg_latency = statistics.mean(latencies) * 1000  # Convert to ms
            min_latency = min(latencies) * 1000
            max_latency = max(latencies) * 1000
            p95_latency = latencies[int(len(latencies) * 0.95)] * 1000
            p99_latency = latencies[int(len(latencies) * 0.99)] * 1000
        else:
            avg_latency = min_latency = max_latency = p95_latency = p99_latency = 0.0
        
        return BenchmarkResult(
            test_name=test.name,
            operations_per_second=ops_per_second,
            average_latency_ms=avg_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            success_rate=success_rate,
            total_operations=total_operations,
            duration_seconds=duration,
            error_count=len(errors),
            errors=errors[:10]  # Keep only first 10 errors
        )
    
    async def _execute_user_operations(
        self, 
        test: BenchmarkTest, 
        user_id: int, 
        pattern: LoadPattern,
        semaphore: asyncio.Semaphore,
        latencies: List[float],
        errors: List[str]
    ) -> int:
        """Execute operations for a single user."""
        successful_ops = 0
        
        async with semaphore:
            for operation_id in range(pattern.operations_per_user):
                try:
                    operation_start = time.time()
                    
                    async with self.db_service.get_session() as session:
                        success = await test.execute_operation(session, user_id, operation_id)
                    
                    operation_time = time.time() - operation_start
                    latencies.append(operation_time)
                    
                    if success:
                        successful_ops += 1
                    else:
                        errors.append(f"Operation failed for user {user_id}, op {operation_id}")
                    
                    # Think time between operations
                    if pattern.think_time_seconds > 0:
                        await asyncio.sleep(pattern.think_time_seconds)
                
                except Exception as e:
                    errors.append(f"Exception for user {user_id}, op {operation_id}: {str(e)}")
        
        return successful_ops
    
    def _aggregate_test_results(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Aggregate results from multiple test iterations."""
        if not results:
            return {}
        
        return {
            'iterations': len(results),
            'average_ops_per_second': statistics.mean([r.operations_per_second for r in results]),
            'best_ops_per_second': max([r.operations_per_second for r in results]),
            'worst_ops_per_second': min([r.operations_per_second for r in results]),
            'average_latency_ms': statistics.mean([r.average_latency_ms for r in results]),
            'best_latency_ms': min([r.average_latency_ms for r in results]),
            'worst_latency_ms': max([r.average_latency_ms for r in results]),
            'average_p95_latency_ms': statistics.mean([r.p95_latency_ms for r in results]),
            'average_p99_latency_ms': statistics.mean([r.p99_latency_ms for r in results]),
            'average_success_rate': statistics.mean([r.success_rate for r in results]),
            'total_errors': sum([r.error_count for r in results]),
            'performance_grade': statistics.mode([r.get_performance_grade() for r in results]),
            'consistency_score': self._calculate_consistency_score(results),
            'detailed_results': [
                {
                    'ops_per_second': r.operations_per_second,
                    'avg_latency_ms': r.average_latency_ms,
                    'p95_latency_ms': r.p95_latency_ms,
                    'success_rate': r.success_rate,
                    'grade': r.get_performance_grade()
                }
                for r in results
            ]
        }
    
    def _calculate_consistency_score(self, results: List[BenchmarkResult]) -> float:
        """Calculate performance consistency score (0-100)."""
        if len(results) < 2:
            return 100.0
        
        ops_per_second = [r.operations_per_second for r in results]
        latencies = [r.average_latency_ms for r in results]
        
        # Calculate coefficient of variation for throughput and latency
        ops_cv = statistics.stdev(ops_per_second) / max(statistics.mean(ops_per_second), 1)
        latency_cv = statistics.stdev(latencies) / max(statistics.mean(latencies), 1)
        
        # Convert to consistency score (lower variation = higher consistency)
        consistency = max(0, 100 - (ops_cv + latency_cv) * 100)
        return round(consistency, 2)
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level performance metrics."""
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'database_service': {},
            'query_optimizer': {},
            'cache': {},
            'connection_pool': {},
            'read_replicas': {}
        }
        
        try:
            # Database service metrics
            db_info = await self.db_service.get_database_info()
            metrics['database_service'] = db_info
        except Exception as e:
            logger.debug(f"Failed to collect database metrics: {e}")
        
        try:
            # Query optimizer metrics
            optimizer = get_query_optimizer()
            if optimizer:
                metrics['query_optimizer'] = optimizer.get_performance_metrics()
        except Exception as e:
            logger.debug(f"Failed to collect optimizer metrics: {e}")
        
        try:
            # Cache metrics
            cache = get_database_cache()
            if cache:
                metrics['cache'] = await cache.get_cache_stats()
        except Exception as e:
            logger.debug(f"Failed to collect cache metrics: {e}")
        
        try:
            # Connection pool metrics
            pool_manager = get_connection_pool_manager()
            if pool_manager:
                metrics['connection_pool'] = await pool_manager.get_pool_statistics()
        except Exception as e:
            logger.debug(f"Failed to collect pool metrics: {e}")
        
        try:
            # Read replica metrics
            replica_manager = get_read_replica_manager()
            if replica_manager:
                metrics['read_replicas'] = replica_manager.get_performance_metrics()
        except Exception as e:
            logger.debug(f"Failed to collect replica metrics: {e}")
        
        return metrics
    
    def _generate_performance_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance summary from benchmark results."""
        test_results = results['test_results']
        
        if not test_results:
            return {'overall_grade': 'F', 'summary': 'No test results available'}
        
        # Calculate overall performance metrics
        avg_ops_per_second = statistics.mean([
            data['average_ops_per_second'] 
            for data in test_results.values() 
            if 'average_ops_per_second' in data
        ])
        
        avg_latency_ms = statistics.mean([
            data['average_latency_ms'] 
            for data in test_results.values() 
            if 'average_latency_ms' in data
        ])
        
        avg_success_rate = statistics.mean([
            data['average_success_rate'] 
            for data in test_results.values() 
            if 'average_success_rate' in data
        ])
        
        # Calculate overall grade
        overall_score = 100
        
        if avg_ops_per_second < 50:
            overall_score -= 30
        elif avg_ops_per_second < 100:
            overall_score -= 15
        
        if avg_latency_ms > 100:
            overall_score -= min(25, (avg_latency_ms - 100) / 10)
        
        if avg_success_rate < 0.95:
            overall_score -= (1 - avg_success_rate) * 100
        
        grades = [data.get('performance_grade', 'F') for data in test_results.values()]
        grade_scores = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
        avg_grade_score = statistics.mean([grade_scores.get(g, 0) for g in grades])
        
        if avg_grade_score >= 3.5:
            overall_grade = 'A'
        elif avg_grade_score >= 2.5:
            overall_grade = 'B'
        elif avg_grade_score >= 1.5:
            overall_grade = 'C'
        elif avg_grade_score >= 0.5:
            overall_grade = 'D'
        else:
            overall_grade = 'F'
        
        return {
            'overall_grade': overall_grade,
            'overall_score': max(0, overall_score),
            'average_throughput_ops_per_second': round(avg_ops_per_second, 2),
            'average_latency_ms': round(avg_latency_ms, 2),
            'average_success_rate': round(avg_success_rate * 100, 2),
            'test_grades': {name: data.get('performance_grade', 'F') for name, data in test_results.items()},
            'performance_characteristics': {
                'throughput_category': self._categorize_throughput(avg_ops_per_second),
                'latency_category': self._categorize_latency(avg_latency_ms),
                'reliability_category': self._categorize_reliability(avg_success_rate)
            }
        }
    
    def _categorize_throughput(self, ops_per_second: float) -> str:
        """Categorize throughput performance."""
        if ops_per_second >= 500:
            return 'Excellent'
        elif ops_per_second >= 200:
            return 'Good'
        elif ops_per_second >= 100:
            return 'Fair'
        elif ops_per_second >= 50:
            return 'Poor'
        else:
            return 'Critical'
    
    def _categorize_latency(self, avg_latency_ms: float) -> str:
        """Categorize latency performance."""
        if avg_latency_ms <= 10:
            return 'Excellent'
        elif avg_latency_ms <= 50:
            return 'Good'
        elif avg_latency_ms <= 100:
            return 'Fair'
        elif avg_latency_ms <= 500:
            return 'Poor'
        else:
            return 'Critical'
    
    def _categorize_reliability(self, success_rate: float) -> str:
        """Categorize reliability performance."""
        if success_rate >= 0.99:
            return 'Excellent'
        elif success_rate >= 0.95:
            return 'Good'
        elif success_rate >= 0.90:
            return 'Fair'
        elif success_rate >= 0.80:
            return 'Poor'
        else:
            return 'Critical'
    
    async def _generate_recommendations(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on benchmark results."""
        recommendations = []
        test_results = results['test_results']
        system_metrics = results['system_metrics']
        
        # Analyze throughput issues
        low_throughput_tests = [
            name for name, data in test_results.items()
            if data.get('average_ops_per_second', 0) < 100
        ]
        
        if low_throughput_tests:
            recommendations.append({
                'type': 'performance',
                'priority': 'high',
                'title': 'Low Throughput Detected',
                'description': f'Tests with low throughput: {", ".join(low_throughput_tests)}',
                'suggestions': [
                    'Consider increasing connection pool size',
                    'Review query optimization opportunities',
                    'Enable query result caching',
                    'Consider read replica scaling'
                ]
            })
        
        # Analyze latency issues
        high_latency_tests = [
            name for name, data in test_results.items()
            if data.get('average_latency_ms', 0) > 100
        ]
        
        if high_latency_tests:
            recommendations.append({
                'type': 'performance',
                'priority': 'high',
                'title': 'High Latency Detected',
                'description': f'Tests with high latency: {", ".join(high_latency_tests)}',
                'suggestions': [
                    'Add database indexes for frequently queried columns',
                    'Optimize slow queries using EXPLAIN ANALYZE',
                    'Consider database connection pooling optimization',
                    'Review application query patterns'
                ]
            })
        
        # Analyze cache effectiveness
        cache_metrics = system_metrics.get('final', {}).get('cache', {})
        if cache_metrics and cache_metrics.get('hit_ratio', 0) < 0.8:
            recommendations.append({
                'type': 'caching',
                'priority': 'medium',
                'title': 'Low Cache Hit Ratio',
                'description': f'Cache hit ratio: {cache_metrics.get("hit_ratio", 0):.2%}',
                'suggestions': [
                    'Review cache configuration and TTL settings',
                    'Implement cache warming for frequently accessed data',
                    'Consider increasing cache size',
                    'Optimize cache key strategies'
                ]
            })
        
        # Analyze connection pool utilization
        pool_metrics = system_metrics.get('final', {}).get('connection_pool', {}).get('pool_status', {})
        utilization = pool_metrics.get('utilization_ratio', 0)
        
        if utilization > 0.8:
            recommendations.append({
                'type': 'infrastructure',
                'priority': 'medium',
                'title': 'High Connection Pool Utilization',
                'description': f'Pool utilization: {utilization:.2%}',
                'suggestions': [
                    'Increase connection pool size',
                    'Review connection timeout settings',
                    'Consider connection multiplexing',
                    'Optimize query execution time'
                ]
            })
        
        # Consistency issues
        inconsistent_tests = [
            name for name, data in test_results.items()
            if data.get('consistency_score', 100) < 80
        ]
        
        if inconsistent_tests:
            recommendations.append({
                'type': 'reliability',
                'priority': 'medium',
                'title': 'Performance Inconsistency',
                'description': f'Tests with inconsistent performance: {", ".join(inconsistent_tests)}',
                'suggestions': [
                    'Review system resource utilization',
                    'Check for background processes affecting performance',
                    'Consider database maintenance scheduling',
                    'Monitor system-level bottlenecks'
                ]
            })
        
        return recommendations
    
    async def detect_performance_regression(self, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect performance regressions compared to historical data."""
        if not self.benchmark_history:
            return {'regression_detected': False, 'message': 'No historical data available'}
        
        # Get most recent comparable benchmark
        recent_benchmark = None
        for historical in reversed(self.benchmark_history):
            if historical['load_pattern'] == current_results.get('load_pattern'):
                recent_benchmark = historical
                break
        
        if not recent_benchmark:
            return {'regression_detected': False, 'message': 'No comparable historical data found'}
        
        regression_analysis = {
            'regression_detected': False,
            'regressions': [],
            'improvements': [],
            'comparison_date': recent_benchmark['timestamp'].isoformat(),
            'overall_change': {}
        }
        
        current_tests = current_results.get('test_results', {})
        historical_tests = recent_benchmark['results'].get('test_results', {})
        
        # Compare each test
        for test_name in current_tests:
            if test_name not in historical_tests:
                continue
            
            current_data = current_tests[test_name]
            historical_data = historical_tests[test_name]
            
            # Compare key metrics
            current_ops = current_data.get('average_ops_per_second', 0)
            historical_ops = historical_data.get('average_ops_per_second', 1)
            ops_change = (current_ops - historical_ops) / historical_ops
            
            current_latency = current_data.get('average_latency_ms', 0)
            historical_latency = historical_data.get('average_latency_ms', 1)
            latency_change = (current_latency - historical_latency) / historical_latency
            
            # Detect significant changes (>10% degradation)
            if ops_change < -0.1 or latency_change > 0.1:
                regression_analysis['regression_detected'] = True
                regression_analysis['regressions'].append({
                    'test_name': test_name,
                    'throughput_change': f'{ops_change:.1%}',
                    'latency_change': f'{latency_change:.1%}',
                    'severity': 'high' if (ops_change < -0.2 or latency_change > 0.2) else 'medium'
                })
            
            # Detect improvements
            elif ops_change > 0.1 or latency_change < -0.1:
                regression_analysis['improvements'].append({
                    'test_name': test_name,
                    'throughput_change': f'{ops_change:.1%}',
                    'latency_change': f'{latency_change:.1%}'
                })
        
        return regression_analysis
    
    def get_benchmark_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get benchmark history for analysis."""
        return [
            {
                'timestamp': entry['timestamp'].isoformat(),
                'load_pattern': entry['load_pattern'],
                'overall_grade': entry['results'].get('performance_summary', {}).get('overall_grade', 'N/A'),
                'average_throughput': entry['results'].get('performance_summary', {}).get('average_throughput_ops_per_second', 0),
                'average_latency': entry['results'].get('performance_summary', {}).get('average_latency_ms', 0)
            }
            for entry in self.benchmark_history[-limit:]
        ]
    
    async def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive performance report."""
        summary = results.get('performance_summary', {})
        recommendations = results.get('recommendations', [])
        
        report_lines = [
            "# Database Performance Benchmark Report",
            f"Generated: {results.get('suite_start_time', 'Unknown')}",
            f"Load Pattern: {results.get('load_pattern', 'Unknown')}",
            "",
            "## Executive Summary",
            f"Overall Performance Grade: **{summary.get('overall_grade', 'N/A')}**",
            f"Average Throughput: **{summary.get('average_throughput_ops_per_second', 0):.1f} ops/sec**",
            f"Average Latency: **{summary.get('average_latency_ms', 0):.1f} ms**",
            f"Success Rate: **{summary.get('average_success_rate', 0):.1f}%**",
            ""
        ]
        
        # Test results
        report_lines.extend([
            "## Test Results",
            ""
        ])
        
        for test_name, test_data in results.get('test_results', {}).items():
            grade = test_data.get('performance_grade', 'N/A')
            ops = test_data.get('average_ops_per_second', 0)
            latency = test_data.get('average_latency_ms', 0)
            
            report_lines.extend([
                f"### {test_name}",
                f"- Grade: **{grade}**",
                f"- Throughput: {ops:.1f} ops/sec",
                f"- Latency: {latency:.1f} ms",
                f"- P95 Latency: {test_data.get('average_p95_latency_ms', 0):.1f} ms",
                ""
            ])
        
        # Recommendations
        if recommendations:
            report_lines.extend([
                "## Recommendations",
                ""
            ])
            
            for i, rec in enumerate(recommendations, 1):
                report_lines.extend([
                    f"### {i}. {rec['title']} ({rec['priority'].upper()} Priority)",
                    f"{rec['description']}",
                    "",
                    "**Suggestions:**"
                ])
                
                for suggestion in rec.get('suggestions', []):
                    report_lines.append(f"- {suggestion}")
                
                report_lines.append("")
        
        return "\n".join(report_lines)


# Global benchmark instance
_performance_benchmark: Optional[DatabasePerformanceBenchmark] = None


def get_performance_benchmark() -> Optional[DatabasePerformanceBenchmark]:
    """Get global performance benchmark instance."""
    return _performance_benchmark


async def setup_performance_benchmark(db_service: DatabaseService) -> DatabasePerformanceBenchmark:
    """Set up database performance benchmark suite."""
    global _performance_benchmark
    
    _performance_benchmark = DatabasePerformanceBenchmark(db_service)
    
    logger.info("Database performance benchmark suite enabled")
    return _performance_benchmark