"""
Performance Benchmarks for claude-tui.

Comprehensive performance benchmarks including:
- Authentication performance benchmarks
- Database operation benchmarks
- AI service response time benchmarks
- Memory usage benchmarks
- Concurrent operation benchmarks
- API endpoint benchmarks
"""

import asyncio
import pytest
import time
import statistics
import psutil
import os
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import pytest_benchmark

from src.auth.jwt_auth import JWTAuthenticator
from src.services.ai_service import AIService
from src.database.models import User
from tests.fixtures.external_service_mocks import (
    MockClaudeCodeIntegration,
    MockClaudeFlowIntegration
)


class BenchmarkMetrics:
    """Utility class for collecting benchmark metrics."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.start_memory = None
        self.measurements = []
    
    def start_measurement(self):
        """Start measuring performance."""
        self.start_time = time.perf_counter()
        self.start_memory = self.process.memory_info().rss
    
    def record_measurement(self, operation: str, additional_data: Dict = None):
        """Record a measurement point."""
        current_time = time.perf_counter()
        current_memory = self.process.memory_info().rss
        
        measurement = {
            "operation": operation,
            "timestamp": current_time,
            "duration_since_start": current_time - (self.start_time or current_time),
            "memory_mb": current_memory / 1024 / 1024,
            "memory_delta_mb": (current_memory - (self.start_memory or current_memory)) / 1024 / 1024,
            **(additional_data or {})
        }
        
        self.measurements.append(measurement)
        return measurement
    
    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark summary."""
        if not self.measurements:
            return {}
        
        durations = [m["duration_since_start"] for m in self.measurements]
        memory_usage = [m["memory_mb"] for m in self.measurements]
        
        return {
            "total_measurements": len(self.measurements),
            "total_duration": max(durations) if durations else 0,
            "avg_memory_usage_mb": statistics.mean(memory_usage),
            "peak_memory_mb": max(memory_usage),
            "measurements": self.measurements
        }


@pytest.fixture
def benchmark_metrics():
    """Provide benchmark metrics collector."""
    return BenchmarkMetrics()


@pytest.fixture
def jwt_authenticator_benchmark():
    """JWT authenticator optimized for benchmarking."""
    return JWTAuthenticator(
        secret_key="benchmark-secret-key",
        access_token_expire_minutes=60
    )


@pytest.fixture
def sample_users_benchmark():
    """Create sample users for benchmarking."""
    users = []
    for i in range(100):
        user = User(
            id=f"benchmark_user_{i}",
            email=f"benchmark{i}@test.com",
            username=f"benchmark{i}",
            full_name=f"Benchmark User {i}",
            is_active=True,
            is_verified=True
        )
        user.set_password("BenchmarkPassword123!")
        users.append(user)
    return users


@pytest.mark.benchmark
class TestAuthenticationBenchmarks:
    """Benchmark authentication operations."""
    
    def test_jwt_token_creation_benchmark(self, benchmark, jwt_authenticator_benchmark, sample_users_benchmark):
        """Benchmark JWT token creation performance."""
        user = sample_users_benchmark[0]
        mock_session_repo = AsyncMock()
        mock_session_repo.create_session.return_value = MagicMock()
        
        async def create_token():
            return await jwt_authenticator_benchmark.create_tokens(
                user=user,
                session_repo=mock_session_repo
            )
        
        def sync_wrapper():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(create_token())
            finally:
                loop.close()
        
        result = benchmark(sync_wrapper)
        assert result.access_token is not None
    
    def test_jwt_token_validation_benchmark(self, benchmark, jwt_authenticator_benchmark, sample_users_benchmark):
        """Benchmark JWT token validation performance."""
        user = sample_users_benchmark[0]
        mock_session_repo = AsyncMock()
        mock_session = MagicMock()
        mock_session.is_active = True
        mock_session.is_expired.return_value = False
        mock_session_repo.get_session.return_value = mock_session
        mock_session_repo.create_session.return_value = mock_session
        
        # Pre-create token
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            token_response = loop.run_until_complete(
                jwt_authenticator_benchmark.create_tokens(
                    user=user,
                    session_repo=mock_session_repo
                )
            )
            token = token_response.access_token
        finally:
            loop.close()
        
        async def validate_token():
            return await jwt_authenticator_benchmark.validate_token(
                token=token,
                session_repo=mock_session_repo
            )
        
        def sync_wrapper():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(validate_token())
            finally:
                loop.close()
        
        result = benchmark(sync_wrapper)
        assert result.user_id is not None
    
    def test_password_hashing_benchmark(self, benchmark, jwt_authenticator_benchmark):
        """Benchmark password hashing performance."""
        password = "BenchmarkPassword123!"
        
        def hash_password():
            return jwt_authenticator_benchmark.hash_password(password)
        
        hashed = benchmark(hash_password)
        assert len(hashed) > 20
        assert jwt_authenticator_benchmark.verify_password(password, hashed)
    
    def test_password_verification_benchmark(self, benchmark, jwt_authenticator_benchmark):
        """Benchmark password verification performance."""
        password = "BenchmarkPassword123!"
        hashed = jwt_authenticator_benchmark.hash_password(password)
        
        def verify_password():
            return jwt_authenticator_benchmark.verify_password(password, hashed)
        
        result = benchmark(verify_password)
        assert result is True


@pytest.mark.benchmark
@pytest.mark.asyncio
class TestConcurrentOperationBenchmarks:
    """Benchmark concurrent operations."""
    
    async def test_concurrent_token_creation_benchmark(self, jwt_authenticator_benchmark, sample_users_benchmark, benchmark_metrics):
        """Benchmark concurrent token creation."""
        benchmark_metrics.start_measurement()
        
        mock_session_repo = AsyncMock()
        mock_session_repo.create_session.return_value = MagicMock()
        
        concurrent_users = 50
        users_to_test = sample_users_benchmark[:concurrent_users]
        
        async def create_token(user):
            start_time = time.perf_counter()
            token_response = await jwt_authenticator_benchmark.create_tokens(
                user=user,
                session_repo=mock_session_repo
            )
            end_time = time.perf_counter()
            return end_time - start_time, token_response
        
        start_time = time.perf_counter()
        tasks = [create_token(user) for user in users_to_test]
        results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()
        
        # Analyze results
        durations = [r[0] for r in results]
        tokens = [r[1] for r in results]
        
        total_time = end_time - start_time
        avg_individual_time = statistics.mean(durations)
        throughput = len(users_to_test) / total_time
        
        benchmark_metrics.record_measurement("concurrent_token_creation", {
            "concurrent_operations": concurrent_users,
            "total_time": total_time,
            "avg_individual_time": avg_individual_time,
            "throughput_ops_per_sec": throughput,
            "successful_operations": len([t for t in tokens if t.access_token])
        })
        
        # Assertions
        assert len(tokens) == concurrent_users
        assert all(t.access_token for t in tokens)
        assert throughput > 10  # Should handle at least 10 tokens per second
        
        print(f"Concurrent Token Creation Benchmark:")
        print(f"  Operations: {concurrent_users}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average individual time: {avg_individual_time:.3f}s")
        print(f"  Throughput: {throughput:.1f} ops/sec")
    
    async def test_concurrent_token_validation_benchmark(self, jwt_authenticator_benchmark, sample_users_benchmark, benchmark_metrics):
        """Benchmark concurrent token validation."""
        benchmark_metrics.start_measurement()
        
        mock_session_repo = AsyncMock()
        mock_session = MagicMock()
        mock_session.is_active = True
        mock_session.is_expired.return_value = False
        mock_session_repo.get_session.return_value = mock_session
        mock_session_repo.create_session.return_value = mock_session
        
        # Pre-create tokens
        num_tokens = 20
        tokens = []
        for user in sample_users_benchmark[:num_tokens]:
            token_response = await jwt_authenticator_benchmark.create_tokens(
                user=user,
                session_repo=mock_session_repo
            )
            tokens.append(token_response.access_token)
        
        # Concurrent validation (more validations than tokens)
        concurrent_validations = 100
        
        async def validate_token(token):
            start_time = time.perf_counter()
            result = await jwt_authenticator_benchmark.validate_token(
                token=token,
                session_repo=mock_session_repo
            )
            end_time = time.perf_counter()
            return end_time - start_time, result
        
        start_time = time.perf_counter()
        validation_tasks = []
        for i in range(concurrent_validations):
            token = tokens[i % len(tokens)]
            validation_tasks.append(validate_token(token))
        
        results = await asyncio.gather(*validation_tasks)
        end_time = time.perf_counter()
        
        # Analyze results
        durations = [r[0] for r in results]
        validations = [r[1] for r in results]
        
        total_time = end_time - start_time
        avg_validation_time = statistics.mean(durations)
        throughput = len(validation_tasks) / total_time
        
        benchmark_metrics.record_measurement("concurrent_token_validation", {
            "concurrent_validations": concurrent_validations,
            "unique_tokens": len(tokens),
            "total_time": total_time,
            "avg_validation_time": avg_validation_time,
            "throughput_ops_per_sec": throughput,
            "successful_validations": len([v for v in validations if v])
        })
        
        # Assertions
        assert len(validations) == concurrent_validations
        assert all(v.user_id for v in validations)
        assert throughput > 50  # Should handle at least 50 validations per second
        
        print(f"Concurrent Token Validation Benchmark:")
        print(f"  Validations: {concurrent_validations}")
        print(f"  Unique tokens: {len(tokens)}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average validation time: {avg_validation_time:.3f}s")
        print(f"  Throughput: {throughput:.1f} ops/sec")


@pytest.mark.benchmark
class TestAIServiceBenchmarks:
    """Benchmark AI service operations."""
    
    def test_ai_service_initialization_benchmark(self, benchmark):
        """Benchmark AI service initialization."""
        async def initialize_ai_service():
            with patch('src.services.ai_service.AIInterface'):
                with patch.object(AIService, '_test_claude_code_connection', return_value=True):
                    with patch.object(AIService, '_test_claude_flow_connection', return_value=True):
                        service = AIService()
                        await service.initialize()
                        return service
        
        def sync_wrapper():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(initialize_ai_service())
            finally:
                loop.close()
        
        service = benchmark(sync_wrapper)
        assert service.is_initialized
    
    def test_code_generation_benchmark(self, benchmark):
        """Benchmark code generation performance."""
        mock_claude_code = MockClaudeCodeIntegration(response_delay=0.01)  # Fast for benchmarking
        
        async def generate_code():
            return await mock_claude_code.generate_code(
                prompt="Create a Python function to calculate fibonacci numbers",
                language="python",
                context={"optimization": "performance"}
            )
        
        def sync_wrapper():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(generate_code())
            finally:
                loop.close()
        
        result = benchmark(sync_wrapper)
        assert result["code"] is not None
        assert result["language"] == "python"
    
    def test_task_orchestration_benchmark(self, benchmark):
        """Benchmark task orchestration performance."""
        mock_claude_flow = MockClaudeFlowIntegration(response_delay=0.02)  # Fast for benchmarking
        
        async def orchestrate_task():
            return await mock_claude_flow.orchestrate_task({
                "task": "Build a REST API with authentication",
                "requirements": {"framework": "FastAPI", "database": "PostgreSQL"},
                "agents": ["backend_developer", "database_architect"],
                "strategy": "parallel"
            })
        
        def sync_wrapper():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(orchestrate_task())
            finally:
                loop.close()
        
        result = benchmark(sync_wrapper)
        assert result["task_id"] is not None
        assert len(result["agents"]) > 0


@pytest.mark.benchmark
@pytest.mark.asyncio
class TestMemoryBenchmarks:
    """Benchmark memory usage patterns."""
    
    async def test_memory_usage_during_token_operations(self, jwt_authenticator_benchmark, sample_users_benchmark, benchmark_metrics):
        """Benchmark memory usage during token operations."""
        import gc
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        mock_session_repo = AsyncMock()
        mock_session_repo.create_session.return_value = MagicMock()
        
        # Perform many token operations
        num_operations = 200
        memory_samples = []
        
        benchmark_metrics.start_measurement()
        
        for i in range(num_operations):
            user = sample_users_benchmark[i % len(sample_users_benchmark)]
            
            # Create token
            token_response = await jwt_authenticator_benchmark.create_tokens(
                user=user,
                session_repo=mock_session_repo
            )
            
            # Validate token
            mock_session = MagicMock()
            mock_session.is_active = True
            mock_session.is_expired.return_value = False
            mock_session_repo.get_session.return_value = mock_session
            
            await jwt_authenticator_benchmark.validate_token(
                token=token_response.access_token,
                session_repo=mock_session_repo
            )
            
            # Sample memory usage periodically
            if i % 20 == 0:
                current_memory = process.memory_info().rss
                memory_samples.append(current_memory / 1024 / 1024)
                
                benchmark_metrics.record_measurement(f"token_operation_{i}", {
                    "memory_mb": current_memory / 1024 / 1024,
                    "operations_completed": i + 1
                })
        
        # Force garbage collection and measure final memory
        gc.collect()
        final_memory = process.memory_info().rss
        
        # Analyze memory usage
        memory_increase = (final_memory - initial_memory) / 1024 / 1024
        peak_memory = max(memory_samples) if memory_samples else 0
        
        # Memory usage should be reasonable
        assert memory_increase < 50, f"Memory increase too high: {memory_increase:.2f}MB"
        assert peak_memory < 200, f"Peak memory too high: {peak_memory:.2f}MB"
        
        print(f"Memory Usage Benchmark:")
        print(f"  Operations: {num_operations}")
        print(f"  Memory increase: {memory_increase:.2f}MB")
        print(f"  Peak memory: {peak_memory:.2f}MB")
        print(f"  Memory per operation: {memory_increase/num_operations:.3f}MB")
    
    async def test_memory_efficiency_with_caching(self, benchmark_metrics):
        """Benchmark memory efficiency with caching enabled."""
        # Simulate caching behavior
        cache = {}
        cache_size_limit = 1000
        
        def cached_operation(key: str, value: Any) -> Any:
            if key in cache:
                return cache[key]
            
            # Simulate computation
            result = {"processed": value, "timestamp": time.time()}
            
            # Implement simple LRU eviction
            if len(cache) >= cache_size_limit:
                oldest_key = min(cache.keys(), key=lambda k: cache[k].get("timestamp", 0))
                del cache[oldest_key]
            
            cache[key] = result
            return result
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        benchmark_metrics.start_measurement()
        
        # Perform operations with caching
        num_operations = 2000
        cache_hits = 0
        cache_misses = 0
        
        for i in range(num_operations):
            key = f"operation_{i % 100}"  # Reuse keys to test caching
            
            if key in cache:
                cache_hits += 1
            else:
                cache_misses += 1
            
            result = cached_operation(key, f"data_{i}")
            
            if i % 100 == 0:
                current_memory = process.memory_info().rss
                benchmark_metrics.record_measurement(f"cached_operation_{i}", {
                    "memory_mb": current_memory / 1024 / 1024,
                    "cache_size": len(cache),
                    "cache_hits": cache_hits,
                    "cache_misses": cache_misses
                })
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024
        cache_hit_rate = cache_hits / (cache_hits + cache_misses)
        
        # Verify caching effectiveness
        assert cache_hit_rate > 0.8, f"Cache hit rate too low: {cache_hit_rate:.2%}"
        assert len(cache) <= cache_size_limit, f"Cache exceeded size limit: {len(cache)}"
        assert memory_increase < 20, f"Memory increase with caching too high: {memory_increase:.2f}MB"
        
        print(f"Memory Caching Benchmark:")
        print(f"  Operations: {num_operations}")
        print(f"  Cache hit rate: {cache_hit_rate:.2%}")
        print(f"  Cache size: {len(cache)}")
        print(f"  Memory increase: {memory_increase:.2f}MB")


@pytest.mark.benchmark
class TestThroughputBenchmarks:
    """Benchmark system throughput under various conditions."""
    
    def test_authentication_throughput_benchmark(self, benchmark, jwt_authenticator_benchmark, sample_users_benchmark):
        """Benchmark authentication throughput."""
        user = sample_users_benchmark[0]
        mock_session_repo = AsyncMock()
        mock_session = MagicMock()
        mock_session.is_active = True
        mock_session.is_expired.return_value = False
        mock_session_repo.get_session.return_value = mock_session
        mock_session_repo.create_session.return_value = mock_session
        
        operations_per_round = 10
        
        async def authentication_round():
            """Perform one round of authentication operations."""
            # Create tokens
            token_creation_tasks = []
            for i in range(operations_per_round):
                task = jwt_authenticator_benchmark.create_tokens(
                    user=user,
                    session_repo=mock_session_repo
                )
                token_creation_tasks.append(task)
            
            token_responses = await asyncio.gather(*token_creation_tasks)
            
            # Validate tokens
            validation_tasks = []
            for token_response in token_responses:
                task = jwt_authenticator_benchmark.validate_token(
                    token=token_response.access_token,
                    session_repo=mock_session_repo
                )
                validation_tasks.append(task)
            
            validation_results = await asyncio.gather(*validation_tasks)
            
            return len(token_responses), len(validation_results)
        
        def sync_wrapper():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(authentication_round())
            finally:
                loop.close()
        
        tokens_created, tokens_validated = benchmark(sync_wrapper)
        
        # Calculate throughput
        execution_time = benchmark.stats['mean']
        total_operations = tokens_created + tokens_validated
        throughput = total_operations / execution_time
        
        assert tokens_created == operations_per_round
        assert tokens_validated == operations_per_round
        assert throughput > 20  # Should handle at least 20 operations per second
        
        print(f"Authentication Throughput Benchmark:")
        print(f"  Operations per round: {operations_per_round * 2}")
        print(f"  Execution time: {execution_time:.3f}s")
        print(f"  Throughput: {throughput:.1f} ops/sec")


@pytest.mark.benchmark
@pytest.mark.parametrize("concurrency_level", [1, 5, 10, 25, 50])
class TestScalabilityBenchmarks:
    """Benchmark system scalability with different concurrency levels."""
    
    @pytest.mark.asyncio
    async def test_scalability_with_concurrency(self, concurrency_level, jwt_authenticator_benchmark, sample_users_benchmark, benchmark_metrics):
        """Test system scalability with different concurrency levels."""
        benchmark_metrics.start_measurement()
        
        mock_session_repo = AsyncMock()
        mock_session = MagicMock()
        mock_session.is_active = True
        mock_session.is_expired.return_value = False
        mock_session_repo.get_session.return_value = mock_session
        mock_session_repo.create_session.return_value = mock_session
        
        users_to_test = sample_users_benchmark[:concurrency_level]
        
        async def process_user(user):
            """Process authentication for one user."""
            start_time = time.perf_counter()
            
            # Create token
            token_response = await jwt_authenticator_benchmark.create_tokens(
                user=user,
                session_repo=mock_session_repo
            )
            
            # Validate token
            validation_result = await jwt_authenticator_benchmark.validate_token(
                token=token_response.access_token,
                session_repo=mock_session_repo
            )
            
            end_time = time.perf_counter()
            return end_time - start_time, token_response, validation_result
        
        # Execute with specified concurrency
        start_time = time.perf_counter()
        tasks = [process_user(user) for user in users_to_test]
        results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()
        
        # Analyze scalability
        total_time = end_time - start_time
        individual_times = [r[0] for r in results]
        avg_individual_time = statistics.mean(individual_times)
        max_individual_time = max(individual_times)
        throughput = len(users_to_test) / total_time
        
        successful_operations = sum(2 for r in results if r[1].access_token and r[2].user_id)
        success_rate = successful_operations / (len(results) * 2)
        
        benchmark_metrics.record_measurement(f"scalability_concurrency_{concurrency_level}", {
            "concurrency_level": concurrency_level,
            "total_time": total_time,
            "avg_individual_time": avg_individual_time,
            "max_individual_time": max_individual_time,
            "throughput_ops_per_sec": throughput,
            "success_rate": success_rate,
            "successful_operations": successful_operations
        })
        
        # Scalability assertions
        assert success_rate >= 0.95, f"Success rate degraded at concurrency {concurrency_level}: {success_rate:.2%}"
        assert throughput > 0, f"Zero throughput at concurrency {concurrency_level}"
        
        # Performance shouldn't degrade too much with higher concurrency
        if concurrency_level <= 10:
            assert avg_individual_time < 0.5, f"Individual operations too slow at concurrency {concurrency_level}: {avg_individual_time:.3f}s"
        
        print(f"Scalability Benchmark (Concurrency: {concurrency_level}):")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average individual time: {avg_individual_time:.3f}s")
        print(f"  Max individual time: {max_individual_time:.3f}s")
        print(f"  Throughput: {throughput:.1f} ops/sec")
        print(f"  Success rate: {success_rate:.2%}")


if __name__ == "__main__":
    # Run benchmarks with pytest-benchmark
    pytest.main([
        __file__,
        "--benchmark-only",
        "--benchmark-sort=mean",
        "--benchmark-columns=min,max,mean,stddev,median,ops,rounds",
        "-v"
    ])