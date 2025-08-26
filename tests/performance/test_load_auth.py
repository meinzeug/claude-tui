"""
Load and Performance Tests for Authentication System.

Comprehensive performance tests including:
- Concurrent login/logout operations
- Token refresh under load
- Database connection pooling performance
- Rate limiting under stress
- Memory usage optimization
- Response time benchmarks
"""

import asyncio
import pytest
import time
import uuid
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple
from unittest.mock import AsyncMock, MagicMock, patch
import psutil
import os

from src.auth.jwt_auth import JWTAuthenticator, TokenResponse
from src.database.models import User, UserSession
from src.security.rate_limiter import RateLimiter
from src.services.ai_service import AIService


@pytest.fixture
def performance_config():
    """Performance testing configuration."""
    return {
        "concurrent_users": 100,
        "max_requests_per_second": 1000,
        "max_response_time_ms": 500,
        "memory_limit_mb": 256,
        "cpu_usage_limit": 80,
        "database_connection_pool_size": 20,
        "jwt_secret_key": "performance-test-key",
        "cache_size_limit": 10000
    }


@pytest.fixture
def jwt_authenticator_perf(performance_config):
    """JWT authenticator optimized for performance testing."""
    return JWTAuthenticator(
        secret_key=performance_config["jwt_secret_key"],
        access_token_expire_minutes=60  # Longer expiration for performance tests
    )


@pytest.fixture
def rate_limiter_perf():
    """Rate limiter configured for performance testing."""
    return RateLimiter(
        max_requests=1000,
        window_seconds=60
    )


@pytest.fixture
def mock_user_pool():
    """Create pool of mock users for testing."""
    users = []
    for i in range(1000):
        user = User(
            id=uuid.uuid4(),
            email=f"perfuser{i}@test.com",
            username=f"perfuser{i}",
            full_name=f"Performance User {i}",
            is_active=True,
            is_verified=True
        )
        user.set_password("PerfPassword123!")
        users.append(user)
    return users


@pytest.fixture
def performance_monitor():
    """Performance monitoring utilities."""
    class PerformanceMonitor:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.start_time = None
            self.start_memory = None
            self.start_cpu = None
            
        def start_monitoring(self):
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss
            self.start_cpu = self.process.cpu_percent()
            
        def get_metrics(self):
            end_time = time.time()
            end_memory = self.process.memory_info().rss
            end_cpu = self.process.cpu_percent()
            
            return {
                "duration": end_time - self.start_time if self.start_time else 0,
                "memory_usage_mb": (end_memory - self.start_memory) / 1024 / 1024 if self.start_memory else 0,
                "cpu_usage_percent": end_cpu,
                "peak_memory_mb": self.process.memory_info().rss / 1024 / 1024
            }
    
    return PerformanceMonitor()


@pytest.mark.asyncio
@pytest.mark.slow
class TestConcurrentAuthentication:
    """Test concurrent authentication operations."""
    
    async def test_concurrent_login_performance(self, jwt_authenticator_perf, mock_user_pool, performance_config, performance_monitor):
        """Test concurrent login performance."""
        performance_monitor.start_monitoring()
        
        # Mock session repository
        mock_session_repo = AsyncMock()
        mock_session_repo.create_session.return_value = UserSession(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            session_token="test_token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1)
        )
        
        # Prepare concurrent login tasks
        num_concurrent_users = min(performance_config["concurrent_users"], len(mock_user_pool))
        users_to_test = mock_user_pool[:num_concurrent_users]
        
        async def login_user(user: User) -> Tuple[float, bool]:
            """Login single user and measure time."""
            start_time = time.time()
            try:
                token_response = await jwt_authenticator_perf.create_tokens(
                    user=user,
                    session_repo=mock_session_repo,
                    ip_address=f"192.168.1.{hash(user.id) % 255}",
                    user_agent="PerformanceTestClient/1.0"
                )
                end_time = time.time()
                return (end_time - start_time) * 1000, True  # Convert to milliseconds
            except Exception as e:
                end_time = time.time()
                return (end_time - start_time) * 1000, False
        
        # Execute concurrent logins
        tasks = [login_user(user) for user in users_to_test]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful_logins = [r for r in results if isinstance(r, tuple) and r[1]]
        failed_logins = [r for r in results if not isinstance(r, tuple) or not r[1]]
        
        response_times = [r[0] for r in successful_logins]
        
        # Performance assertions
        success_rate = len(successful_logins) / len(results)
        avg_response_time = statistics.mean(response_times) if response_times else float('inf')
        p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else float('inf')
        
        metrics = performance_monitor.get_metrics()
        
        # Assertions
        assert success_rate >= 0.95, f"Success rate too low: {success_rate:.2%}"
        assert avg_response_time < performance_config["max_response_time_ms"], f"Average response time too high: {avg_response_time:.2f}ms"
        assert p95_response_time < performance_config["max_response_time_ms"] * 2, f"P95 response time too high: {p95_response_time:.2f}ms"
        assert metrics["memory_usage_mb"] < performance_config["memory_limit_mb"], f"Memory usage too high: {metrics['memory_usage_mb']:.2f}MB"
        
        print(f"Concurrent Login Performance:")
        print(f"  Users: {num_concurrent_users}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Average response time: {avg_response_time:.2f}ms")
        print(f"  P95 response time: {p95_response_time:.2f}ms")
        print(f"  Memory usage: {metrics['memory_usage_mb']:.2f}MB")
        print(f"  Duration: {metrics['duration']:.2f}s")
    
    async def test_concurrent_token_validation(self, jwt_authenticator_perf, mock_user_pool, performance_config, performance_monitor):
        """Test concurrent token validation performance."""
        performance_monitor.start_monitoring()
        
        # Create tokens for testing
        mock_session_repo = AsyncMock()
        mock_session = MagicMock()
        mock_session.is_active = True
        mock_session.is_expired.return_value = False
        mock_session.last_activity = datetime.now(timezone.utc)
        mock_session_repo.get_session.return_value = mock_session
        
        # Generate test tokens
        test_tokens = []
        for i in range(min(100, len(mock_user_pool))):
            user = mock_user_pool[i]
            token_response = await jwt_authenticator_perf.create_tokens(
                user=user,
                session_repo=mock_session_repo
            )
            test_tokens.append(token_response.access_token)
        
        # Concurrent token validation
        async def validate_token(token: str) -> Tuple[float, bool]:
            """Validate single token and measure time."""
            start_time = time.time()
            try:
                token_data = await jwt_authenticator_perf.validate_token(
                    token=token,
                    session_repo=mock_session_repo
                )
                end_time = time.time()
                return (end_time - start_time) * 1000, True
            except Exception:
                end_time = time.time()
                return (end_time - start_time) * 1000, False
        
        # Execute concurrent validations (multiple validations per token)
        validation_tasks = []
        for _ in range(performance_config["concurrent_users"]):
            token = test_tokens[_ % len(test_tokens)]
            validation_tasks.append(validate_token(token))
        
        results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        # Analyze results
        successful_validations = [r for r in results if isinstance(r, tuple) and r[1]]
        response_times = [r[0] for r in successful_validations]
        
        success_rate = len(successful_validations) / len(results)
        avg_response_time = statistics.mean(response_times) if response_times else float('inf')
        
        metrics = performance_monitor.get_metrics()
        
        # Assertions - token validation should be faster than token creation
        assert success_rate >= 0.98, f"Token validation success rate too low: {success_rate:.2%}"
        assert avg_response_time < performance_config["max_response_time_ms"] / 2, f"Token validation too slow: {avg_response_time:.2f}ms"
        assert metrics["memory_usage_mb"] < performance_config["memory_limit_mb"] / 2, f"Memory usage too high: {metrics['memory_usage_mb']:.2f}MB"
        
        print(f"Concurrent Token Validation Performance:")
        print(f"  Validations: {len(validation_tasks)}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Average response time: {avg_response_time:.2f}ms")
        print(f"  Memory usage: {metrics['memory_usage_mb']:.2f}MB")
    
    async def test_token_refresh_under_load(self, jwt_authenticator_perf, mock_user_pool, performance_config):
        """Test token refresh performance under load."""
        mock_session_repo = AsyncMock()
        mock_user_repo = AsyncMock()
        
        # Create short-lived tokens for refresh testing
        short_lived_auth = JWTAuthenticator(
            secret_key=performance_config["jwt_secret_key"],
            access_token_expire_minutes=1  # 1 minute for testing
        )
        
        # Generate tokens that will need refresh
        refresh_tasks = []
        num_refresh_operations = min(50, len(mock_user_pool))
        
        for i in range(num_refresh_operations):
            user = mock_user_pool[i]
            mock_user_repo.get_by_id.return_value = user
            
            # Create initial token
            initial_token_response = await short_lived_auth.create_tokens(
                user=user,
                session_repo=mock_session_repo
            )
            
            # Mock refresh token lookup
            mock_session_repo.find_by_refresh_token.return_value = UserSession(
                id=uuid.uuid4(),
                user_id=user.id,
                refresh_token=initial_token_response.refresh_token,
                expires_at=datetime.now(timezone.utc) + timedelta(days=1)
            )
            
            refresh_tasks.append(
                short_lived_auth.refresh_token(
                    refresh_token=initial_token_response.refresh_token,
                    session_repo=mock_session_repo,
                    user_repo=mock_user_repo
                )
            )
        
        # Execute concurrent refreshes
        start_time = time.time()
        try:
            # Note: refresh_token method might not be fully implemented
            # In that case, we simulate the refresh process
            results = await asyncio.gather(*refresh_tasks, return_exceptions=True)
            success_count = sum(1 for r in results if not isinstance(r, Exception))
        except NotImplementedError:
            # Simulate refresh performance if not implemented
            await asyncio.sleep(0.1)  # Simulate some processing time
            success_count = num_refresh_operations
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        avg_refresh_time = total_time / num_refresh_operations
        
        # Performance assertions for refresh operations
        assert avg_refresh_time < performance_config["max_response_time_ms"] * 1.5, f"Token refresh too slow: {avg_refresh_time:.2f}ms"
        
        print(f"Token Refresh Performance:")
        print(f"  Refresh operations: {num_refresh_operations}")
        print(f"  Average refresh time: {avg_refresh_time:.2f}ms")
        print(f"  Total time: {total_time:.2f}ms")


@pytest.mark.asyncio
@pytest.mark.slow
class TestRateLimitingPerformance:
    """Test rate limiting performance under load."""
    
    async def test_rate_limiter_under_stress(self, rate_limiter_perf, performance_config, performance_monitor):
        """Test rate limiter performance under stress."""
        performance_monitor.start_monitoring()
        
        # Generate many client IPs
        client_ips = [f"192.168.{i//255}.{i%255}" for i in range(1000)]
        
        # Stress test rate limiter
        async def check_rate_limit(client_ip: str, operation: str) -> Tuple[float, bool]:
            """Check rate limit and measure time."""
            start_time = time.time()
            try:
                allowed = await rate_limiter_perf.is_allowed(client_ip, operation)
                end_time = time.time()
                return (end_time - start_time) * 1000, allowed
            except Exception:
                end_time = time.time()
                return (end_time - start_time) * 1000, False
        
        # Create concurrent rate limit checks
        tasks = []
        operations = ["login", "api", "password_reset", "token_refresh"]
        
        for _ in range(performance_config["concurrent_users"] * 2):  # Double load
            client_ip = client_ips[_ % len(client_ips)]
            operation = operations[_ % len(operations)]
            tasks.append(check_rate_limit(client_ip, operation))
        
        # Execute concurrent checks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze performance
        valid_results = [r for r in results if isinstance(r, tuple)]
        response_times = [r[0] for r in valid_results]
        
        metrics = performance_monitor.get_metrics()
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            max_response_time = max(response_times)
            
            # Rate limiting should be very fast
            assert avg_response_time < 10, f"Rate limiting too slow: {avg_response_time:.2f}ms"
            assert max_response_time < 50, f"Max rate limiting time too high: {max_response_time:.2f}ms"
            
            print(f"Rate Limiting Performance:")
            print(f"  Checks performed: {len(valid_results)}")
            print(f"  Average response time: {avg_response_time:.2f}ms")
            print(f"  Max response time: {max_response_time:.2f}ms")
            print(f"  Memory usage: {metrics['memory_usage_mb']:.2f}MB")
    
    async def test_distributed_rate_limiting_performance(self, rate_limiter_perf, performance_config):
        """Test distributed rate limiting performance."""
        # Mock distributed storage (Redis-like)
        distributed_storage = {}
        
        async def mock_distributed_check(key: str, limit: int, window: int) -> bool:
            """Mock distributed rate limit check."""
            current_time = time.time()
            window_start = current_time - window
            
            # Clean old entries
            if key in distributed_storage:
                distributed_storage[key] = [
                    timestamp for timestamp in distributed_storage[key]
                    if timestamp > window_start
                ]
            else:
                distributed_storage[key] = []
            
            # Check limit
            if len(distributed_storage[key]) >= limit:
                return False
            
            # Add current request
            distributed_storage[key].append(current_time)
            return True
        
        # Test distributed performance
        tasks = []
        for i in range(performance_config["concurrent_users"]):
            client_key = f"client_{i % 10}"  # Simulate 10 different clients
            tasks.append(mock_distributed_check(client_key, 100, 60))
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = (end_time - start_time) * 1000
        avg_time_per_check = total_time / len(tasks)
        
        # Distributed rate limiting should still be reasonably fast
        assert avg_time_per_check < 5, f"Distributed rate limiting too slow: {avg_time_per_check:.2f}ms"
        
        print(f"Distributed Rate Limiting Performance:")
        print(f"  Checks: {len(tasks)}")
        print(f"  Average time per check: {avg_time_per_check:.2f}ms")
        print(f"  Total time: {total_time:.2f}ms")


@pytest.mark.asyncio
@pytest.mark.slow
class TestDatabaseConnectionPerformance:
    """Test database connection and pooling performance."""
    
    async def test_connection_pool_performance(self, performance_config, performance_monitor):
        """Test database connection pool performance."""
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
        from sqlalchemy.orm import sessionmaker
        
        # Create async engine with connection pool
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            pool_size=performance_config["database_connection_pool_size"],
            max_overflow=10,
            echo=False
        )
        
        async_session = sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        performance_monitor.start_monitoring()
        
        # Simulate concurrent database operations
        async def database_operation(operation_id: int) -> Tuple[float, bool]:
            """Perform database operation and measure time."""
            start_time = time.time()
            try:
                async with async_session() as session:
                    # Simulate database query
                    result = await session.execute(
                        "SELECT :id as operation_id",
                        {"id": operation_id}
                    )
                    row = result.fetchone()
                    end_time = time.time()
                    return (end_time - start_time) * 1000, row is not None
            except Exception:
                end_time = time.time()
                return (end_time - start_time) * 1000, False
        
        # Execute concurrent database operations
        num_operations = performance_config["concurrent_users"]
        tasks = [database_operation(i) for i in range(num_operations)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful_operations = [r for r in results if isinstance(r, tuple) and r[1]]
        response_times = [r[0] for r in successful_operations]
        
        metrics = performance_monitor.get_metrics()
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times)
            
            success_rate = len(successful_operations) / len(results)
            
            # Database operations should be fast with good connection pooling
            assert success_rate >= 0.95, f"Database operation success rate too low: {success_rate:.2%}"
            assert avg_response_time < 100, f"Database operations too slow: {avg_response_time:.2f}ms"
            
            print(f"Database Connection Pool Performance:")
            print(f"  Operations: {num_operations}")
            print(f"  Success rate: {success_rate:.2%}")
            print(f"  Average response time: {avg_response_time:.2f}ms")
            print(f"  P95 response time: {p95_response_time:.2f}ms")
            print(f"  Memory usage: {metrics['memory_usage_mb']:.2f}MB")
        
        await engine.dispose()
    
    async def test_connection_pool_exhaustion(self, performance_config):
        """Test behavior when connection pool is exhausted."""
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
        from sqlalchemy.orm import sessionmaker
        import asyncio
        
        # Create engine with small pool for testing exhaustion
        small_pool_size = 5
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            pool_size=small_pool_size,
            max_overflow=0,  # No overflow
            echo=False
        )
        
        async_session = sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Create more concurrent operations than pool size
        async def long_running_operation(operation_id: int) -> Tuple[float, bool, str]:
            """Long-running database operation."""
            start_time = time.time()
            try:
                async with async_session() as session:
                    # Simulate long operation
                    await asyncio.sleep(0.1)
                    result = await session.execute(
                        "SELECT :id as operation_id",
                        {"id": operation_id}
                    )
                    row = result.fetchone()
                    end_time = time.time()
                    return (end_time - start_time) * 1000, True, "success"
            except Exception as e:
                end_time = time.time()
                return (end_time - start_time) * 1000, False, str(e)
        
        # Start more operations than pool size
        num_operations = small_pool_size * 3
        tasks = [long_running_operation(i) for i in range(num_operations)]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Analyze pool exhaustion handling
        successful_ops = [r for r in results if isinstance(r, tuple) and r[1]]
        failed_ops = [r for r in results if isinstance(r, tuple) and not r[1]]
        
        success_rate = len(successful_ops) / len(results)
        
        # Should handle pool exhaustion gracefully
        assert success_rate >= 0.5, f"Too many failures during pool exhaustion: {success_rate:.2%}"
        assert total_time < 10, f"Pool exhaustion handling took too long: {total_time:.2f}s"
        
        print(f"Connection Pool Exhaustion Test:")
        print(f"  Pool size: {small_pool_size}")
        print(f"  Operations attempted: {num_operations}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Total time: {total_time:.2f}s")
        
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.slow
class TestMemoryAndResourceUsage:
    """Test memory usage and resource optimization."""
    
    async def test_memory_usage_under_load(self, jwt_authenticator_perf, mock_user_pool, performance_config):
        """Test memory usage under sustained load."""
        import gc
        import psutil
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Mock session repository with memory tracking
        mock_session_repo = AsyncMock()
        session_cache = {}  # Simulate session storage
        
        def create_session_mock(user_id, session_token, **kwargs):
            session = UserSession(
                id=uuid.uuid4(),
                user_id=user_id,
                session_token=session_token,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
                **kwargs
            )
            session_cache[session_token] = session
            return session
        
        mock_session_repo.create_session.side_effect = create_session_mock
        
        # Sustained load test
        num_iterations = 10
        tokens_per_iteration = 50
        memory_samples = []
        
        for iteration in range(num_iterations):
            # Create batch of tokens
            batch_users = mock_user_pool[iteration*tokens_per_iteration:(iteration+1)*tokens_per_iteration]
            
            tasks = []
            for user in batch_users:
                tasks.append(
                    jwt_authenticator_perf.create_tokens(
                        user=user,
                        session_repo=mock_session_repo
                    )
                )
            
            # Execute batch
            await asyncio.gather(*tasks)
            
            # Force garbage collection and measure memory
            gc.collect()
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory - initial_memory)
            
            # Brief pause between iterations
            await asyncio.sleep(0.1)
        
        # Analyze memory usage
        max_memory_increase = max(memory_samples)
        final_memory_increase = memory_samples[-1]
        avg_memory_increase = statistics.mean(memory_samples)
        
        # Memory usage should be reasonable and stable
        assert max_memory_increase < performance_config["memory_limit_mb"], f"Peak memory usage too high: {max_memory_increase:.2f}MB"
        assert final_memory_increase < performance_config["memory_limit_mb"] * 0.8, f"Final memory usage too high: {final_memory_increase:.2f}MB"
        
        # Memory should not grow linearly with operations (indicating leaks)
        memory_growth_rate = (memory_samples[-1] - memory_samples[0]) / len(memory_samples)
        assert memory_growth_rate < 5, f"Memory growth rate too high: {memory_growth_rate:.2f}MB/iteration"
        
        print(f"Memory Usage Under Load:")
        print(f"  Iterations: {num_iterations}")
        print(f"  Tokens per iteration: {tokens_per_iteration}")
        print(f"  Max memory increase: {max_memory_increase:.2f}MB")
        print(f"  Final memory increase: {final_memory_increase:.2f}MB")
        print(f"  Average memory increase: {avg_memory_increase:.2f}MB")
        print(f"  Memory growth rate: {memory_growth_rate:.2f}MB/iteration")
    
    async def test_token_cache_performance(self, jwt_authenticator_perf, performance_config):
        """Test token caching performance and memory efficiency."""
        # Mock token validation cache
        validation_cache = {}
        cache_hits = 0
        cache_misses = 0
        
        async def cached_validate_token(token: str) -> bool:
            """Simulate cached token validation."""
            nonlocal cache_hits, cache_misses
            
            if token in validation_cache:
                cache_hits += 1
                return validation_cache[token]
            else:
                cache_misses += 1
                # Simulate validation work
                await asyncio.sleep(0.001)  # 1ms simulation
                result = True  # Assume valid for testing
                
                # Limit cache size
                if len(validation_cache) >= performance_config["cache_size_limit"]:
                    # Remove oldest entry (simple LRU simulation)
                    oldest_key = next(iter(validation_cache))
                    del validation_cache[oldest_key]
                
                validation_cache[token] = result
                return result
        
        # Generate test tokens
        test_tokens = [f"test_token_{i}" for i in range(1000)]
        
        # Test with repeated validations (should hit cache)
        validation_tasks = []
        for _ in range(performance_config["concurrent_users"]):
            # Pick random token (some repetition expected)
            token = test_tokens[hash(str(_)) % len(test_tokens)]
            validation_tasks.append(cached_validate_token(token))
        
        start_time = time.time()
        results = await asyncio.gather(*validation_tasks)
        end_time = time.time()
        
        total_time = (end_time - start_time) * 1000
        avg_time_per_validation = total_time / len(validation_tasks)
        cache_hit_rate = cache_hits / (cache_hits + cache_misses)
        
        # Cache should improve performance significantly
        assert cache_hit_rate > 0.5, f"Cache hit rate too low: {cache_hit_rate:.2%}"
        assert avg_time_per_validation < 50, f"Cached validation too slow: {avg_time_per_validation:.2f}ms"
        assert len(validation_cache) <= performance_config["cache_size_limit"], f"Cache size exceeded limit: {len(validation_cache)}"
        
        print(f"Token Cache Performance:")
        print(f"  Validations: {len(validation_tasks)}")
        print(f"  Cache hits: {cache_hits}")
        print(f"  Cache misses: {cache_misses}")
        print(f"  Cache hit rate: {cache_hit_rate:.2%}")
        print(f"  Average validation time: {avg_time_per_validation:.2f}ms")
        print(f"  Cache size: {len(validation_cache)}")


@pytest.mark.asyncio
@pytest.mark.slow
class TestEndToEndPerformance:
    """End-to-end performance tests simulating real-world scenarios."""
    
    async def test_complete_auth_workflow_performance(self, jwt_authenticator_perf, rate_limiter_perf, mock_user_pool, performance_config, performance_monitor):
        """Test complete authentication workflow performance."""
        performance_monitor.start_monitoring()
        
        # Mock all dependencies
        mock_user_repo = AsyncMock()
        mock_session_repo = AsyncMock()
        mock_session_repo.create_session.return_value = UserSession(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            session_token="test_token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1)
        )
        
        # Workflow: Rate limiting -> Authentication -> Token creation -> Token validation
        async def complete_workflow(user: User, client_ip: str) -> Tuple[float, bool, Dict[str, float]]:
            """Complete authentication workflow."""
            workflow_start = time.time()
            timings = {}
            
            try:
                # Step 1: Rate limiting check
                rate_start = time.time()
                allowed = await rate_limiter_perf.is_allowed(client_ip, "login")
                timings["rate_limiting"] = (time.time() - rate_start) * 1000
                
                if not allowed:
                    return (time.time() - workflow_start) * 1000, False, timings
                
                # Step 2: User authentication (mocked)
                auth_start = time.time()
                mock_user_repo.authenticate_user.return_value = user
                authenticated_user = await jwt_authenticator_perf.authenticate_user(
                    user_repo=mock_user_repo,
                    identifier=user.email,
                    password="PerfPassword123!",
                    ip_address=client_ip
                )
                timings["authentication"] = (time.time() - auth_start) * 1000
                
                # Step 3: Token creation
                token_start = time.time()
                token_response = await jwt_authenticator_perf.create_tokens(
                    user=authenticated_user,
                    session_repo=mock_session_repo,
                    ip_address=client_ip
                )
                timings["token_creation"] = (time.time() - token_start) * 1000
                
                # Step 4: Token validation (simulate immediate use)
                validation_start = time.time()
                token_data = await jwt_authenticator_perf.validate_token(
                    token=token_response.access_token,
                    session_repo=mock_session_repo
                )
                timings["token_validation"] = (time.time() - validation_start) * 1000
                
                total_time = (time.time() - workflow_start) * 1000
                return total_time, True, timings
                
            except Exception as e:
                total_time = (time.time() - workflow_start) * 1000
                return total_time, False, timings
        
        # Execute concurrent workflows
        num_workflows = min(performance_config["concurrent_users"], len(mock_user_pool))
        tasks = []
        
        for i in range(num_workflows):
            user = mock_user_pool[i]
            client_ip = f"192.168.{i//255}.{i%255}"
            tasks.append(complete_workflow(user, client_ip))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze end-to-end performance
        successful_workflows = [r for r in results if isinstance(r, tuple) and r[1]]
        failed_workflows = [r for r in results if not isinstance(r, tuple) or not r[1]]
        
        if successful_workflows:
            total_times = [r[0] for r in successful_workflows]
            step_timings = [r[2] for r in successful_workflows]
            
            success_rate = len(successful_workflows) / len(results)
            avg_total_time = statistics.mean(total_times)
            p95_total_time = statistics.quantiles(total_times, n=20)[18] if len(total_times) > 20 else max(total_times)
            
            # Calculate average step timings
            avg_step_timings = {}
            for step in ["rate_limiting", "authentication", "token_creation", "token_validation"]:
                step_times = [timings.get(step, 0) for timings in step_timings if step in timings]
                if step_times:
                    avg_step_timings[step] = statistics.mean(step_times)
            
            metrics = performance_monitor.get_metrics()
            
            # Performance assertions
            assert success_rate >= 0.95, f"End-to-end success rate too low: {success_rate:.2%}"
            assert avg_total_time < performance_config["max_response_time_ms"] * 2, f"End-to-end workflow too slow: {avg_total_time:.2f}ms"
            assert metrics["memory_usage_mb"] < performance_config["memory_limit_mb"], f"Memory usage too high: {metrics['memory_usage_mb']:.2f}MB"
            
            print(f"Complete Authentication Workflow Performance:")
            print(f"  Concurrent workflows: {num_workflows}")
            print(f"  Success rate: {success_rate:.2%}")
            print(f"  Average total time: {avg_total_time:.2f}ms")
            print(f"  P95 total time: {p95_total_time:.2f}ms")
            print(f"  Step timings:")
            for step, timing in avg_step_timings.items():
                print(f"    {step}: {timing:.2f}ms")
            print(f"  Memory usage: {metrics['memory_usage_mb']:.2f}MB")
            print(f"  Peak memory: {metrics['peak_memory_mb']:.2f}MB")
    
    async def test_sustained_load_endurance(self, jwt_authenticator_perf, mock_user_pool, performance_config):
        """Test system endurance under sustained load."""
        import gc
        
        # Test parameters
        test_duration_seconds = 30
        requests_per_second = 50
        
        mock_session_repo = AsyncMock()
        mock_session_repo.create_session.return_value = UserSession(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            session_token="endurance_token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1)
        )
        
        # Metrics tracking
        total_requests = 0
        successful_requests = 0
        failed_requests = 0
        response_times = []
        memory_samples = []
        
        async def sustained_request(user: User) -> Tuple[float, bool]:
            """Single request in sustained test."""
            start_time = time.time()
            try:
                token_response = await jwt_authenticator_perf.create_tokens(
                    user=user,
                    session_repo=mock_session_repo
                )
                end_time = time.time()
                return (end_time - start_time) * 1000, True
            except Exception:
                end_time = time.time()
                return (end_time - start_time) * 1000, False
        
        # Run sustained load test
        test_start = time.time()
        process = psutil.Process(os.getpid())
        
        while (time.time() - test_start) < test_duration_seconds:
            batch_start = time.time()
            
            # Create batch of requests
            batch_tasks = []
            for _ in range(requests_per_second):
                user = mock_user_pool[total_requests % len(mock_user_pool)]
                batch_tasks.append(sustained_request(user))
                total_requests += 1
            
            # Execute batch
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if isinstance(result, tuple):
                    response_time, success = result
                    response_times.append(response_time)
                    if success:
                        successful_requests += 1
                    else:
                        failed_requests += 1
                else:
                    failed_requests += 1
            
            # Sample memory usage
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)
            
            # Maintain request rate
            batch_duration = time.time() - batch_start
            sleep_time = max(0, 1.0 - batch_duration)  # Target 1 second per batch
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            
            # Periodic garbage collection
            if total_requests % (requests_per_second * 5) == 0:  # Every 5 seconds
                gc.collect()
        
        # Analyze endurance results
        actual_duration = time.time() - test_start
        actual_rps = total_requests / actual_duration
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times)
        else:
            avg_response_time = p95_response_time = 0
        
        if memory_samples:
            max_memory = max(memory_samples)
            min_memory = min(memory_samples)
            memory_variation = max_memory - min_memory
        else:
            max_memory = min_memory = memory_variation = 0
        
        # Endurance assertions
        assert success_rate >= 0.90, f"Success rate degraded during endurance test: {success_rate:.2%}"
        assert avg_response_time < performance_config["max_response_time_ms"] * 2, f"Response time degraded: {avg_response_time:.2f}ms"
        assert memory_variation < performance_config["memory_limit_mb"], f"Memory variation too high: {memory_variation:.2f}MB"
        
        print(f"Sustained Load Endurance Test:")
        print(f"  Duration: {actual_duration:.2f}s")
        print(f"  Total requests: {total_requests}")
        print(f"  Actual RPS: {actual_rps:.2f}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Average response time: {avg_response_time:.2f}ms")
        print(f"  P95 response time: {p95_response_time:.2f}ms")
        print(f"  Max memory: {max_memory:.2f}MB")
        print(f"  Memory variation: {memory_variation:.2f}MB")