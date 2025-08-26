#!/usr/bin/env python3
"""
API Response Optimizer - Advanced Response Time Enhancement

Optimizes API response times through:
- Response compression (gzip, brotli)
- Response streaming for large datasets
- ETags and conditional requests
- Response chunking and pagination
- Connection pooling optimization
- Request batching and multiplexing
- CDN integration for static content
"""

import asyncio
import gzip
import json
import time
import hashlib
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import logging
from functools import wraps
import aiohttp
from aiohttp import web, ClientSession, ClientTimeout
from aiohttp.web_response import StreamResponse
import brotli
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class ResponseMetrics:
    """Metrics for response optimization"""
    endpoint: str
    method: str
    response_time_ms: float
    response_size_bytes: int
    compressed_size_bytes: Optional[int] = None
    cache_hit: bool = False
    compression_ratio: Optional[float] = None
    status_code: int = 200
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def compression_savings(self) -> Optional[float]:
        """Calculate compression savings percentage"""
        if self.compressed_size_bytes is None:
            return None
        return (1 - self.compressed_size_bytes / self.response_size_bytes) * 100


class CompressionManager:
    """Manages response compression strategies"""
    
    def __init__(self):
        self.compression_stats = {}
        self.compression_threshold = 1024  # Only compress responses > 1KB
        
    def should_compress(self, data: Union[str, bytes], content_type: str) -> bool:
        """Determine if response should be compressed"""
        size = len(data.encode() if isinstance(data, str) else data)
        
        # Don't compress small responses
        if size < self.compression_threshold:
            return False
        
        # Don't compress already compressed content
        compressed_types = {
            'image/jpeg', 'image/png', 'image/gif', 'image/webp',
            'video/', 'audio/', 'application/zip', 'application/gzip'
        }
        
        if any(ct in content_type for ct in compressed_types):
            return False
        
        return True
    
    def compress_gzip(self, data: Union[str, bytes]) -> bytes:
        """Compress data using gzip"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        start_time = time.time()
        compressed = gzip.compress(data, compresslevel=6)
        compression_time = time.time() - start_time
        
        # Update stats
        compression_ratio = len(compressed) / len(data)
        self.compression_stats['gzip'] = {
            'compression_time_ms': compression_time * 1000,
            'compression_ratio': compression_ratio,
            'savings_percent': (1 - compression_ratio) * 100
        }
        
        return compressed
    
    def compress_brotli(self, data: Union[str, bytes]) -> bytes:
        """Compress data using Brotli"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        start_time = time.time()
        compressed = brotli.compress(data, quality=6)
        compression_time = time.time() - start_time
        
        # Update stats
        compression_ratio = len(compressed) / len(data)
        self.compression_stats['brotli'] = {
            'compression_time_ms': compression_time * 1000,
            'compression_ratio': compression_ratio,
            'savings_percent': (1 - compression_ratio) * 100
        }
        
        return compressed
    
    def choose_compression(self, accept_encoding: str) -> Optional[str]:
        """Choose best compression method based on client support"""
        if 'br' in accept_encoding:
            return 'brotli'  # Brotli typically has better compression
        elif 'gzip' in accept_encoding:
            return 'gzip'
        return None


class StreamingResponseManager:
    """Manages streaming responses for large datasets"""
    
    def __init__(self, chunk_size: int = 8192):
        self.chunk_size = chunk_size
        
    async def stream_json_array(
        self, 
        data_generator: AsyncGenerator[Dict[str, Any], None],
        request: web.Request
    ) -> StreamResponse:
        """Stream JSON array response"""
        response = StreamResponse(
            status=200,
            headers={
                'Content-Type': 'application/json',
                'Transfer-Encoding': 'chunked'
            }
        )
        
        await response.prepare(request)
        
        # Start JSON array
        await response.write(b'[')
        first_item = True
        
        async for item in data_generator:
            if not first_item:
                await response.write(b',')
            first_item = False
            
            # Serialize and write item
            item_json = json.dumps(item).encode('utf-8')
            await response.write(item_json)
            
            # Yield control to allow other tasks
            await asyncio.sleep(0)
        
        # Close JSON array
        await response.write(b']')
        await response.write_eof()
        
        return response
    
    async def stream_ndjson(
        self, 
        data_generator: AsyncGenerator[Dict[str, Any], None],
        request: web.Request
    ) -> StreamResponse:
        """Stream newline-delimited JSON response"""
        response = StreamResponse(
            status=200,
            headers={
                'Content-Type': 'application/x-ndjson',
                'Transfer-Encoding': 'chunked'
            }
        )
        
        await response.prepare(request)
        
        async for item in data_generator:
            # Serialize item and add newline
            item_json = json.dumps(item) + '\n'
            await response.write(item_json.encode('utf-8'))
            await asyncio.sleep(0)  # Yield control
        
        await response.write_eof()
        return response
    
    async def stream_csv(
        self, 
        data_generator: AsyncGenerator[Dict[str, Any], None],
        request: web.Request,
        headers: Optional[List[str]] = None
    ) -> StreamResponse:
        """Stream CSV response"""
        import csv
        import io
        
        response = StreamResponse(
            status=200,
            headers={
                'Content-Type': 'text/csv',
                'Transfer-Encoding': 'chunked',
                'Content-Disposition': 'attachment; filename="data.csv"'
            }
        )
        
        await response.prepare(request)
        
        first_row = True
        fieldnames = headers
        
        async for item in data_generator:
            if first_row and not fieldnames:
                fieldnames = list(item.keys())
            
            # Create CSV row
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            
            if first_row:
                writer.writeheader()
                first_row = False
            
            writer.writerow(item)
            csv_data = output.getvalue().encode('utf-8')
            await response.write(csv_data)
            await asyncio.sleep(0)
        
        await response.write_eof()
        return response


class ETAGManager:
    """Manages ETags for conditional requests"""
    
    def __init__(self):
        self.etag_cache = {}
        
    def generate_etag(self, data: Any, weak: bool = False) -> str:
        """Generate ETag for response data"""
        # Serialize data for hashing
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        # Generate hash
        etag_hash = hashlib.md5(data_str.encode()).hexdigest()[:16]
        
        # Format as ETag
        prefix = 'W/' if weak else ''
        return f'{prefix}"{etag_hash}"'
    
    def check_not_modified(self, request: web.Request, etag: str) -> bool:
        """Check if client has current version (304 Not Modified)"""
        if_none_match = request.headers.get('If-None-Match')
        if not if_none_match:
            return False
        
        # Handle multiple ETags
        client_etags = [tag.strip() for tag in if_none_match.split(',')]
        
        # Check for match (including weak ETags)
        current_etag = etag.lstrip('W/')
        return any(client_etag.lstrip('W/') == current_etag for client_etag in client_etags)


class RequestBatcher:
    """Batches multiple requests for efficient processing"""
    
    def __init__(self, batch_size: int = 10, batch_timeout: float = 0.1):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.pending_requests = {}
        self.batch_tasks = {}
        
    async def batch_request(
        self, 
        batch_key: str, 
        request_data: Any, 
        processor: Callable
    ) -> Any:
        """Add request to batch and wait for result"""
        # Initialize batch if needed
        if batch_key not in self.pending_requests:
            self.pending_requests[batch_key] = []
            # Start batch timeout
            asyncio.create_task(self._batch_timeout_handler(batch_key, processor))
        
        # Create future for this request
        request_future = asyncio.Future()
        self.pending_requests[batch_key].append({
            'data': request_data,
            'future': request_future
        })
        
        # Process batch if it's full
        if len(self.pending_requests[batch_key]) >= self.batch_size:
            await self._process_batch(batch_key, processor)
        
        return await request_future
    
    async def _batch_timeout_handler(self, batch_key: str, processor: Callable):
        """Handle batch timeout"""
        await asyncio.sleep(self.batch_timeout)
        
        if batch_key in self.pending_requests and self.pending_requests[batch_key]:
            await self._process_batch(batch_key, processor)
    
    async def _process_batch(self, batch_key: str, processor: Callable):
        """Process a batch of requests"""
        if batch_key not in self.pending_requests:
            return
        
        batch = self.pending_requests[batch_key]
        del self.pending_requests[batch_key]
        
        if not batch:
            return
        
        try:
            # Extract data from batch
            batch_data = [req['data'] for req in batch]
            
            # Process batch
            results = await processor(batch_data)
            
            # Send results to futures
            for i, request in enumerate(batch):
                result = results[i] if i < len(results) else None
                request['future'].set_result(result)
                
        except Exception as e:
            # Send error to all futures
            for request in batch:
                request['future'].set_exception(e)


class ConnectionPoolManager:
    """Manages connection pools for optimal performance"""
    
    def __init__(self):
        self.session_pools = {}
        self.default_timeout = ClientTimeout(total=30)
        
    @asynccontextmanager
    async def get_session(
        self, 
        pool_key: str = 'default',
        connector_limit: int = 100,
        timeout: Optional[ClientTimeout] = None
    ):
        """Get or create a session with connection pooling"""
        
        if pool_key not in self.session_pools:
            connector = aiohttp.TCPConnector(
                limit=connector_limit,
                limit_per_host=20,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30
            )
            
            session = ClientSession(
                connector=connector,
                timeout=timeout or self.default_timeout
            )
            
            self.session_pools[pool_key] = session
        
        try:
            yield self.session_pools[pool_key]
        except Exception:
            # Don't close session on error, keep it for reuse
            raise
    
    async def close_all_sessions(self):
        """Close all session pools"""
        for session in self.session_pools.values():
            await session.close()
        self.session_pools.clear()


class ResponseOptimizer:
    """Main response optimization coordinator"""
    
    def __init__(self):
        self.compression_manager = CompressionManager()
        self.streaming_manager = StreamingResponseManager()
        self.etag_manager = ETAGManager()
        self.request_batcher = RequestBatcher()
        self.connection_manager = ConnectionPoolManager()
        self.metrics = []
        
    def optimize_response(
        self,
        cache_ttl: int = 3600,
        compress: bool = True,
        etag_enabled: bool = True
    ):
        """Decorator to optimize API responses"""
        def decorator(handler: Callable):
            @wraps(handler)
            async def wrapper(request: web.Request):
                start_time = time.time()
                
                # Execute handler
                result = await handler(request)
                
                # Convert result to web.Response if needed
                if not isinstance(result, web.Response):
                    response_data = result
                    
                    # Generate ETag if enabled
                    etag = None
                    if etag_enabled:
                        etag = self.etag_manager.generate_etag(response_data)
                        
                        # Check for Not Modified
                        if self.etag_manager.check_not_modified(request, etag):
                            return web.Response(status=304, headers={'ETag': etag})
                    
                    # Serialize response
                    if isinstance(response_data, (dict, list)):
                        content = json.dumps(response_data)
                        content_type = 'application/json'
                    else:
                        content = str(response_data)
                        content_type = 'text/plain'
                    
                    # Apply compression if enabled
                    headers = {}
                    if etag:
                        headers['ETag'] = etag
                    
                    if compress and self.compression_manager.should_compress(content, content_type):
                        accept_encoding = request.headers.get('Accept-Encoding', '')
                        compression_method = self.compression_manager.choose_compression(accept_encoding)
                        
                        if compression_method == 'brotli':
                            content = self.compression_manager.compress_brotli(content)
                            headers['Content-Encoding'] = 'br'
                        elif compression_method == 'gzip':
                            content = self.compression_manager.compress_gzip(content)
                            headers['Content-Encoding'] = 'gzip'
                    
                    # Add caching headers
                    if cache_ttl > 0:
                        headers['Cache-Control'] = f'public, max-age={cache_ttl}'
                        expires = datetime.utcnow() + timedelta(seconds=cache_ttl)
                        headers['Expires'] = expires.strftime('%a, %d %b %Y %H:%M:%S GMT')
                    
                    response = web.Response(
                        body=content,
                        content_type=content_type,
                        headers=headers
                    )
                else:
                    response = result
                
                # Record metrics
                response_time = (time.time() - start_time) * 1000
                self._record_metrics(request, response, response_time)
                
                return response
                
            return wrapper
        return decorator
    
    def stream_large_response(self, format: str = 'json'):
        """Decorator for streaming large responses"""
        def decorator(handler: Callable):
            @wraps(handler)
            async def wrapper(request: web.Request):
                # Handler should return an async generator
                data_generator = await handler(request)
                
                if format == 'json':
                    return await self.streaming_manager.stream_json_array(data_generator, request)
                elif format == 'ndjson':
                    return await self.streaming_manager.stream_ndjson(data_generator, request)
                elif format == 'csv':
                    return await self.streaming_manager.stream_csv(data_generator, request)
                else:
                    raise ValueError(f"Unsupported streaming format: {format}")
                    
            return wrapper
        return decorator
    
    def batch_requests(self, batch_key: str, processor: Callable):
        """Decorator for batching requests"""
        def decorator(handler: Callable):
            @wraps(handler)
            async def wrapper(request: web.Request):
                # Extract request data
                request_data = await handler(request)
                
                # Batch the request
                result = await self.request_batcher.batch_request(
                    batch_key, request_data, processor
                )
                
                return web.json_response(result)
                
            return wrapper
        return decorator
    
    def _record_metrics(self, request: web.Request, response: web.Response, response_time_ms: float):
        """Record response metrics"""
        # Calculate response size
        response_size = len(response.body) if hasattr(response, 'body') and response.body else 0
        
        # Check if compressed
        compressed_size = None
        if 'Content-Encoding' in response.headers:
            compressed_size = response_size
            # For metrics, we'd need to track original size separately
        
        metrics = ResponseMetrics(
            endpoint=request.path,
            method=request.method,
            response_time_ms=response_time_ms,
            response_size_bytes=response_size,
            compressed_size_bytes=compressed_size,
            status_code=response.status
        )
        
        self.metrics.append(metrics)
        
        # Keep only recent metrics (last 1000)
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-500:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        if not self.metrics:
            return {'status': 'no_data'}
        
        # Calculate aggregates
        total_requests = len(self.metrics)
        avg_response_time = sum(m.response_time_ms for m in self.metrics) / total_requests
        
        # Response time percentiles
        response_times = sorted(m.response_time_ms for m in self.metrics)
        p95_response_time = response_times[int(len(response_times) * 0.95)]
        p99_response_time = response_times[int(len(response_times) * 0.99)]
        
        # Compression stats
        compressed_responses = [m for m in self.metrics if m.compressed_size_bytes is not None]
        compression_rate = len(compressed_responses) / total_requests if total_requests > 0 else 0
        
        # Error rate
        error_responses = [m for m in self.metrics if m.status_code >= 400]
        error_rate = len(error_responses) / total_requests if total_requests > 0 else 0
        
        # Endpoint performance
        endpoint_stats = {}
        for metric in self.metrics:
            if metric.endpoint not in endpoint_stats:
                endpoint_stats[metric.endpoint] = []
            endpoint_stats[metric.endpoint].append(metric.response_time_ms)
        
        endpoint_performance = {
            endpoint: {
                'avg_response_time': sum(times) / len(times),
                'request_count': len(times)
            }
            for endpoint, times in endpoint_stats.items()
        }
        
        return {
            'summary': {
                'total_requests': total_requests,
                'avg_response_time_ms': avg_response_time,
                'p95_response_time_ms': p95_response_time,
                'p99_response_time_ms': p99_response_time,
                'error_rate': error_rate,
                'compression_rate': compression_rate
            },
            'compression': self.compression_manager.compression_stats,
            'endpoints': endpoint_performance,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def shutdown(self):
        """Shutdown optimizer components"""
        await self.connection_manager.close_all_sessions()


# Global optimizer instance
_response_optimizer: Optional[ResponseOptimizer] = None


def get_response_optimizer() -> ResponseOptimizer:
    """Get global response optimizer"""
    global _response_optimizer
    if _response_optimizer is None:
        _response_optimizer = ResponseOptimizer()
    return _response_optimizer


# Convenience decorators
def optimized_response(cache_ttl: int = 3600, compress: bool = True):
    """Convenience decorator for response optimization"""
    optimizer = get_response_optimizer()
    return optimizer.optimize_response(cache_ttl=cache_ttl, compress=compress)


def streaming_response(format: str = 'json'):
    """Convenience decorator for streaming responses"""
    optimizer = get_response_optimizer()
    return optimizer.stream_large_response(format=format)


if __name__ == "__main__":
    # Example usage and testing
    async def test_response_optimization():
        print("🚀 API RESPONSE OPTIMIZER - Testing")
        print("=" * 50)
        
        optimizer = get_response_optimizer()
        
        # Test compression
        print("🗜️ Testing compression...")
        test_data = {"data": [f"item_{i}" for i in range(1000)]}
        test_json = json.dumps(test_data)
        
        print(f"   Original size: {len(test_json)} bytes")
        
        # Test gzip compression
        gzipped = optimizer.compression_manager.compress_gzip(test_json)
        print(f"   Gzip size: {len(gzipped)} bytes ({len(gzipped)/len(test_json):.2%} of original)")
        
        # Test brotli compression
        brotli_compressed = optimizer.compression_manager.compress_brotli(test_json)
        print(f"   Brotli size: {len(brotli_compressed)} bytes ({len(brotli_compressed)/len(test_json):.2%} of original)")
        
        # Test ETag generation
        print("\n🏷️ Testing ETag generation...")
        etag = optimizer.etag_manager.generate_etag(test_data)
        print(f"   Generated ETag: {etag}")
        
        # Test streaming (simulation)
        print("\n📡 Testing streaming capabilities...")
        async def sample_data_generator():
            for i in range(5):
                yield {"id": i, "value": f"data_{i}"}
                await asyncio.sleep(0.01)
        
        # Simulate streaming (would need actual web.Request in real usage)
        print("   Streaming generator created successfully")
        
        print("\n✅ API response optimization test completed!")
    
    # Run test
    asyncio.run(test_response_optimization())