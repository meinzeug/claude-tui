"""
High-Performance Response Compression Middleware.

Implements intelligent compression with gzip/brotli support
for optimal API response times and bandwidth usage.
"""

import gzip
import logging
import time
from typing import Dict, Any, List, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse
import asyncio

try:
    import brotli
    BROTLI_AVAILABLE = True
except ImportError:
    BROTLI_AVAILABLE = False

logger = logging.getLogger(__name__)


class CompressionMiddleware(BaseHTTPMiddleware):
    """
    Intelligent response compression middleware.
    
    Features:
    - Auto-detection of compression support
    - Configurable compression levels
    - Size-based compression decisions
    - Content-type filtering
    - Performance monitoring
    """
    
    def __init__(
        self,
        app,
        minimum_size: int = 500,
        gzip_level: int = 6,
        brotli_level: int = 4,
        excluded_types: Optional[List[str]] = None,
        excluded_paths: Optional[List[str]] = None
    ):
        super().__init__(app)
        self.minimum_size = minimum_size
        self.gzip_level = gzip_level
        self.brotli_level = brotli_level
        
        # Default excluded content types (already compressed)
        self.excluded_types = excluded_types or [
            'image/', 'video/', 'audio/', 'application/zip',
            'application/gzip', 'application/pdf'
        ]
        
        # Excluded paths
        self.excluded_paths = excluded_paths or [
            '/health', '/metrics', '/docs', '/redoc'
        ]
        
        # Performance stats
        self.compression_stats = {
            'total_responses': 0,
            'compressed_responses': 0,
            'bytes_saved': 0,
            'compression_time': 0,
            'gzip_used': 0,
            'brotli_used': 0
        }
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Main compression middleware dispatch."""
        # Check if compression should be applied
        if self._should_skip_compression(request):
            return await call_next(request)
        
        # Execute request
        start_time = time.time()
        response = await call_next(request)
        
        self.compression_stats['total_responses'] += 1
        
        # Skip compression for non-200 responses or streaming responses
        if (response.status_code != 200 or 
            isinstance(response, StreamingResponse)):
            return response
        
        # Get response body
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk
        
        original_size = len(response_body)
        
        # Check if response is large enough to compress
        if original_size < self.minimum_size:
            # Return original response
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=response.headers,
                media_type=response.media_type
            )
        
        # Check content type
        content_type = response.headers.get('content-type', '')
        if self._should_skip_content_type(content_type):
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=response.headers,
                media_type=response.media_type
            )
        
        # Determine best compression method
        compression_method = self._get_best_compression(request)
        
        # Compress response
        compression_start = time.time()
        compressed_body, encoding = await self._compress_response(
            response_body, compression_method
        )
        compression_time = time.time() - compression_start
        
        if compressed_body and len(compressed_body) < original_size:
            # Update stats
            self.compression_stats['compressed_responses'] += 1
            self.compression_stats['bytes_saved'] += (original_size - len(compressed_body))
            self.compression_stats['compression_time'] += compression_time
            
            if encoding == 'gzip':
                self.compression_stats['gzip_used'] += 1
            elif encoding == 'br':
                self.compression_stats['brotli_used'] += 1
            
            # Create compressed response
            compressed_response = Response(
                content=compressed_body,
                status_code=response.status_code,
                media_type=response.media_type
            )
            
            # Copy original headers
            for name, value in response.headers.items():
                if name.lower() not in ['content-length', 'content-encoding']:
                    compressed_response.headers[name] = value
            
            # Add compression headers
            compressed_response.headers['content-encoding'] = encoding
            compressed_response.headers['content-length'] = str(len(compressed_body))
            compressed_response.headers['x-compression-ratio'] = f"{original_size}/{len(compressed_body)}"
            compressed_response.headers['x-compression-time'] = f"{compression_time:.4f}s"
            
            logger.debug(
                f"Compressed response: {original_size}→{len(compressed_body)} bytes "
                f"({encoding}) in {compression_time:.4f}s"
            )
            
            return compressed_response
        
        # Return original if compression didn't help
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=response.headers,
            media_type=response.media_type
        )
    
    def _should_skip_compression(self, request: Request) -> bool:
        """Check if compression should be skipped for this request."""
        # Skip if path is excluded
        path = request.url.path
        if any(path.startswith(excluded) for excluded in self.excluded_paths):
            return True
        
        # Skip if client doesn't accept compressed responses
        accept_encoding = request.headers.get('accept-encoding', '').lower()
        if not ('gzip' in accept_encoding or 'br' in accept_encoding):
            return True
        
        return False
    
    def _should_skip_content_type(self, content_type: str) -> bool:
        """Check if content type should be excluded from compression."""
        content_type_lower = content_type.lower()
        return any(excluded in content_type_lower for excluded in self.excluded_types)
    
    def _get_best_compression(self, request: Request) -> str:
        """Determine the best compression method for the client."""
        accept_encoding = request.headers.get('accept-encoding', '').lower()
        
        # Prefer Brotli if available and supported
        if BROTLI_AVAILABLE and 'br' in accept_encoding:
            return 'brotli'
        elif 'gzip' in accept_encoding:
            return 'gzip'
        
        return 'none'
    
    async def _compress_response(
        self, 
        response_body: bytes, 
        method: str
    ) -> tuple[Optional[bytes], str]:
        """Compress response body using specified method."""
        if method == 'brotli' and BROTLI_AVAILABLE:
            try:
                # Run compression in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                compressed = await loop.run_in_executor(
                    None, 
                    lambda: brotli.compress(response_body, quality=self.brotli_level)
                )
                return compressed, 'br'
            except Exception as e:
                logger.warning(f"Brotli compression failed: {e}")
                # Fall back to gzip
                method = 'gzip'
        
        if method == 'gzip':
            try:
                # Run compression in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                compressed = await loop.run_in_executor(
                    None,
                    lambda: gzip.compress(response_body, compresslevel=self.gzip_level)
                )
                return compressed, 'gzip'
            except Exception as e:
                logger.warning(f"Gzip compression failed: {e}")
        
        return None, 'none'
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression performance statistics."""
        total = self.compression_stats['total_responses']
        compressed = self.compression_stats['compressed_responses']
        
        compression_rate = (compressed / total) if total > 0 else 0
        avg_compression_time = (
            self.compression_stats['compression_time'] / compressed
            if compressed > 0 else 0
        )
        
        return {
            'total_responses': total,
            'compressed_responses': compressed,
            'compression_rate': compression_rate,
            'bytes_saved': self.compression_stats['bytes_saved'],
            'average_compression_time': avg_compression_time,
            'gzip_used': self.compression_stats['gzip_used'],
            'brotli_used': self.compression_stats['brotli_used'],
            'brotli_available': BROTLI_AVAILABLE
        }


# Streaming compression for large responses
class StreamingCompressionMiddleware(BaseHTTPMiddleware):
    """
    Streaming compression middleware for large responses.
    
    Compresses data as it's being generated to reduce memory usage
    and improve time-to-first-byte for large responses.
    """
    
    def __init__(self, app, chunk_size: int = 8192):
        super().__init__(app)
        self.chunk_size = chunk_size
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Dispatch with streaming compression support."""
        response = await call_next(request)
        
        # Only handle streaming responses
        if not isinstance(response, StreamingResponse):
            return response
        
        # Check compression support
        accept_encoding = request.headers.get('accept-encoding', '').lower()
        if 'gzip' not in accept_encoding:
            return response
        
        # Create compressed streaming response
        async def compress_stream():
            compressor = gzip.GzipFile(fileobj=None, mode='wb')
            
            async for chunk in response.body_iterator:
                if isinstance(chunk, str):
                    chunk = chunk.encode('utf-8')
                
                compressed_chunk = compressor.compress(chunk)
                if compressed_chunk:
                    yield compressed_chunk
            
            # Final chunk
            final_chunk = compressor.flush()
            if final_chunk:
                yield final_chunk
        
        # Create new streaming response
        compressed_response = StreamingResponse(
            compress_stream(),
            status_code=response.status_code,
            media_type=response.media_type
        )
        
        # Copy headers
        for name, value in response.headers.items():
            if name.lower() not in ['content-length', 'content-encoding']:
                compressed_response.headers[name] = value
        
        # Add compression headers
        compressed_response.headers['content-encoding'] = 'gzip'
        compressed_response.headers['transfer-encoding'] = 'chunked'
        
        return compressed_response


def setup_compression_middleware(
    app,
    minimum_size: int = 500,
    gzip_level: int = 6,
    brotli_level: int = 4,
    enable_streaming: bool = True
):
    """Setup compression middleware for FastAPI app."""
    # Add main compression middleware
    app.add_middleware(
        CompressionMiddleware,
        minimum_size=minimum_size,
        gzip_level=gzip_level,
        brotli_level=brotli_level
    )
    
    # Add streaming compression if enabled
    if enable_streaming:
        app.add_middleware(StreamingCompressionMiddleware)
    
    logger.info(
        f"Compression middleware enabled: gzip_level={gzip_level}, "
        f"brotli_level={brotli_level}, brotli_available={BROTLI_AVAILABLE}"
    )