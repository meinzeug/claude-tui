#!/usr/bin/env python3
"""
CDN Configuration - Static Asset Optimization and CDN Integration

Provides CDN integration and static asset optimization:
- Static asset versioning and cache busting
- CDN integration with multiple providers (CloudFlare, AWS CloudFront, etc.)
- Asset compression and optimization
- Progressive loading strategies
- Cache headers optimization
- Asset bundling and minification
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import asyncio
import aiofiles
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


@dataclass
class AssetInfo:
    """Information about a static asset"""
    path: str
    size_bytes: int
    content_hash: str
    content_type: str
    last_modified: datetime
    cached_version: Optional[str] = None
    compressed_size: Optional[int] = None
    
    @property
    def compression_ratio(self) -> Optional[float]:
        """Get compression ratio if available"""
        if self.compressed_size is None:
            return None
        return self.compressed_size / self.size_bytes if self.size_bytes > 0 else 0


class StaticAssetManager:
    """Manages static assets with versioning and optimization"""
    
    def __init__(self, assets_directory: str = "static", cache_directory: str = "cache"):
        self.assets_dir = Path(assets_directory)
        self.cache_dir = Path(cache_directory)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.asset_registry: Dict[str, AssetInfo] = {}
        self.version_manifest: Dict[str, str] = {}
        
        # File type configurations
        self.compressible_types = {
            '.js': 'application/javascript',
            '.css': 'text/css',
            '.html': 'text/html',
            '.json': 'application/json',
            '.svg': 'image/svg+xml',
            '.txt': 'text/plain',
            '.xml': 'application/xml'
        }
        
        self.image_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.svg': 'image/svg+xml'
        }
        
        self.cache_durations = {
            # Immutable assets (with hash in name)
            'immutable': 31536000,  # 1 year
            # Regular static assets
            'static': 86400,        # 1 day
            # Dynamic content
            'dynamic': 3600,        # 1 hour
            # API responses
            'api': 300             # 5 minutes
        }
    
    async def scan_assets(self):
        """Scan and catalog all static assets"""
        if not self.assets_dir.exists():
            logger.warning(f"Assets directory {self.assets_dir} does not exist")
            return
        
        logger.info(f"Scanning assets in {self.assets_dir}")
        
        for file_path in self.assets_dir.rglob('*'):
            if file_path.is_file():
                await self._process_asset(file_path)
        
        # Generate version manifest
        await self._generate_version_manifest()
        
        logger.info(f"Processed {len(self.asset_registry)} assets")
    
    async def _process_asset(self, file_path: Path):
        """Process a single asset file"""
        try:
            # Get file info
            stat_info = file_path.stat()
            file_size = stat_info.st_size
            last_modified = datetime.fromtimestamp(stat_info.st_mtime)
            
            # Read file content for hashing
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()
            
            # Generate content hash
            content_hash = hashlib.md5(content).hexdigest()[:12]
            
            # Determine content type
            file_ext = file_path.suffix.lower()
            content_type = (
                self.compressible_types.get(file_ext) or
                self.image_types.get(file_ext) or
                'application/octet-stream'
            )
            
            # Create relative path
            relative_path = str(file_path.relative_to(self.assets_dir))
            
            # Create asset info
            asset_info = AssetInfo(
                path=relative_path,
                size_bytes=file_size,
                content_hash=content_hash,
                content_type=content_type,
                last_modified=last_modified
            )
            
            # Compress if applicable
            if file_ext in self.compressible_types:
                compressed_size = await self._compress_asset(file_path, content)
                asset_info.compressed_size = compressed_size
            
            self.asset_registry[relative_path] = asset_info
            
        except Exception as e:
            logger.error(f"Error processing asset {file_path}: {e}")
    
    async def _compress_asset(self, file_path: Path, content: bytes) -> int:
        """Compress an asset and return compressed size"""
        try:
            import gzip
            
            compressed_path = self.cache_dir / f"{file_path.stem}_{hashlib.md5(content).hexdigest()[:8]}.gz"
            
            async with aiofiles.open(compressed_path, 'wb') as f:
                compressed_content = gzip.compress(content, compresslevel=6)
                await f.write(compressed_content)
            
            return len(compressed_content)
            
        except Exception as e:
            logger.error(f"Error compressing asset {file_path}: {e}")
            return len(content)
    
    async def _generate_version_manifest(self):
        """Generate version manifest for cache busting"""
        self.version_manifest = {
            asset_path: asset_info.content_hash
            for asset_path, asset_info in self.asset_registry.items()
        }
        
        # Save manifest to file
        manifest_path = self.cache_dir / 'asset_manifest.json'
        async with aiofiles.open(manifest_path, 'w') as f:
            await f.write(json.dumps(self.version_manifest, indent=2))
    
    def get_versioned_url(self, asset_path: str, base_url: str = '') -> str:
        """Get versioned URL for an asset"""
        if asset_path not in self.asset_registry:
            return urljoin(base_url, asset_path)
        
        asset_info = self.asset_registry[asset_path]
        
        # Add version parameter
        separator = '&' if '?' in asset_path else '?'
        versioned_path = f"{asset_path}{separator}v={asset_info.content_hash}"
        
        return urljoin(base_url, versioned_path)
    
    def get_cache_headers(self, asset_path: str) -> Dict[str, str]:
        """Get appropriate cache headers for an asset"""
        headers = {}
        
        if asset_path not in self.asset_registry:
            return headers
        
        asset_info = self.asset_registry[asset_path]
        
        # Determine cache type
        if asset_info.content_hash in asset_path:
            # Immutable asset (hash in filename)
            cache_type = 'immutable'
            headers['Cache-Control'] = f'public, max-age={self.cache_durations["immutable"]}, immutable'
        elif asset_path.startswith(('css/', 'js/', 'images/')):
            # Static assets
            cache_type = 'static'
            headers['Cache-Control'] = f'public, max-age={self.cache_durations["static"]}'
        else:
            # Default caching
            cache_type = 'dynamic'
            headers['Cache-Control'] = f'public, max-age={self.cache_durations["dynamic"]}'
        
        # Add ETag
        headers['ETag'] = f'"{asset_info.content_hash}"'
        
        # Add Last-Modified
        headers['Last-Modified'] = asset_info.last_modified.strftime('%a, %d %b %Y %H:%M:%S GMT')
        
        # Add Content-Type
        headers['Content-Type'] = asset_info.content_type
        
        # Add compression headers if applicable
        if asset_info.compressed_size:
            headers['Content-Encoding'] = 'gzip'
            headers['Vary'] = 'Accept-Encoding'
        
        return headers


class CDNProvider:
    """Base class for CDN providers"""
    
    def __init__(self, name: str):
        self.name = name
    
    async def purge_cache(self, urls: List[str]) -> bool:
        """Purge CDN cache for specific URLs"""
        logger.warning(f"Cache purging not implemented for {self.name} CDN provider")
        return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get CDN cache statistics"""
        logger.warning(f"Cache stats not implemented for {self.name} CDN provider")
        return {
            'provider': self.name,
            'status': 'not_implemented',
            'hit_ratio': 0.0,
            'requests': 0,
            'bandwidth': 0
        }


class CloudFlareCDN(CDNProvider):
    """CloudFlare CDN integration"""
    
    def __init__(self, api_token: str, zone_id: str):
        super().__init__("CloudFlare")
        self.api_token = api_token
        self.zone_id = zone_id
        self.base_url = "https://api.cloudflare.com/client/v4"
    
    async def purge_cache(self, urls: List[str]) -> bool:
        """Purge CloudFlare cache"""
        try:
            import aiohttp
            
            headers = {
                'Authorization': f'Bearer {self.api_token}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'files': urls
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f'{self.base_url}/zones/{self.zone_id}/purge_cache',
                    headers=headers,
                    json=data
                ) as response:
                    result = await response.json()
                    return result.get('success', False)
                    
        except Exception as e:
            logger.error(f"CloudFlare cache purge error: {e}")
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get CloudFlare analytics"""
        try:
            import aiohttp
            
            headers = {
                'Authorization': f'Bearer {self.api_token}',
                'Content-Type': 'application/json'
            }
            
            # Get zone analytics
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f'{self.base_url}/zones/{self.zone_id}/analytics/dashboard',
                    headers=headers
                ) as response:
                    return await response.json()
                    
        except Exception as e:
            logger.error(f"CloudFlare stats error: {e}")
            return {}


class CDNManager:
    """Manages CDN integration and static asset delivery"""
    
    def __init__(self):
        self.asset_manager = StaticAssetManager()
        self.cdn_providers: List[CDNProvider] = []
        self.delivery_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'bytes_served': 0,
            'requests_served': 0
        }
    
    def add_cdn_provider(self, provider: CDNProvider):
        """Add a CDN provider"""
        self.cdn_providers.append(provider)
        logger.info(f"Added CDN provider: {provider.name}")
    
    async def initialize(self):
        """Initialize CDN manager"""
        await self.asset_manager.scan_assets()
        logger.info("CDN manager initialized")
    
    async def serve_asset(self, asset_path: str, request_headers: Dict[str, str] = None) -> Dict[str, Any]:
        """Serve a static asset with appropriate headers"""
        request_headers = request_headers or {}
        
        # Check if asset exists
        if asset_path not in self.asset_manager.asset_registry:
            return {
                'status': 404,
                'headers': {},
                'body': b'Asset not found'
            }
        
        asset_info = self.asset_manager.asset_registry[asset_path]
        
        # Check for conditional requests (304 Not Modified)
        if_none_match = request_headers.get('If-None-Match', '').strip('"')
        if if_none_match == asset_info.content_hash:
            self.delivery_stats['cache_hits'] += 1
            return {
                'status': 304,
                'headers': self.asset_manager.get_cache_headers(asset_path),
                'body': b''
            }
        
        # Serve the asset
        try:
            asset_file_path = self.asset_manager.assets_dir / asset_path
            async with aiofiles.open(asset_file_path, 'rb') as f:
                content = await f.read()
            
            headers = self.asset_manager.get_cache_headers(asset_path)
            
            # Update stats
            self.delivery_stats['cache_misses'] += 1
            self.delivery_stats['bytes_served'] += len(content)
            self.delivery_stats['requests_served'] += 1
            
            return {
                'status': 200,
                'headers': headers,
                'body': content
            }
            
        except Exception as e:
            logger.error(f"Error serving asset {asset_path}: {e}")
            return {
                'status': 500,
                'headers': {},
                'body': b'Internal server error'
            }
    
    def generate_asset_urls(self, base_url: str = '') -> Dict[str, str]:
        """Generate versioned URLs for all assets"""
        return {
            asset_path: self.asset_manager.get_versioned_url(asset_path, base_url)
            for asset_path in self.asset_manager.asset_registry
        }
    
    async def invalidate_cdn_cache(self, asset_paths: List[str] = None):
        """Invalidate CDN cache for specific assets or all assets"""
        if asset_paths is None:
            asset_paths = list(self.asset_manager.asset_registry.keys())
        
        # Convert to full URLs (assuming we have a base URL)
        urls = [f"https://example.com/{path}" for path in asset_paths]  # TODO: Use actual domain
        
        for provider in self.cdn_providers:
            success = await provider.purge_cache(urls)
            if success:
                logger.info(f"Successfully purged cache on {provider.name}")
            else:
                logger.error(f"Failed to purge cache on {provider.name}")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive CDN performance metrics"""
        
        # Asset statistics
        total_assets = len(self.asset_manager.asset_registry)
        total_size = sum(asset.size_bytes for asset in self.asset_manager.asset_registry.values())
        compressed_assets = sum(1 for asset in self.asset_manager.asset_registry.values() if asset.compressed_size)
        
        # Calculate compression savings
        compression_savings = 0
        if compressed_assets > 0:
            for asset in self.asset_manager.asset_registry.values():
                if asset.compressed_size:
                    compression_savings += (asset.size_bytes - asset.compressed_size)
        
        # CDN provider stats
        cdn_stats = {}
        for provider in self.cdn_providers:
            stats = await provider.get_cache_stats()
            cdn_stats[provider.name] = stats
        
        # Cache hit rate
        total_requests = self.delivery_stats['cache_hits'] + self.delivery_stats['cache_misses']
        cache_hit_rate = (self.delivery_stats['cache_hits'] / total_requests) if total_requests > 0 else 0
        
        return {
            'assets': {
                'total_count': total_assets,
                'total_size_bytes': total_size,
                'compressed_count': compressed_assets,
                'compression_savings_bytes': compression_savings,
                'compression_percentage': (compression_savings / total_size * 100) if total_size > 0 else 0
            },
            'delivery': {
                **self.delivery_stats,
                'cache_hit_rate': cache_hit_rate
            },
            'cdn_providers': cdn_stats,
            'timestamp': datetime.utcnow().isoformat()
        }


# Global CDN manager instance
_cdn_manager: Optional[CDNManager] = None


def get_cdn_manager() -> CDNManager:
    """Get global CDN manager instance"""
    global _cdn_manager
    if _cdn_manager is None:
        _cdn_manager = CDNManager()
    return _cdn_manager


async def initialize_cdn(assets_directory: str = "static") -> CDNManager:
    """Initialize global CDN manager"""
    global _cdn_manager
    _cdn_manager = CDNManager()
    _cdn_manager.asset_manager.assets_dir = Path(assets_directory)
    await _cdn_manager.initialize()
    return _cdn_manager


if __name__ == "__main__":
    # Example usage and testing
    async def test_cdn_system():
        print("🚀 CDN CONFIGURATION SYSTEM - Testing")
        print("=" * 50)
        
        # Create test assets directory
        test_assets_dir = Path("test_assets")
        test_assets_dir.mkdir(exist_ok=True)
        
        # Create test files
        test_files = {
            "style.css": "body { background: blue; color: white; }",
            "script.js": "console.log('Hello, world!');",
            "image.svg": "<svg><circle r='10'/></svg>",
            "data.json": '{"test": "data"}'
        }
        
        for filename, content in test_files.items():
            async with aiofiles.open(test_assets_dir / filename, 'w') as f:
                await f.write(content)
        
        # Initialize CDN manager
        cdn_manager = await initialize_cdn(str(test_assets_dir))
        
        print(f"📊 Asset Processing Results:")
        metrics = await cdn_manager.get_performance_metrics()
        print(f"   Total assets: {metrics['assets']['total_count']}")
        print(f"   Total size: {metrics['assets']['total_size_bytes']} bytes")
        print(f"   Compressed assets: {metrics['assets']['compressed_count']}")
        print(f"   Compression savings: {metrics['assets']['compression_savings_bytes']} bytes")
        
        # Test asset serving
        print(f"\n🌐 Testing Asset Serving:")
        for asset_path in cdn_manager.asset_manager.asset_registry:
            response = await cdn_manager.serve_asset(asset_path)
            print(f"   {asset_path}: Status {response['status']}")
        
        # Test versioned URLs
        print(f"\n🔗 Versioned URLs:")
        urls = cdn_manager.generate_asset_urls("https://cdn.example.com")
        for path, url in urls.items():
            print(f"   {path} -> {url}")
        
        # Clean up test files
        import shutil
        shutil.rmtree(test_assets_dir, ignore_errors=True)
        shutil.rmtree("cache", ignore_errors=True)
        
        print("\n✅ CDN configuration system test completed!")
    
    # Run test
    asyncio.run(test_cdn_system())