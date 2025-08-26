#!/usr/bin/env python3
"""
Streaming File Processor - Scalability Solution for 10,000+ Files
Implements memory-efficient streaming processing to handle massive file collections

SCALABILITY TARGETS:
- Current: ~260 files processed
- Target: 10,000+ files (38x increase)
- Memory usage: Constant O(1) instead of O(n)
- Processing time: Sub-linear scaling through parallelization
"""

import asyncio
import aiofiles
import logging
import time
import os
from typing import AsyncGenerator, Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from queue import Queue
import threading
import psutil

logger = logging.getLogger(__name__)


@dataclass
class StreamingStats:
    """Statistics for streaming file processing"""
    files_processed: int = 0
    bytes_processed: int = 0
    processing_time_ms: float = 0
    memory_peak_mb: float = 0
    throughput_files_per_sec: float = 0
    throughput_mb_per_sec: float = 0
    errors: int = 0
    batch_count: int = 0


@dataclass
class ProcessingResult:
    """Result from file processing"""
    file_path: str
    success: bool
    processing_time_ms: float
    file_size_bytes: int
    result_data: Any = None
    error_message: str = ""


class StreamingFileProcessor:
    """
    Memory-efficient streaming file processor for massive file collections
    """
    
    def __init__(
        self,
        batch_size: int = 100,
        max_workers: int = None,
        max_memory_mb: int = 200,
        enable_multiprocessing: bool = True
    ):
        self.batch_size = batch_size
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) * 4)
        self.max_memory_mb = max_memory_mb
        self.enable_multiprocessing = enable_multiprocessing
        
        # Processing resources
        self.thread_executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="file_processor"
        )
        
        if enable_multiprocessing:
            self.process_executor = ProcessPoolExecutor(
                max_workers=min(8, os.cpu_count() or 1)
            )
        else:
            self.process_executor = None
            
        # Statistics tracking
        self.stats = StreamingStats()
        self.processing_queue = asyncio.Queue(maxsize=1000)
        self.result_queue = asyncio.Queue()
        
        # Memory monitoring
        self.memory_monitor_active = False
        self.memory_warnings = 0
        
    async def process_files_streaming(
        self,
        file_paths: List[str],
        processor_func: Callable[[str], Any],
        chunk_processor: Optional[Callable[[List[Any]], Any]] = None
    ) -> AsyncGenerator[ProcessingResult, None]:
        """
        Process files in streaming fashion with constant memory usage
        
        Args:
            file_paths: List of file paths to process
            processor_func: Function to process each file
            chunk_processor: Optional function to process results in chunks
        """
        logger.info(f"🚀 Starting streaming processing of {len(file_paths)} files")
        logger.info(f"   Batch size: {self.batch_size}")
        logger.info(f"   Max workers: {self.max_workers}")
        logger.info(f"   Memory limit: {self.max_memory_mb}MB")
        
        # Start memory monitoring
        memory_task = asyncio.create_task(self._monitor_memory())
        
        # Start statistics tracking
        processing_start = time.time()
        
        try:
            # Process files in batches
            async for batch_results in self._process_batches_streaming(file_paths, processor_func):
                # Yield individual results
                for result in batch_results:
                    self.stats.files_processed += 1
                    self.stats.bytes_processed += result.file_size_bytes
                    
                    if not result.success:
                        self.stats.errors += 1
                        
                    yield result
                    
                # Process chunk results if chunk processor provided
                if chunk_processor:
                    try:
                        chunk_result = await asyncio.get_event_loop().run_in_executor(
                            self.thread_executor,
                            chunk_processor,
                            [r.result_data for r in batch_results if r.success]
                        )
                        logger.debug(f"Chunk processing completed")
                    except Exception as e:
                        logger.error(f"Chunk processing failed: {e}")
                        
                # Memory check after each batch
                await self._check_memory_usage()
                
        finally:
            # Stop memory monitoring
            self.memory_monitor_active = False
            memory_task.cancel()
            
            # Update final statistics
            processing_time = (time.time() - processing_start) * 1000
            self.stats.processing_time_ms = processing_time
            
            if processing_time > 0:
                self.stats.throughput_files_per_sec = (self.stats.files_processed / processing_time) * 1000
                self.stats.throughput_mb_per_sec = (self.stats.bytes_processed / 1024 / 1024 / processing_time) * 1000
                
            logger.info(f"✅ Streaming processing completed:")
            logger.info(f"   Files processed: {self.stats.files_processed}")
            logger.info(f"   Processing time: {processing_time:.1f}ms")
            logger.info(f"   Throughput: {self.stats.throughput_files_per_sec:.1f} files/sec")
            logger.info(f"   Memory peak: {self.stats.memory_peak_mb:.1f}MB")
            logger.info(f"   Errors: {self.stats.errors}")
            
    async def _process_batches_streaming(
        self,
        file_paths: List[str], 
        processor_func: Callable[[str], Any]
    ) -> AsyncGenerator[List[ProcessingResult], None]:
        """Process files in streaming batches"""
        
        batch = []
        batch_number = 0
        
        for file_path in file_paths:
            batch.append(file_path)
            
            # Process batch when full
            if len(batch) >= self.batch_size:
                batch_number += 1
                logger.debug(f"Processing batch {batch_number} ({len(batch)} files)")
                
                batch_results = await self._process_batch_parallel(batch, processor_func)
                yield batch_results
                
                # Clear batch and force garbage collection
                batch.clear()
                
                # Yield control to event loop
                await asyncio.sleep(0)
                
        # Process remaining files in final batch
        if batch:
            batch_number += 1
            logger.debug(f"Processing final batch {batch_number} ({len(batch)} files)")
            batch_results = await self._process_batch_parallel(batch, processor_func)
            yield batch_results
            
        self.stats.batch_count = batch_number
        
    async def _process_batch_parallel(
        self,
        file_paths: List[str],
        processor_func: Callable[[str], Any]
    ) -> List[ProcessingResult]:
        """Process a batch of files in parallel"""
        
        batch_start = time.time()
        
        # Create processing tasks
        tasks = []
        for file_path in file_paths:
            if self.enable_multiprocessing and self.process_executor:
                # Use process pool for CPU-intensive tasks
                task = asyncio.get_event_loop().run_in_executor(
                    self.process_executor,
                    self._process_single_file_sync,
                    file_path, processor_func
                )
            else:
                # Use thread pool
                task = asyncio.get_event_loop().run_in_executor(
                    self.thread_executor,
                    self._process_single_file_sync,
                    file_path, processor_func
                )
            tasks.append(task)
            
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ProcessingResult(
                    file_path=file_paths[i],
                    success=False,
                    processing_time_ms=0,
                    file_size_bytes=0,
                    error_message=str(result)
                ))
            else:
                processed_results.append(result)
                
        batch_time = (time.time() - batch_start) * 1000
        logger.debug(f"Batch completed in {batch_time:.1f}ms")
        
        return processed_results
        
    def _process_single_file_sync(
        self,
        file_path: str,
        processor_func: Callable[[str], Any]
    ) -> ProcessingResult:
        """Process a single file synchronously"""
        
        start_time = time.time()
        
        try:
            # Get file size
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            # Process file
            result_data = processor_func(file_path)
            
            processing_time = (time.time() - start_time) * 1000
            
            return ProcessingResult(
                file_path=file_path,
                success=True,
                processing_time_ms=processing_time,
                file_size_bytes=file_size,
                result_data=result_data
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            
            return ProcessingResult(
                file_path=file_path,
                success=False,
                processing_time_ms=processing_time,
                file_size_bytes=0,
                error_message=str(e)
            )
            
    async def _monitor_memory(self):
        """Monitor memory usage during processing"""
        self.memory_monitor_active = True
        
        while self.memory_monitor_active:
            try:
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                # Update peak memory
                if memory_mb > self.stats.memory_peak_mb:
                    self.stats.memory_peak_mb = memory_mb
                    
                # Check if approaching memory limit
                if memory_mb > self.max_memory_mb * 0.9:
                    self.memory_warnings += 1
                    logger.warning(f"⚠️ High memory usage: {memory_mb:.1f}MB")
                    
                    # Trigger garbage collection
                    import gc
                    gc.collect()
                    
                # Check if exceeding memory limit
                if memory_mb > self.max_memory_mb:
                    logger.error(f"🚨 Memory limit exceeded: {memory_mb:.1f}MB")
                    # Could implement emergency memory cleanup here
                    
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                await asyncio.sleep(5)
                
    async def _check_memory_usage(self):
        """Check current memory usage"""
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.max_memory_mb * 0.8:
                logger.debug(f"Memory usage: {memory_mb:.1f}MB")
                
                # Force garbage collection
                import gc
                gc.collect()
                
        except Exception as e:
            logger.debug(f"Memory check error: {e}")
            
    async def process_directory_streaming(
        self,
        directory_path: str,
        file_pattern: str = "**/*.py",
        processor_func: Callable[[str], Any] = None
    ) -> AsyncGenerator[ProcessingResult, None]:
        """Process all files in a directory using streaming"""
        
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")
            
        # Find all matching files
        file_paths = [str(p) for p in directory.glob(file_pattern) if p.is_file()]
        
        logger.info(f"Found {len(file_paths)} files matching pattern '{file_pattern}'")
        
        # Use default processor if none provided
        if processor_func is None:
            processor_func = self._default_file_processor
            
        # Process files using streaming
        async for result in self.process_files_streaming(file_paths, processor_func):
            yield result
            
    def _default_file_processor(self, file_path: str) -> Dict[str, Any]:
        """Default file processor - analyzes file metadata"""
        try:
            path = Path(file_path)
            
            return {
                'file_path': file_path,
                'file_name': path.name,
                'file_size': path.stat().st_size,
                'file_type': path.suffix,
                'modified_time': path.stat().st_mtime,
                'line_count': self._count_lines(file_path) if path.suffix == '.py' else 0
            }
            
        except Exception as e:
            return {'error': str(e)}
            
    def _count_lines(self, file_path: str) -> int:
        """Count lines in a text file efficiently"""
        try:
            with open(file_path, 'rb') as f:
                return sum(1 for _ in f)
        except Exception:
            return 0
            
    def get_statistics(self) -> StreamingStats:
        """Get current processing statistics"""
        return self.stats
        
    async def cleanup(self):
        """Cleanup resources"""
        try:
            self.memory_monitor_active = False
            
            if self.thread_executor:
                self.thread_executor.shutdown(wait=True)
                
            if self.process_executor:
                self.process_executor.shutdown(wait=True)
                
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# Specialized processors for different file types

class CodeAnalysisStreamingProcessor(StreamingFileProcessor):
    """Specialized streaming processor for code analysis"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    async def analyze_codebase_streaming(
        self,
        codebase_path: str,
        language_extensions: List[str] = None
    ) -> Dict[str, Any]:
        """Analyze entire codebase using streaming processing"""
        
        if language_extensions is None:
            language_extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c']
            
        analysis_results = {
            'total_files': 0,
            'total_lines': 0,
            'total_size_bytes': 0,
            'language_breakdown': {},
            'largest_files': [],
            'analysis_time_ms': 0
        }
        
        start_time = time.time()
        
        # Find all code files
        codebase = Path(codebase_path)
        file_paths = []
        
        for ext in language_extensions:
            files = list(codebase.rglob(f"*{ext}"))
            file_paths.extend([str(f) for f in files if f.is_file()])
            
        logger.info(f"Found {len(file_paths)} code files to analyze")
        
        # Process files using streaming
        async for result in self.process_files_streaming(file_paths, self._analyze_code_file):
            if result.success and result.result_data:
                data = result.result_data
                
                analysis_results['total_files'] += 1
                analysis_results['total_lines'] += data.get('line_count', 0)
                analysis_results['total_size_bytes'] += result.file_size_bytes
                
                # Language breakdown
                file_ext = Path(result.file_path).suffix
                if file_ext not in analysis_results['language_breakdown']:
                    analysis_results['language_breakdown'][file_ext] = {
                        'files': 0, 'lines': 0, 'size_bytes': 0
                    }
                    
                analysis_results['language_breakdown'][file_ext]['files'] += 1
                analysis_results['language_breakdown'][file_ext]['lines'] += data.get('line_count', 0)
                analysis_results['language_breakdown'][file_ext]['size_bytes'] += result.file_size_bytes
                
                # Track largest files
                analysis_results['largest_files'].append({
                    'path': result.file_path,
                    'size_bytes': result.file_size_bytes,
                    'lines': data.get('line_count', 0)
                })
                
                # Keep only top 10 largest files
                analysis_results['largest_files'].sort(key=lambda x: x['size_bytes'], reverse=True)
                analysis_results['largest_files'] = analysis_results['largest_files'][:10]
                
        analysis_results['analysis_time_ms'] = (time.time() - start_time) * 1000
        
        return analysis_results
        
    def _analyze_code_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single code file"""
        try:
            path = Path(file_path)
            
            # Count lines efficiently
            line_count = 0
            with open(file_path, 'rb') as f:
                line_count = sum(1 for _ in f)
                
            # Basic code analysis
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            return {
                'file_path': file_path,
                'line_count': line_count,
                'char_count': len(content),
                'language': path.suffix,
                'complexity_estimate': self._estimate_complexity(content),
                'import_count': content.count('import ') + content.count('from '),
                'function_count': content.count('def ') + content.count('function '),
                'class_count': content.count('class ')
            }
            
        except Exception as e:
            return {'error': str(e)}
            
    def _estimate_complexity(self, content: str) -> int:
        """Estimate code complexity based on keywords"""
        complexity_keywords = ['if', 'for', 'while', 'try', 'except', 'elif', 'else']
        return sum(content.count(keyword) for keyword in complexity_keywords)


# Convenience functions

async def process_files_fast(
    file_paths: List[str],
    processor_func: Callable[[str], Any],
    batch_size: int = 100,
    max_workers: int = None
) -> List[ProcessingResult]:
    """Fast file processing with streaming"""
    
    processor = StreamingFileProcessor(
        batch_size=batch_size,
        max_workers=max_workers
    )
    
    results = []
    async for result in processor.process_files_streaming(file_paths, processor_func):
        results.append(result)
        
    await processor.cleanup()
    return results


async def analyze_codebase_fast(codebase_path: str) -> Dict[str, Any]:
    """Fast codebase analysis using streaming"""
    
    processor = CodeAnalysisStreamingProcessor(batch_size=50)
    result = await processor.analyze_codebase_streaming(codebase_path)
    await processor.cleanup()
    
    return result


if __name__ == "__main__":
    async def main():
        # Test streaming file processing
        print("🚀 Testing streaming file processor...")
        
        # Analyze current codebase
        result = await analyze_codebase_fast("/home/tekkadmin/claude-tui/src")
        
        print(f"📊 Codebase Analysis Results:")
        print(f"   Total files: {result['total_files']}")
        print(f"   Total lines: {result['total_lines']:,}")
        print(f"   Total size: {result['total_size_bytes'] / 1024 / 1024:.1f}MB")
        print(f"   Analysis time: {result['analysis_time_ms']:.1f}ms")
        print(f"   Languages: {list(result['language_breakdown'].keys())}")
        
    asyncio.run(main())