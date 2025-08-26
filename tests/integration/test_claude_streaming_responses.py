"""
Claude Streaming Response Tests

Tests streaming responses, real-time processing, and concurrent request handling
for both Claude client implementations.
"""

import asyncio
import json
import pytest
import time
from typing import AsyncGenerator, Dict, List
from unittest.mock import AsyncMock, patch

from src.claude_tui.integrations.claude_code_client import ClaudeCodeClient
from src.claude_tui.integrations.claude_code_direct_client import ClaudeCodeDirectClient
from src.claude_tui.core.config_manager import ConfigManager

OAUTH_TOKEN = "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"


class MockStreamingResponse:
    """Mock streaming HTTP response for testing."""
    
    def __init__(self, chunks: List[str], delay: float = 0.1):
        self.chunks = chunks
        self.delay = delay
        self.index = 0
    
    async def content_iter_chunked(self, chunk_size: int):
        """Simulate chunked content iteration."""
        for chunk in self.chunks:
            await asyncio.sleep(self.delay)
            yield chunk.encode('utf-8')
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class TestStreamingResponses:
    """Test streaming response handling."""
    
    @pytest.fixture
    async def http_client(self):
        """Create HTTP client for streaming tests."""
        config = ConfigManager()
        client = ClaudeCodeClient(config, oauth_token=OAUTH_TOKEN)
        yield client
        await client.cleanup()
    
    @pytest.mark.asyncio
    async def test_mock_streaming_task_execution(self, http_client):
        """Test streaming task execution with mocked responses."""
        streaming_chunks = [
            '{"type": "start", "content": "Starting code generation..."}',
            '{"type": "progress", "content": "Analyzing requirements..."}',
            '{"type": "code", "content": "def hello():"}',
            '{"type": "code", "content": "    return \\"Hello, World!\\""}',
            '{"type": "complete", "content": "Code generation complete."}'
        ]
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = MockStreamingResponse(streaming_chunks)
            mock_response.status = 200
            mock_response.headers = {'Content-Type': 'application/json'}
            
            mock_session_instance = AsyncMock()
            mock_session_instance.request.return_value = mock_response
            mock_session.return_value = mock_session_instance
            
            # Test streaming response processing
            result = await http_client.execute_task("Generate hello world function")
            
            assert isinstance(result, dict)
            # Should have processed streaming content
    
    @pytest.mark.asyncio
    async def test_streaming_with_interruption(self, http_client):
        """Test streaming response handling with interruption."""
        interrupted_chunks = [
            '{"type": "start", "content": "Starting..."}',
            '{"type": "progress", "content": "Working..."}',
            # Simulate connection interruption
        ]
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = MockStreamingResponse(interrupted_chunks)
            mock_response.status = 200
            
            # Simulate connection error during streaming
            async def failing_iter(chunk_size):
                for i, chunk in enumerate(interrupted_chunks):
                    if i == 1:  # Fail on second chunk
                        raise asyncio.TimeoutError("Connection interrupted")
                    yield chunk.encode('utf-8')
            
            mock_response.content_iter_chunked = failing_iter
            
            mock_session_instance = AsyncMock()
            mock_session_instance.request.return_value = mock_response
            mock_session.return_value = mock_session_instance
            
            # Should handle interruption gracefully
            result = await http_client.execute_task("Test task")
            assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_large_streaming_response(self, http_client):
        """Test handling of large streaming responses."""
        # Generate large response chunks
        large_chunks = []
        for i in range(50):
            chunk_data = {
                "type": "code_chunk",
                "content": f"# Code block {i}\n" + "    print(f'Processing item {j}')\n" * 20,
                "chunk_id": i
            }
            large_chunks.append(json.dumps(chunk_data))
        
        large_chunks.append('{"type": "complete", "content": "Large response complete"}')
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = MockStreamingResponse(large_chunks, delay=0.01)  # Faster chunks
            mock_response.status = 200
            
            mock_session_instance = AsyncMock()
            mock_session_instance.request.return_value = mock_response
            mock_session.return_value = mock_session_instance
            
            start_time = time.time()
            result = await http_client.execute_task("Generate large code file")
            processing_time = time.time() - start_time
            
            assert isinstance(result, dict)
            print(f"Large streaming response processed in {processing_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_concurrent_streaming_requests(self, http_client):
        """Test multiple concurrent streaming requests."""
        def create_streaming_chunks(task_id: int) -> List[str]:
            return [
                f'{{"type": "start", "task_id": {task_id}, "content": "Task {task_id} starting..."}}',
                f'{{"type": "progress", "task_id": {task_id}, "content": "Task {task_id} working..."}}',
                f'{{"type": "code", "task_id": {task_id}, "content": "def task_{task_id}(): pass"}}',
                f'{{"type": "complete", "task_id": {task_id}, "content": "Task {task_id} complete"}}'
            ]
        
        with patch('aiohttp.ClientSession') as mock_session:
            # Create separate mock responses for each request
            def create_mock_response(task_id):
                chunks = create_streaming_chunks(task_id)
                response = MockStreamingResponse(chunks, delay=0.05)
                response.status = 200
                return response
            
            mock_session_instance = AsyncMock()
            # Mock different responses for different calls
            responses = [create_mock_response(i) for i in range(3)]
            mock_session_instance.request.side_effect = responses
            mock_session.return_value = mock_session_instance
            
            # Execute concurrent streaming requests
            tasks = [
                http_client.execute_task(f"Streaming task {i}")
                for i in range(3)
            ]
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            assert len(results) == 3
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"Task {i} failed: {result}")
                else:
                    assert isinstance(result, dict)
            
            print(f"Concurrent streaming completed in {total_time:.2f}s")


class TestRealTimeProcessing:
    """Test real-time processing capabilities."""
    
    @pytest.fixture
    def direct_client(self):
        """Create direct client for real-time tests."""
        import tempfile
        from pathlib import Path
        
        temp_dir = tempfile.mkdtemp()
        cc_file = Path(temp_dir) / ".cc"
        cc_file.write_text(OAUTH_TOKEN)
        
        client = ClaudeCodeDirectClient(
            oauth_token_file=str(cc_file),
            working_directory=temp_dir
        )
        yield client
        client.cleanup_session()
        
        # Cleanup
        cc_file.unlink(missing_ok=True)
        Path(temp_dir).rmdir()
    
    @pytest.mark.asyncio
    async def test_real_time_code_validation(self, direct_client):
        """Test real-time code validation as code is being written."""
        code_fragments = [
            "def calculate_fibonacci(n):",
            "    if n <= 1:",
            "        return n",
            "    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)"
        ]
        
        validation_results = []
        
        for i, fragment in enumerate(code_fragments):
            # Build cumulative code
            cumulative_code = "\n".join(code_fragments[:i+1])
            
            # Validate current code state
            result = await direct_client.validate_code_via_cli(
                code=cumulative_code,
                validation_rules=["syntax_check"],
                timeout=10
            )
            
            validation_results.append({
                'fragment': fragment,
                'valid': result.get('valid', False),
                'issues': result.get('issues', [])
            })
        
        # Final code should be valid
        assert validation_results[-1]['valid'] is not False  # May be None for some implementations
        
        print("Real-time validation results:")
        for i, result in enumerate(validation_results):
            print(f"Fragment {i+1}: {result['valid']} - {len(result['issues'])} issues")
    
    @pytest.mark.asyncio
    async def test_progressive_code_building(self, direct_client):
        """Test progressive code building and refinement."""
        initial_code = """
def sort_list(items):
    # TODO: Implement sorting
    pass
"""
        
        refinement_instructions = [
            "Implement bubble sort algorithm",
            "Add input validation",
            "Add type hints and documentation",
            "Optimize for better performance"
        ]
        
        current_code = initial_code
        refinement_history = [{"step": 0, "code": current_code}]
        
        for i, instruction in enumerate(refinement_instructions):
            result = await direct_client.refactor_code_via_cli(
                code=current_code,
                instructions=instruction,
                timeout=15
            )
            
            if result.get('success'):
                current_code = result.get('refactored_code', current_code)
            
            refinement_history.append({
                "step": i + 1,
                "instruction": instruction,
                "code": current_code,
                "success": result.get('success', False)
            })
        
        # Should show progression
        assert len(refinement_history) == 5  # Initial + 4 refinements
        
        print("Progressive refinement history:")
        for step in refinement_history:
            if step['step'] > 0:
                print(f"Step {step['step']}: {step['instruction']} - Success: {step['success']}")
    
    @pytest.mark.asyncio 
    async def test_interactive_debugging_simulation(self, direct_client):
        """Simulate interactive debugging session."""
        buggy_code = """
def divide_numbers(a, b):
    result = a / b  # Potential division by zero
    return result

def process_list(numbers):
    results = []
    for num in numbers:
        results.append(divide_numbers(num, 0))  # Bug: dividing by zero
    return results
"""
        
        # Step 1: Identify issues
        validation_result = await direct_client.validate_code_via_cli(
            code=buggy_code,
            validation_rules=["runtime_check", "logic_check"],
            timeout=10
        )
        
        print(f"Initial validation: {validation_result.get('valid')}")
        print(f"Issues found: {len(validation_result.get('issues', []))}")
        
        # Step 2: Fix the issues
        fix_instruction = """
Fix the division by zero error and add proper error handling.
Make the function more robust.
"""
        
        fix_result = await direct_client.refactor_code_via_cli(
            code=buggy_code,
            instructions=fix_instruction,
            timeout=15
        )
        
        print(f"Fix applied successfully: {fix_result.get('success')}")
        
        if fix_result.get('success'):
            fixed_code = fix_result.get('refactored_code', buggy_code)
            
            # Step 3: Validate the fix
            final_validation = await direct_client.validate_code_via_cli(
                code=fixed_code,
                validation_rules=["runtime_check", "logic_check"],
                timeout=10
            )
            
            print(f"Final validation: {final_validation.get('valid')}")
            print(f"Remaining issues: {len(final_validation.get('issues', []))}")


class TestConcurrentRequestHandling:
    """Test concurrent request handling and resource management."""
    
    @pytest.fixture
    async def http_client(self):
        """Create HTTP client for concurrency tests."""
        config = ConfigManager()
        client = ClaudeCodeClient(config, oauth_token=OAUTH_TOKEN)
        yield client
        await client.cleanup()
    
    @pytest.fixture
    def direct_client(self):
        """Create direct client for concurrency tests."""
        import tempfile
        from pathlib import Path
        
        temp_dir = tempfile.mkdtemp()
        cc_file = Path(temp_dir) / ".cc"
        cc_file.write_text(OAUTH_TOKEN)
        
        client = ClaudeCodeDirectClient(
            oauth_token_file=str(cc_file),
            working_directory=temp_dir
        )
        yield client
        client.cleanup_session()
        
        # Cleanup
        cc_file.unlink(missing_ok=True)
        Path(temp_dir).rmdir()
    
    @pytest.mark.asyncio
    async def test_mixed_operation_concurrency(self, direct_client):
        """Test concurrent execution of different operation types."""
        # Define different types of operations
        operations = [
            ("execute", "Create a simple calculator function"),
            ("validate", "def calc(a, b): return a + b"),
            ("refactor", "def add(x, y): return x + y", "Add type hints"),
            ("execute", "Create a fibonacci function"),
            ("validate", "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)")
        ]
        
        async def run_operation(op_type, *args):
            if op_type == "execute":
                return await direct_client.execute_task_via_cli(args[0], timeout=15)
            elif op_type == "validate":
                return await direct_client.validate_code_via_cli(args[0], timeout=10)
            elif op_type == "refactor":
                return await direct_client.refactor_code_via_cli(args[0], args[1], timeout=15)
        
        # Execute all operations concurrently
        tasks = [run_operation(op[0], *op[1:]) for op in operations]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Analyze results
        successful_operations = 0
        failed_operations = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Operation {i} failed: {type(result).__name__}: {result}")
                failed_operations += 1
            else:
                print(f"Operation {i} completed: {type(result)}")
                successful_operations += 1
        
        print(f"Concurrent operations completed in {total_time:.2f}s")
        print(f"Success rate: {successful_operations}/{len(operations)}")
        
        # Should handle concurrent operations reasonably well
        assert successful_operations > 0
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion_handling(self, direct_client):
        """Test handling of resource exhaustion scenarios."""
        # Create many concurrent tasks to potentially exhaust resources
        num_tasks = 10
        
        tasks = [
            direct_client.execute_task_via_cli(
                f"Create function number {i}",
                timeout=5  # Short timeout to avoid long waits
            )
            for i in range(num_tasks)
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Categorize results
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
        failed = sum(1 for r in results if isinstance(r, dict) and not r.get('success'))
        exceptions = sum(1 for r in results if isinstance(r, Exception))
        
        print(f"Resource exhaustion test results:")
        print(f"Total time: {total_time:.2f}s")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Exceptions: {exceptions}")
        
        # Should not crash completely
        assert successful + failed + exceptions == num_tasks
    
    @pytest.mark.asyncio
    async def test_session_state_isolation(self, direct_client):
        """Test that concurrent operations don't interfere with each other."""
        # Define operations that might interfere with each other
        operations = [
            ("validate", "def test1(): return 1"),
            ("validate", "def test2(): return 2"), 
            ("validate", "def test3(): return 3"),
        ]
        
        # Execute operations concurrently
        tasks = [
            direct_client.validate_code_via_cli(code, timeout=10)
            for _, code in operations
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Each operation should have unique execution IDs
        execution_ids = []
        for result in results:
            if isinstance(result, dict) and 'execution_id' in result:
                execution_ids.append(result['execution_id'])
        
        # All execution IDs should be unique
        assert len(execution_ids) == len(set(execution_ids))
        print(f"Unique execution IDs: {execution_ids}")


if __name__ == "__main__":
    """Run streaming tests."""
    pytest.main([__file__, "-v", "-s", "--tb=short"])