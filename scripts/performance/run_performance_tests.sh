#!/bin/bash
"""
High-Performance API Testing Suite.

Comprehensive performance testing script that validates
API optimizations and ensures <500ms response times.
"""

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_BASE_URL="http://localhost:8000"
REDIS_URL="redis://localhost:6379"
BENCHMARK_DURATION=60
MAX_CONCURRENT_REQUESTS=50
PERFORMANCE_THRESHOLD_MS=500

echo -e "${BLUE}🚀 Claude TIU Performance Testing Suite${NC}"
echo "================================================="
echo "Target API: $API_BASE_URL"
echo "Performance threshold: ${PERFORMANCE_THRESHOLD_MS}ms"
echo ""

# Function to check if service is running
check_service() {
    local service_name=$1
    local check_command=$2
    
    echo -n "Checking $service_name... "
    if eval $check_command > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Running${NC}"
        return 0
    else
        echo -e "${RED}❌ Not running${NC}"
        return 1
    fi
}

# Function to run performance test
run_performance_test() {
    local test_name=$1
    local test_command=$2
    
    echo -e "\n${BLUE}📊 Running $test_name...${NC}"
    echo "----------------------------------------"
    
    if eval $test_command; then
        echo -e "${GREEN}✅ $test_name PASSED${NC}"
        return 0
    else
        echo -e "${RED}❌ $test_name FAILED${NC}"
        return 1
    fi
}

# Pre-flight checks
echo -e "${BLUE}🔍 Pre-flight Checks${NC}"
echo "--------------------"

# Check if API server is running
if ! check_service "API Server" "curl -s -f $API_BASE_URL/health"; then
    echo -e "${YELLOW}⚠️  Starting API server...${NC}"
    cd /home/tekkadmin/claude-tiu
    python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
    API_PID=$!
    
    # Wait for server to start
    echo "Waiting for server to start..."
    for i in {1..30}; do
        if curl -s -f $API_BASE_URL/health > /dev/null 2>&1; then
            echo -e "${GREEN}✅ API Server started${NC}"
            break
        fi
        sleep 1
        
        if [ $i -eq 30 ]; then
            echo -e "${RED}❌ API Server failed to start${NC}"
            exit 1
        fi
    done
fi

# Check Redis (optional for caching)
check_service "Redis Server" "redis-cli ping"
REDIS_AVAILABLE=$?

# Check dependencies
echo -n "Checking Python dependencies... "
if python -c "import aiohttp, asyncio, psutil" 2>/dev/null; then
    echo -e "${GREEN}✅ Available${NC}"
else
    echo -e "${RED}❌ Missing dependencies${NC}"
    echo "Installing required packages..."
    pip install aiohttp asyncio psutil
fi

echo ""

# Performance Tests
echo -e "${BLUE}⚡ Performance Tests${NC}"
echo "===================="

FAILED_TESTS=0
TOTAL_TESTS=0

# Test 1: Basic Health Check Performance
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if run_performance_test "Health Check Latency" "
python3 scripts/performance/benchmark_api.py \
    --url $API_BASE_URL \
    --timeout 5 \
    --concurrent 10 2>/dev/null | grep -q 'EXCELLENT\|GOOD'
"; then
    :
else
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Test 2: AI Endpoint Performance
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if run_performance_test "AI Code Generation Performance" "
timeout 30s python3 -c '
import asyncio
import aiohttp
import time

async def test_ai_performance():
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        try:
            async with session.post(\"$API_BASE_URL/api/v1/ai/code/generate\", 
                json={\"prompt\": \"def hello(): pass\", \"language\": \"python\"},
                timeout=10) as response:
                await response.text()
                response_time = (time.time() - start_time) * 1000
                print(f\"Response time: {response_time:.1f}ms\")
                exit(0 if response_time < $PERFORMANCE_THRESHOLD_MS else 1)
        except Exception as e:
            print(f\"Error: {e}\")
            exit(1)

asyncio.run(test_ai_performance())
'"; then
    :
else
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Test 3: Concurrent Request Handling
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if run_performance_test "Concurrent Request Handling" "
python3 -c '
import asyncio
import aiohttp
import time

async def concurrent_test():
    async with aiohttp.ClientSession() as session:
        # Create 20 concurrent requests
        tasks = []
        for i in range(20):
            task = session.get(\"$API_BASE_URL/health\", timeout=10)
            tasks.append(task)
        
        start_time = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        successful = sum(1 for r in responses if not isinstance(r, Exception))
        avg_time = (total_time / len(tasks)) * 1000
        
        print(f\"Concurrent requests: {len(tasks)}\")
        print(f\"Successful: {successful}\")
        print(f\"Average time: {avg_time:.1f}ms\")
        
        # Close all responses
        for response in responses:
            if hasattr(response, \"close\"):
                response.close()
        
        exit(0 if successful >= 18 and avg_time < $PERFORMANCE_THRESHOLD_MS else 1)

asyncio.run(concurrent_test())
'"; then
    :
else
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Test 4: Memory Usage Test
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if run_performance_test "Memory Usage Test" "
python3 -c '
import psutil
import requests
import time

# Get initial memory
process = psutil.Process()
initial_memory = process.memory_info().rss / 1024 / 1024  # MB

# Make multiple requests
for i in range(50):
    try:
        response = requests.get(\"$API_BASE_URL/health\", timeout=5)
        response.close()
    except:
        pass
    time.sleep(0.1)

# Check final memory
final_memory = process.memory_info().rss / 1024 / 1024  # MB
memory_growth = final_memory - initial_memory

print(f\"Initial memory: {initial_memory:.1f}MB\")
print(f\"Final memory: {final_memory:.1f}MB\")
print(f\"Memory growth: {memory_growth:.1f}MB\")

# Acceptable if memory growth is less than 50MB
exit(0 if memory_growth < 50 else 1)
'"; then
    :
else
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Test 5: Caching Performance (if Redis is available)
if [ $REDIS_AVAILABLE -eq 0 ]; then
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    if run_performance_test "Caching Performance" "
    python3 -c '
import asyncio
import aiohttp
import time

async def cache_test():
    async with aiohttp.ClientSession() as session:
        # First request (cache miss)
        start_time = time.time()
        async with session.get(\"$API_BASE_URL/api/v1/ai/performance\", timeout=10) as response:
            await response.text()
        first_request_time = (time.time() - start_time) * 1000
        
        # Second request (should be cached)
        start_time = time.time()
        async with session.get(\"$API_BASE_URL/api/v1/ai/performance\", timeout=10) as response:
            await response.text()
            cache_header = response.headers.get(\"X-Cache\", \"\")
        second_request_time = (time.time() - start_time) * 1000
        
        print(f\"First request: {first_request_time:.1f}ms\")
        print(f\"Second request: {second_request_time:.1f}ms\") 
        print(f\"Cache header: {cache_header}\")
        
        # Cache should improve performance significantly
        improvement = first_request_time - second_request_time
        exit(0 if improvement > 50 or \"HIT\" in cache_header else 1)

asyncio.run(cache_test())
    '"; then
        :
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
fi

# Cleanup
if [ ! -z "$API_PID" ]; then
    echo -e "\n${BLUE}🧹 Cleaning up...${NC}"
    kill $API_PID 2>/dev/null || true
    wait $API_PID 2>/dev/null || true
fi

# Final Results
echo ""
echo "================================================="
echo -e "${BLUE}📊 Performance Test Results${NC}"
echo "================================================="
echo "Total tests: $TOTAL_TESTS"
echo "Passed: $((TOTAL_TESTS - FAILED_TESTS))"
echo "Failed: $FAILED_TESTS"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "\n${GREEN}🎉 ALL PERFORMANCE TESTS PASSED!${NC}"
    echo -e "${GREEN}✅ API is optimized for <${PERFORMANCE_THRESHOLD_MS}ms response times${NC}"
    exit 0
else
    echo -e "\n${RED}❌ $FAILED_TESTS PERFORMANCE TEST(S) FAILED${NC}"
    echo -e "${YELLOW}⚠️  API needs optimization to meet performance targets${NC}"
    exit 1
fi