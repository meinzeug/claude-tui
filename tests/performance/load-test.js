import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

// Custom metrics
export const errorRate = new Rate('errors');

// Test configuration
export const options = {
  stages: [
    // Ramp-up
    { duration: '2m', target: 20 }, // Ramp-up to 20 users over 2 minutes
    { duration: '5m', target: 20 }, // Stay at 20 users for 5 minutes
    { duration: '2m', target: 50 }, // Ramp-up to 50 users over 2 minutes
    { duration: '5m', target: 50 }, // Stay at 50 users for 5 minutes
    { duration: '2m', target: 100 }, // Ramp-up to 100 users over 2 minutes
    { duration: '5m', target: 100 }, // Stay at 100 users for 5 minutes
    // Ramp-down
    { duration: '3m', target: 0 }, // Ramp-down to 0 users over 3 minutes
  ],
  thresholds: {
    // Performance thresholds
    http_req_duration: ['p(50)<500', 'p(95)<2000', 'p(99)<5000'], // 50% under 500ms, 95% under 2s, 99% under 5s
    http_req_failed: ['rate<0.05'], // Error rate under 5%
    errors: ['rate<0.05'], // Custom error rate under 5%
    http_reqs: ['rate>10'], // Minimum request rate of 10/sec
  },
};

// Base URL from environment variable
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

// Test data
const testData = {
  users: [
    { email: 'test1@example.com', password: 'testpass123' },
    { email: 'test2@example.com', password: 'testpass123' },
    { email: 'test3@example.com', password: 'testpass123' },
  ],
  projects: [
    { name: 'Test Project 1', description: 'Load test project 1' },
    { name: 'Test Project 2', description: 'Load test project 2' },
  ],
};

export default function () {
  // Test 1: Health Check
  const healthResponse = http.get(`${BASE_URL}/health`);
  check(healthResponse, {
    'health check status is 200': (r) => r.status === 200,
    'health check response time < 100ms': (r) => r.timings.duration < 100,
  }) || errorRate.add(1);

  sleep(1);

  // Test 2: API Health Check
  const apiHealthResponse = http.get(`${BASE_URL}/api/v1/health`);
  check(apiHealthResponse, {
    'API health check status is 200': (r) => r.status === 200,
    'API health response time < 200ms': (r) => r.timings.duration < 200,
  }) || errorRate.add(1);

  sleep(1);

  // Test 3: User Registration
  const userData = testData.users[Math.floor(Math.random() * testData.users.length)];
  const registerPayload = {
    email: `${userData.email}_${__VU}_${__ITER}`,
    password: userData.password,
    full_name: `Test User ${__VU}`,
  };

  const registerResponse = http.post(`${BASE_URL}/api/v1/auth/register`, JSON.stringify(registerPayload), {
    headers: {
      'Content-Type': 'application/json',
    },
  });

  check(registerResponse, {
    'registration status is 201 or 400': (r) => r.status === 201 || r.status === 400, // 400 if user already exists
    'registration response time < 1000ms': (r) => r.timings.duration < 1000,
  }) || errorRate.add(1);

  sleep(1);

  // Test 4: User Login
  const loginPayload = {
    username: userData.email,
    password: userData.password,
  };

  const loginResponse = http.post(`${BASE_URL}/api/v1/auth/token`, loginPayload, {
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
  });

  let token = '';
  const loginSuccess = check(loginResponse, {
    'login status is 200': (r) => r.status === 200,
    'login response time < 1000ms': (r) => r.timings.duration < 1000,
    'login returns access token': (r) => {
      if (r.status === 200) {
        const body = JSON.parse(r.body);
        token = body.access_token;
        return !!token;
      }
      return false;
    },
  });

  if (!loginSuccess) {
    errorRate.add(1);
  }

  sleep(1);

  // Test 5: Authenticated API calls (if login successful)
  if (token) {
    const authHeaders = {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    };

    // Get user profile
    const profileResponse = http.get(`${BASE_URL}/api/v1/users/me`, {
      headers: authHeaders,
    });

    check(profileResponse, {
      'profile status is 200': (r) => r.status === 200,
      'profile response time < 500ms': (r) => r.timings.duration < 500,
    }) || errorRate.add(1);

    sleep(1);

    // Create a project
    const projectData = testData.projects[Math.floor(Math.random() * testData.projects.length)];
    const projectPayload = {
      name: `${projectData.name} ${__VU}_${__ITER}`,
      description: projectData.description,
    };

    const createProjectResponse = http.post(`${BASE_URL}/api/v1/projects/`, JSON.stringify(projectPayload), {
      headers: authHeaders,
    });

    check(createProjectResponse, {
      'create project status is 201': (r) => r.status === 201,
      'create project response time < 1000ms': (r) => r.timings.duration < 1000,
    }) || errorRate.add(1);

    sleep(1);

    // List projects
    const listProjectsResponse = http.get(`${BASE_URL}/api/v1/projects/`, {
      headers: authHeaders,
    });

    check(listProjectsResponse, {
      'list projects status is 200': (r) => r.status === 200,
      'list projects response time < 500ms': (r) => r.timings.duration < 500,
    }) || errorRate.add(1);
  }

  sleep(2);
}

export function handleSummary(data) {
  return {
    'performance-results.json': JSON.stringify(data, null, 2),
    'stdout': textSummary(data, { indent: ' ', enableColors: true }),
  };
}

function textSummary(data, options) {
  const indent = options.indent || '';
  const enableColors = options.enableColors || false;
  
  let output = `${indent}
     ✓ checks.........................: ${data.metrics.checks.values.passes}/${data.metrics.checks.values.passes + data.metrics.checks.values.fails}
     data_received..................: ${(data.metrics.data_received.values.count / 1024 / 1024).toFixed(2)} MB
     data_sent......................: ${(data.metrics.data_sent.values.count / 1024 / 1024).toFixed(2)} MB
     http_req_blocked...............: avg=${data.metrics.http_req_blocked.values.avg.toFixed(2)}ms
     http_req_connecting............: avg=${data.metrics.http_req_connecting.values.avg.toFixed(2)}ms
     http_req_duration..............: avg=${data.metrics.http_req_duration.values.avg.toFixed(2)}ms
     http_req_failed................: ${(data.metrics.http_req_failed.values.rate * 100).toFixed(2)}%
     http_req_receiving.............: avg=${data.metrics.http_req_receiving.values.avg.toFixed(2)}ms
     http_req_sending...............: avg=${data.metrics.http_req_sending.values.avg.toFixed(2)}ms
     http_req_tls_handshaking.......: avg=${data.metrics.http_req_tls_handshaking.values.avg.toFixed(2)}ms
     http_req_waiting...............: avg=${data.metrics.http_req_waiting.values.avg.toFixed(2)}ms
     http_reqs......................: ${data.metrics.http_reqs.values.count} ${data.metrics.http_reqs.values.rate.toFixed(2)}/s
     iteration_duration.............: avg=${data.metrics.iteration_duration.values.avg.toFixed(2)}ms
     iterations.....................: ${data.metrics.iterations.values.count}
     vus............................: ${data.metrics.vus.values.value}
     vus_max........................: ${data.metrics.vus_max.values.value}
`;
  
  return output;
}