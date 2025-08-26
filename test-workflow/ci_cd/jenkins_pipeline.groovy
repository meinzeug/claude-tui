#!/usr/bin/env groovy

/**
 * Jenkins Pipeline for Test Workflow Framework
 * Comprehensive CI/CD pipeline with parallel execution and comprehensive reporting
 */

pipeline {
    agent {
        label 'python-agent'
    }
    
    environment {
        PYTHON_VERSION = '3.9'
        PIP_CACHE_DIR = "${WORKSPACE}/.pip-cache"
        PYTHONPATH = "${WORKSPACE}"
        TEST_RESULTS_DIR = "test-results"
        COVERAGE_THRESHOLD = '85'
        PERFORMANCE_THRESHOLD = '95'  // 95th percentile
    }
    
    options {
        timestamps()
        timeout(time: 45, unit: 'MINUTES')
        buildDiscarder(logRotator(
            numToKeepStr: '50',
            artifactNumToKeepStr: '20'
        ))
        skipStagesAfterUnstable()
        parallelsAlwaysFailFast()
    }
    
    triggers {
        // Poll SCM every 5 minutes during business hours
        pollSCM('H/5 8-18 * * 1-5')
        // Daily build at midnight
        cron('H 0 * * *')
    }
    
    stages {
        stage('Environment Setup') {
            steps {
                script {
                    echo "Setting up environment for Test Workflow Framework"
                    echo "Branch: ${env.GIT_BRANCH}"
                    echo "Commit: ${env.GIT_COMMIT}"
                    echo "Build Number: ${env.BUILD_NUMBER}"
                }
                
                // Clean workspace
                cleanWs()
                
                // Checkout code
                checkout scm
                
                // Setup Python virtual environment
                sh '''
                    python${PYTHON_VERSION} -m venv venv
                    source venv/bin/activate
                    pip install --upgrade pip setuptools wheel
                    pip install --cache-dir=${PIP_CACHE_DIR} -r requirements.txt
                    pip install --cache-dir=${PIP_CACHE_DIR} -r requirements-dev.txt
                    pip install --cache-dir=${PIP_CACHE_DIR} -e .
                '''
                
                // Create results directories
                sh 'mkdir -p ${TEST_RESULTS_DIR}/{unit,integration,e2e,performance,security,coverage}'
            }
        }
        
        stage('Code Quality & Security') {
            parallel {
                stage('Linting & Formatting') {
                    steps {
                        sh '''
                            source venv/bin/activate
                            
                            # Run flake8
                            flake8 test-workflow/ --format=junit-xml --output-file=${TEST_RESULTS_DIR}/flake8.xml || true
                            
                            # Run mypy
                            mypy test-workflow/ --junit-xml=${TEST_RESULTS_DIR}/mypy.xml --ignore-missing-imports || true
                            
                            # Check formatting
                            black --check --diff test-workflow/ > ${TEST_RESULTS_DIR}/black-report.txt || true
                            isort --check-only --diff test-workflow/ > ${TEST_RESULTS_DIR}/isort-report.txt || true
                        '''
                    }
                    post {
                        always {
                            publishTestResults testResultsPattern: "${TEST_RESULTS_DIR}/flake8.xml,${TEST_RESULTS_DIR}/mypy.xml"
                            archiveArtifacts artifacts: "${TEST_RESULTS_DIR}/*-report.txt", allowEmptyArchive: true
                        }
                    }
                }
                
                stage('Security Scan') {
                    steps {
                        sh '''
                            source venv/bin/activate
                            
                            # Install security tools
                            pip install bandit safety
                            
                            # Run Bandit security scan
                            bandit -r test-workflow/ -f json -o ${TEST_RESULTS_DIR}/security/bandit-report.json || true
                            bandit -r test-workflow/ -f txt -o ${TEST_RESULTS_DIR}/security/bandit-report.txt || true
                            
                            # Run Safety dependency check
                            safety check --json --output ${TEST_RESULTS_DIR}/security/safety-report.json || true
                            safety check --output ${TEST_RESULTS_DIR}/security/safety-report.txt || true
                        '''
                    }
                    post {
                        always {
                            archiveArtifacts artifacts: "${TEST_RESULTS_DIR}/security/*", allowEmptyArchive: true
                        }
                    }
                }
            }
        }
        
        stage('Testing') {
            parallel {
                stage('Unit Tests') {
                    steps {
                        sh '''
                            source venv/bin/activate
                            
                            python -m pytest tests/unit/ \
                                --cov=test_workflow \
                                --cov-config=.coveragerc \
                                --cov-report=xml:${TEST_RESULTS_DIR}/coverage/unit-coverage.xml \
                                --cov-report=html:${TEST_RESULTS_DIR}/coverage/unit-html \
                                --cov-report=term \
                                --junitxml=${TEST_RESULTS_DIR}/unit/junit.xml \
                                --html=${TEST_RESULTS_DIR}/unit/report.html \
                                --self-contained-html \
                                --tb=short \
                                -v \
                                --maxfail=5 \
                                --durations=10
                        '''
                    }
                    post {
                        always {
                            publishTestResults testResultsPattern: "${TEST_RESULTS_DIR}/unit/junit.xml"
                            publishHTML([
                                allowMissing: false,
                                alwaysLinkToLastBuild: true,
                                keepAll: true,
                                reportDir: "${TEST_RESULTS_DIR}/unit",
                                reportFiles: 'report.html',
                                reportName: 'Unit Test Report'
                            ])
                        }
                    }
                }
                
                stage('Integration Tests') {
                    environment {
                        DATABASE_URL = 'postgresql://testuser:testpass@localhost:5432/testdb'
                        REDIS_URL = 'redis://localhost:6379/0'
                    }
                    steps {
                        // Start required services
                        sh '''
                            # Start PostgreSQL in Docker
                            docker run -d --name postgres-test \
                                -e POSTGRES_USER=testuser \
                                -e POSTGRES_PASSWORD=testpass \
                                -e POSTGRES_DB=testdb \
                                -p 5432:5432 \
                                postgres:13
                            
                            # Start Redis in Docker
                            docker run -d --name redis-test \
                                -p 6379:6379 \
                                redis:7
                            
                            # Wait for services to be ready
                            sleep 30
                        '''
                        
                        sh '''
                            source venv/bin/activate
                            
                            python -m pytest tests/integration/ \
                                --cov=test_workflow \
                                --cov-append \
                                --cov-config=.coveragerc \
                                --cov-report=xml:${TEST_RESULTS_DIR}/coverage/integration-coverage.xml \
                                --cov-report=html:${TEST_RESULTS_DIR}/coverage/integration-html \
                                --junitxml=${TEST_RESULTS_DIR}/integration/junit.xml \
                                --html=${TEST_RESULTS_DIR}/integration/report.html \
                                --self-contained-html \
                                --tb=short \
                                -v \
                                --maxfail=3 \
                                --durations=10
                        '''
                    }
                    post {
                        always {
                            // Cleanup services
                            sh '''
                                docker stop postgres-test redis-test || true
                                docker rm postgres-test redis-test || true
                            '''
                            publishTestResults testResultsPattern: "${TEST_RESULTS_DIR}/integration/junit.xml"
                            publishHTML([
                                allowMissing: false,
                                alwaysLinkToLastBuild: true,
                                keepAll: true,
                                reportDir: "${TEST_RESULTS_DIR}/integration",
                                reportFiles: 'report.html',
                                reportName: 'Integration Test Report'
                            ])
                        }
                    }
                }
                
                stage('Performance Tests') {
                    steps {
                        sh '''
                            source venv/bin/activate
                            pip install pytest-benchmark
                            
                            python -m pytest tests/performance/ \
                                --benchmark-json=${TEST_RESULTS_DIR}/performance/benchmark.json \
                                --benchmark-histogram=${TEST_RESULTS_DIR}/performance/histogram \
                                --benchmark-compare-fail=mean:${PERFORMANCE_THRESHOLD}% \
                                --junitxml=${TEST_RESULTS_DIR}/performance/junit.xml \
                                --html=${TEST_RESULTS_DIR}/performance/report.html \
                                --self-contained-html \
                                -v
                        '''
                    }
                    post {
                        always {
                            publishTestResults testResultsPattern: "${TEST_RESULTS_DIR}/performance/junit.xml"
                            archiveArtifacts artifacts: "${TEST_RESULTS_DIR}/performance/*", allowEmptyArchive: true
                            publishHTML([
                                allowMissing: false,
                                alwaysLinkToLastBuild: true,
                                keepAll: true,
                                reportDir: "${TEST_RESULTS_DIR}/performance",
                                reportFiles: 'report.html',
                                reportName: 'Performance Test Report'
                            ])
                        }
                    }
                }
            }
        }
        
        stage('E2E Tests') {
            when {
                anyOf {
                    branch 'main'
                    branch 'develop'
                    changeRequest()
                }
            }
            steps {
                sh '''
                    source venv/bin/activate
                    
                    # Install E2E dependencies
                    pip install playwright
                    playwright install
                    
                    # Start application
                    python -m test_workflow.examples.demo_app &
                    APP_PID=$!
                    echo $APP_PID > app.pid
                    
                    # Wait for app to start
                    sleep 15
                    
                    # Run E2E tests
                    python -m pytest tests/e2e/ \
                        --junitxml=${TEST_RESULTS_DIR}/e2e/junit.xml \
                        --html=${TEST_RESULTS_DIR}/e2e/report.html \
                        --self-contained-html \
                        --tb=short \
                        -v \
                        --maxfail=1 \
                        --capture=no
                '''
            }
            post {
                always {
                    sh '''
                        # Stop application
                        if [ -f app.pid ]; then
                            kill $(cat app.pid) || true
                            rm app.pid
                        fi
                    '''
                    publishTestResults testResultsPattern: "${TEST_RESULTS_DIR}/e2e/junit.xml"
                    archiveArtifacts artifacts: "${TEST_RESULTS_DIR}/e2e/**", allowEmptyArchive: true
                }
            }
        }
        
        stage('Coverage Analysis') {
            steps {
                sh '''
                    source venv/bin/activate
                    
                    # Combine coverage reports
                    coverage combine ${TEST_RESULTS_DIR}/coverage/.coverage* || true
                    coverage xml -o ${TEST_RESULTS_DIR}/coverage/combined-coverage.xml
                    coverage html -d ${TEST_RESULTS_DIR}/coverage/combined-html
                    coverage report --show-missing > ${TEST_RESULTS_DIR}/coverage/coverage-report.txt
                    
                    # Check coverage threshold
                    COVERAGE_PERCENT=$(coverage report --show-missing | tail -1 | awk '{print $4}' | sed 's/%//')
                    echo "Coverage: ${COVERAGE_PERCENT}%"
                    
                    if [ "${COVERAGE_PERCENT%.*}" -lt "${COVERAGE_THRESHOLD}" ]; then
                        echo "Coverage ${COVERAGE_PERCENT}% is below threshold ${COVERAGE_THRESHOLD}%"
                        exit 1
                    fi
                '''
            }
            post {
                always {
                    publishCoverage adapters: [cobertura('${TEST_RESULTS_DIR}/coverage/combined-coverage.xml')], 
                                   sourceFileResolver: sourceFiles('STORE_LAST_BUILD')
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: "${TEST_RESULTS_DIR}/coverage/combined-html",
                        reportFiles: 'index.html',
                        reportName: 'Coverage Report'
                    ])
                }
            }
        }
        
        stage('Package & Artifacts') {
            steps {
                sh '''
                    source venv/bin/activate
                    
                    # Build package
                    pip install build
                    python -m build --outdir dist/
                    
                    # Check package
                    pip install twine
                    python -m twine check dist/*
                    
                    # Test installation
                    pip install dist/*.whl
                    python -c "import test_workflow; print(f'Version: {test_workflow.__version__}')"
                '''
            }
            post {
                always {
                    archiveArtifacts artifacts: 'dist/*', fingerprint: true
                }
            }
        }
        
        stage('Quality Gates') {
            steps {
                script {
                    // Parse test results
                    def unitTests = readFile("${TEST_RESULTS_DIR}/unit/junit.xml")
                    def integrationTests = readFile("${TEST_RESULTS_DIR}/integration/junit.xml")
                    
                    // Check if critical tests passed
                    if (currentBuild.result == 'UNSTABLE') {
                        error("Quality gates failed: Tests are unstable")
                    }
                    
                    echo "✅ All quality gates passed!"
                }
            }
        }
    }
    
    post {
        always {
            // Archive all test results
            archiveArtifacts artifacts: "${TEST_RESULTS_DIR}/**", allowEmptyArchive: true
            
            // Clean up
            sh '''
                # Stop any remaining processes
                pkill -f "test_workflow" || true
                
                # Clean Docker containers
                docker stop $(docker ps -aq) || true
                docker rm $(docker ps -aq) || true
                
                # Clean Python cache
                find . -type d -name "__pycache__" -exec rm -rf {} + || true
                find . -type f -name "*.pyc" -delete || true
            '''
        }
        
        success {
            script {
                if (env.BRANCH_NAME == 'main') {
                    echo "🎉 Main branch build successful - ready for release!"
                    
                    // Trigger deployment pipeline
                    build job: 'test-workflow-deploy', 
                          parameters: [
                              string(name: 'BUILD_NUMBER', value: env.BUILD_NUMBER),
                              string(name: 'GIT_COMMIT', value: env.GIT_COMMIT)
                          ],
                          wait: false
                }
            }
        }
        
        failure {
            script {
                // Send notifications
                emailext (
                    subject: "❌ Build Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                    body: """
                        Build failed for ${env.JOB_NAME} - ${env.BUILD_NUMBER}
                        
                        Branch: ${env.GIT_BRANCH}
                        Commit: ${env.GIT_COMMIT}
                        
                        Check the console output at:
                        ${env.BUILD_URL}console
                        
                        Test results:
                        ${env.BUILD_URL}testReport/
                    """,
                    recipientProviders: [developers(), requestor()]
                )
            }
        }
        
        unstable {
            script {
                emailext (
                    subject: "⚠️ Build Unstable: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                    body: """
                        Build is unstable for ${env.JOB_NAME} - ${env.BUILD_NUMBER}
                        
                        Some tests may have failed. Please review:
                        ${env.BUILD_URL}testReport/
                    """,
                    recipientProviders: [developers(), requestor()]
                )
            }
        }
    }
}