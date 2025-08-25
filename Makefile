# Claude TIU - DevOps Makefile
# Comprehensive development and deployment automation

.PHONY: help install install-dev clean test test-full lint format security-scan \
        docker-build docker-run docker-stop docker-clean deploy deploy-staging \
        deploy-production backup restore monitoring logs docs release

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
PROJECT_NAME := claude-tiu
DOCKER_IMAGE := $(PROJECT_NAME)
DOCKER_TAG := latest
ENVIRONMENT := development
COMPOSE_FILE := docker-compose.yml

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Help target
help: ## Show this help message
	@echo "$(GREEN)Claude TIU - DevOps Makefile$(NC)"
	@echo "=============================="
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation targets
install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install -r requirements.txt -r requirements-dev.txt
	$(PIP) install -e .
	pre-commit install
	@echo "$(GREEN)Development environment ready!$(NC)"

# Cleanup targets
clean: ## Clean temporary files and caches
	@echo "$(BLUE)Cleaning temporary files...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/
	@echo "$(GREEN)Cleanup complete!$(NC)"

clean-all: clean ## Clean everything including docker images and volumes
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	-docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG) 2>/dev/null || true
	-docker system prune -f
	-docker volume prune -f

# Testing targets
test: ## Run unit tests
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(PYTHON) -m pytest tests/unit/ -v --tb=short

test-integration: ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(PYTHON) -m pytest tests/integration/ -v --tb=short

test-e2e: ## Run end-to-end tests
	@echo "$(BLUE)Running E2E tests...$(NC)"
	$(PYTHON) -m pytest tests/e2e/ -v --tb=short

test-full: ## Run all tests with coverage
	@echo "$(BLUE)Running full test suite with coverage...$(NC)"
	$(PYTHON) -m pytest tests/ -v --cov=claude_tiu --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

test-performance: ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(NC)"
	$(PYTHON) -m pytest tests/performance/ -v --benchmark-only

# Code quality targets
lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(NC)"
	black --check --diff src/ tests/
	isort --check-only --diff src/ tests/
	flake8 src/ tests/
	mypy src/
	@echo "$(GREEN)Linting passed!$(NC)"

format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	black src/ tests/
	isort src/ tests/
	@echo "$(GREEN)Code formatted!$(NC)"

type-check: ## Run type checking
	@echo "$(BLUE)Running type checking...$(NC)"
	mypy src/ --strict

# Security targets
security-scan: ## Run security vulnerability scan
	@echo "$(BLUE)Running security scan...$(NC)"
	$(PYTHON) scripts/devops/security_audit.py --format both --verbose
	@echo "$(GREEN)Security scan complete! Check security_reports/$(NC)"

security-quick: ## Run quick security scan
	@echo "$(BLUE)Running quick security scan...$(NC)"
	bandit -r src/ -f json -o security_reports/bandit_quick.json || true
	safety check --json --output security_reports/safety_quick.json || true
	@echo "$(GREEN)Quick security scan complete!$(NC)"

# Docker targets
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "$(GREEN)Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)$(NC)"

docker-build-dev: ## Build development Docker image
	@echo "$(BLUE)Building development Docker image...$(NC)"
	docker build --target development -t $(DOCKER_IMAGE):dev .

docker-build-prod: ## Build production Docker image
	@echo "$(BLUE)Building production Docker image...$(NC)"
	docker build --target production -t $(DOCKER_IMAGE):prod .

docker-build-all: ## Build all Docker images (dev, prod, test)
	@echo "$(BLUE)Building all Docker images...$(NC)"
	$(MAKE) docker-build-dev
	$(MAKE) docker-build-prod
	docker build --target testing -t $(DOCKER_IMAGE):test .
	@echo "$(GREEN)All Docker images built successfully!$(NC)"

docker-security-scan: ## Run security scan on Docker images
	@echo "$(BLUE)Running Docker security scans...$(NC)"
	docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
		-v $(PWD):/src aquasec/trivy image $(DOCKER_IMAGE):prod
	@echo "$(GREEN)Docker security scan completed!$(NC)"

docker-run: ## Run application in Docker
	@echo "$(BLUE)Starting application with Docker Compose...$(NC)"
	docker-compose -f $(COMPOSE_FILE) up -d
	@echo "$(GREEN)Application started! Access at http://localhost:8000$(NC)"

docker-run-dev: ## Run development environment with Docker
	@echo "$(BLUE)Starting development environment...$(NC)"
	docker-compose -f $(COMPOSE_FILE) --profile dev up -d
	@echo "$(GREEN)Development environment started!$(NC)"

docker-stop: ## Stop Docker containers
	@echo "$(BLUE)Stopping Docker containers...$(NC)"
	docker-compose -f $(COMPOSE_FILE) down

docker-logs: ## View Docker logs
	docker-compose -f $(COMPOSE_FILE) logs -f

docker-clean: ## Clean Docker resources
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	docker-compose -f $(COMPOSE_FILE) down --volumes --rmi all
	docker system prune -f

# Deployment targets
deploy: ## Deploy to environment (default: development)
	@echo "$(BLUE)Deploying to $(ENVIRONMENT)...$(NC)"
	./scripts/devops/deploy.sh -e $(ENVIRONMENT) -t docker-compose

deploy-staging: ## Deploy to staging environment
	@echo "$(BLUE)Deploying to staging...$(NC)"
	./scripts/devops/deploy.sh -e staging -t kubernetes

deploy-production: ## Deploy to production environment
	@echo "$(YELLOW)Deploying to production...$(NC)"
	@echo "$(RED)WARNING: This will deploy to production!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo ""; \
		./scripts/devops/deploy.sh -e production -t kubernetes; \
	else \
		echo ""; \
		echo "Deployment cancelled"; \
	fi

deploy-dry-run: ## Dry run deployment
	@echo "$(BLUE)Performing deployment dry run...$(NC)"
	./scripts/devops/deploy.sh -e $(ENVIRONMENT) -t docker-compose -d

# Kubernetes targets
k8s-validate: ## Validate Kubernetes manifests
	@echo "$(BLUE)Validating Kubernetes manifests...$(NC)"
	kubectl --dry-run=client apply -f k8s/
	@echo "$(GREEN)Kubernetes manifests are valid!$(NC)"

k8s-lint: ## Lint Kubernetes manifests
	@echo "$(BLUE)Linting Kubernetes manifests...$(NC)"
	docker run --rm -v $(PWD):/data garethr/kubeval /data/k8s/*.yaml

k8s-deploy-staging: ## Deploy to Kubernetes staging
	@echo "$(BLUE)Deploying to Kubernetes staging...$(NC)"
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/staging/ -n claude-tiu-staging
	kubectl rollout status deployment/claude-tiu-staging -n claude-tiu-staging

k8s-deploy-production: ## Deploy to Kubernetes production
	@echo "$(YELLOW)Deploying to Kubernetes production...$(NC)"
	@read -p "Are you sure? This will deploy to production! [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo ""; \
		kubectl apply -f k8s/namespace.yaml; \
		kubectl apply -f k8s/production/ -n claude-tiu; \
		kubectl rollout status deployment/claude-tiu -n claude-tiu; \
	else \
		echo ""; \
		echo "Production deployment cancelled"; \
	fi

k8s-status: ## Check Kubernetes deployment status
	@echo "$(BLUE)Kubernetes Status:$(NC)"
	kubectl get pods -n claude-tiu-staging
	kubectl get pods -n claude-tiu

k8s-logs: ## View Kubernetes logs
	@echo "$(BLUE)Kubernetes Logs (Staging):$(NC)"
	kubectl logs -l app=claude-tiu -n claude-tiu-staging --tail=100

k8s-rollback: ## Rollback Kubernetes deployment
	@echo "$(BLUE)Rolling back Kubernetes deployment...$(NC)"
	kubectl rollout undo deployment/claude-tiu -n $(ENV:-claude-tiu-staging)

# Database targets
db-migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	alembic upgrade head

db-migration: ## Create new database migration
	@read -p "Enter migration message: " msg; \
	alembic revision --autogenerate -m "$$msg"

db-reset: ## Reset database (WARNING: destroys data)
	@echo "$(RED)WARNING: This will destroy all database data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo ""; \
		alembic downgrade base; \
		alembic upgrade head; \
		echo "$(GREEN)Database reset complete!$(NC)"; \
	else \
		echo ""; \
		echo "Database reset cancelled"; \
	fi

# Backup and restore targets
backup: ## Create backup
	@echo "$(BLUE)Creating backup...$(NC)"
	mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	docker-compose exec -T db pg_dump -U claude_user claude_tiu > backups/$(shell date +%Y%m%d_%H%M%S)/database.sql
	@echo "$(GREEN)Backup created in backups/$(NC)"

restore: ## Restore from backup (provide BACKUP_DIR)
ifndef BACKUP_DIR
	@echo "$(RED)Error: Please specify BACKUP_DIR=backups/YYYYMMDD_HHMMSS$(NC)"
	@exit 1
endif
	@echo "$(BLUE)Restoring from $(BACKUP_DIR)...$(NC)"
	docker-compose exec -T db psql -U claude_user claude_tiu < $(BACKUP_DIR)/database.sql
	@echo "$(GREEN)Restore complete!$(NC)"

# Monitoring targets
monitoring: ## Start monitoring stack
	@echo "$(BLUE)Starting monitoring stack...$(NC)"
	docker-compose -f $(COMPOSE_FILE) --profile monitoring up -d
	@echo "$(GREEN)Monitoring stack started!$(NC)"
	@echo "Grafana: http://localhost:3000 (admin/admin123)"
	@echo "Prometheus: http://localhost:9090"

logs: ## View application logs
	@echo "$(BLUE)Application logs:$(NC)"
	docker-compose logs -f claude-tiu

logs-all: ## View all service logs
	@echo "$(BLUE)All service logs:$(NC)"
	docker-compose logs -f

# Documentation targets
docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	sphinx-apidoc -o docs/api src/claude_tiu --force
	sphinx-build -b html docs docs/_build/html
	@echo "$(GREEN)Documentation generated in docs/_build/html$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8080$(NC)"
	cd docs/_build/html && $(PYTHON) -m http.server 8080

# Release targets
release-patch: ## Create patch release
	@echo "$(BLUE)Creating patch release...$(NC)"
	bump2version patch
	git push --tags

release-minor: ## Create minor release
	@echo "$(BLUE)Creating minor release...$(NC)"
	bump2version minor
	git push --tags

release-major: ## Create major release
	@echo "$(BLUE)Creating major release...$(NC)"
	bump2version major
	git push --tags

# CI/CD targets
ci-check: ## Run all CI checks locally
	@echo "$(BLUE)Running CI checks...$(NC)"
	$(MAKE) lint
	$(MAKE) test-full
	$(MAKE) security-scan
	$(MAKE) docker-build-all
	@echo "$(GREEN)All CI checks passed!$(NC)"

ci-matrix: ## Run CI matrix tests (multiple Python versions)
	@echo "$(BLUE)Running CI matrix tests...$(NC)"
	@for version in 3.9 3.10 3.11 3.12; do \
		echo "Testing with Python $$version..."; \
		docker run --rm -v $(PWD):/app -w /app python:$$version-slim bash -c \
			"pip install -r requirements.txt -r requirements-dev.txt && python -m pytest tests/ -v"; \
	done

pre-commit: ## Run pre-commit hooks
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files

# Development helpers
dev-setup: install-dev ## Set up development environment
	@echo "$(GREEN)Development environment setup complete!$(NC)"
	@echo "Next steps:"
	@echo "1. Copy .env.example to .env and configure"
	@echo "2. Run 'make docker-run-dev' to start services"
	@echo "3. Run 'make test' to verify everything works"

dev-start: docker-run-dev ## Start development environment

dev-stop: docker-stop ## Stop development environment

dev-reset: docker-clean dev-start ## Reset and restart development environment

# Health checks
health-check: ## Check application health
	@echo "$(BLUE)Checking application health...$(NC)"
	curl -f http://localhost:8000/health || (echo "$(RED)Health check failed!$(NC)" && exit 1)
	@echo "$(GREEN)Application is healthy!$(NC)"

status: ## Show service status
	@echo "$(BLUE)Service Status:$(NC)"
	docker-compose ps

# Performance targets
benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(NC)"
	$(PYTHON) -m pytest tests/performance/ --benchmark-only --benchmark-json=benchmark_results.json
	@echo "$(GREEN)Benchmark results saved to benchmark_results.json$(NC)"

load-test: ## Run load tests
	@echo "$(BLUE)Running load tests...$(NC)"
	locust -f tests/performance/locustfile.py --host=http://localhost:8000

# Utility targets
version: ## Show current version
	@$(PYTHON) -c "import toml; print(f\"Claude TIU v{toml.load('pyproject.toml')['project']['version']}\")"

env: ## Show environment information
	@echo "$(BLUE)Environment Information:$(NC)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pip: $(shell $(PIP) --version)"
	@echo "Docker: $(shell docker --version)"
	@echo "Docker Compose: $(shell docker-compose --version)"
	@echo "Git: $(shell git --version)"
	@echo "Current Branch: $(shell git branch --show-current)"
	@echo "Current Environment: $(ENVIRONMENT)"

check-deps: ## Check for outdated dependencies
	@echo "$(BLUE)Checking for outdated dependencies...$(NC)"
	$(PIP) list --outdated

update-deps: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -r requirements.txt -r requirements-dev.txt

# Git hooks
install-hooks: ## Install git hooks
	@echo "$(BLUE)Installing git hooks...$(NC)"
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "$(GREEN)Git hooks installed!$(NC)"

# Quick commands
quick-test: ## Quick test run (unit tests only)
	$(PYTHON) -m pytest tests/unit/ -x -v

quick-start: ## Quick start (no deps install)
	docker-compose up -d claude-tiu db cache

# Cleanup on exit
.ONESHELL:
cleanup:
	@echo "$(BLUE)Performing cleanup...$(NC)"