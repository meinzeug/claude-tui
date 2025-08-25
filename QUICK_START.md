# 🚀 Claude TUI Quick Start Guide

## Prerequisites

- Python 3.9+
- Docker & Docker Compose
- Git
- Make (optional but recommended)

## 🎯 Quick Start (5 Minutes)

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/claude-tui.git
cd claude-tui

# Start with Docker Compose
docker-compose up -d

# Access the application
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
# TUI: docker exec -it claude-tui-app python -m claude_tui
```

### Option 2: Local Development

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup environment
cp .env.example .env
# Edit .env with your API keys

# Initialize database
python -m alembic upgrade head

# Run TUI Application
python run_tui.py

# Or run API Server
uvicorn src.api.main:app --reload
```

### Option 3: Using Make

```bash
# Complete setup
make setup

# Run tests
make test

# Start development
make dev

# Deploy to production
make deploy ENV=production
```

## 🎮 Using the TUI

### Navigation
- `↑/↓` or `j/k` - Navigate menus
- `Enter` - Select option
- `Tab` - Switch between panels
- `Esc` - Go back/Cancel
- `q` - Quit application
- `?` - Show help

### Main Features

1. **Create New Project**
   - Press `n` from main menu
   - Select project template
   - Configure settings
   - Watch AI generate code

2. **Validate Code**
   - Press `v` from main menu
   - Select file or paste code
   - View real vs fake progress
   - Auto-fix placeholders

3. **Task Management**
   - Press `t` for task dashboard
   - Create AI-powered tasks
   - Monitor execution
   - View progress intelligence

4. **Settings**
   - Press `s` for settings
   - Configure AI providers
   - Set validation thresholds
   - Customize interface

## 🔑 API Authentication

### Get JWT Token
```bash
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin"}'
```

### Use Token
```bash
curl -X GET http://localhost:8000/api/v1/projects \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## 📊 Monitoring

### Access Monitoring Stack
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- API Metrics: http://localhost:8000/metrics

## 🧪 Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific categories
pytest -m unit           # Unit tests
pytest -m security      # Security tests
pytest -m performance   # Performance tests

# Anti-hallucination tests
pytest tests/validation/test_anti_hallucination.py
```

## 🐛 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Change port in docker-compose.yml or .env
   PORT=8001 docker-compose up
   ```

2. **Database connection error**
   ```bash
   # Reset database
   make db-reset
   ```

3. **API key not working**
   ```bash
   # Verify in .env file
   CLAUDE_API_KEY=sk-ant-...
   ```

4. **TUI not displaying correctly**
   ```bash
   # Ensure terminal supports 256 colors
   export TERM=xterm-256color
   ```

## 📚 Documentation

- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Project Docs**: See `/docs` directory
- **Architecture**: `/docs/architecture.md`
- **Security**: `/docs/security.md`

## 🆘 Getting Help

- **GitHub Issues**: [Report bugs](https://github.com/claude-tui/claude-tui/issues)
- **Documentation**: [Full docs](https://docs.claude-tui.ai)
- **Discord**: [Join community](https://discord.gg/claude-tui)

## 🎉 Next Steps

1. **Explore the TUI** - Try creating a project
2. **Test the API** - Check out `/docs` for interactive API
3. **Run validation** - Test anti-hallucination features
4. **Customize settings** - Configure for your needs
5. **Deploy to production** - Use `make deploy`

---

**Welcome to the future of AI-powered development! 🚀**

*No hallucinations, just real code.*