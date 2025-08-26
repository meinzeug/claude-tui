# Claude-Flow Project

AI-powered development with Claude-Flow v2.0.0 and Hive Mind orchestration.

## 🚀 Quick Start

```bash
# 1. Authenticate Claude (first time only)
./claude-flow.sh auth

# 2. Start Claude-Flow
./claude-flow.sh start

# 3. Run a task
./claude-flow.sh swarm "Build a feature"

# 4. Check status
./claude-flow.sh status
```

## 📚 Commands

| Command | Description |
|---------|-------------|
| `start` | Start all services |
| `stop` | Stop all services |
| `status` | Check system status |
| `swarm <task>` | Execute swarm task |
| `sparc <research>` | Research with SPARC |
| `auth` | Setup OAuth authentication |
| `doctor` | Run diagnostics |
| `init` | Initialize/reset |

## 🐝 Features

- **Hive Mind** orchestration with Queen/Worker pattern
- **27+ Neural Models** with WASM acceleration
- **SQLite Memory** for persistent context
- **OAuth Authentication** via Anthropic Console
- **Auto-checkpointing** for safe rollback

## 📁 Structure

```
.hive-mind/     # Hive configuration
.swarm/         # Memory database
.mcp/           # MCP servers (future)
memory/         # Agent memories
logs/           # System logs
CLAUDE.md       # Claude configuration
```

## 🔐 Authentication

First time setup:
```bash
claude  # Opens browser for OAuth
```

## 📖 Links

- [Claude-Flow](https://github.com/ruvnet/claude-flow)
- [Claude Code](https://docs.anthropic.com/claude-code)
- [Anthropic Console](https://console.anthropic.com)
