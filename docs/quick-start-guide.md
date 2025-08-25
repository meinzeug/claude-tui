# Claude-TUI Quick Start Guide

## 🚀 Get Up and Running in 10 Minutes

Welcome to Claude-TUI, the AI-powered terminal user interface that revolutionizes software development! This guide will have you creating AI-powered projects in just 10 minutes.

---

## Prerequisites (2 minutes)

### Required Software
- **Python 3.9+** - [Download Python](https://python.org/downloads)
- **Node.js 16+** - [Download Node.js](https://nodejs.org)  
- **Git** - [Download Git](https://git-scm.com/downloads)
- **4GB+ RAM** recommended

### Get Your API Key
1. Visit [https://claude-tui.dev/signup](https://claude-tui.dev/signup)
2. Create your free account (30-day trial)
3. Copy your API key from the dashboard
4. Keep it handy - you'll need it in step 3!

### Quick Environment Check
```bash
# Verify your environment
python --version  # Should show 3.9+
node --version    # Should show 16+
git --version     # Should show 2.0+
```

---

## Step 1: Installation (2 minutes)

### Option A: Quick Install (Recommended)
```bash
# Install Claude-TUI with all dependencies
curl -sSL https://install.claude-tui.dev | bash

# Or via pip
pip install claude-tui[all]
```

### Option B: Manual Install
```bash
# Clone and install
git clone https://github.com/your-username/claude-tui.git
cd claude-tui

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Verify Installation
```bash
# Check Claude-TUI installation
claude-tui --version
# Expected: claude-tui 1.0.0

# Check Claude Flow installation  
npx claude-flow@alpha --version
# Expected: claude-flow 2.0.0-alpha.x
```

---

## Step 2: Configuration (1 minute)

### Set Up API Key
```bash
# Set your Claude API key
export CLAUDE_API_KEY="your_api_key_here"

# Or create config file
claude-tui config set api-key YOUR_API_KEY
```

### Quick Configuration Check
```bash
# Test your setup
claude-tui doctor

# Expected output:
# ✅ Python environment: OK
# ✅ Claude API key: Valid
# ✅ Claude Flow: Connected
# ✅ System resources: Sufficient
# 🚀 Ready to create amazing projects!
```

---

## Step 3: Create Your First AI-Powered Project (3 minutes)

### Option 1: Interactive Mode (Recommended for beginners)
```bash
# Launch interactive project creator
claude-tui create --interactive

# Follow the prompts:
# 📝 Project name: My First AI Project
# 🎯 Project type: [react/fastapi/django/nodejs] 
# 📁 Location: ./my-first-project
# ✨ Features: [Select with spacebar, confirm with enter]
```

### Option 2: Command Line (For experienced developers)
```bash
# Create a React TypeScript project with AI features
claude-tui create \
  --name "AI-Powered Dashboard" \
  --type react \
  --template react-typescript-advanced \
  --features "authentication,real-time-data,ai-insights,charts" \
  --path ./ai-dashboard \
  --ai-creativity 0.8

# Create a FastAPI backend with AI endpoints
claude-tui create \
  --name "Intelligent API" \
  --type fastapi \
  --template api-microservices \
  --features "authentication,ai-endpoints,database,testing" \
  --path ./intelligent-api
```

### What Happens During Creation
```bash
🔄 Creating project structure...
🤖 AI analyzing requirements...
📝 Generating intelligent code...
🧪 Creating comprehensive tests...
📚 Writing documentation...
🔍 Validating implementation...
✅ Project created successfully!

📁 Project location: ./ai-dashboard
🌐 Development server: http://localhost:3000
📖 Documentation: ./ai-dashboard/docs/README.md
```

---

## Step 4: Launch and Explore (2 minutes)

### Start the TUI Interface
```bash
# Navigate to your project
cd ./ai-dashboard

# Launch Claude-TUI interface
python -m claude_tui

# Or use the run script
python run_tui.py

# Or if installed globally
claude-tui
```

### TUI Interface Overview
```
┌─ Claude-TUI AI Development Environment ─────────────────────────────────┐
│                                                                         │
│ 📊 Dashboard    📁 Projects    🤖 AI Tools    ⚙️ Settings             │
│ ────────────────────────────────────────────────────────────────────── │
│                                                                         │
│ 🎯 Current Project: AI-Powered Dashboard                               │
│ 📈 Health Score: 95%    🔥 AI Confidence: High                        │
│                                                                         │
│ 🔧 Quick Actions:                                                      │
│   [G] Generate Component    [R] Review Code    [T] Run Tests           │
│   [V] Validate Project     [D] Deploy         [H] Help                 │
│                                                                         │
│ 💡 AI Suggestions:                                                     │
│   • Add user authentication system                                     │
│   • Implement real-time notifications                                  │
│   • Optimize database queries                                          │
│                                                                         │
│ 📊 Project Status:                                                     │
│   Components: 12/15 ✅    Tests: 95% ✅    Security: A+ ✅            │
│                                                                         │
└─ Press ? for help, Q to quit ──────────────────────────────────────────┘
```

### Essential Keyboard Shortcuts
| Key | Action | Description |
|-----|--------|-------------|
| `G` | Generate Code | AI-powered code generation |
| `R` | Review Code | Intelligent code review |
| `T` | Run Tests | Execute test suite |
| `V` | Validate | Anti-hallucination validation |
| `D` | Deploy | Deploy to cloud platforms |
| `?` | Help | Show all shortcuts |
| `Q` | Quit | Exit Claude-TUI |

---

## Step 5: Generate Your First AI Code (2 minutes)

### Using the TUI Interface
1. Press `G` in the TUI to open AI Code Generator
2. Enter your prompt: *"Create a user dashboard with charts and real-time data"*
3. Select language/framework (auto-detected from project)
4. Watch AI generate comprehensive code with tests!

### Using Command Line
```bash
# Generate a React component with AI
claude-tui generate \
  --prompt "Create a responsive user dashboard with data visualization" \
  --type component \
  --language typescript \
  --framework react \
  --include-tests true

# Generate API endpoints
claude-tui generate \
  --prompt "Create REST API for user management with authentication" \
  --type api \
  --language python \
  --framework fastapi \
  --include-docs true
```

### Example AI Generated Code
```typescript
// Generated in seconds by Claude-TUI AI
import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

interface DashboardData {
  timestamp: string;
  users: number;
  revenue: number;
}

export const UserDashboard: React.FC = () => {
  const [data, setData] = useState<DashboardData[]>([]);
  const [loading, setLoading] = useState(true);

  // Real-time data fetching with WebSocket
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/dashboard');
    ws.onmessage = (event) => {
      const newData = JSON.parse(event.data);
      setData(prev => [...prev.slice(-24), newData]);
    };
    return () => ws.close();
  }, []);

  if (loading) return <LoadingSpinner />;

  return (
    <div className="dashboard-container">
      <h1>Real-Time Dashboard</h1>
      <div className="charts-grid">
        <LineChart width={600} height={300} data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="timestamp" />
          <YAxis />
          <Tooltip />
          <Line type="monotone" dataKey="users" stroke="#8884d8" />
          <Line type="monotone" dataKey="revenue" stroke="#82ca9d" />
        </LineChart>
      </div>
    </div>
  );
};

// Comprehensive tests included!
// docs/UserDashboard.md documentation generated!
// Accessibility features added automatically!
```

---

## Congratulations! 🎉

You've successfully:
- ✅ Installed Claude-TUI in under 10 minutes
- ✅ Created your first AI-powered project
- ✅ Generated intelligent code with comprehensive tests
- ✅ Learned the essential TUI navigation

## Next Steps

### Explore Advanced Features (Optional)
```bash
# Try the SPARC methodology for complex features
npx claude-flow sparc tdd "user authentication system"

# Create a workflow for full-stack development  
claude-tui workflow create \
  --name "Full Stack Feature" \
  --steps "backend,frontend,tests,docs"

# Validate your code for authenticity
claude-tui validate --level comprehensive --auto-fix
```

### Join the Community
- 💬 [Discord Community](https://discord.gg/claude-tui)
- 📚 [Complete Documentation](https://docs.claude-tui.dev)
- 🎥 [Video Tutorials](https://learn.claude-tui.dev)
- 🐛 [Report Issues](https://github.com/claude-tui/issues)

---

## Common Quick Start Issues & Solutions

### "Command not found: claude-tui"
```bash
# Solution 1: Add to PATH
export PATH=$PATH:~/.local/bin

# Solution 2: Reinstall with --user flag
pip install --user claude-tui[all]

# Solution 3: Use python -m
python -m claude_tui --version
```

### "API Key Invalid"
```bash
# Verify your key format (should start with 'sk-')
echo $CLAUDE_API_KEY

# Set key explicitly
claude-tui config set api-key sk-your-actual-key-here

# Test connection
claude-tui test-connection
```

### "Port Already in Use"
```bash
# Find and kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
claude-tui --port 8001
```

### "Memory Issues"
```bash
# Check available memory
claude-tui doctor --memory

# Use lightweight mode for systems <4GB RAM
claude-tui --mode lightweight

# Clear cache if needed
claude-tui cache clear
```

---

## Project Templates Available

| Template | Description | Best For |
|----------|-------------|----------|
| `react-basic` | Simple React app | Learning React |
| `react-typescript` | React with TypeScript | Type-safe frontends |
| `react-advanced` | React with all features | Production apps |
| `fastapi-basic` | Simple FastAPI | API prototyping |
| `fastapi-advanced` | Full FastAPI setup | Production APIs |
| `django-cms` | Django CMS | Content management |
| `nodejs-express` | Express.js API | Node.js backends |
| `python-ml` | ML/AI project template | Data science |
| `mobile-react-native` | React Native app | Mobile apps |
| `game-unity` | Unity game template | Game development |

### Create from Custom Template
```bash
# Use community template
claude-tui create \
  --template community/ecommerce-fullstack \
  --name "My Store"

# Create from GitHub repository
claude-tui create \
  --template github:username/template-repo \
  --name "Custom Project"
```

---

## Performance Tips for New Users

### Optimize for Your Hardware
```bash
# For systems with 8GB+ RAM (recommended)
claude-tui config set performance-mode high
claude-tui config set ai-creativity 0.8
claude-tui config set parallel-tasks 4

# For systems with 4GB RAM
claude-tui config set performance-mode balanced
claude-tui config set ai-creativity 0.6
claude-tui config set parallel-tasks 2

# For systems with <4GB RAM
claude-tui config set performance-mode low
claude-tui config set ai-creativity 0.4
claude-tui config set parallel-tasks 1
```

### Caching for Speed
```bash
# Enable intelligent caching (recommended)
claude-tui config set cache-enabled true
claude-tui config set cache-ttl 3600

# Pre-warm cache with common patterns
claude-tui cache warm --patterns "react,fastapi,authentication"
```

---

## What You've Built

After following this quick start, you have:

1. **🏗️ Production-Ready Project Structure**
   - Intelligent directory organization
   - Configuration files optimized for your stack
   - CI/CD pipeline ready to deploy

2. **🤖 AI-Generated Code**  
   - Components with proper TypeScript types
   - Comprehensive test coverage (80%+)
   - Documentation automatically generated
   - Security best practices implemented

3. **🔍 Quality Assurance**
   - Anti-hallucination validation passed
   - Code style and linting configured
   - Performance optimizations applied
   - Accessibility features included

4. **🚀 Ready for Development**
   - Development server configured
   - Hot reloading enabled
   - Debugging tools integrated
   - Deployment scripts generated

---

## Your Next 10 Minutes

Now that you're set up, here's what to try next:

### Minute 1-2: Explore the Generated Code
```bash
# Browse the generated project structure
tree ./ai-dashboard

# Look at the AI-generated components
cat ./ai-dashboard/src/components/UserDashboard.tsx
```

### Minute 3-4: Run Tests and Validation
```bash
# Run the comprehensive test suite
cd ./ai-dashboard
npm test

# Validate code quality
claude-tui validate --comprehensive
```

### Minute 5-6: Try Advanced AI Features
```bash
# Generate a complex feature
claude-tui generate \
  --prompt "Add user authentication with social login" \
  --type feature \
  --include-tests true

# Review generated code with AI
claude-tui review --ai-powered
```

### Minute 7-8: Customize and Extend
```bash
# Modify AI creativity for different results
claude-tui config set ai-creativity 0.9

# Regenerate with higher creativity
claude-tui generate \
  --prompt "Create an innovative data visualization" \
  --regenerate
```

### Minute 9-10: Deploy Your First Version
```bash
# Build for production
npm run build

# Deploy to Vercel/Netlify (if configured)
claude-tui deploy --platform vercel

# Or preview locally
npm run preview
```

---

## Welcome to the Future of Development! 🚀

You're now equipped with Claude-TUI, the most advanced AI-powered development environment. Whether you're building web apps, APIs, mobile apps, or complex systems, Claude-TUI accelerates your development while maintaining the highest quality standards.

**Happy coding!** 🎉

---

*Need help? Join our [Discord community](https://discord.gg/claude-tui) or check the [full documentation](https://docs.claude-tui.dev) for advanced features.*