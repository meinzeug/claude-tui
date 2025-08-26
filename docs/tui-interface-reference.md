# Claude-TUI Terminal User Interface Reference

## 🖥️ Complete TUI Navigation & Keyboard Shortcuts

The Claude-TUI Terminal User Interface provides a powerful, keyboard-driven development environment. This reference covers all keyboard shortcuts, interface elements, and advanced navigation techniques.

---

## Table of Contents

1. [Interface Overview](#interface-overview)
2. [Global Keyboard Shortcuts](#global-keyboard-shortcuts)
3. [Screen-Specific Commands](#screen-specific-commands)
4. [Advanced Navigation](#advanced-navigation)
5. [Customization Options](#customization-options)
6. [Performance Tips](#performance-tips)

---

## Interface Overview

### Main Interface Layout
```
┌─ Claude-TUI AI Development Environment ─ [v1.0.0] ──────────────────────┐
│                                                                         │
│ 📊 Dashboard │ 📁 Projects │ 🤖 AI Tools │ 🔧 Tasks │ ⚙️ Settings      │
│ ────────────────────────────────────────────────────────────────────── │
│                                                                         │
│ 🎯 Active Project: E-commerce API                    🔥 Health: 94%    │
│ 📂 Location: /home/user/projects/ecommerce-api      💾 Auto-save: ON   │
│                                                                         │
│ ┌─ Quick Actions ─────────────┐  ┌─ Real-time Status ──────────────┐   │
│ │ [G] Generate Code           │  │ 🤖 AI Agent: Ready              │   │
│ │ [R] Review & Analyze        │  │ 📊 Progress: 87%                │   │
│ │ [T] Run Tests               │  │ ⚡ Performance: Optimal          │   │
│ │ [V] Validate Project        │  │ 🔍 Last Validation: 2 min ago   │   │
│ │ [D] Deploy Application      │  │ 🌐 Server: Running (port 8000)  │   │
│ └─────────────────────────────┘  └──────────────────────────────────┘   │
│                                                                         │
│ ┌─ AI Insights & Suggestions ───────────────────────────────────────┐   │
│ │ 💡 Consider implementing rate limiting for public endpoints       │   │
│ │ 🔒 Add input validation to user registration endpoint             │   │
│ │ 🚀 Database queries could be optimized for better performance     │   │
│ │ 📚 API documentation coverage: 76% (Good, could be improved)      │   │
│ └────────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│ ┌─ Project Files ────────────────┐  ┌─ Console Output ───────────────┐  │
│ │ 📁 src/                        │  │ $ Running tests...              │  │
│ │   ├── api/                     │  │ ✅ test_auth.py::test_login     │  │
│ │   │   ├── auth.py              │  │ ✅ test_auth.py::test_register  │  │
│ │   │   └── models.py            │  │ ✅ test_api.py::test_users      │  │
│ │   ├── tests/                   │  │ 📊 Coverage: 94%                │  │
│ │   └── main.py                  │  │ 🎉 All tests passed!           │  │
│ │ 📄 requirements.txt            │  │                                 │  │
│ │ ⚙️ config.yaml                 │  └─────────────────────────────────┘  │
│ └────────────────────────────────┘                                      │
│                                                                         │
│ Status: 🟢 Ready │ Memory: 2.1GB │ CPU: 15% │ Press ? for help, Q to quit │
└─────────────────────────────────────────────────────────────────────────┘
```

### Screen Hierarchy
```
Main Dashboard
├── 📊 Dashboard (Overview & Metrics)
├── 📁 Projects (Project Management)
│   ├── Project List
│   ├── Project Details  
│   ├── Project Creation Wizard
│   └── Project Settings
├── 🤖 AI Tools (AI-Powered Features)
│   ├── Code Generation
│   ├── Code Review
│   ├── Validation Engine
│   └── Agent Orchestration
├── 🔧 Tasks (Task Management)
│   ├── Active Tasks
│   ├── Task History
│   └── Workflow Builder
└── ⚙️ Settings (Configuration)
    ├── General Settings
    ├── AI Configuration
    ├── Performance Tuning
    └── Keyboard Shortcuts
```

---

## Global Keyboard Shortcuts

### Navigation & Core Actions
| Shortcut | Action | Description |
|----------|--------|-------------|
| `Tab` / `Shift+Tab` | Navigate | Move between interface elements |
| `1-5` | Quick Screen | Jump to main screens (Dashboard, Projects, AI, Tasks, Settings) |
| `Ctrl+Tab` | Screen Forward | Cycle through screens forward |
| `Ctrl+Shift+Tab` | Screen Backward | Cycle through screens backward |
| `Esc` | Back/Cancel | Go back or cancel current action |
| `Enter` | Confirm/Select | Confirm action or select item |
| `Space` | Toggle/Activate | Toggle checkboxes or activate buttons |
| `?` / `F1` | Help | Show context-sensitive help |
| `Ctrl+Q` / `Q` | Quit | Exit Claude-TUI (with confirmation) |

### File & Project Operations
| Shortcut | Action | Description |
|----------|--------|-------------|
| `Ctrl+N` | New Project | Create new AI-powered project |
| `Ctrl+O` | Open Project | Open existing project |
| `Ctrl+S` | Save | Save current changes |
| `Ctrl+Shift+S` | Save All | Save all modified files |
| `Ctrl+R` | Refresh | Refresh current view/data |
| `Ctrl+W` | Close | Close current project/tab |
| `F5` | Reload | Force reload interface |

### AI & Development Actions
| Shortcut | Action | Description |
|----------|--------|-------------|
| `G` | Generate Code | Open AI code generation dialog |
| `Ctrl+G` | Quick Generate | Quick code generation with last settings |
| `R` | Review Code | Start AI code review |
| `Ctrl+R` | Auto Review | Review all modified files |
| `T` | Run Tests | Execute test suite |
| `Ctrl+T` | Test Current | Test current file/component |
| `V` | Validate | Run validation engine |
| `Ctrl+V` | Quick Validate | Validate with last settings |
| `D` | Deploy | Open deployment options |
| `Ctrl+D` | Quick Deploy | Deploy with last configuration |

### Search & Navigation
| Shortcut | Action | Description |
|----------|--------|-------------|
| `Ctrl+F` | Search | Search in current view |
| `Ctrl+Shift+F` | Global Search | Search across entire project |
| `Ctrl+P` | Quick Open | Quick project/file picker |
| `Ctrl+Shift+P` | Command Palette | Show all available commands |
| `/` | Filter | Quick filter current list |
| `Ctrl+/` | Toggle Filter | Show/hide filter bar |

### Window Management
| Shortcut | Action | Description |
|----------|--------|-------------|
| `F11` | Fullscreen | Toggle fullscreen mode |
| `Ctrl++` | Zoom In | Increase interface scale |
| `Ctrl+-` | Zoom Out | Decrease interface scale |
| `Ctrl+0` | Reset Zoom | Reset to default scale |
| `Ctrl+Shift+D` | Toggle Dark Mode | Switch between light/dark themes |

---

## Screen-Specific Commands

### 📊 Dashboard Screen
| Shortcut | Action | Description |
|----------|--------|-------------|
| `R` | Refresh Metrics | Update all dashboard metrics |
| `A` | View Analytics | Open detailed analytics view |
| `L` | View Logs | Show recent activity logs |
| `S` | System Status | Detailed system health check |
| `H` | Health Check | Run comprehensive health check |
| `↑↓` | Navigate Widgets | Move between dashboard widgets |
| `Enter` | Expand Widget | Expand selected widget to full view |
| `M` | Minimize All | Minimize all expanded widgets |

```
Dashboard Navigation:
┌─ Performance Metrics ──┐ ← R to refresh
│ CPU: ████▁▁▁▁▁▁ 40%    │
│ RAM: ██████▁▁▁▁ 60%    │
│ Network: Low           │
└────────────────────────┘
         ↕ Use ↑↓ to navigate
┌─ AI Agent Status ──────┐
│ 🤖 Primary: Active     │ ← Enter to expand
│ 🔄 Queue: 3 tasks      │
│ ⚡ Response: 150ms     │
└────────────────────────┘
```

### 📁 Projects Screen
| Shortcut | Action | Description |
|----------|--------|-------------|
| `N` | New Project | Create new project |
| `O` | Open Selected | Open selected project |
| `E` | Edit Project | Edit project settings |
| `C` | Clone Project | Clone/duplicate project |
| `Delete` | Delete Project | Delete selected project (with confirmation) |
| `I` | Import Project | Import existing project |
| `X` | Export Project | Export project as template |
| `F` | Toggle Favorites | Add/remove from favorites |
| `↑↓` | Navigate List | Move through project list |
| `PageUp/Down` | Page Navigation | Navigate page by page |
| `Home/End` | First/Last | Jump to first/last project |

```
Project List Navigation:
📁 Projects (12 total)
┌─────────────────────────────────────────┐
│ ⭐ E-commerce API        │ FastAPI │ 94% │ ← Selected (press O to open)
│    Blog Platform         │ Django  │ 87% │
│    Mobile App            │ React   │ 92% │
│    Data Pipeline         │ Python  │ 81% │
└─────────────────────────────────────────┘
         ↑↓ Navigate    F=Favorite    N=New
```

### 🤖 AI Tools Screen
| Shortcut | Action | Description |
|----------|--------|-------------|
| `1` | Code Generation | Open code generation tool |
| `2` | Code Review | Start code review process |
| `3` | Validation Engine | Run anti-hallucination validation |
| `4` | Agent Orchestration | Manage AI agent swarms |
| `5` | Training Mode | Train custom AI patterns |
| `C` | Configuration | AI behavior configuration |
| `H` | History | View AI interaction history |
| `B` | Benchmarks | View AI performance benchmarks |
| `Ctrl+Enter` | Execute Prompt | Execute current AI prompt |
| `Ctrl+↑↓` | Prompt History | Navigate through prompt history |

```
AI Code Generation Interface:
┌─ Prompt Input ─────────────────────────────────────────────────┐
│ Create a user authentication system with JWT tokens           │ ← Type here
│ and password reset functionality                               │
└────────────────────────────────────────────────────────────────┘
┌─ Configuration ────────────┐  ┌─ Preview ──────────────────────┐
│ Language: Python           │  │ Will generate:                 │
│ Framework: FastAPI         │  │ • Authentication endpoints     │
│ Creativity: ████████▁▁ 80% │  │ • Password hashing utilities   │
│ Include Tests: ✓           │  │ • JWT token management         │
│ Include Docs: ✓            │  │ • Email service integration    │
└────────────────────────────┘  └────────────────────────────────┘
                                    Ctrl+Enter to generate
```

### 🔧 Tasks Screen
| Shortcut | Action | Description |
|----------|--------|-------------|
| `N` | New Task | Create new development task |
| `R` | Run Selected | Execute selected task |
| `P` | Pause Task | Pause running task |
| `S` | Stop Task | Stop running task |
| `K` | Kill Task | Force kill task (emergency) |
| `L` | View Logs | Show task execution logs |
| `D` | Task Details | Show detailed task information |
| `H` | Task History | View completed task history |
| `F` | Filter Tasks | Filter by status/type |
| `↑↓` | Navigate Tasks | Move through task list |

```
Task Management Interface:
Status: ⚡ 3 Running, 📋 5 Queued, ✅ 12 Completed

┌─ Active Tasks ─────────────────────────────────────────────────────┐
│ 🟢 Generating user models...     │ 45% │ ETA: 2m │ [P] Pause      │
│ 🟡 Running test suite...         │ 78% │ ETA: 1m │ [L] View Logs  │ ← Selected
│ 🔴 Deploying to staging...       │ 23% │ ETA: 5m │ [S] Stop       │
└────────────────────────────────────────────────────────────────────┘
┌─ Task Queue ───────────────────────────────────────────────────────┐
│ 📋 Code review pending...                                         │
│ 📋 Documentation generation...                                    │
│ 📋 Security scan...                                               │
└────────────────────────────────────────────────────────────────────┘
```

### ⚙️ Settings Screen
| Shortcut | Action | Description |
|----------|--------|-------------|
| `1-9` | Category Navigation | Jump to settings category |
| `R` | Reset to Defaults | Reset current category to defaults |
| `A` | Apply Changes | Apply and save all changes |
| `U` | Undo Changes | Undo unsaved changes |
| `E` | Export Config | Export configuration to file |
| `I` | Import Config | Import configuration from file |
| `T` | Test Settings | Test current configuration |
| `↑↓` | Navigate Options | Move through setting options |
| `Space` | Toggle Boolean | Toggle boolean settings |
| `Enter` | Edit Value | Edit selected setting value |

```
Settings Navigation:
┌─ Categories ──────┐ ┌─ AI Configuration ────────────────────────┐
│ 1. General        │ │ Creativity Level:  ████████▁▁ 80%        │ ← Selected
│ 2. AI Config   ←  │ │ Model Selection:   Claude-3-Sonnet       │
│ 3. Performance    │ │ Response Timeout:  30 seconds            │
│ 4. Shortcuts      │ │ Auto-validation:   ✓ Enabled             │
│ 5. Theme          │ │ Cache Results:     ✓ Enabled             │
│ 6. Advanced       │ │ Debug Mode:        ✗ Disabled            │
└───────────────────┘ └───────────────────────────────────────────┘
                               ↑↓ Navigate  Space=Toggle  Enter=Edit
```

---

## Advanced Navigation

### Multi-Panel Navigation
| Shortcut | Action | Description |
|----------|--------|-------------|
| `Ctrl+1-4` | Focus Panel | Focus specific panel (Files, Editor, Console, etc.) |
| `Ctrl+Shift+←→` | Resize Panels | Adjust panel sizes |
| `Alt+←→` | Switch Panels | Quick switch between adjacent panels |
| `Ctrl+Alt+←→` | Move Panel | Move current panel left/right |
| `F6` | Next Panel | Cycle focus through all panels |
| `Shift+F6` | Previous Panel | Reverse cycle through panels |

### Context Menus & Popups
| Shortcut | Action | Description |
|----------|--------|-------------|
| `Right Click` / `Menu` | Context Menu | Open context-sensitive menu |
| `Ctrl+Space` | Quick Actions | Show available actions for selected item |
| `Alt+Enter` | Properties | Show properties/details of selected item |
| `Ctrl+I` | Information | Show detailed information popup |
| `Esc` | Close Popup | Close any open popup or menu |

### Power User Features
| Shortcut | Action | Description |
|----------|--------|-------------|
| `Ctrl+Shift+P` | Command Palette | Access all commands via search |
| `Ctrl+K` | Quick Command | Execute command by name |
| `Alt+F1-F12` | Custom Shortcuts | User-defined shortcuts (configurable) |
| `Ctrl+Alt+T` | Terminal Mode | Switch to terminal-style input |
| `Ctrl+Alt+M` | Macro Record | Start/stop macro recording |
| `Ctrl+Alt+P` | Macro Playback | Play back recorded macro |

---

## Customization Options

### Keyboard Shortcut Customization
Access via Settings → Keyboard Shortcuts or `Ctrl+Alt+K`:

```
┌─ Keyboard Shortcuts Customization ───────────────────────────────────┐
│                                                                       │
│ Search shortcuts: [_______________] 🔍                               │
│                                                                       │
│ ┌─ Action ──────────────────┐ ┌─ Current ─┐ ┌─ Custom ──────────────┐│
│ │ Generate Code             │ │ G         │ │ [Click to customize]  ││
│ │ Review Code               │ │ R         │ │ Ctrl+Alt+R           ││ ← Modified
│ │ Run Tests                 │ │ T         │ │ [Click to customize]  ││
│ │ Validate Project          │ │ V         │ │ F9                   ││ ← Modified
│ │ Deploy Application        │ │ D         │ │ [Click to customize]  ││
│ └───────────────────────────┘ └───────────┘ └───────────────────────┘│
│                                                                       │
│ [Reset to Defaults] [Export Config] [Import Config] [Apply Changes]  │
└───────────────────────────────────────────────────────────────────────┘
```

### Theme & Appearance
Access via Settings → Theme or `Ctrl+Shift+T`:

```
┌─ Theme Configuration ─────────────────────────────────────────────────┐
│                                                                       │
│ Color Scheme:  ⚫ Dark Mode    ⚪ Light Mode    🌓 Auto (System)     │
│                                                                       │
│ ┌─ Color Palette ──────────┐ ┌─ Preview ──────────────────────────┐  │
│ │ Background: #1a1a1a      │ │ ┌─ Sample Window ─────────────────┐ │  │
│ │ Foreground: #ffffff      │ │ │ 📊 Dashboard │ 🤖 AI Tools       │ │  │
│ │ Accent:     #007acc      │ │ │ ──────────────────────────────── │ │  │
│ │ Success:    #28a745      │ │ │ Status: Ready                   │ │  │
│ │ Warning:    #ffc107      │ │ │ ✅ Tests passed                 │ │  │
│ │ Error:      #dc3545      │ │ │ 💡 AI suggestions available     │ │  │
│ └──────────────────────────┘ │ └─────────────────────────────────┘ │  │
│                              └─────────────────────────────────────┘  │
│                                                                       │
│ Font Family: JetBrains Mono    Font Size: 12    Line Height: 1.4     │
│                                                                       │
│ Interface Scale: ████████▁▁ 100%    Animation Speed: ██████▁▁▁▁ 60%  │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### Layout Customization
Access via Settings → Layout or `Ctrl+Shift+L`:

```
┌─ Layout Configuration ────────────────────────────────────────────────┐
│                                                                       │
│ Default Layout: [Dashboard] [Projects] [AI Tools] [Tasks] [Settings]  │
│                                                                       │
│ Panel Configuration:                                                  │
│ ┌─ Left Panel (25%) ───────┐ ┌─ Main Panel (50%) ─┐ ┌─ Right (25%) ─┐│
│ │ ✓ Project Tree           │ │ ✓ Main Content      │ │ ✓ AI Assistant││
│ │ ✓ File Explorer          │ │ ✓ Code Editor       │ │ ✓ Task Queue  ││
│ │ ✗ Bookmarks              │ │ ✗ Terminal          │ │ ✓ Output Log  ││
│ └──────────────────────────┘ └─────────────────────┘ └───────────────┘│
│                                                                       │
│ ┌─ Bottom Panel (20%) ───────────────────────────────────────────────┐│
│ │ ✓ Console Output    ✓ Error List    ✓ Task Progress    ✗ Debug     ││
│ └────────────────────────────────────────────────────────────────────┘│
│                                                                       │
│ Auto-hide panels: ✓    Show line numbers: ✓    Minimap: ✗            │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Performance Tips

### Memory-Efficient Navigation
- Use `Ctrl+W` to close unused project tabs
- Enable auto-cleanup in Settings → Performance
- Use `Ctrl+Shift+R` to refresh and clear memory caches
- Set lower AI creativity levels for faster responses

### Keyboard-Only Workflow
```bash
# Example power-user workflow using only keyboard:

1. Ctrl+N          # Create new project
2. Type project details using Tab navigation
3. Enter           # Confirm project creation
4. G               # Generate initial code
5. Type AI prompt
6. Ctrl+Enter      # Execute generation
7. V               # Validate generated code
8. T               # Run tests
9. R               # Review code quality
10. D              # Deploy when ready
```

### Quick Actions Setup
Configure frequently used commands in Settings → Quick Actions:

```
┌─ Quick Actions Configuration ─────────────────────────────────────────┐
│                                                                       │
│ Slot 1 (F1):  Generate API endpoint      [Configure]                 │
│ Slot 2 (F2):  Run full test suite        [Configure]                 │
│ Slot 3 (F3):  Deploy to staging          [Configure]                 │
│ Slot 4 (F4):  AI code review             [Configure]                 │
│ Slot 5 (F5):  Validate & fix issues      [Configure]                 │
│ Slot 6 (F6):  Generate documentation     [Configure]                 │
│                                                                       │
│ ✓ Show quick action hints    ✓ Enable F-key shortcuts               │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Accessibility Features

### Visual Accessibility
| Feature | Shortcut | Description |
|---------|----------|-------------|
| High Contrast Mode | `Ctrl+Alt+H` | Enable high contrast theme |
| Zoom Interface | `Ctrl++/-` | Increase/decrease interface scale |
| Large Cursor | `Ctrl+Alt+C` | Enable large cursor mode |
| Screen Reader | `Ctrl+Alt+S` | Toggle screen reader compatibility |

### Keyboard Navigation
- All interface elements are keyboard accessible
- Tab order is logical and predictable
- Focus indicators are clearly visible
- Skip links available for main sections

### Screen Reader Support
Claude-TUI supports popular screen readers:
- NVDA (Windows)
- JAWS (Windows)
- VoiceOver (macOS)
- Orca (Linux)

Enable screen reader mode: Settings → Accessibility → Screen Reader Support

---

## Troubleshooting Interface Issues

### Interface Not Responding
```bash
# Force refresh interface
Ctrl+F5

# Reset to default layout
Ctrl+Alt+0

# Safe mode restart
claude-tui --safe-mode
```

### Keyboard Shortcuts Not Working
1. Check if shortcuts are conflicting with system shortcuts
2. Reset shortcuts: Settings → Keyboard → Reset to Defaults
3. Clear keyboard buffer: `Esc` → `Ctrl+Alt+K`
4. Restart in keyboard-only mode: `claude-tui --keyboard-mode`

### Display Issues
```bash
# Reset zoom level
Ctrl+0

# Switch to basic theme
Ctrl+Shift+D (toggle to light mode)

# Clear interface cache
claude-tui --clear-cache

# Terminal compatibility mode
claude-tui --terminal-mode
```

---

## Advanced Keyboard Combinations

### Multi-Key Sequences
| Sequence | Action | Description |
|----------|--------|-------------|
| `Ctrl+K, P` | Project Commands | Show project-specific commands |
| `Ctrl+K, A` | AI Commands | Show AI-related commands |
| `Ctrl+K, T` | Task Commands | Show task management commands |
| `Ctrl+K, S` | Setting Commands | Show configuration commands |
| `Ctrl+K, H` | Help Commands | Show help and documentation |

### Chord Shortcuts (Hold first key)
| Chord | Action | Description |
|-------|--------|-------------|
| `Alt+G, C` | Generate Component | Generate React/Vue component |
| `Alt+G, A` | Generate API | Generate API endpoint |
| `Alt+G, T` | Generate Tests | Generate test suite |
| `Alt+R, C` | Review Code | AI code review |
| `Alt+R, S` | Review Security | Security analysis |
| `Alt+R, P` | Review Performance | Performance analysis |

---

## Context-Sensitive Help

The help system (`?` or `F1`) provides context-aware assistance:

```
┌─ Context Help: Project Creation ─────────────────────────────────────┐
│                                                                       │
│ 🎯 Current Context: Creating New Project                             │
│                                                                       │
│ Available Shortcuts:                                                  │
│ • Tab/Shift+Tab - Navigate between fields                           │
│ • Ctrl+Space - Show template suggestions                             │
│ • Ctrl+T - Show template preview                                     │
│ • F2 - Configure advanced options                                    │
│ • Enter - Create project with current settings                       │
│ • Esc - Cancel and return to projects list                          │
│                                                                       │
│ 💡 Tips:                                                             │
│ • Use template suggestions for faster setup                          │
│ • Enable AI features for intelligent code generation                 │
│ • Consider adding testing from the start                            │
│                                                                       │
│ 📖 Related Documentation:                                            │
│ • Project Templates Guide                                            │
│ • AI Configuration Best Practices                                   │
│ • Testing Setup Instructions                                        │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Summary of Essential Shortcuts

### Must-Know Shortcuts (Top 20)
1. `?` - Help (most important!)
2. `G` - Generate Code
3. `R` - Review Code  
4. `T` - Run Tests
5. `V` - Validate Project
6. `D` - Deploy
7. `Ctrl+N` - New Project
8. `Ctrl+O` - Open Project
9. `Ctrl+S` - Save
10. `Ctrl+F` - Search
11. `Ctrl+P` - Quick Open
12. `Ctrl+Shift+P` - Command Palette
13. `Tab` - Navigate Forward
14. `Shift+Tab` - Navigate Backward
15. `Esc` - Back/Cancel
16. `Enter` - Confirm/Select
17. `1-5` - Switch Main Screens
18. `Ctrl+Q` - Quit
19. `F5` - Refresh
20. `Ctrl+/` - Toggle Comments

### Productivity Shortcuts (Advanced Users)
- `Ctrl+K` sequences for quick commands
- `Alt+G` combinations for generation
- `Alt+R` combinations for review
- `F1-F6` for custom quick actions
- `Ctrl+Alt+` combinations for system functions

---

**Master these shortcuts to become a Claude-TUI power user! 🚀**

*This reference is also available within the application by pressing `?` or `F1` for context-sensitive help.*