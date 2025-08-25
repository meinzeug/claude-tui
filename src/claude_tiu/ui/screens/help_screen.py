"""
Help Screen - Context-sensitive help and documentation.

Provides comprehensive help documentation, keyboard shortcuts,
tutorials, and troubleshooting information.
"""

import logging
from typing import Dict, List, Tuple

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Static, Button, Tree, Tabs, TabbedContent, TabPane,
    Markdown, Label
)
from textual.screen import ModalScreen
from textual.binding import Binding

logger = logging.getLogger(__name__)


class HelpScreen(ModalScreen):
    """
    Comprehensive help and documentation screen.
    
    Sections:
    - Getting Started
    - Keyboard Shortcuts  
    - Features Guide
    - Troubleshooting
    - About
    """
    
    BINDINGS = [
        Binding("escape", "dismiss", "Close Help"),
        Binding("q", "dismiss", "Close Help"),
    ]
    
    def compose(self) -> ComposeResult:
        """Compose the help screen layout."""
        with Container(classes="help-container"):
            # Header
            with Horizontal(classes="help-header"):
                yield Label("📚 Claude TIU Help & Documentation", classes="help-title")
                yield Button("❌ Close", id="close_btn", variant="default")
            
            # Main content with tabs
            with TabbedContent(initial="getting_started"):
                with TabPane("Getting Started", id="getting_started"):
                    yield self._create_getting_started_content()
                
                with TabPane("Keyboard Shortcuts", id="shortcuts"):
                    yield self._create_shortcuts_content()
                
                with TabPane("Features", id="features"):
                    yield self._create_features_content()
                
                with TabPane("AI Assistant", id="ai_help"):
                    yield self._create_ai_help_content()
                
                with TabPane("Troubleshooting", id="troubleshooting"):
                    yield self._create_troubleshooting_content()
                
                with TabPane("About", id="about"):
                    yield self._create_about_content()
    
    def _create_getting_started_content(self) -> ScrollableContainer:
        """Create getting started content."""
        content = ScrollableContainer(classes="help-content")
        
        getting_started_md = """
# 🚀 Getting Started with Claude TIU

## Welcome!
Claude TIU is an AI-powered Terminal User Interface that enhances your development workflow with intelligent assistance, project management, and automated tasks.

## First Steps

### 1. Create Your First Project
- Press `Ctrl+N` or click the "New Project" button
- Follow the project wizard to select a template
- Configure your project settings
- Let Claude TIU set up your development environment

### 2. Explore the Interface
- **File Browser**: Navigate your project files (left panel)
- **Editor**: Edit code with syntax highlighting (center)
- **AI Assistant**: Get help and generate code (right panel)
- **Task Monitor**: Track progress of automated tasks

### 3. AI Integration
- Use the AI Assistant to:
  - Generate code snippets
  - Review and improve existing code
  - Get explanations for complex logic
  - Debug issues and errors

### 4. Workflow Automation
- Create custom workflows for repetitive tasks
- Use templates for common project structures  
- Integrate with Git for version control
- Set up automated testing and validation

## Quick Tips
- Use `F1` to open this help at any time
- `Ctrl+S` saves the current file
- `F5` refreshes the workspace
- Tab through interface elements
- Most actions have keyboard shortcuts (see Shortcuts tab)

## Need Help?
- Check the Troubleshooting tab for common issues
- Use the AI Assistant for context-specific help
- Visit our documentation for detailed guides
"""
        
        with content:
            yield Markdown(getting_started_md)
        
        return content
    
    def _create_shortcuts_content(self) -> ScrollableContainer:
        """Create keyboard shortcuts content."""
        content = ScrollableContainer(classes="help-content")
        
        shortcuts = [
            ("Global", [
                ("F1", "Show this help screen"),
                ("F2", "Open settings"),
                ("F5", "Refresh workspace"),
                ("Ctrl+Q", "Quit application"),
                ("Escape", "Close modal/dialog"),
            ]),
            ("File Management", [
                ("Ctrl+N", "New project/file"),
                ("Ctrl+O", "Open project/file"),
                ("Ctrl+S", "Save current file"),
                ("Ctrl+Shift+S", "Save all files"),
            ]),
            ("Editor", [
                ("Ctrl+Z", "Undo"),
                ("Ctrl+Y", "Redo"),
                ("Ctrl+F", "Find in file"),
                ("Ctrl+H", "Find and replace"),
                ("Ctrl+/", "Toggle comment"),
                ("Tab", "Indent"),
                ("Shift+Tab", "Unindent"),
            ]),
            ("Navigation", [
                ("Tab", "Next element"),
                ("Shift+Tab", "Previous element"),
                ("Arrow Keys", "Navigate"),
                ("Enter", "Activate/Select"),
                ("Space", "Toggle checkbox/radio"),
            ]),
            ("AI Assistant", [
                ("Ctrl+Enter", "Send message to AI"),
                ("Ctrl+G", "Generate code"),
                ("Ctrl+R", "Review current file"),
                ("Ctrl+A", "Ask AI about selection"),
            ]),
        ]
        
        shortcut_text = "# ⌨️ Keyboard Shortcuts\n\n"
        
        for category, category_shortcuts in shortcuts:
            shortcut_text += f"## {category}\n\n"
            for key, description in category_shortcuts:
                shortcut_text += f"- **{key}**: {description}\n"
            shortcut_text += "\n"
        
        with content:
            yield Markdown(shortcut_text)
        
        return content
    
    def _create_features_content(self) -> ScrollableContainer:
        """Create features guide content."""
        content = ScrollableContainer(classes="help-content")
        
        features_md = """
# ✨ Features Guide

## 🤖 AI-Powered Development

### Code Generation
- Describe what you want in natural language
- AI generates production-ready code
- Supports multiple programming languages
- Follows best practices and patterns

### Code Review & Analysis  
- Automated code quality checks
- Security vulnerability detection
- Performance optimization suggestions
- Documentation completeness review

### Intelligent Completion
- Context-aware code completion
- API usage suggestions
- Error detection and fixes
- Refactoring recommendations

## 📁 Project Management

### Project Templates
- Pre-configured project structures
- Language-specific templates
- Framework integration
- Dependency management

### Workspace Organization
- File tree navigation
- Multi-file editing
- Project-wide search
- Git integration

## 🔄 Workflow Automation

### Task Engine
- Define custom workflows
- Parallel task execution
- Dependency management
- Progress tracking

### CI/CD Integration
- Automated testing
- Build pipeline setup
- Deployment automation
- Quality gates

## 🛡️ Validation & Quality

### Anti-Hallucination Engine
- Detects incomplete code
- Validates AI outputs
- Ensures production readiness
- Continuous quality monitoring

### Real-time Validation
- Syntax checking
- Type validation
- Style consistency
- Security scanning

## 🎨 User Interface

### Responsive Design
- Adaptive layout
- Multiple themes
- Customizable interface
- Accessibility features

### Real-time Updates
- Live progress tracking
- Instant feedback
- Collaborative editing
- Notification system
"""
        
        with content:
            yield Markdown(features_md)
        
        return content
    
    def _create_ai_help_content(self) -> ScrollableContainer:
        """Create AI assistant help content."""
        content = ScrollableContainer(classes="help-content")
        
        ai_help_md = """
# 🤖 AI Assistant Guide

## Getting Started with AI

### Basic Usage
1. Type your question or request in the AI panel
2. Press Enter or click Send
3. AI will analyze your request and provide assistance
4. Follow up with clarifying questions as needed

### Example Prompts

#### Code Generation
```
Create a Python function that reads a CSV file and returns a pandas DataFrame
```

```
Generate a React component for a user profile card with props
```

#### Code Review
```
Review this function for performance issues and suggest improvements
```

```
Check this code for security vulnerabilities
```

#### Debugging
```
This function is throwing an IndexError, can you help me fix it?
```

```
Why isn't my API endpoint returning the expected response?
```

#### Explanation
```
Explain how this algorithm works step by step
```

```
What design patterns are used in this code?
```

## Advanced Features

### Context Awareness
- AI understands your current file and project
- Provides relevant suggestions based on your codebase
- Maintains conversation history for follow-ups

### Multi-language Support
- Python, JavaScript, TypeScript, Java, C++, and more
- Framework-specific knowledge (React, Django, Express, etc.)
- Best practices for each language and framework

### Integration Features
- Automatic code insertion at cursor
- File creation and modification
- Project structure generation
- Dependency recommendations

## Tips for Better Results

### Be Specific
❌ "Make this better"
✅ "Optimize this function for better memory usage"

### Provide Context
❌ "Fix this error"
✅ "This function throws TypeError when input is None, how do I handle it?"

### Ask Follow-up Questions
- "Can you explain this part in more detail?"
- "What are the trade-offs of this approach?"
- "Show me an alternative implementation"

## Limitations

### What AI Can Do
- Generate and review code
- Explain concepts and algorithms
- Suggest improvements and fixes
- Provide documentation and examples

### What AI Cannot Do
- Execute code directly
- Access external systems without permission
- Make destructive changes without confirmation
- Replace proper testing and validation
"""
        
        with content:
            yield Markdown(ai_help_md)
        
        return content
    
    def _create_troubleshooting_content(self) -> ScrollableContainer:
        """Create troubleshooting content."""
        content = ScrollableContainer(classes="help-content")
        
        troubleshooting_md = """
# 🔧 Troubleshooting

## Common Issues

### AI Services Not Working

#### Symptoms
- "AI services disconnected" message
- No response from AI assistant
- Error messages when asking questions

#### Solutions
1. **Check API Key**
   - Ensure `CLAUDE_CODE_OAUTH_TOKEN` is set
   - Verify the token is valid and not expired
   - Check environment variable in terminal

2. **Network Connection**
   - Verify internet connectivity
   - Check firewall settings
   - Try pinging api.anthropic.com

3. **Restart Application**
   - Close and reopen Claude TIU
   - Check system resources (memory, CPU)

### File Operations Failing

#### Symptoms
- Cannot open or save files
- Permission denied errors
- File tree not loading

#### Solutions
1. **Check Permissions**
   - Ensure read/write access to project directory
   - Run with appropriate user permissions
   - Check file ownership

2. **File Path Issues**
   - Verify path exists and is accessible
   - Check for special characters in filenames
   - Ensure adequate disk space

### Performance Issues

#### Symptoms
- Slow response times
- UI freezing or lag
- High memory usage

#### Solutions
1. **System Resources**
   - Close unnecessary applications
   - Increase available memory
   - Check CPU usage

2. **Project Size**
   - Large projects may be slower
   - Consider excluding node_modules or similar
   - Use .gitignore patterns

### Installation Problems

#### Symptoms
- Missing dependencies
- Import errors
- Module not found errors

#### Solutions
1. **Python Environment**
   ```bash
   pip install -r requirements.txt
   pip install --upgrade claude-tiu
   ```

2. **Node.js Dependencies**
   ```bash
   npm install -g claude-flow@alpha
   ```

## Getting Help

### System Diagnostics
Run the built-in system checker:
```bash
claude-tiu doctor
```

### Debug Mode
Enable debug logging:
```bash
claude-tiu --debug
```

### Log Files
Check log files at:
- Linux/Mac: `~/.claude-tiu/logs/`
- Windows: `%APPDATA%\\claude-tiu\\logs\\`

### Contact Support
If problems persist:
- Check GitHub issues
- Create new issue with logs
- Include system information
- Describe steps to reproduce

## FAQ

### Q: How do I update Claude TIU?
A: Use pip to update: `pip install --upgrade claude-tiu`

### Q: Can I use Claude TIU offline?
A: Basic features work offline, but AI requires internet connection

### Q: How do I backup my configuration?
A: Configuration is stored in `~/.claude-tiu/config.yaml`

### Q: Is my code sent to external servers?
A: Only when explicitly using AI features, and only the relevant context
"""
        
        with content:
            yield Markdown(troubleshooting_md)
        
        return content
    
    def _create_about_content(self) -> ScrollableContainer:
        """Create about content."""
        content = ScrollableContainer(classes="help-content")
        
        about_md = """
# ℹ️ About Claude TIU

## What is Claude TIU?

Claude TIU (Terminal User Interface) is an intelligent AI-powered development environment that combines the power of artificial intelligence with sophisticated terminal interfaces for enhanced software development workflows.

## Key Features

- **🤖 AI Integration**: Powered by Claude for intelligent code generation and analysis
- **🎯 Anti-Hallucination**: Advanced validation to ensure code quality
- **📊 Real-time Monitoring**: Track progress and system status
- **🔄 Workflow Automation**: Orchestrate complex development tasks
- **🛡️ Security First**: Sandboxed execution and security validation
- **🎨 Modern UI**: Beautiful terminal interface with rich interactions

## Technology Stack

- **Frontend**: Textual (Python TUI framework)
- **AI**: Claude Code integration
- **Backend**: FastAPI + SQLAlchemy
- **Database**: SQLite/PostgreSQL
- **Validation**: Custom anti-hallucination engine
- **Orchestration**: Claude Flow

## Version Information

- **Version**: 0.1.0
- **Status**: Alpha Release
- **Python**: 3.9+ Required
- **License**: MIT

## Credits

### Development Team
- Claude TIU Core Team
- AI Integration Specialists  
- UX/UI Designers
- Quality Assurance Engineers

### Special Thanks
- Anthropic for Claude AI
- Textual framework creators
- Open source community
- Beta testers and early adopters

### Third-Party Libraries
- Textual: Modern TUI framework
- Rich: Text formatting and styling
- FastAPI: Web framework
- SQLAlchemy: Database ORM
- Pydantic: Data validation
- Click: Command line interface

## License

MIT License - see LICENSE file for details

## Contributing

We welcome contributions! Please see our contributing guidelines:

- Fork the repository
- Create feature branch
- Add tests for new functionality
- Ensure all tests pass
- Submit pull request with clear description

## Support

- **Documentation**: https://claude-tiu.readthedocs.io
- **GitHub**: https://github.com/claude-tiu/claude-tiu
- **Issues**: https://github.com/claude-tiu/claude-tiu/issues
- **Discussions**: https://github.com/claude-tiu/claude-tiu/discussions

## Roadmap

### Upcoming Features
- Visual workflow designer
- Plugin system
- Team collaboration tools
- Cloud integration
- Mobile companion app
- Advanced debugging tools

### Long-term Vision
Create the most intelligent and intuitive development environment that empowers developers to build better software faster while maintaining the highest quality standards.

---

*Built with ❤️ by the Claude TIU team*
"""
        
        with content:
            yield Markdown(about_md)
        
        return content
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle help screen button presses."""
        if event.button.id == "close_btn":
            self.dismiss()
    
    def action_dismiss(self) -> None:
        """Dismiss the help screen."""
        self.dismiss()
        
    def on_mount(self) -> None:
        """Initialize the help screen."""
        logger.info("Help screen opened")