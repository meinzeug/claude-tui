# Claude-TUI Launcher Guide

This guide explains how to use the robust TUI launcher system for the Claude-TUI application.

## Quick Start

### Simple Usage (Recommended for most users)

```bash
# Start in interactive mode (default)
python3 run_tui.py

# Start in headless mode
python3 run_tui.py headless

# Start in debug mode
python3 run_tui.py debug
```

### Advanced Usage (Full control)

```bash
# Basic test mode
python3 launch_tui.py --test-mode

# Headless mode with logging
python3 launch_tui.py --headless --log-level INFO

# Debug mode with file logging
python3 launch_tui.py --debug --log-file /tmp/claude-tui.log

# Non-interactive mode with custom UI type
python3 launch_tui.py --non-interactive --ui-type ui --retry-attempts 5
```

## Launcher Features

### 1. Robust Error Handling
- **Graceful degradation**: Falls back to simpler implementations when advanced features fail
- **Retry logic**: Automatically retries failed operations (configurable attempts)
- **Comprehensive logging**: Detailed error reporting and debugging information
- **Recovery mechanisms**: Attempts multiple strategies to initialize components

### 2. Multiple Operation Modes

#### Interactive Mode (Default)
- Full TUI experience with all features
- Real-time progress monitoring
- User interaction capabilities
- Visual interface

#### Headless Mode
- No visual interface
- Suitable for server deployments
- API-only operation
- Minimal resource usage

#### Test Mode
- Non-blocking initialization
- Suitable for automated testing
- Quick startup and shutdown
- Minimal logging

#### Debug Mode
- Verbose logging
- Detailed error information
- Performance metrics
- Development assistance

### 3. Dependency Management
- **Automatic dependency checking**: Validates required packages
- **Graceful handling of missing dependencies**: Continues with available components
- **Optional dependency warnings**: Informs about missing optional features
- **Version compatibility**: Checks Python version requirements

### 4. Health Monitoring
- **System resource checking**: Memory and disk space validation
- **Component status tracking**: Monitors individual component health
- **Performance metrics**: Tracks initialization times
- **Environment validation**: Checks system configuration

### 5. UI Implementation Flexibility
- **Multiple UI backends**: Supports different UI implementations
- **Automatic fallback**: Tries multiple UI types until one works
- **Custom UI selection**: Allows specifying preferred UI type
- **Minimal fallback**: Emergency console mode when all else fails

## Command Line Options

### Mode Options
- `--interactive` / `--non-interactive`: Control interactive mode (default: interactive)
- `--headless`: Run without visual interface
- `--test-mode`: Run in test mode (non-blocking)
- `--background-init`: Initialize in background thread

### UI Options
- `--ui-type {auto,ui,claude_tui}`: Specify UI implementation
  - `auto`: Try all available implementations (default)
  - `ui`: Use src/ui implementation
  - `claude_tui`: Use src/claude_tui/ui implementation

### Debugging Options
- `--debug`: Enable debug mode with verbose logging
- `--log-level {DEBUG,INFO,WARNING,ERROR}`: Set logging verbosity
- `--log-file LOG_FILE`: Write logs to file

### Recovery Options
- `--timeout TIMEOUT`: Timeout for operations (default: 30s)
- `--retry-attempts N`: Number of retry attempts (default: 3)
- `--no-fallback`: Disable fallback mode
- `--no-health-check`: Skip system health check

## Architecture

### Component Initialization Order
1. **Dependency Check**: Validates required packages
2. **Health Check**: Monitors system resources
3. **Bridge Initialization**: Sets up integration layer
4. **App Instance Creation**: Creates UI application
5. **Application Initialization**: Initializes core systems
6. **Application Execution**: Runs in appropriate mode

### Fallback Strategy
1. **Bridge-based initialization**: Preferred method using integration bridge
2. **Direct import attempts**: Fallback to direct module imports
3. **Minimal fallback app**: Emergency console mode

### Error Recovery
- **Progressive degradation**: Continues with available components
- **Retry mechanisms**: Automatic retry with exponential backoff
- **Detailed logging**: Comprehensive error reporting
- **Graceful shutdown**: Clean cleanup on failures

## Integration with Existing Code

### Programmatic Usage

```python
from launch_tui import TUILauncher, LauncherConfig

# Create custom configuration
config = LauncherConfig(
    interactive=True,
    headless=False,
    debug=True,
    ui_type="auto",
    retry_attempts=5
)

# Launch application
launcher = TUILauncher(config)
success, app = launcher.launch()

if success:
    print("Application started successfully!")
    # Use app instance...
else:
    print("Failed to start application")
```

### Entry Point Functions

```python
from launch_tui import launch_interactive, launch_headless, launch_test_mode

# Simple entry points
app = launch_interactive()  # Full interactive mode
app = launch_headless()     # Headless mode
app = launch_test_mode()    # Test mode
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
**Symptoms**: `ImportError` or `ModuleNotFoundError`
**Solutions**:
- Check that all required dependencies are installed: `pip install -r requirements.txt`
- Verify Python path includes src directory
- Use `--debug` flag for detailed import information

#### 2. Initialization Failures
**Symptoms**: "Failed to initialize integration bridge"
**Solutions**:
- Check system resources (memory, disk space)
- Try `--retry-attempts 5` for unstable networks
- Use `--no-health-check` to skip resource validation
- Enable `--debug` mode for detailed diagnostics

#### 3. UI Creation Failures
**Symptoms**: "No UI implementation available"
**Solutions**:
- Try different UI types: `--ui-type ui` or `--ui-type claude_tui`
- Enable fallback mode (default) for emergency console mode
- Check terminal compatibility for textual applications

#### 4. Permission Errors
**Symptoms**: Permission denied errors
**Solutions**:
- Ensure write permissions for log files
- Check config directory permissions (~/.config/claude-tui)
- Run with appropriate user permissions

### Debug Information

Enable debug logging to get comprehensive information:

```bash
python3 launch_tui.py --debug --log-level DEBUG --log-file debug.log
```

This provides:
- Detailed component initialization steps
- Import attempt information
- System resource status
- Error stack traces
- Performance metrics

### Recovery Options

If standard launch fails:

```bash
# Try minimal configuration
python3 launch_tui.py --no-health-check --retry-attempts 1 --ui-type auto

# Emergency fallback mode
python3 launch_tui.py --headless --no-health-check --fallback-mode

# Console debugging
python3 launch_tui.py --debug --log-level DEBUG --non-interactive
```

## Advanced Configuration

### Custom Launcher Configuration

```python
from launch_tui import LauncherConfig, TUILauncher

config = LauncherConfig(
    interactive=True,
    debug=False,
    headless=False,
    test_mode=False,
    ui_type="auto",
    log_level="INFO",
    log_file="/var/log/claude-tui.log",
    timeout=60.0,
    retry_attempts=5,
    fallback_mode=True,
    health_check=True,
    background_init=False
)

launcher = TUILauncher(config)
success, app = launcher.launch()
```

### Environment Variables

Set environment variables for default behavior:

```bash
export CLAUDE_TUI_LOG_LEVEL=DEBUG
export CLAUDE_TUI_UI_TYPE=ui
export CLAUDE_TUI_HEADLESS=true
```

### Integration with systemd

Create a service file for headless deployment:

```ini
[Unit]
Description=Claude TUI Service
After=network.target

[Service]
Type=simple
User=claude-tui
WorkingDirectory=/opt/claude-tui
ExecStart=/usr/bin/python3 launch_tui.py --headless --log-file /var/log/claude-tui.log
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Performance Considerations

### Resource Usage
- **Interactive mode**: Higher memory usage due to UI components
- **Headless mode**: Minimal resource usage
- **Debug mode**: Additional CPU for logging operations
- **Background init**: Faster startup but requires thread management

### Optimization Tips
1. Use headless mode for server deployments
2. Disable health checks for faster startup (if system is stable)
3. Reduce retry attempts for faster failure detection
4. Use appropriate log levels to balance debugging and performance

## Security Considerations

### File Permissions
- Log files should have appropriate permissions
- Config directories should be user-accessible only
- Temporary files are created with secure permissions

### Network Security
- API tokens are never logged
- Network timeouts prevent hanging connections
- SSL/TLS verification is enforced where applicable

### Process Security
- Signal handlers for graceful shutdown
- Process isolation in headless mode
- Memory cleanup on exit

## Support and Maintenance

### Logging
All launcher operations are logged with appropriate levels:
- **ERROR**: Critical failures that prevent operation
- **WARNING**: Issues that don't prevent operation but should be addressed
- **INFO**: Important operational information
- **DEBUG**: Detailed diagnostic information

### Monitoring
The launcher provides metrics for:
- Initialization time
- Component success rates
- Error frequencies
- Resource usage patterns

### Updates
The launcher is designed to be forward-compatible:
- New UI implementations are automatically detected
- Additional components can be added to the bridge
- Configuration options can be extended without breaking changes

For additional support, check the logs and use debug mode for detailed diagnostics.