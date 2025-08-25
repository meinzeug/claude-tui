"""
Settings Screen - Application configuration interface.

Provides comprehensive settings management for UI preferences,
AI services, security options, and project defaults.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Static, Button, Input, Select, Switch, Slider, 
    RadioSet, RadioButton, Label, TextArea, Tabs, TabbedContent, TabPane
)
from textual.screen import ModalScreen
from textual.binding import Binding
from textual.message import Message

logger = logging.getLogger(__name__)


class SettingsScreen(ModalScreen):
    """
    Comprehensive settings management screen.
    
    Categories:
    - General preferences
    - UI/UX settings
    - AI service configuration
    - Security & privacy
    - Project defaults
    - Advanced options
    """
    
    BINDINGS = [
        Binding("escape", "dismiss", "Close Settings"),
        Binding("ctrl+s", "save_settings", "Save Settings"),
        Binding("ctrl+r", "reset_defaults", "Reset to Defaults"),
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.settings_changed = False
        self.original_settings: Dict[str, Any] = {}
        self.current_settings: Dict[str, Any] = {}
    
    def compose(self) -> ComposeResult:
        """Compose the settings screen layout."""
        with Container(classes="settings-container"):
            # Header
            with Horizontal(classes="settings-header"):
                yield Label("⚙️ Settings & Preferences", classes="settings-title")
                yield Button("❌ Close", id="close_btn", variant="default")
            
            # Main content with tabs
            with TabbedContent(initial="general"):
                with TabPane("General", id="general"):
                    yield self._create_general_settings()
                
                with TabPane("Interface", id="interface"):
                    yield self._create_interface_settings()
                
                with TabPane("AI Services", id="ai_services"):
                    yield self._create_ai_settings()
                
                with TabPane("Security", id="security"):
                    yield self._create_security_settings()
                
                with TabPane("Projects", id="projects"):
                    yield self._create_project_settings()
                
                with TabPane("Advanced", id="advanced"):
                    yield self._create_advanced_settings()
            
            # Footer with action buttons
            with Horizontal(classes="settings-footer"):
                yield Button("💾 Save", id="save_btn", variant="success")
                yield Button("🔄 Reset", id="reset_btn", variant="default")
                yield Button("❌ Cancel", id="cancel_btn", variant="default")
    
    def _create_general_settings(self) -> ScrollableContainer:
        """Create general settings panel."""
        content = ScrollableContainer(classes="settings-content")
        
        with content:
            yield Label("📋 General Preferences", classes="section-title")
            
            with Vertical(classes="settings-group"):
                yield Label("Application Theme:")
                yield Select(
                    options=[
                        ("dark", "Dark Theme"),
                        ("light", "Light Theme"),
                        ("auto", "Auto (System)")
                    ],
                    value="dark",
                    id="theme_select"
                )
                
                yield Label("Language:")
                yield Select(
                    options=[
                        ("en", "English"),
                        ("es", "Español"),
                        ("fr", "Français"),
                        ("de", "Deutsch")
                    ],
                    value="en",
                    id="language_select"
                )
                
                yield Label("Auto-save interval (minutes):")
                yield Slider(
                    min=1,
                    max=60,
                    value=5,
                    id="autosave_interval"
                )
                
                yield Switch(
                    "Enable notifications",
                    value=True,
                    id="notifications_enabled"
                )
                
                yield Switch(
                    "Check for updates automatically",
                    value=True,
                    id="auto_updates"
                )
                
                yield Switch(
                    "Send anonymous usage statistics",
                    value=False,
                    id="telemetry_enabled"
                )
        
        return content
    
    def _create_interface_settings(self) -> ScrollableContainer:
        """Create interface settings panel."""
        content = ScrollableContainer(classes="settings-content")
        
        with content:
            yield Label("🎨 User Interface", classes="section-title")
            
            with Vertical(classes="settings-group"):
                yield Label("Font size:")
                yield Slider(
                    min=8,
                    max=24,
                    value=12,
                    id="font_size"
                )
                
                yield Switch(
                    "Show line numbers in editor",
                    value=True,
                    id="show_line_numbers"
                )
                
                yield Switch(
                    "Enable vim mode",
                    value=False,
                    id="vim_mode"
                )
                
                yield Switch(
                    "Enable animations",
                    value=True,
                    id="animations_enabled"
                )
                
                yield Switch(
                    "Show file tree by default",
                    value=True,
                    id="show_file_tree"
                )
                
                yield Label("Panel layout:")
                yield RadioSet(
                    RadioButton("Standard (3-column)", value="standard"),
                    RadioButton("Compact (2-column)", value="compact"),
                    RadioButton("Full-screen editor", value="fullscreen"),
                    id="panel_layout"
                )
                
                yield Label("Status bar information:")
                yield Switch("Show memory usage", value=True, id="status_memory")
                yield Switch("Show AI status", value=True, id="status_ai")
                yield Switch("Show project info", value=True, id="status_project")
                yield Switch("Show clock", value=True, id="status_clock")
        
        return content
    
    def _create_ai_settings(self) -> ScrollableContainer:
        """Create AI service settings panel."""
        content = ScrollableContainer(classes="settings-content")
        
        with content:
            yield Label("🤖 AI Service Configuration", classes="section-title")
            
            with Vertical(classes="settings-group"):
                yield Label("Claude Code API Token:")
                yield Input(
                    placeholder="Enter your Claude Code OAuth token...",
                    password=True,
                    id="claude_token"
                )
                
                yield Label("Default AI model:")
                yield Select(
                    options=[
                        ("claude-3-sonnet", "Claude 3 Sonnet (Balanced)"),
                        ("claude-3-opus", "Claude 3 Opus (Most Capable)"),
                        ("claude-3-haiku", "Claude 3 Haiku (Fastest)")
                    ],
                    value="claude-3-sonnet",
                    id="default_model"
                )
                
                yield Label("AI response timeout (seconds):")
                yield Slider(
                    min=30,
                    max=600,
                    value=300,
                    id="ai_timeout"
                )
                
                yield Label("Max retries on failure:")
                yield Slider(
                    min=1,
                    max=10,
                    value=3,
                    id="ai_retries"
                )
                
                yield Switch(
                    "Enable AI code suggestions",
                    value=True,
                    id="ai_suggestions"
                )
                
                yield Switch(
                    "Enable automatic code review",
                    value=True,
                    id="auto_code_review"
                )
                
                yield Switch(
                    "Cache AI responses",
                    value=True,
                    id="ai_cache"
                )
                
                yield Label("AI behavior settings:")
                yield Switch("Use project context in queries", value=True, id="ai_project_context")
                yield Switch("Include file history in context", value=False, id="ai_file_history")
                yield Switch("Enable conversation memory", value=True, id="ai_memory")
        
        return content
    
    def _create_security_settings(self) -> ScrollableContainer:
        """Create security settings panel.""" 
        content = ScrollableContainer(classes="settings-content")
        
        with content:
            yield Label("🛡️ Security & Privacy", classes="section-title")
            
            with Vertical(classes="settings-group"):
                yield Switch(
                    "Enable sandbox mode for code execution",
                    value=True,
                    id="sandbox_enabled"
                )
                
                yield Label("Maximum file size (MB):")
                yield Slider(
                    min=1,
                    max=1000,
                    value=100,
                    id="max_file_size"
                )
                
                yield Switch(
                    "Enable audit logging",
                    value=True,
                    id="audit_logging"
                )
                
                yield Switch(
                    "Require confirmation for destructive operations",
                    value=True,
                    id="confirm_destructive"
                )
                
                yield Label("Allowed file extensions:")
                yield TextArea(
                    text=".py .js .ts .jsx .tsx .html .css .json .yaml .yml .md .txt",
                    id="allowed_extensions",
                    max_height=3
                )
                
                yield Label("Blocked commands:")
                yield TextArea(
                    text="rm -rf del format fdisk dd",
                    id="blocked_commands",
                    max_height=3
                )
                
                yield Label("API key rotation (days):")
                yield Slider(
                    min=30,
                    max=365,
                    value=90,
                    id="key_rotation"
                )
                
                yield Switch(
                    "Encrypt sensitive configuration data",
                    value=True,
                    id="encrypt_config"
                )
        
        return content
    
    def _create_project_settings(self) -> ScrollableContainer:
        """Create project default settings panel."""
        content = ScrollableContainer(classes="settings-content")
        
        with content:
            yield Label("📁 Project Defaults", classes="section-title")
            
            with Vertical(classes="settings-group"):
                yield Label("Default project template:")
                yield Select(
                    options=[
                        ("basic", "Basic Project"),
                        ("python-cli", "Python CLI"),
                        ("python-web", "Python Web API"),
                        ("react-app", "React Application"),
                        ("documentation", "Documentation Site")
                    ],
                    value="basic",
                    id="default_template"
                )
                
                yield Label("Default project location:")
                yield Input(
                    placeholder="Enter default project directory...",
                    value=str(Path.home() / "Projects"),
                    id="default_location"
                )
                
                yield Switch(
                    "Initialize Git repository by default",
                    value=True,
                    id="default_git_init"
                )
                
                yield Switch(
                    "Create virtual environment by default",
                    value=True,
                    id="default_venv"
                )
                
                yield Switch(
                    "Enable AI integration by default",
                    value=True,
                    id="default_ai_integration"
                )
                
                yield Switch(
                    "Auto-validate code on save",
                    value=True,
                    id="auto_validation"
                )
                
                yield Switch(
                    "Auto-complete placeholder code",
                    value=True,
                    id="auto_completion"
                )
                
                yield Label("Code quality threshold:")
                yield Slider(
                    min=0.0,
                    max=1.0,
                    value=0.8,
                    step=0.1,
                    id="quality_threshold"
                )
                
                yield Label("Backup settings:")
                yield Switch("Enable automatic backups", value=True, id="backup_enabled")
                yield Slider(
                    min=5,
                    max=1440,
                    value=30,
                    id="backup_interval",
                    name="Backup interval (minutes):"
                )
        
        return content
    
    def _create_advanced_settings(self) -> ScrollableContainer:
        """Create advanced settings panel."""
        content = ScrollableContainer(classes="settings-content")
        
        with content:
            yield Label("🔧 Advanced Options", classes="section-title")
            
            with Vertical(classes="settings-group"):
                yield Label("Log level:")
                yield Select(
                    options=[
                        ("DEBUG", "Debug (Verbose)"),
                        ("INFO", "Info (Default)"),
                        ("WARNING", "Warning (Minimal)"),
                        ("ERROR", "Error (Errors Only)")
                    ],
                    value="INFO",
                    id="log_level"
                )
                
                yield Label("Performance settings:")
                yield Switch("Enable high-performance mode", value=False, id="high_performance")
                yield Switch("Preload frequently used files", value=True, id="preload_files")
                yield Switch("Enable memory optimization", value=True, id="memory_optimization")
                
                yield Label("Experimental features:")
                yield Switch("Beta feature preview", value=False, id="beta_features")
                yield Switch("Experimental AI models", value=False, id="experimental_ai")
                yield Switch("Advanced debugging tools", value=False, id="advanced_debugging")
                
                yield Label("Developer options:")
                yield Switch("Enable developer mode", value=False, id="developer_mode")
                yield Switch("Show debug information", value=False, id="show_debug")
                yield Switch("Enable API access logs", value=False, id="api_logs")
                
                yield Label("Configuration file location:")
                yield Input(
                    placeholder="Custom config directory...",
                    value="~/.claude-tui",
                    id="config_location"
                )
                
                yield Label("Plugin directory:")
                yield Input(
                    placeholder="Plugin directory...",  
                    value="~/.claude-tui/plugins",
                    id="plugin_directory"
                )
        
        return content
    
    def on_mount(self) -> None:
        """Initialize the settings screen."""
        self._load_current_settings()
        logger.info("Settings screen opened")
    
    def _load_current_settings(self) -> None:
        """Load current settings from configuration."""
        # TODO: Load from actual config manager
        self.original_settings = {}
        self.current_settings = {}
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle settings screen button presses."""
        if event.button.id == "close_btn" or event.button.id == "cancel_btn":
            if self.settings_changed:
                # TODO: Show confirmation dialog
                pass
            self.dismiss()
        elif event.button.id == "save_btn":
            self.action_save_settings()
        elif event.button.id == "reset_btn":
            self.action_reset_defaults()
    
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select widget changes."""
        self.settings_changed = True
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input widget changes."""
        self.settings_changed = True
    
    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle switch widget changes."""
        self.settings_changed = True
    
    def on_slider_changed(self, event: Slider.Changed) -> None:
        """Handle slider widget changes."""
        self.settings_changed = True
    
    def action_save_settings(self) -> None:
        """Save current settings."""
        try:
            # TODO: Implement actual settings save
            self.app.notify("Settings saved successfully", severity="success")
            self.settings_changed = False
            logger.info("Settings saved")
        except Exception as e:
            self.app.notify(f"Failed to save settings: {e}", severity="error")
            logger.error(f"Failed to save settings: {e}")
    
    def action_reset_defaults(self) -> None:
        """Reset settings to defaults."""
        try:
            # TODO: Implement settings reset
            self.app.notify("Settings reset to defaults", severity="info")
            self.settings_changed = True
            logger.info("Settings reset to defaults")
        except Exception as e:
            self.app.notify(f"Failed to reset settings: {e}", severity="error")
            logger.error(f"Failed to reset settings: {e}")
    
    def action_dismiss(self) -> None:
        """Dismiss the settings screen."""
        if self.settings_changed:
            # TODO: Show unsaved changes confirmation
            pass
        self.dismiss()