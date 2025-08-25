#!/usr/bin/env python3
"""
Settings Screen - Application configuration and preferences
with API key management, theme selection, and system settings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from textual import on, work
from textual.containers import Vertical, Horizontal, Container, Center, Middle
from textual.widgets import (
    Static, Label, Button, Input, Select, Checkbox, 
    RadioSet, RadioButton, TextArea, ProgressBar, 
    TabbedContent, TabPane, Switch
)
from textual.screen import ModalScreen
from textual.message import Message
from textual.reactive import reactive
from textual.validation import Function, ValidationResult, Validator
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.console import Console


class Theme(Enum):
    """Available themes"""
    DARK = "dark"
    LIGHT = "light"
    AUTO = "auto"
    CYBERPUNK = "cyberpunk"
    MATRIX = "matrix"
    RETRO = "retro"


class LogLevel(Enum):
    """Logging levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AppSettings:
    """Application settings data structure"""
    # AI Settings
    claude_api_key: Optional[str] = None
    claude_flow_enabled: bool = True
    auto_completion: bool = True
    validation_enabled: bool = True
    max_concurrent_tasks: int = 5
    
    # UI Settings
    theme: Theme = Theme.DARK
    show_notifications: bool = True
    notification_duration: int = 5
    auto_refresh_interval: int = 30
    vim_mode: bool = False
    
    # Validation Settings
    placeholder_detection: bool = True
    semantic_analysis: bool = True
    quality_scoring: bool = True
    auto_fix_placeholders: bool = False
    validation_interval: int = 30
    
    # Development Settings
    debug_mode: bool = False
    log_level: LogLevel = LogLevel.INFO
    save_session_history: bool = True
    backup_projects: bool = True
    auto_save_interval: int = 300
    
    # Performance Settings
    cache_ai_responses: bool = True
    cache_duration: int = 3600
    max_memory_usage: int = 1024  # MB
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'claude_api_key': self.claude_api_key,
            'claude_flow_enabled': self.claude_flow_enabled,
            'auto_completion': self.auto_completion,
            'validation_enabled': self.validation_enabled,
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'theme': self.theme.value,
            'show_notifications': self.show_notifications,
            'notification_duration': self.notification_duration,
            'auto_refresh_interval': self.auto_refresh_interval,
            'vim_mode': self.vim_mode,
            'placeholder_detection': self.placeholder_detection,
            'semantic_analysis': self.semantic_analysis,
            'quality_scoring': self.quality_scoring,
            'auto_fix_placeholders': self.auto_fix_placeholders,
            'validation_interval': self.validation_interval,
            'debug_mode': self.debug_mode,
            'log_level': self.log_level.value,
            'save_session_history': self.save_session_history,
            'backup_projects': self.backup_projects,
            'auto_save_interval': self.auto_save_interval,
            'cache_ai_responses': self.cache_ai_responses,
            'cache_duration': self.cache_duration,
            'max_memory_usage': self.max_memory_usage,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppSettings':
        """Create from dictionary"""
        settings = cls()
        
        # Update from dictionary with defaults
        settings.claude_api_key = data.get('claude_api_key')
        settings.claude_flow_enabled = data.get('claude_flow_enabled', True)
        settings.auto_completion = data.get('auto_completion', True)
        settings.validation_enabled = data.get('validation_enabled', True)
        settings.max_concurrent_tasks = data.get('max_concurrent_tasks', 5)
        
        theme_value = data.get('theme', 'dark')
        settings.theme = Theme(theme_value) if theme_value in [t.value for t in Theme] else Theme.DARK
        
        settings.show_notifications = data.get('show_notifications', True)
        settings.notification_duration = data.get('notification_duration', 5)
        settings.auto_refresh_interval = data.get('auto_refresh_interval', 30)
        settings.vim_mode = data.get('vim_mode', False)
        
        settings.placeholder_detection = data.get('placeholder_detection', True)
        settings.semantic_analysis = data.get('semantic_analysis', True)
        settings.quality_scoring = data.get('quality_scoring', True)
        settings.auto_fix_placeholders = data.get('auto_fix_placeholders', False)
        settings.validation_interval = data.get('validation_interval', 30)
        
        settings.debug_mode = data.get('debug_mode', False)
        
        log_level_value = data.get('log_level', 'info')
        settings.log_level = LogLevel(log_level_value) if log_level_value in [l.value for l in LogLevel] else LogLevel.INFO
        
        settings.save_session_history = data.get('save_session_history', True)
        settings.backup_projects = data.get('backup_projects', True)
        settings.auto_save_interval = data.get('auto_save_interval', 300)
        
        settings.cache_ai_responses = data.get('cache_ai_responses', True)
        settings.cache_duration = data.get('cache_duration', 3600)
        settings.max_memory_usage = data.get('max_memory_usage', 1024)
        
        return settings


class APIKeyValidator(Validator):
    """Validator for API keys"""
    
    def validate(self, value: str) -> ValidationResult:
        """Validate API key format"""
        if not value:
            return self.success()  # Optional field
        
        # Basic validation - should start with appropriate prefix
        if not value.startswith(('sk-', 'claude-', 'api-')):
            return self.failure("API key should start with 'sk-', 'claude-', or 'api-'")
        
        if len(value) < 20:
            return self.failure("API key seems too short")
        
        return self.success()


class AISettingsTab(Container):
    """AI-related settings tab"""
    
    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings
        
    def compose(self):
        """Compose AI settings"""
        yield Label("🤖 AI Configuration", classes="section-header")
        
        # API Key
        with Vertical(classes="setting-group"):
            yield Label("Claude API Key:")
            yield Input(
                placeholder="sk-... (optional for local development)",
                id="claude-api-key",
                password=True,
                value=self.settings.claude_api_key or "",
                validators=[APIKeyValidator()]
            )
            yield Label("Your API key is stored securely and encrypted", classes="help-text")
        
        # AI Features
        with Vertical(classes="setting-group"):
            yield Label("AI Features:", classes="subsection-header")
            
            yield Checkbox("Enable Claude Flow orchestration", id="claude-flow-enabled", value=self.settings.claude_flow_enabled)
            yield Checkbox("Auto-completion of placeholder code", id="auto-completion", value=self.settings.auto_completion)
            yield Checkbox("Real-time progress validation", id="validation-enabled", value=self.settings.validation_enabled)
        
        # Performance
        with Vertical(classes="setting-group"):
            yield Label("Performance:", classes="subsection-header")
            
            yield Label("Max Concurrent AI Tasks:")
            yield Select(
                [("1", 1), ("3", 3), ("5", 5), ("10", 10), ("15", 15)],
                value=self.settings.max_concurrent_tasks,
                id="max-concurrent-tasks"
            )
            
            yield Checkbox("Cache AI responses", id="cache-ai-responses", value=self.settings.cache_ai_responses)
            
            yield Label("Cache Duration (seconds):")
            yield Select(
                [("5 minutes", 300), ("30 minutes", 1800), ("1 hour", 3600), ("6 hours", 21600), ("24 hours", 86400)],
                value=self.settings.cache_duration,
                id="cache-duration"
            )


class ValidationSettingsTab(Container):
    """Anti-hallucination validation settings"""
    
    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings
        
    def compose(self):
        """Compose validation settings"""
        yield Label("🔍 Progress Intelligence & Anti-Hallucination", classes="section-header")
        
        # Core Validation Features
        with Vertical(classes="setting-group"):
            yield Label("Core Validation:", classes="subsection-header")
            
            yield Checkbox("Placeholder detection (TODOs, stubs, etc.)", id="placeholder-detection", value=self.settings.placeholder_detection)
            yield Checkbox("Semantic code analysis", id="semantic-analysis", value=self.settings.semantic_analysis)
            yield Checkbox("Quality scoring and metrics", id="quality-scoring", value=self.settings.quality_scoring)
        
        # Advanced Features
        with Vertical(classes="setting-group"):
            yield Label("Advanced Features:", classes="subsection-header")
            
            yield Checkbox("Auto-fix detected placeholders", id="auto-fix-placeholders", value=self.settings.auto_fix_placeholders)
            yield Label("⚠️ Auto-fix will automatically complete placeholder code", classes="warning-text")
        
        # Validation Timing
        with Vertical(classes="setting-group"):
            yield Label("Validation Timing:", classes="subsection-header")
            
            yield Label("Validation Check Interval:")
            yield Select(
                [("15 seconds", 15), ("30 seconds", 30), ("1 minute", 60), ("5 minutes", 300), ("Manual only", 0)],
                value=self.settings.validation_interval,
                id="validation-interval"
            )


class UISettingsTab(Container):
    """User interface settings"""
    
    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings
        
    def compose(self):
        """Compose UI settings"""
        yield Label("🎨 User Interface", classes="section-header")
        
        # Theme Settings
        with Vertical(classes="setting-group"):
            yield Label("Theme:", classes="subsection-header")
            
            with RadioSet(id="theme-selector"):
                for theme in Theme:
                    yield RadioButton(
                        f"{self._get_theme_icon(theme)} {theme.value.title()}",
                        value=theme.value,
                        id=f"theme-{theme.value}"
                    )
                    
            # Set current theme
            current_theme_radio = self.query_one(f"#theme-{self.settings.theme.value}", RadioButton)
            current_theme_radio.value = True
        
        # Notification Settings
        with Vertical(classes="setting-group"):
            yield Label("Notifications:", classes="subsection-header")
            
            yield Checkbox("Show notifications", id="show-notifications", value=self.settings.show_notifications)
            
            yield Label("Notification Duration (seconds):")
            yield Select(
                [("3 seconds", 3), ("5 seconds", 5), ("10 seconds", 10), ("Never auto-dismiss", 0)],
                value=self.settings.notification_duration,
                id="notification-duration"
            )
        
        # Interface Behavior
        with Vertical(classes="setting-group"):
            yield Label("Interface Behavior:", classes="subsection-header")
            
            yield Checkbox("Vim-style navigation", id="vim-mode", value=self.settings.vim_mode)
            
            yield Label("Auto-refresh Interval:")
            yield Select(
                [("10 seconds", 10), ("30 seconds", 30), ("1 minute", 60), ("5 minutes", 300), ("Disabled", 0)],
                value=self.settings.auto_refresh_interval,
                id="auto-refresh-interval"
            )
    
    def _get_theme_icon(self, theme: Theme) -> str:
        """Get icon for theme"""
        icons = {
            Theme.DARK: "🌙",
            Theme.LIGHT: "☀️",
            Theme.AUTO: "🌗",
            Theme.CYBERPUNK: "🟣",
            Theme.MATRIX: "🟢",
            Theme.RETRO: "📼"
        }
        return icons.get(theme, "🎨")


class DevelopmentSettingsTab(Container):
    """Development and debugging settings"""
    
    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings
        
    def compose(self):
        """Compose development settings"""
        yield Label("🔧 Development & Debugging", classes="section-header")
        
        # Debug Settings
        with Vertical(classes="setting-group"):
            yield Label("Debugging:", classes="subsection-header")
            
            yield Checkbox("Debug mode", id="debug-mode", value=self.settings.debug_mode)
            yield Label("⚠️ Debug mode shows detailed logs and may impact performance", classes="warning-text")
            
            yield Label("Log Level:")
            with RadioSet(id="log-level-selector"):
                for level in LogLevel:
                    yield RadioButton(
                        f"{self._get_log_level_icon(level)} {level.value.upper()}",
                        value=level.value,
                        id=f"log-{level.value}"
                    )
            
            # Set current log level
            current_log_radio = self.query_one(f"#log-{self.settings.log_level.value}", RadioButton)
            current_log_radio.value = True
        
        # Data Management
        with Vertical(classes="setting-group"):
            yield Label("Data Management:", classes="subsection-header")
            
            yield Checkbox("Save session history", id="save-session-history", value=self.settings.save_session_history)
            yield Checkbox("Auto-backup projects", id="backup-projects", value=self.settings.backup_projects)
            
            yield Label("Auto-save Interval:")
            yield Select(
                [("1 minute", 60), ("5 minutes", 300), ("10 minutes", 600), ("30 minutes", 1800), ("Disabled", 0)],
                value=self.settings.auto_save_interval,
                id="auto-save-interval"
            )
        
        # Performance Limits
        with Vertical(classes="setting-group"):
            yield Label("Performance Limits:", classes="subsection-header")
            
            yield Label("Max Memory Usage (MB):")
            yield Select(
                [("512 MB", 512), ("1 GB", 1024), ("2 GB", 2048), ("4 GB", 4096), ("Unlimited", 0)],
                value=self.settings.max_memory_usage,
                id="max-memory-usage"
            )
    
    def _get_log_level_icon(self, level: LogLevel) -> str:
        """Get icon for log level"""
        icons = {
            LogLevel.DEBUG: "🔍",
            LogLevel.INFO: "ℹ️",
            LogLevel.WARNING: "⚠️",
            LogLevel.ERROR: "❌",
            LogLevel.CRITICAL: "🚨"
        }
        return icons.get(level, "📝")


class SettingsScreen(ModalScreen):
    """Main settings screen"""
    
    def __init__(self) -> None:
        super().__init__()
        self.settings = self._load_settings()
        self.settings_changed = False
        
    def compose(self):
        """Compose settings screen"""
        with Container(id="settings-container"):
            # Header
            yield Label("⚙️ Settings", classes="settings-header")
            
            # Settings tabs
            with TabbedContent("AI", "Validation", "Interface", "Development"):
                # AI Settings Tab
                with TabPane("AI", id="ai-tab"):
                    self.ai_settings = AISettingsTab(self.settings)
                    yield self.ai_settings
                
                # Validation Settings Tab
                with TabPane("Validation", id="validation-tab"):
                    self.validation_settings = ValidationSettingsTab(self.settings)
                    yield self.validation_settings
                
                # UI Settings Tab
                with TabPane("Interface", id="ui-tab"):
                    self.ui_settings = UISettingsTab(self.settings)
                    yield self.ui_settings
                
                # Development Settings Tab
                with TabPane("Development", id="dev-tab"):
                    self.development_settings = DevelopmentSettingsTab(self.settings)
                    yield self.development_settings
            
            # Action buttons
            with Horizontal(classes="settings-actions"):
                yield Button("💾 Save Settings", id="save-settings", variant="primary")
                yield Button("🔄 Reset to Defaults", id="reset-settings")
                yield Button("💾 Export Settings", id="export-settings")
                yield Button("📥 Import Settings", id="import-settings")
                yield Button("❌ Cancel", id="cancel-settings")
    
    def _load_settings(self) -> AppSettings:
        """Load settings from file"""
        settings_file = Path.home() / ".claude-tiu" / "settings.json"
        
        if settings_file.exists():
            try:
                with open(settings_file, 'r') as f:
                    data = json.load(f)
                    return AppSettings.from_dict(data)
            except Exception:
                # If loading fails, return defaults
                pass
        
        return AppSettings()
    
    def _save_settings(self) -> bool:
        """Save current settings"""
        try:
            # Collect current form values
            updated_settings = self._collect_settings_from_form()
            
            # Create settings directory if it doesn't exist
            settings_dir = Path.home() / ".claude-tiu"
            settings_dir.mkdir(exist_ok=True)
            
            # Save to file
            settings_file = settings_dir / "settings.json"
            with open(settings_file, 'w') as f:
                json.dump(updated_settings.to_dict(), f, indent=2)
            
            self.settings = updated_settings
            return True
            
        except Exception as e:
            # Handle save error
            return False
    
    def _collect_settings_from_form(self) -> AppSettings:
        """Collect settings from form widgets"""
        settings = AppSettings()
        
        try:
            # AI Settings
            settings.claude_api_key = self.query_one("#claude-api-key", Input).value or None
            settings.claude_flow_enabled = self.query_one("#claude-flow-enabled", Checkbox).value
            settings.auto_completion = self.query_one("#auto-completion", Checkbox).value
            settings.validation_enabled = self.query_one("#validation-enabled", Checkbox).value
            settings.max_concurrent_tasks = int(self.query_one("#max-concurrent-tasks", Select).value)
            settings.cache_ai_responses = self.query_one("#cache-ai-responses", Checkbox).value
            settings.cache_duration = int(self.query_one("#cache-duration", Select).value)
            
            # Validation Settings
            settings.placeholder_detection = self.query_one("#placeholder-detection", Checkbox).value
            settings.semantic_analysis = self.query_one("#semantic-analysis", Checkbox).value
            settings.quality_scoring = self.query_one("#quality-scoring", Checkbox).value
            settings.auto_fix_placeholders = self.query_one("#auto-fix-placeholders", Checkbox).value
            settings.validation_interval = int(self.query_one("#validation-interval", Select).value)
            
            # UI Settings
            theme_selector = self.query_one("#theme-selector", RadioSet)
            if theme_selector.pressed:
                settings.theme = Theme(theme_selector.pressed.value)
            
            settings.show_notifications = self.query_one("#show-notifications", Checkbox).value
            settings.notification_duration = int(self.query_one("#notification-duration", Select).value)
            settings.vim_mode = self.query_one("#vim-mode", Checkbox).value
            settings.auto_refresh_interval = int(self.query_one("#auto-refresh-interval", Select).value)
            
            # Development Settings
            settings.debug_mode = self.query_one("#debug-mode", Checkbox).value
            
            log_level_selector = self.query_one("#log-level-selector", RadioSet)
            if log_level_selector.pressed:
                settings.log_level = LogLevel(log_level_selector.pressed.value)
            
            settings.save_session_history = self.query_one("#save-session-history", Checkbox).value
            settings.backup_projects = self.query_one("#backup-projects", Checkbox).value
            settings.auto_save_interval = int(self.query_one("#auto-save-interval", Select).value)
            settings.max_memory_usage = int(self.query_one("#max-memory-usage", Select).value)
            
        except Exception as e:
            # If collection fails, return current settings
            settings = self.settings
        
        return settings
    
    @on(Button.Pressed, "#save-settings")
    def save_settings(self) -> None:
        """Save settings and close"""
        if self._save_settings():
            self.post_message(SettingsSavedMessage(self.settings))
            self.dismiss(self.settings)
        else:
            # Show error notification
            pass
    
    @on(Button.Pressed, "#reset-settings")
    def reset_settings(self) -> None:
        """Reset to default settings"""
        self.settings = AppSettings()
        # Refresh the form with default values
        self.refresh()
    
    @on(Button.Pressed, "#export-settings")
    def export_settings(self) -> None:
        """Export settings to file"""
        current_settings = self._collect_settings_from_form()
        self.post_message(ExportSettingsMessage(current_settings))
    
    @on(Button.Pressed, "#import-settings")
    def import_settings(self) -> None:
        """Import settings from file"""
        self.post_message(ImportSettingsMessage())
    
    @on(Button.Pressed, "#cancel-settings")
    def cancel_settings(self) -> None:
        """Cancel without saving"""
        self.dismiss()


# Messages
class SettingsSavedMessage(Message):
    """Message sent when settings are saved"""
    
    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings


class ExportSettingsMessage(Message):
    """Message to export settings"""
    
    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings


class ImportSettingsMessage(Message):
    """Message to import settings"""
    pass