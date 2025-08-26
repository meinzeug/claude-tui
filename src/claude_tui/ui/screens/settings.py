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
    Static, Button, Input, Select, Switch,  # Slider removed - not in Textual 5.3.0
    RadioSet, RadioButton, Label, TextArea, Tabs, TabbedContent, TabPane
)
from textual.screen import ModalScreen
from textual.binding import Binding
from textual.message import Message

# Import ConfigManager for settings persistence
from ...core.config_manager import ConfigManager

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
    
    def __init__(self, config_manager: Optional[ConfigManager] = None, **kwargs):
        super().__init__(**kwargs)
        self.config_manager = config_manager or ConfigManager()
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
                yield Input(  # Using Input instead of Slider (not in Textual 5.3.0)
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
                yield Input(  # Using Input instead of Slider (not in Textual 5.3.0)
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
                yield Input(  # Using Input instead of Slider (not in Textual 5.3.0)
                    min=30,
                    max=600,
                    value=300,
                    id="ai_timeout"
                )
                
                yield Label("Max retries on failure:")
                yield Input(  # Using Input instead of Slider (not in Textual 5.3.0)
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
                yield Input(  # Using Input instead of Slider (not in Textual 5.3.0)
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
                yield Input(  # Using Input instead of Slider (not in Textual 5.3.0)
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
                yield Input(  # Using Input instead of Slider (not in Textual 5.3.0)
                    min=0.0,
                    max=1.0,
                    value=0.8,
                    step=0.1,
                    id="quality_threshold"
                )
                
                yield Label("Backup settings:")
                yield Switch("Enable automatic backups", value=True, id="backup_enabled")
                yield Input(  # Using Input instead of Slider (not in Textual 5.3.0)
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
        try:
            if not self.config_manager.config:
                # Initialize config if not loaded
                import asyncio
                loop = asyncio.get_event_loop()
                loop.create_task(self.config_manager.initialize())
                return
            
            config = self.config_manager.config
            
            # Extract all settings from config
            self.current_settings = {
                # General settings
                'theme': config.ui_preferences.theme,
                'language': 'en',  # Default language
                'autosave_interval': 5,  # Default autosave interval
                'notifications_enabled': True,
                'auto_updates': True,
                'telemetry_enabled': False,
                
                # Interface settings
                'font_size': config.ui_preferences.font_size,
                'show_line_numbers': config.ui_preferences.show_line_numbers,
                'vim_mode': config.ui_preferences.vim_mode,
                'animations_enabled': config.ui_preferences.animations_enabled,
                'show_file_tree': True,
                'panel_layout': 'standard',
                'status_memory': True,
                'status_ai': True,
                'status_project': True,
                'status_clock': True,
                
                # AI service settings
                'claude_token': '',  # Don't expose token in UI
                'default_model': 'claude-3-sonnet',
                'ai_timeout': 300,
                'ai_retries': 3,
                'ai_suggestions': True,
                'auto_code_review': True,
                'ai_cache': True,
                'ai_project_context': True,
                'ai_file_history': False,
                'ai_memory': True,
                
                # Security settings
                'sandbox_enabled': config.security.sandbox_enabled,
                'max_file_size': config.security.max_file_size_mb,
                'audit_logging': config.security.audit_logging,
                'confirm_destructive': True,
                'allowed_extensions': ' '.join(config.security.allowed_file_extensions),
                'blocked_commands': ' '.join(config.security.blocked_commands),
                'key_rotation': config.security.api_key_rotation_days,
                'encrypt_config': True,
                
                # Project settings
                'default_template': config.project_defaults.default_template,
                'default_location': str(Path.home() / "Projects"),
                'default_git_init': True,
                'default_venv': True,
                'default_ai_integration': True,
                'auto_validation': config.project_defaults.auto_validation,
                'auto_completion': config.project_defaults.auto_completion,
                'quality_threshold': config.project_defaults.code_quality_threshold,
                'backup_enabled': config.project_defaults.backup_enabled,
                'backup_interval': config.project_defaults.backup_interval_minutes,
                
                # Advanced settings
                'log_level': config.ui_preferences.log_level,
                'high_performance': False,
                'preload_files': True,
                'memory_optimization': True,
                'beta_features': False,
                'experimental_ai': False,
                'advanced_debugging': False,
                'developer_mode': False,
                'show_debug': False,
                'api_logs': False,
                'config_location': str(self.config_manager.get_config_dir()),
                'plugin_directory': str(self.config_manager.get_config_dir() / "plugins")
            }
            
            # Store original settings for change detection
            self.original_settings = self.current_settings.copy()
            
            # Apply settings to UI components
            self.call_after_refresh(self._apply_settings_to_ui, self.current_settings)
            
            logger.info("Settings loaded successfully from ConfigManager")
            
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            # Use defaults if loading fails
            self.original_settings = {}
            self.current_settings = {}
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle settings screen button presses."""
        if event.button.id == "close_btn" or event.button.id == "cancel_btn":
            if self.settings_changed:
                self._show_unsaved_changes_dialog()
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
    
    # def on_slider_changed(self, event: Slider.Changed) -> None:  # Slider not available in Textual 5.3.0
        """Handle slider widget changes."""
        self.settings_changed = True
    
    def action_save_settings(self) -> None:
        """Save current settings."""
        try:
            # Collect current settings from UI
            settings = self._collect_current_settings()
            
            # Save settings using ConfigManager
            import asyncio
            loop = asyncio.get_event_loop()
            
            async def save_async():
                try:
                    # Update ConfigManager settings
                    await self._save_settings_to_config(settings)
                    await self.config_manager.save_config()
                    
                    # Update tracking variables
                    self.original_settings = settings.copy()
                    self.settings_changed = False
                    
                    self.app.notify("Settings saved successfully", severity="success")
                    logger.info("Settings saved successfully")
                    
                except Exception as e:
                    self.app.notify(f"Failed to save settings: {e}", severity="error")
                    logger.error(f"Failed to save settings: {e}")
            
            # Run async save
            loop.create_task(save_async())
            
        except Exception as e:
            self.app.notify(f"Failed to save settings: {e}", severity="error")
            logger.error(f"Failed to save settings: {e}")
    
    def action_reset_defaults(self) -> None:
        """Reset settings to defaults."""
        try:
            # Reset to defaults using ConfigManager
            import asyncio
            loop = asyncio.get_event_loop()
            
            async def reset_async():
                try:
                    await self.config_manager.reset_to_defaults()
                    
                    # Reload settings from defaults
                    self._load_current_settings()
                    
                    # Apply to UI
                    self._apply_settings_to_ui(self.current_settings)
                    
                    self.settings_changed = True
                    self.app.notify("Settings reset to defaults", severity="info")
                    logger.info("Settings reset to defaults")
                    
                except Exception as e:
                    self.app.notify(f"Failed to reset settings: {e}", severity="error")
                    logger.error(f"Failed to reset settings: {e}")
            
            loop.create_task(reset_async())
            
        except Exception as e:
            self.app.notify(f"Failed to reset settings: {e}", severity="error")
            logger.error(f"Failed to reset settings: {e}")
    
    def action_dismiss(self) -> None:
        """Dismiss the settings screen."""
        if self.settings_changed:
            # Show confirmation for unsaved changes
            def confirm_dismiss(confirmed: bool) -> None:
                if confirmed:
                    self.dismiss()
            
            self.app.push_screen(
                'confirmation_dialog',
                message="You have unsaved changes. Close without saving?",
                callback=confirm_dismiss
            )
            return
        self.dismiss()
    
    def _collect_current_settings(self) -> Dict[str, Any]:
        """Collect current settings from UI components."""
        settings = {}
        try:
            # Try to get values from UI components
            ai_service_select = self.query_one("#ai_service_select", expect_type=Select)
            if ai_service_select.value != Select.BLANK:
                settings['ai_service'] = ai_service_select.value
            
            auto_validation_switch = self.query_one("#auto_validation_switch", expect_type=Switch)
            settings['auto_validation'] = auto_validation_switch.value
            
            log_level_select = self.query_one("#log_level_select", expect_type=Select)
            if log_level_select.value != Select.BLANK:
                settings['log_level'] = log_level_select.value
            
            theme_select = self.query_one("#theme_select", expect_type=Select)
            if theme_select.value != Select.BLANK:
                settings['theme'] = theme_select.value
            
            timeout_input = self.query_one("#timeout_input", expect_type=Input)
            if timeout_input.value:
                try:
                    settings['timeout'] = int(timeout_input.value)
                except ValueError:
                    settings['timeout'] = 300  # Default fallback
        
        except Exception as e:
            logger.warning(f"Could not collect all settings from UI: {e}")
            # Return current_settings as fallback
            return self.current_settings.copy()
        
        return settings
    
    def _apply_settings_to_ui(self, settings: Dict[str, Any]) -> None:
        """Apply settings to UI components."""
        try:
            if 'ai_service' in settings:
                ai_service_select = self.query_one("#ai_service_select", expect_type=Select)
                ai_service_select.value = settings['ai_service']
            
            if 'auto_validation' in settings:
                auto_validation_switch = self.query_one("#auto_validation_switch", expect_type=Switch)
                auto_validation_switch.value = settings['auto_validation']
            
            if 'log_level' in settings:
                log_level_select = self.query_one("#log_level_select", expect_type=Select)
                log_level_select.value = settings['log_level']
            
            if 'theme' in settings:
                theme_select = self.query_one("#theme_select", expect_type=Select)
                theme_select.value = settings['theme']
            
            if 'timeout' in settings:
                timeout_input = self.query_one("#timeout_input", expect_type=Input)
                timeout_input.value = str(settings['timeout'])
        
        except Exception as e:
            logger.warning(f"Could not apply all settings to UI: {e}")
    
    def _show_unsaved_changes_dialog(self) -> None:
        """Show confirmation dialog for unsaved changes."""
        try:
            from textual.widgets import Static
            from textual.containers import Center, Middle
            from textual.screen import ModalScreen
            
            class ConfirmationDialog(ModalScreen):
                """Confirmation dialog for unsaved changes."""
                
                def compose(self) -> ComposeResult:
                    with Center():
                        with Middle():
                            with Container(classes="dialog"):
                                yield Static("⚠️ Unsaved Changes", classes="dialog-title")
                                yield Static("You have unsaved changes. Close without saving?", classes="dialog-message")
                                with Horizontal(classes="dialog-buttons"):
                                    yield Button("Don't Save", id="dont_save", variant="default")
                                    yield Button("Cancel", id="cancel", variant="primary")
                
                def on_button_pressed(self, event: Button.Pressed) -> None:
                    if event.button.id == "dont_save":
                        self.dismiss(True)
                    else:
                        self.dismiss(False)
            
            def handle_confirmation(confirmed: Optional[bool]) -> None:
                if confirmed:
                    self.dismiss()
            
            self.app.push_screen(ConfirmationDialog(), handle_confirmation)
            
        except Exception as e:
            logger.error(f"Failed to show confirmation dialog: {e}")
            # Fall back to direct dismiss
            self.dismiss()
    
    async def _save_settings_to_config(self, settings: Dict[str, Any]) -> None:
        """Save settings to ConfigManager."""
        try:
            # Map UI settings to config structure
            if not self.config_manager.config:
                await self.config_manager.initialize()
            
            # Update UI preferences
            await self.config_manager.update_setting('ui_preferences.theme', settings.get('theme', 'dark'))
            await self.config_manager.update_setting('ui_preferences.font_size', int(settings.get('font_size', 12)))
            await self.config_manager.update_setting('ui_preferences.show_line_numbers', settings.get('show_line_numbers', True))
            await self.config_manager.update_setting('ui_preferences.vim_mode', settings.get('vim_mode', False))
            await self.config_manager.update_setting('ui_preferences.animations_enabled', settings.get('animations_enabled', True))
            await self.config_manager.update_setting('ui_preferences.log_level', settings.get('log_level', 'INFO'))
            
            # Update security settings
            await self.config_manager.update_setting('security.sandbox_enabled', settings.get('sandbox_enabled', True))
            await self.config_manager.update_setting('security.max_file_size_mb', int(settings.get('max_file_size', 100)))
            await self.config_manager.update_setting('security.audit_logging', settings.get('audit_logging', True))
            
            # Parse file extensions and commands
            if 'allowed_extensions' in settings:
                extensions = [ext.strip() for ext in settings['allowed_extensions'].split() if ext.strip()]
                await self.config_manager.update_setting('security.allowed_file_extensions', extensions)
            
            if 'blocked_commands' in settings:
                commands = [cmd.strip() for cmd in settings['blocked_commands'].split() if cmd.strip()]
                await self.config_manager.update_setting('security.blocked_commands', commands)
            
            await self.config_manager.update_setting('security.api_key_rotation_days', int(settings.get('key_rotation', 90)))
            
            # Update project defaults
            await self.config_manager.update_setting('project_defaults.default_template', settings.get('default_template', 'basic'))
            await self.config_manager.update_setting('project_defaults.auto_validation', settings.get('auto_validation', True))
            await self.config_manager.update_setting('project_defaults.auto_completion', settings.get('auto_completion', True))
            await self.config_manager.update_setting('project_defaults.code_quality_threshold', float(settings.get('quality_threshold', 0.8)))
            await self.config_manager.update_setting('project_defaults.backup_enabled', settings.get('backup_enabled', True))
            await self.config_manager.update_setting('project_defaults.backup_interval_minutes', int(settings.get('backup_interval', 30)))
            
            # Store API token securely if provided
            if 'claude_token' in settings and settings['claude_token']:
                await self.config_manager.store_api_key('claude_code', settings['claude_token'])
            
            logger.info("Settings successfully mapped to ConfigManager")
            
        except Exception as e:
            logger.error(f"Failed to save settings to config: {e}")
            raise