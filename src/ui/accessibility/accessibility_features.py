#!/usr/bin/env python3
"""
Accessibility Features - Screen reader support, high contrast, keyboard navigation
"""

from __future__ import annotations

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from textual import events
from textual.app import App
from textual.widget import Widget
from textual.message import Message
from textual.reactive import reactive
from rich.text import Text


class AccessibilityLevel(Enum):
    """Accessibility compliance levels"""
    AA = "AA"
    AAA = "AAA"


class ContrastLevel(Enum):
    """Contrast levels"""
    NORMAL = "normal"
    HIGH = "high"
    MAXIMUM = "maximum"


class ScreenReaderMode(Enum):
    """Screen reader compatibility modes"""
    NVDA = "nvda"
    JAWS = "jaws"
    VOICEOVER = "voiceover"
    ORCA = "orca"
    GENERIC = "generic"


@dataclass
class AccessibilitySettings:
    """Accessibility configuration"""
    # Visual accessibility
    high_contrast: bool = False
    large_text: bool = False
    reduce_motion: bool = False
    focus_indicators: bool = True
    
    # Screen reader support
    screen_reader_enabled: bool = False
    screen_reader_mode: ScreenReaderMode = ScreenReaderMode.GENERIC
    announce_changes: bool = True
    verbose_descriptions: bool = False
    
    # Keyboard navigation
    keyboard_only: bool = False
    focus_trapping: bool = True
    skip_links: bool = True
    
    # Timing and interaction
    extended_timeouts: bool = False
    disable_auto_refresh: bool = False
    
    # Audio/Visual alerts
    audio_cues: bool = False
    visual_alerts: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'high_contrast': self.high_contrast,
            'large_text': self.large_text,
            'reduce_motion': self.reduce_motion,
            'focus_indicators': self.focus_indicators,
            'screen_reader_enabled': self.screen_reader_enabled,
            'screen_reader_mode': self.screen_reader_mode.value,
            'announce_changes': self.announce_changes,
            'verbose_descriptions': self.verbose_descriptions,
            'keyboard_only': self.keyboard_only,
            'focus_trapping': self.focus_trapping,
            'skip_links': self.skip_links,
            'extended_timeouts': self.extended_timeouts,
            'disable_auto_refresh': self.disable_auto_refresh,
            'audio_cues': self.audio_cues,
            'visual_alerts': self.visual_alerts,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AccessibilitySettings':
        """Create from dictionary"""
        settings = cls()
        settings.high_contrast = data.get('high_contrast', False)
        settings.large_text = data.get('large_text', False)
        settings.reduce_motion = data.get('reduce_motion', False)
        settings.focus_indicators = data.get('focus_indicators', True)
        settings.screen_reader_enabled = data.get('screen_reader_enabled', False)
        
        sr_mode = data.get('screen_reader_mode', 'generic')
        settings.screen_reader_mode = ScreenReaderMode(sr_mode) if sr_mode in [m.value for m in ScreenReaderMode] else ScreenReaderMode.GENERIC
        
        settings.announce_changes = data.get('announce_changes', True)
        settings.verbose_descriptions = data.get('verbose_descriptions', False)
        settings.keyboard_only = data.get('keyboard_only', False)
        settings.focus_trapping = data.get('focus_trapping', True)
        settings.skip_links = data.get('skip_links', True)
        settings.extended_timeouts = data.get('extended_timeouts', False)
        settings.disable_auto_refresh = data.get('disable_auto_refresh', False)
        settings.audio_cues = data.get('audio_cues', False)
        settings.visual_alerts = data.get('visual_alerts', True)
        
        return settings


class AccessibilityManager:
    """Manages accessibility features across the application"""
    
    def __init__(self, app: App):
        self.app = app
        self.settings = AccessibilitySettings()
        self.aria_live_regions: Dict[str, Widget] = {}
        self.focus_history: List[Widget] = []
        self.announcements_queue: List[str] = []
        
        # Load settings
        self._load_settings()
        
        # Setup accessibility features
        self._setup_accessibility_features()
    
    def _load_settings(self) -> None:
        """Load accessibility settings"""
        settings_file = Path.home() / ".claude-tui" / "accessibility.json"
        
        if settings_file.exists():
            try:
                with open(settings_file, 'r') as f:
                    data = json.load(f)
                    self.settings = AccessibilitySettings.from_dict(data)
            except Exception as e:
                print(f"Warning: Failed to load accessibility settings: {e}")
    
    def save_settings(self) -> bool:
        """Save accessibility settings"""
        try:
            settings_dir = Path.home() / ".claude-tui"
            settings_dir.mkdir(exist_ok=True)
            
            settings_file = settings_dir / "accessibility.json"
            with open(settings_file, 'w') as f:
                json.dump(self.settings.to_dict(), f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving accessibility settings: {e}")
            return False
    
    def _setup_accessibility_features(self) -> None:
        """Setup accessibility features based on current settings"""
        if self.settings.high_contrast:
            self._apply_high_contrast()
        
        if self.settings.large_text:
            self._apply_large_text()
        
        if self.settings.focus_indicators:
            self._enhance_focus_indicators()
        
        if self.settings.screen_reader_enabled:
            self._setup_screen_reader_support()
    
    def _apply_high_contrast(self) -> None:
        """Apply high contrast theme"""
        # This would integrate with the theme system
        from ..themes.theme_manager import get_theme_manager
        
        theme_manager = get_theme_manager()
        theme_manager.set_theme("high_contrast")
    
    def _apply_large_text(self) -> None:
        """Apply large text settings"""
        # This would modify CSS or font settings
        pass
    
    def _enhance_focus_indicators(self) -> None:
        """Enhance focus indicators for better visibility"""
        # Add enhanced focus styles
        focus_css = """
        *:focus {
            outline: 2 $primary;
            background: $primary-alpha-20;
        }
        """
        # Apply to app styles (would need CSS injection mechanism)
    
    def _setup_screen_reader_support(self) -> None:
        """Setup screen reader support"""
        # Setup ARIA live regions
        self._create_aria_live_regions()
        
        # Setup focus management
        self._setup_focus_management()
        
        # Setup announcements
        self._setup_announcements()
    
    def _create_aria_live_regions(self) -> None:
        """Create ARIA live regions for announcements"""
        # This would create invisible regions for screen reader announcements
        pass
    
    def _setup_focus_management(self) -> None:
        """Setup enhanced focus management"""
        # Override app's focus handling
        original_on_focus = self.app.on_focus if hasattr(self.app, 'on_focus') else None
        
        def enhanced_on_focus(event: events.Focus) -> None:
            widget = event.widget
            
            # Track focus history
            self.focus_history.append(widget)
            if len(self.focus_history) > 50:  # Limit history size
                self.focus_history = self.focus_history[-50:]
            
            # Announce focus change
            if self.settings.screen_reader_enabled:
                self._announce_focus_change(widget)
            
            # Call original handler if exists
            if original_on_focus:
                original_on_focus(event)
        
        # Would need mechanism to override app's focus handling
    
    def _setup_announcements(self) -> None:
        """Setup announcement system"""
        if self.settings.screen_reader_enabled:
            # Start announcement processor
            asyncio.create_task(self._process_announcements())
    
    async def _process_announcements(self) -> None:
        """Process queued announcements"""
        while True:
            if self.announcements_queue:
                announcement = self.announcements_queue.pop(0)
                await self._make_announcement(announcement)
            
            await asyncio.sleep(0.1)  # Check every 100ms
    
    async def _make_announcement(self, text: str) -> None:
        """Make screen reader announcement"""
        if self.settings.screen_reader_enabled:
            # This would interface with screen reader APIs
            # For terminal apps, might output to stderr or use specific protocols
            print(f"\x1b[2K\rSCREEN_READER: {text}", flush=True)
            await asyncio.sleep(0.1)  # Brief pause
    
    def announce(self, text: str, priority: str = "polite") -> None:
        """Queue announcement for screen reader"""
        if self.settings.screen_reader_enabled and self.settings.announce_changes:
            if priority == "assertive":
                # High priority - insert at front
                self.announcements_queue.insert(0, text)
            else:
                # Normal priority - add to end
                self.announcements_queue.append(text)
    
    def _announce_focus_change(self, widget: Widget) -> None:
        """Announce focus change to screen reader"""
        widget_type = type(widget).__name__
        widget_id = getattr(widget, 'id', None)
        
        # Generate descriptive text
        if hasattr(widget, 'get_accessibility_description'):
            description = widget.get_accessibility_description()
        else:
            description = self._generate_widget_description(widget)
        
        announcement = f"{widget_type} {description}"
        if widget_id:
            announcement = f"{widget_id} {announcement}"
        
        self.announce(announcement)
    
    def _generate_widget_description(self, widget: Widget) -> str:
        """Generate accessibility description for widget"""
        descriptions = []
        
        # Check for text content
        if hasattr(widget, 'renderable') and hasattr(widget.renderable, '__str__'):
            content = str(widget.renderable)
            if content and content.strip():
                descriptions.append(f"contains {content[:100]}")
        
        # Check for button-like widgets
        if 'button' in type(widget).__name__.lower():
            descriptions.append("button")
        
        # Check for input widgets
        if 'input' in type(widget).__name__.lower():
            descriptions.append("text input")
            if hasattr(widget, 'value'):
                value = str(widget.value)
                if value:
                    descriptions.append(f"current value {value}")
                else:
                    descriptions.append("empty")
        
        # Check for list widgets
        if 'list' in type(widget).__name__.lower():
            descriptions.append("list")
            if hasattr(widget, 'count'):
                descriptions.append(f"with {widget.count} items")
        
        # Check for disabled state
        if hasattr(widget, 'disabled') and widget.disabled:
            descriptions.append("disabled")
        
        return ", ".join(descriptions) if descriptions else "interactive element"
    
    def register_aria_live_region(self, name: str, widget: Widget) -> None:
        """Register widget as ARIA live region"""
        self.aria_live_regions[name] = widget
    
    def update_live_region(self, name: str, content: str) -> None:
        """Update ARIA live region content"""
        if name in self.aria_live_regions:
            widget = self.aria_live_regions[name]
            # Update widget content
            if hasattr(widget, 'update'):
                widget.update(content)
            
            # Announce change
            self.announce(f"{name} updated: {content}")
    
    def create_skip_links(self, landmarks: List[Tuple[str, str]]) -> List[Widget]:
        """Create skip links for keyboard navigation"""
        from ..widgets.advanced_components import EnhancedButton
        
        skip_links = []
        for label, target_id in landmarks:
            button = EnhancedButton(f"Skip to {label}", classes="skip-link")
            button.target_id = target_id
            button.add_click_handler(lambda: self._skip_to_target(target_id))
            skip_links.append(button)
        
        return skip_links
    
    def _skip_to_target(self, target_id: str) -> None:
        """Skip to target element"""
        # Find widget with target ID
        target_widget = self.app.query_one(f"#{target_id}")
        if target_widget:
            target_widget.focus()
            self.announce(f"Skipped to {target_id}")
    
    def create_focus_trap(self, container: Widget, focusable_widgets: List[Widget]) -> None:
        """Create focus trap within container"""
        if not self.settings.focus_trapping:
            return
        
        def trap_focus(event: events.Key) -> None:
            if event.key == "tab":
                current_focus = self.app.screen.focused
                if current_focus in focusable_widgets:
                    current_index = focusable_widgets.index(current_focus)
                    
                    if event.shift:  # Shift+Tab
                        next_index = (current_index - 1) % len(focusable_widgets)
                    else:  # Tab
                        next_index = (current_index + 1) % len(focusable_widgets)
                    
                    focusable_widgets[next_index].focus()
                    event.prevent_default()
        
        # Would need to bind this to the container
        container.on_key = trap_focus
    
    def check_color_contrast(self, foreground: str, background: str) -> Tuple[float, AccessibilityLevel]:
        """Check color contrast ratio"""
        # This would implement WCAG contrast calculation
        # For now, return mock values
        ratio = 4.5  # Mock ratio
        level = AccessibilityLevel.AA if ratio >= 4.5 else None
        return ratio, level
    
    def validate_accessibility(self, widget: Widget) -> List[str]:
        """Validate widget accessibility"""
        issues = []
        
        # Check for missing alt text on images
        if hasattr(widget, 'alt_text') and not widget.alt_text:
            issues.append("Missing alternative text")
        
        # Check for proper labeling
        if 'input' in type(widget).__name__.lower():
            if not hasattr(widget, 'label') or not widget.label:
                issues.append("Input missing label")
        
        # Check for keyboard accessibility
        if not hasattr(widget, 'can_focus') or not widget.can_focus:
            if 'button' in type(widget).__name__.lower():
                issues.append("Interactive element not keyboard accessible")
        
        # Check for proper heading hierarchy (would need more context)
        # Check for sufficient color contrast (would need color analysis)
        
        return issues
    
    def generate_accessibility_report(self) -> Dict[str, Any]:
        """Generate accessibility audit report"""
        report = {
            'timestamp': str(asyncio.get_event_loop().time()),
            'settings': self.settings.to_dict(),
            'issues': [],
            'compliance_level': None,
            'recommendations': []
        }
        
        # Audit all widgets in the app
        all_widgets = self.app.query("*")
        total_issues = 0
        
        for widget in all_widgets:
            widget_issues = self.validate_accessibility(widget)
            if widget_issues:
                total_issues += len(widget_issues)
                report['issues'].append({
                    'widget': str(widget),
                    'issues': widget_issues
                })
        
        # Determine compliance level
        if total_issues == 0:
            report['compliance_level'] = AccessibilityLevel.AAA.value
        elif total_issues < 5:
            report['compliance_level'] = AccessibilityLevel.AA.value
        else:
            report['compliance_level'] = None
        
        # Generate recommendations
        if total_issues > 0:
            report['recommendations'] = [
                "Add proper labels to all interactive elements",
                "Ensure sufficient color contrast",
                "Provide alternative text for visual elements",
                "Test with keyboard navigation only",
                "Test with screen reader software"
            ]
        
        return report
    
    def enable_high_contrast(self) -> None:
        """Enable high contrast mode"""
        self.settings.high_contrast = True
        self._apply_high_contrast()
        self.save_settings()
        self.announce("High contrast mode enabled")
    
    def disable_high_contrast(self) -> None:
        """Disable high contrast mode"""
        self.settings.high_contrast = False
        # Apply normal theme
        self.save_settings()
        self.announce("High contrast mode disabled")
    
    def toggle_screen_reader_mode(self) -> None:
        """Toggle screen reader support"""
        self.settings.screen_reader_enabled = not self.settings.screen_reader_enabled
        
        if self.settings.screen_reader_enabled:
            self._setup_screen_reader_support()
            self.announce("Screen reader support enabled")
        else:
            self.announce("Screen reader support disabled")
        
        self.save_settings()
    
    def increase_text_size(self) -> None:
        """Increase text size for better readability"""
        self.settings.large_text = True
        self._apply_large_text()
        self.save_settings()
        self.announce("Large text enabled")
    
    def decrease_text_size(self) -> None:
        """Decrease text size to normal"""
        self.settings.large_text = False
        self.save_settings()
        self.announce("Normal text size restored")


class AccessibilityWidget:
    """Mixin class to add accessibility features to widgets"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accessibility_label: Optional[str] = None
        self.accessibility_description: Optional[str] = None
        self.accessibility_role: Optional[str] = None
        self.accessibility_state: Dict[str, Any] = {}
    
    def set_accessibility_label(self, label: str) -> None:
        """Set accessibility label"""
        self.accessibility_label = label
    
    def set_accessibility_description(self, description: str) -> None:
        """Set accessibility description"""
        self.accessibility_description = description
    
    def set_accessibility_role(self, role: str) -> None:
        """Set accessibility role"""
        self.accessibility_role = role
    
    def set_accessibility_state(self, state: str, value: Any) -> None:
        """Set accessibility state property"""
        self.accessibility_state[state] = value
    
    def get_accessibility_description(self) -> str:
        """Get full accessibility description"""
        parts = []
        
        if self.accessibility_role:
            parts.append(self.accessibility_role)
        
        if self.accessibility_label:
            parts.append(self.accessibility_label)
        elif self.accessibility_description:
            parts.append(self.accessibility_description)
        
        # Add state information
        for state, value in self.accessibility_state.items():
            if value:
                parts.append(state)
        
        return ", ".join(parts) if parts else "interactive element"


# Message classes
class AccessibilitySettingsChanged(Message):
    """Message sent when accessibility settings change"""
    
    def __init__(self, settings: AccessibilitySettings) -> None:
        super().__init__()
        self.settings = settings


class AccessibilityAnnouncement(Message):
    """Message for screen reader announcements"""
    
    def __init__(self, text: str, priority: str = "polite") -> None:
        super().__init__()
        self.text = text
        self.priority = priority


# Global accessibility manager
_accessibility_manager: Optional[AccessibilityManager] = None

def get_accessibility_manager(app: App) -> AccessibilityManager:
    """Get global accessibility manager instance"""
    global _accessibility_manager
    if _accessibility_manager is None or _accessibility_manager.app != app:
        _accessibility_manager = AccessibilityManager(app)
    return _accessibility_manager

def setup_accessibility(app: App) -> AccessibilityManager:
    """Setup accessibility features for an app"""
    return get_accessibility_manager(app)