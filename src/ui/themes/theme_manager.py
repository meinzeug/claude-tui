#!/usr/bin/env python3
"""
Theme Manager - Modern theme system with dark/light modes and advanced customization
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import colorsys

from textual.app import App


class ThemeMode(Enum):
    """Available theme modes"""
    DARK = "dark"
    LIGHT = "light"
    AUTO = "auto"
    HIGH_CONTRAST = "high_contrast"
    CYBERPUNK = "cyberpunk"
    MATRIX = "matrix"
    RETRO = "retro"
    CUSTOM = "custom"


@dataclass
class ColorPalette:
    """Color palette definition"""
    # Core colors
    primary: str
    primary_variant: str
    secondary: str
    secondary_variant: str
    
    # Background colors
    background: str
    surface: str
    surface_variant: str
    
    # Text colors
    on_background: str
    on_surface: str
    on_primary: str
    on_secondary: str
    
    # Status colors
    success: str
    warning: str
    error: str
    info: str
    
    # Accent colors
    accent: str
    highlight: str
    
    # Border and outline
    border: str
    outline: str
    outline_variant: str
    
    # Special states
    disabled: str
    shadow: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for CSS variables"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'ColorPalette':
        """Create from dictionary"""
        return cls(**data)
    
    def generate_variants(self) -> Dict[str, str]:
        """Generate color variants for hover, active states etc."""
        variants = {}
        base_colors = ['primary', 'secondary', 'success', 'warning', 'error', 'info']
        
        for color_name in base_colors:
            base_color = getattr(self, color_name)
            if base_color.startswith('#'):
                # Generate lighter and darker variants
                variants[f"{color_name}_light"] = self._lighten_color(base_color, 0.2)
                variants[f"{color_name}_dark"] = self._darken_color(base_color, 0.2)
                variants[f"{color_name}_alpha_50"] = base_color + "80"  # 50% alpha
                variants[f"{color_name}_alpha_20"] = base_color + "33"  # 20% alpha
        
        return variants
    
    def _lighten_color(self, hex_color: str, factor: float) -> str:
        """Lighten a hex color by a factor"""
        rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        h, l, s = colorsys.rgb_to_hls(rgb[0]/255, rgb[1]/255, rgb[2]/255)
        l = min(1.0, l + factor)
        rgb = colorsys.hls_to_rgb(h, l, s)
        return f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
    
    def _darken_color(self, hex_color: str, factor: float) -> str:
        """Darken a hex color by a factor"""
        rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        h, l, s = colorsys.rgb_to_hls(rgb[0]/255, rgb[1]/255, rgb[2]/255)
        l = max(0.0, l - factor)
        rgb = colorsys.hls_to_rgb(h, l, s)
        return f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"


@dataclass 
class Typography:
    """Typography settings"""
    base_font_size: int = 14
    line_height: float = 1.5
    font_family: str = "monospace"
    heading_scale: float = 1.25
    
    # Text styles
    bold_weight: str = "bold"
    normal_weight: str = "normal"
    italic_style: str = "italic"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Spacing:
    """Spacing and layout settings"""
    base_unit: int = 4  # Base spacing unit in pixels/chars
    
    # Padding scales
    padding_xs: int = 1
    padding_sm: int = 2  
    padding_md: int = 4
    padding_lg: int = 6
    padding_xl: int = 8
    
    # Margin scales
    margin_xs: int = 1
    margin_sm: int = 2
    margin_md: int = 4
    margin_lg: int = 6
    margin_xl: int = 8
    
    # Border radius
    border_radius_sm: int = 1
    border_radius_md: int = 2
    border_radius_lg: int = 3
    
    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class AnimationSettings:
    """Animation and transition settings"""
    duration_fast: str = "150ms"
    duration_normal: str = "300ms"
    duration_slow: str = "600ms"
    
    easing_ease: str = "ease"
    easing_ease_in: str = "ease-in"
    easing_ease_out: str = "ease-out"
    easing_ease_in_out: str = "ease-in-out"
    
    # Textual-specific (simplified)
    enable_smooth_scrolling: bool = True
    enable_hover_effects: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Theme:
    """Complete theme definition"""
    name: str
    mode: ThemeMode
    colors: ColorPalette
    typography: Typography
    spacing: Spacing
    animations: AnimationSettings
    
    # Theme metadata
    description: str = ""
    author: str = ""
    version: str = "1.0.0"
    
    def to_css_variables(self) -> str:
        """Generate CSS variables for the theme"""
        css_vars = []
        
        # Color variables
        for name, value in self.colors.to_dict().items():
            css_vars.append(f"${name.replace('_', '-')}: {value};")
        
        # Color variants
        for name, value in self.colors.generate_variants().items():
            css_vars.append(f"${name.replace('_', '-')}: {value};")
        
        # Typography variables
        for name, value in self.typography.to_dict().items():
            css_vars.append(f"${name.replace('_', '-')}: {value};")
        
        # Spacing variables
        for name, value in self.spacing.to_dict().items():
            css_vars.append(f"${name.replace('_', '-')}: {value};")
        
        return "\n".join(css_vars)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'mode': self.mode.value,
            'colors': self.colors.to_dict(),
            'typography': self.typography.to_dict(),
            'spacing': self.spacing.to_dict(),
            'animations': self.animations.to_dict(),
            'description': self.description,
            'author': self.author,
            'version': self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Theme':
        """Create from dictionary"""
        return cls(
            name=data['name'],
            mode=ThemeMode(data['mode']),
            colors=ColorPalette.from_dict(data['colors']),
            typography=Typography(**data['typography']),
            spacing=Spacing(**data['spacing']),
            animations=AnimationSettings(**data['animations']),
            description=data.get('description', ''),
            author=data.get('author', ''),
            version=data.get('version', '1.0.0')
        )


class ThemeManager:
    """Advanced theme management system"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path.home() / ".claude-tui" / "themes"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self._themes: Dict[str, Theme] = {}
        self._current_theme: Optional[Theme] = None
        self._theme_observers: List[callable] = []
        
        # Load built-in themes
        self._load_builtin_themes()
        
        # Load custom themes
        self._load_custom_themes()
        
        # Set default theme
        self._current_theme = self._themes.get('dark_modern') or list(self._themes.values())[0]
    
    def _load_builtin_themes(self):
        """Load built-in theme definitions"""
        
        # Modern Dark Theme
        dark_modern = Theme(
            name="dark_modern",
            mode=ThemeMode.DARK,
            colors=ColorPalette(
                primary="#0ea5e9",
                primary_variant="#0284c7",
                secondary="#8b5cf6", 
                secondary_variant="#7c3aed",
                background="#0f172a",
                surface="#1e293b",
                surface_variant="#334155",
                on_background="#f8fafc",
                on_surface="#f8fafc",
                on_primary="#ffffff",
                on_secondary="#ffffff",
                success="#10b981",
                warning="#f59e0b",
                error="#ef4444",
                info="#3b82f6",
                accent="#8b5cf6",
                highlight="#fbbf24",
                border="#475569",
                outline="#64748b",
                outline_variant="#475569",
                disabled="#64748b",
                shadow="#000000"
            ),
            typography=Typography(),
            spacing=Spacing(),
            animations=AnimationSettings(),
            description="Modern dark theme with blue accents",
            author="Claude-TUI Team"
        )
        
        # Modern Light Theme
        light_modern = Theme(
            name="light_modern",
            mode=ThemeMode.LIGHT,
            colors=ColorPalette(
                primary="#0ea5e9",
                primary_variant="#0284c7",
                secondary="#8b5cf6",
                secondary_variant="#7c3aed",
                background="#ffffff",
                surface="#f8fafc",
                surface_variant="#f1f5f9",
                on_background="#0f172a",
                on_surface="#1e293b",
                on_primary="#ffffff",
                on_secondary="#ffffff",
                success="#059669",
                warning="#d97706",
                error="#dc2626",
                info="#2563eb",
                accent="#8b5cf6",
                highlight="#f59e0b",
                border="#e2e8f0",
                outline="#cbd5e1",
                outline_variant="#e2e8f0",
                disabled="#94a3b8",
                shadow="#64748b"
            ),
            typography=Typography(),
            spacing=Spacing(),
            animations=AnimationSettings(),
            description="Clean light theme with blue accents",
            author="Claude-TUI Team"
        )
        
        # High Contrast Theme
        high_contrast = Theme(
            name="high_contrast",
            mode=ThemeMode.HIGH_CONTRAST,
            colors=ColorPalette(
                primary="#ffffff",
                primary_variant="#e5e5e5",
                secondary="#00ff00",
                secondary_variant="#00cc00",
                background="#000000",
                surface="#1a1a1a",
                surface_variant="#2a2a2a",
                on_background="#ffffff",
                on_surface="#ffffff",
                on_primary="#000000",
                on_secondary="#000000",
                success="#00ff00",
                warning="#ffff00",
                error="#ff0000",
                info="#00ffff",
                accent="#ffffff",
                highlight="#ffff00",
                border="#ffffff",
                outline="#ffffff",
                outline_variant="#cccccc",
                disabled="#666666",
                shadow="#000000"
            ),
            typography=Typography(base_font_size=16),
            spacing=Spacing(base_unit=6),
            animations=AnimationSettings(enable_hover_effects=False),
            description="High contrast theme for accessibility",
            author="Claude-TUI Team"
        )
        
        # Cyberpunk Theme
        cyberpunk = Theme(
            name="cyberpunk",
            mode=ThemeMode.CYBERPUNK,
            colors=ColorPalette(
                primary="#ff0066",
                primary_variant="#cc0052",
                secondary="#00ffff",
                secondary_variant="#00cccc",
                background="#0a0a0a",
                surface="#1a0d1a",
                surface_variant="#2a1a2a",
                on_background="#ff00ff",
                on_surface="#ff00ff",
                on_primary="#000000",
                on_secondary="#000000",
                success="#00ff00",
                warning="#ff9900",
                error="#ff0066",
                info="#00ffff",
                accent="#ff00ff",
                highlight="#ffff00",
                border="#ff0066",
                outline="#ff00ff",
                outline_variant="#cc00cc",
                disabled="#666666",
                shadow="#ff0066"
            ),
            typography=Typography(font_family="monospace"),
            spacing=Spacing(),
            animations=AnimationSettings(),
            description="Cyberpunk neon aesthetic",
            author="Claude-TUI Team"
        )
        
        # Matrix Theme
        matrix = Theme(
            name="matrix",
            mode=ThemeMode.MATRIX,
            colors=ColorPalette(
                primary="#00ff00",
                primary_variant="#00cc00",
                secondary="#008000",
                secondary_variant="#006400",
                background="#000000",
                surface="#001100",
                surface_variant="#002200",
                on_background="#00ff00",
                on_surface="#00ff00",
                on_primary="#000000",
                on_secondary="#000000",
                success="#00ff00",
                warning="#ffff00",
                error="#ff4444",
                info="#44ff44",
                accent="#00ff00",
                highlight="#ffffff",
                border="#00ff00",
                outline="#00cc00",
                outline_variant="#008000",
                disabled="#004400",
                shadow="#000000"
            ),
            typography=Typography(font_family="monospace"),
            spacing=Spacing(),
            animations=AnimationSettings(),
            description="Matrix digital rain theme",
            author="Claude-TUI Team"
        )
        
        # Retro Theme
        retro = Theme(
            name="retro",
            mode=ThemeMode.RETRO,
            colors=ColorPalette(
                primary="#ff6b35",
                primary_variant="#e55a2b",
                secondary="#4ecdc4",
                secondary_variant="#44b3ac",
                background="#2e2e2e",
                surface="#3a3a3a",
                surface_variant="#4a4a4a",
                on_background="#f7f7f7",
                on_surface="#f7f7f7",
                on_primary="#ffffff",
                on_secondary="#ffffff",
                success="#95e1d3",
                warning="#f38ba8",
                error="#e06b74",
                info="#74c0fc",
                accent="#ff6b35",
                highlight="#ffd93d",
                border="#5a5a5a",
                outline="#6a6a6a",
                outline_variant="#5a5a5a",
                disabled="#7a7a7a",
                shadow="#1e1e1e"
            ),
            typography=Typography(),
            spacing=Spacing(),
            animations=AnimationSettings(),
            description="Warm retro color scheme",
            author="Claude-TUI Team"
        )
        
        # Store themes
        self._themes = {
            'dark_modern': dark_modern,
            'light_modern': light_modern,
            'high_contrast': high_contrast,
            'cyberpunk': cyberpunk,
            'matrix': matrix,
            'retro': retro
        }
    
    def _load_custom_themes(self):
        """Load custom themes from config directory"""
        for theme_file in self.config_dir.glob("*.json"):
            try:
                with open(theme_file, 'r') as f:
                    theme_data = json.load(f)
                    theme = Theme.from_dict(theme_data)
                    self._themes[theme.name] = theme
            except Exception as e:
                print(f"Warning: Failed to load theme from {theme_file}: {e}")
    
    def get_available_themes(self) -> List[str]:
        """Get list of available theme names"""
        return list(self._themes.keys())
    
    def get_theme(self, name: str) -> Optional[Theme]:
        """Get theme by name"""
        return self._themes.get(name)
    
    def get_current_theme(self) -> Theme:
        """Get currently active theme"""
        return self._current_theme
    
    def set_theme(self, theme_name: str) -> bool:
        """Set active theme by name"""
        if theme_name in self._themes:
            self._current_theme = self._themes[theme_name]
            self._notify_observers()
            return True
        return False
    
    def create_custom_theme(self, theme: Theme) -> bool:
        """Create and save a custom theme"""
        try:
            # Add to memory
            self._themes[theme.name] = theme
            
            # Save to file
            theme_file = self.config_dir / f"{theme.name}.json"
            with open(theme_file, 'w') as f:
                json.dump(theme.to_dict(), f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving custom theme: {e}")
            return False
    
    def delete_custom_theme(self, theme_name: str) -> bool:
        """Delete a custom theme"""
        if theme_name in ['dark_modern', 'light_modern', 'high_contrast', 'cyberpunk', 'matrix', 'retro']:
            return False  # Can't delete built-in themes
        
        try:
            # Remove from memory
            if theme_name in self._themes:
                del self._themes[theme_name]
            
            # Remove file
            theme_file = self.config_dir / f"{theme_name}.json"
            if theme_file.exists():
                theme_file.unlink()
            
            # Switch to default if current theme was deleted
            if self._current_theme and self._current_theme.name == theme_name:
                self.set_theme('dark_modern')
            
            return True
        except Exception as e:
            print(f"Error deleting custom theme: {e}")
            return False
    
    def generate_theme_css(self, theme_name: Optional[str] = None) -> str:
        """Generate CSS for a theme"""
        theme = self._themes.get(theme_name) if theme_name else self._current_theme
        if not theme:
            return ""
        
        return f"""/* {theme.name} Theme - {theme.description} */
/* Generated by Claude-TUI Theme Manager */

{theme.to_css_variables()}

/* Theme-specific component styles */
Screen {{
    background: $background;
    color: $on-background;
}}

/* Enhanced component styling based on theme */
Button {{
    background: $surface;
    color: $on-surface;
    border: round $outline;
}}

Button.primary {{
    background: $primary;
    color: $on-primary;
}}

Button:hover {{
    background: $primary-light;
}}

Input {{
    background: $surface-variant;
    color: $on-surface;
    border: round $outline;
}}

Input:focus {{
    border: round $primary;
    background: $surface;
}}
"""
    
    def add_theme_observer(self, callback: callable):
        """Add observer for theme changes"""
        self._theme_observers.append(callback)
    
    def remove_theme_observer(self, callback: callable):
        """Remove theme observer"""
        if callback in self._theme_observers:
            self._theme_observers.remove(callback)
    
    def _notify_observers(self):
        """Notify all observers of theme change"""
        for observer in self._theme_observers:
            try:
                observer(self._current_theme)
            except Exception as e:
                print(f"Error in theme observer: {e}")
    
    def auto_detect_theme(self) -> str:
        """Auto-detect appropriate theme based on system settings"""
        # This would integrate with system theme detection
        # For now, return dark_modern as default
        return "dark_modern"
    
    def export_theme(self, theme_name: str, export_path: Path) -> bool:
        """Export theme to file"""
        theme = self._themes.get(theme_name)
        if not theme:
            return False
        
        try:
            with open(export_path, 'w') as f:
                json.dump(theme.to_dict(), f, indent=2)
            return True
        except Exception:
            return False
    
    def import_theme(self, import_path: Path) -> Optional[str]:
        """Import theme from file"""
        try:
            with open(import_path, 'r') as f:
                theme_data = json.load(f)
                theme = Theme.from_dict(theme_data)
                
                # Add to collection
                self._themes[theme.name] = theme
                
                # Optionally save as custom theme
                self.create_custom_theme(theme)
                
                return theme.name
        except Exception as e:
            print(f"Error importing theme: {e}")
            return None


# Global theme manager instance
_theme_manager: Optional[ThemeManager] = None

def get_theme_manager() -> ThemeManager:
    """Get global theme manager instance"""
    global _theme_manager
    if _theme_manager is None:
        _theme_manager = ThemeManager()
    return _theme_manager

def apply_theme_to_app(app: App, theme: Optional[Theme] = None):
    """Apply theme to Textual app"""
    theme_manager = get_theme_manager()
    active_theme = theme or theme_manager.get_current_theme()
    
    # Generate CSS content
    css_content = theme_manager.generate_theme_css(active_theme.name)
    
    # Apply to app (would need custom CSS loading mechanism)
    # This is a placeholder - actual implementation would depend on 
    # how Textual loads and applies CSS
    pass