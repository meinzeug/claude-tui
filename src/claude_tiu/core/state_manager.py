"""
State Manager - Simple state management for claude_tiu core module.

This module provides basic state management functionality to resolve import dependencies.
"""

import logging
from typing import Any, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class StateManager:
    """Basic state management for the claude_tiu core module."""
    
    def __init__(self):
        """Initialize state manager."""
        self._state: Dict[str, Any] = {}
        self._created_at = datetime.utcnow()
        self.logger = logger
    
    def get_state(self, key: str, default: Optional[Any] = None) -> Any:
        """Get state value by key."""
        return self._state.get(key, default)
    
    def set_state(self, key: str, value: Any) -> None:
        """Set state value by key."""
        self._state[key] = value
        self.logger.debug(f"State updated: {key} = {value}")
    
    def clear_state(self) -> None:
        """Clear all state."""
        self._state.clear()
        self.logger.info("State cleared")
    
    def get_all_state(self) -> Dict[str, Any]:
        """Get all state data."""
        return self._state.copy()
    
    @property
    def created_at(self) -> datetime:
        """Get state manager creation time."""
        return self._created_at