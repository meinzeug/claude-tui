"""File Tree Widget for Claude-TUI."""

import logging
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any
import os
import stat

try:
    from textual.widgets import DirectoryTree, Tree
    from textual import events
    from textual.message import Message
    from textual.reactive import reactive
    
    from ...interfaces.ui_interfaces import TreeInterface
    
    logger = logging.getLogger(__name__)
    
    class FileTree(DirectoryTree):
        """Enhanced file tree with filtering and actions."""
        
        class FileSelected(Message):
            """Message sent when a file is selected."""
            
            def __init__(self, sender: "FileTree", path: Path) -> None:
                self.path = path
                super().__init__(sender)
        
        class DirectoryChanged(Message):
            """Message sent when directory changes."""
            
            def __init__(self, sender: "FileTree", path: Path) -> None:
                self.path = path
                super().__init__(sender)
        
        def __init__(
            self, 
            path: str = ".", 
            show_hidden: bool = False,
            file_filter: Optional[Callable[[Path], bool]] = None,
            *args, 
            **kwargs
        ):
            super().__init__(path, *args, **kwargs)
            self.root_path = Path(path)
            self.show_hidden = show_hidden
            self.file_filter = file_filter
            self._file_selected_callbacks: List[Callable[[Path], None]] = []
            self._initialized = False
            self._expanded_nodes: set[str] = set()
            
        def initialize(self) -> None:
            """Initialize the file tree."""
            if self._initialized:
                return
                
            try:
                # Ensure root path exists
                if not self.root_path.exists():
                    logger.warning(f"Root path does not exist: {self.root_path}")
                    self.root_path = Path.cwd()
                    
                # Set up file filtering
                if not self.file_filter:
                    self.file_filter = self._default_file_filter
                    
                self._initialized = True
                logger.debug(f"File tree initialized with root: {self.root_path}")
                
            except Exception as e:
                logger.error(f"Failed to initialize file tree: {e}")
        
        def cleanup(self) -> None:
            """Cleanup tree resources."""
            self._file_selected_callbacks.clear()
            self._expanded_nodes.clear()
            logger.debug("File tree cleanup completed")
        
        def set_focus(self) -> None:
            """Set focus to the tree."""
            if hasattr(self, 'focus'):
                self.focus()
        
        def set_root_path(self, path: Path) -> None:
            """Set root directory path."""
            try:
                if path.exists() and path.is_dir():
                    self.root_path = path
                    self.path = str(path)
                    self.reload()
                    self.post_message(self.DirectoryChanged(self, path))
                    logger.debug(f"Root path changed to: {path}")
                else:
                    logger.error(f"Invalid directory path: {path}")
            except Exception as e:
                logger.error(f"Failed to set root path: {e}")
        
        def refresh(self) -> None:
            """Refresh tree contents."""
            try:
                # Store expanded state
                expanded_paths = self._get_expanded_paths()
                
                # Reload the tree
                self.reload()
                
                # Restore expanded state
                for path in expanded_paths:
                    self.expand_node(Path(path))
                    
                logger.debug("File tree refreshed")
            except Exception as e:
                logger.error(f"Failed to refresh tree: {e}")
        
        def expand_node(self, path: Path) -> None:
            """Expand tree node at path."""
            try:
                rel_path = path.relative_to(self.root_path)
                self._expanded_nodes.add(str(rel_path))
                
                # Use DirectoryTree's expand functionality if available
                if hasattr(self, 'expand'):
                    # This would need to map to actual node IDs
                    pass
                    
                logger.debug(f"Expanded node: {path}")
            except Exception as e:
                logger.error(f"Failed to expand node {path}: {e}")
        
        def collapse_node(self, path: Path) -> None:
            """Collapse tree node at path."""
            try:
                rel_path = path.relative_to(self.root_path)
                self._expanded_nodes.discard(str(rel_path))
                
                # Use DirectoryTree's collapse functionality if available
                if hasattr(self, 'collapse'):
                    # This would need to map to actual node IDs
                    pass
                    
                logger.debug(f"Collapsed node: {path}")
            except Exception as e:
                logger.error(f"Failed to collapse node {path}: {e}")
        
        def select_node(self, path: Path) -> None:
            """Select tree node at path."""
            try:
                # Use DirectoryTree's selection functionality if available
                if hasattr(self, 'select_node'):
                    # This would need to map to actual node
                    pass
                    
                self.post_message(self.FileSelected(self, path))
                logger.debug(f"Selected node: {path}")
            except Exception as e:
                logger.error(f"Failed to select node {path}: {e}")
        
        def get_selected_path(self) -> Optional[Path]:
            """Get currently selected path."""
            # This would need access to DirectoryTree's selection state
            # For now, return None as placeholder
            return None
        
        def on_file_selected(self, callback: Callable[[Path], None]) -> None:
            """Register callback for file selection."""
            if callback not in self._file_selected_callbacks:
                self._file_selected_callbacks.append(callback)
        
        def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
            """Handle file selection from Textual."""
            self.post_message(self.FileSelected(self, event.path))
            self._notify_file_selected(event.path)
            
        def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
            """Handle directory selection from Textual."""
            self.post_message(self.DirectoryChanged(self, event.path))
            
        def _notify_file_selected(self, path: Path) -> None:
            """Notify all registered callbacks of file selection."""
            for callback in self._file_selected_callbacks:
                try:
                    callback(path)
                except Exception as e:
                    logger.error(f"File selection callback error: {e}")
        
        def _default_file_filter(self, path: Path) -> bool:
            """Default file filtering logic."""
            try:
                # Skip hidden files/directories if not showing hidden
                if not self.show_hidden and path.name.startswith('.'):
                    return False
                
                # Skip common build/cache directories
                skip_dirs = {
                    '__pycache__', '.git', '.svn', '.hg', 'node_modules',
                    '.vscode', '.idea', 'build', 'dist', '.pytest_cache'
                }
                
                if path.is_dir() and path.name in skip_dirs:
                    return False
                
                return True
                
            except Exception as e:
                logger.error(f"File filter error for {path}: {e}")
                return True
        
        def _get_expanded_paths(self) -> List[str]:
            """Get list of currently expanded paths."""
            return list(self._expanded_nodes)
        
        def get_file_info(self, path: Path) -> Dict[str, Any]:
            """Get detailed file information."""
            try:
                if not path.exists():
                    return {}
                
                stat_info = path.stat()
                
                info = {
                    'name': path.name,
                    'path': str(path),
                    'is_file': path.is_file(),
                    'is_dir': path.is_dir(),
                    'size': stat_info.st_size if path.is_file() else 0,
                    'modified': stat_info.st_mtime,
                    'permissions': oct(stat_info.st_mode)[-3:],
                    'extension': path.suffix.lower() if path.is_file() else None
                }
                
                return info
                
            except Exception as e:
                logger.error(f"Failed to get file info for {path}: {e}")
                return {}
        
        def search_files(self, query: str, case_sensitive: bool = False) -> List[Path]:
            """Search for files matching query."""
            matches = []
            search_func = str.find if case_sensitive else lambda s, q: s.lower().find(q.lower())
            
            try:
                for root, dirs, files in os.walk(self.root_path):
                    root_path = Path(root)
                    
                    # Filter directories
                    dirs[:] = [d for d in dirs if self.file_filter(root_path / d)]
                    
                    # Search files
                    for file in files:
                        file_path = root_path / file
                        if self.file_filter(file_path) and search_func(file, query) != -1:
                            matches.append(file_path)
            
            except Exception as e:
                logger.error(f"File search error: {e}")
            
            return matches
                
except ImportError as e:
    # Enhanced fallback implementation
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to import Textual widgets: {e}. Using enhanced fallback.")
    
    from ...interfaces.ui_interfaces import TreeInterface
    
    class FileTree(TreeInterface):
        """Enhanced fallback file tree widget."""
        
        def __init__(
            self, 
            path: str = ".", 
            show_hidden: bool = False,
            file_filter: Optional[Callable[[Path], bool]] = None,
            *args, 
            **kwargs
        ):
            self.root_path = Path(path)
            self.show_hidden = show_hidden
            self.file_filter = file_filter or self._default_file_filter
            self._file_selected_callbacks: List[Callable[[Path], None]] = []
            self._expanded_nodes: set[str] = set()
            self._selected_path: Optional[Path] = None
            self._initialized = False
            
        def initialize(self) -> None:
            """Initialize the fallback tree."""
            if not self.root_path.exists():
                self.root_path = Path.cwd()
            self._initialized = True
            logger.info(f"Fallback file tree initialized with root: {self.root_path}")
        
        def cleanup(self) -> None:
            """Cleanup tree resources."""
            self._file_selected_callbacks.clear()
            self._expanded_nodes.clear()
        
        def set_focus(self) -> None:
            """Set focus (no-op in fallback)."""
            pass
        
        def set_root_path(self, path: Path) -> None:
            """Set root directory path."""
            if path.exists() and path.is_dir():
                self.root_path = path
                logger.info(f"Root path changed to: {path}")
        
        def refresh(self) -> None:
            """Refresh tree contents (logged in fallback)."""
            logger.info("Would refresh file tree")
        
        def expand_node(self, path: Path) -> None:
            """Expand tree node at path."""
            try:
                rel_path = path.relative_to(self.root_path)
                self._expanded_nodes.add(str(rel_path))
                logger.info(f"Would expand node: {path}")
            except ValueError:
                logger.error(f"Path not relative to root: {path}")
        
        def collapse_node(self, path: Path) -> None:
            """Collapse tree node at path."""
            try:
                rel_path = path.relative_to(self.root_path)
                self._expanded_nodes.discard(str(rel_path))
                logger.info(f"Would collapse node: {path}")
            except ValueError:
                logger.error(f"Path not relative to root: {path}")
        
        def select_node(self, path: Path) -> None:
            """Select tree node at path."""
            self._selected_path = path
            self._notify_file_selected(path)
            logger.info(f"Selected node: {path}")
        
        def get_selected_path(self) -> Optional[Path]:
            """Get currently selected path."""
            return self._selected_path
        
        def on_file_selected(self, callback: Callable[[Path], None]) -> None:
            """Register callback for file selection."""
            if callback not in self._file_selected_callbacks:
                self._file_selected_callbacks.append(callback)
        
        def _notify_file_selected(self, path: Path) -> None:
            """Notify all registered callbacks of file selection."""
            for callback in self._file_selected_callbacks:
                try:
                    callback(path)
                except Exception as e:
                    logger.error(f"File selection callback error: {e}")
        
        def _default_file_filter(self, path: Path) -> bool:
            """Default file filtering logic."""
            if not self.show_hidden and path.name.startswith('.'):
                return False
            
            skip_dirs = {
                '__pycache__', '.git', '.svn', '.hg', 'node_modules',
                '.vscode', '.idea', 'build', 'dist', '.pytest_cache'
            }
            
            return not (path.is_dir() and path.name in skip_dirs)
        
        def get_file_info(self, path: Path) -> Dict[str, Any]:
            """Get detailed file information."""
            try:
                if not path.exists():
                    return {}
                
                stat_info = path.stat()
                
                return {
                    'name': path.name,
                    'path': str(path),
                    'is_file': path.is_file(),
                    'is_dir': path.is_dir(),
                    'size': stat_info.st_size if path.is_file() else 0,
                    'modified': stat_info.st_mtime,
                    'permissions': oct(stat_info.st_mode)[-3:],
                    'extension': path.suffix.lower() if path.is_file() else None
                }
                
            except Exception as e:
                logger.error(f"Failed to get file info for {path}: {e}")
                return {}
        
        def search_files(self, query: str, case_sensitive: bool = False) -> List[Path]:
            """Search for files matching query."""
            matches = []
            search_func = str.find if case_sensitive else lambda s, q: s.lower().find(q.lower())
            
            try:
                for root, dirs, files in os.walk(self.root_path):
                    root_path = Path(root)
                    
                    # Filter directories
                    dirs[:] = [d for d in dirs if self.file_filter(root_path / d)]
                    
                    # Search files
                    for file in files:
                        file_path = root_path / file
                        if self.file_filter(file_path) and search_func(file, query) != -1:
                            matches.append(file_path)
            
            except Exception as e:
                logger.error(f"File search error: {e}")
            
            return matches