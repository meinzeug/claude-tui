"""
IDE Intelligence Bridge
Seamless integration layer for VS Code, IntelliJ IDEA, Vim/Neovim and other IDEs
"""

import asyncio
import json
import logging
import subprocess
import tempfile
import time
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import websocket
import requests
from threading import Thread, Event

from .universal_environment_adapter import (
    AdapterPlugin, EnvironmentContext, IntegrationStatus, 
    EnvironmentCapability, create_adapter_plugin
)


class IDEType(str, Enum):
    """Supported IDE types"""
    VSCODE = "vscode"
    INTELLIJ = "intellij"
    VIM = "vim"
    NEOVIM = "neovim"
    SUBLIME = "sublime"
    ATOM = "atom"
    EMACS = "emacs"
    PYCHARM = "pycharm"
    WEBSTORM = "webstorm"


class IDEFeature(str, Enum):
    """IDE feature capabilities"""
    SYNTAX_HIGHLIGHTING = "syntax_highlighting"
    AUTO_COMPLETION = "auto_completion"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    TERMINAL = "terminal"
    GIT_INTEGRATION = "git_integration"
    EXTENSION_API = "extension_api"
    LANGUAGE_SERVER = "language_server"
    PROJECT_MANAGEMENT = "project_management"
    CODE_NAVIGATION = "code_navigation"


@dataclass
class IDEEvent:
    """IDE event data structure"""
    event_type: str
    timestamp: float
    file_path: Optional[str] = None
    content: Optional[str] = None
    cursor_position: Optional[Tuple[int, int]] = None
    selection: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "file_path": self.file_path,
            "content": self.content,
            "cursor_position": self.cursor_position,
            "selection": self.selection,
            "metadata": self.metadata or {}
        }


class BaseIDEAdapter(AdapterPlugin):
    """Base adapter for IDE integration"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ide_type = IDEType(config.get("ide_type", "vscode"))
        self.workspace_path = config.get("workspace_path")
        self.ide_path = config.get("ide_path")
        self.port = config.get("port", 9000)
        self.host = config.get("host", "localhost")
        
        # Event handling
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._stop_event = Event()
        self._monitor_thread: Optional[Thread] = None
        
        # State tracking
        self._current_files: Set[str] = set()
        self._active_file: Optional[str] = None
        self._cursor_position: Optional[Tuple[int, int]] = None
        
    @abstractmethod
    async def _establish_connection(self) -> bool:
        """Establish connection to IDE"""
        pass
        
    @abstractmethod
    async def _close_connection(self):
        """Close connection to IDE"""
        pass
        
    @abstractmethod
    async def _send_ide_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send command to IDE"""
        pass
        
    async def initialize(self) -> bool:
        """Initialize IDE adapter"""
        try:
            self._status = IntegrationStatus.CONNECTING
            
            # Detect IDE capabilities
            await self._detect_capabilities()
            
            # Start monitoring
            self._start_monitoring()
            
            self._status = IntegrationStatus.ACTIVE
            self.logger.info(f"IDE adapter initialized: {self.ide_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize IDE adapter: {e}")
            self._status = IntegrationStatus.ERROR
            return False
            
    async def connect(self, context: EnvironmentContext) -> bool:
        """Connect to IDE"""
        try:
            self.workspace_path = context.workspace_path
            
            if await self._establish_connection():
                self.logger.info(f"Connected to {self.ide_type}")
                return True
            else:
                self.logger.error(f"Failed to connect to {self.ide_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False
            
    async def disconnect(self) -> bool:
        """Disconnect from IDE"""
        try:
            self._stop_event.set()
            
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5)
                
            await self._close_connection()
            
            self._status = IntegrationStatus.INACTIVE
            self.logger.info(f"Disconnected from {self.ide_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Disconnection error: {e}")
            return False
            
    async def send_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send command to IDE"""
        try:
            if self._status != IntegrationStatus.ACTIVE:
                raise RuntimeError(f"IDE not connected: {self._status}")
                
            return await self._send_ide_command(command, params)
            
        except Exception as e:
            self.logger.error(f"Command failed: {e}")
            raise
            
    async def receive_events(self) -> List[Dict[str, Any]]:
        """Receive events from IDE"""
        events = []
        
        try:
            while not self._event_queue.empty():
                event = await self._event_queue.get()
                events.append(event.to_dict())
                
        except Exception as e:
            self.logger.error(f"Error receiving events: {e}")
            
        return events
        
    async def sync_state(self, state: Dict[str, Any]) -> bool:
        """Sync state with IDE"""
        try:
            # Sync open files
            if "open_files" in state:
                for file_path in state["open_files"]:
                    await self._send_ide_command("open_file", {"path": file_path})
                    
            # Sync active file
            if "active_file" in state:
                await self._send_ide_command("activate_file", {"path": state["active_file"]})
                
            # Sync cursor position
            if "cursor_position" in state:
                await self._send_ide_command("set_cursor", {"position": state["cursor_position"]})
                
            return True
            
        except Exception as e:
            self.logger.error(f"State sync failed: {e}")
            return False
            
    async def _detect_capabilities(self):
        """Detect IDE capabilities"""
        capabilities = set()
        
        # Common capabilities for most IDEs
        capabilities.add(EnvironmentCapability(
            name=IDEFeature.SYNTAX_HIGHLIGHTING.value,
            version="1.0",
            features={"languages", "themes"}
        ))
        
        capabilities.add(EnvironmentCapability(
            name=IDEFeature.CODE_NAVIGATION.value,
            version="1.0", 
            features={"goto_definition", "find_references", "search"}
        ))
        
        # IDE-specific capabilities
        if self.ide_type in [IDEType.VSCODE, IDEType.INTELLIJ]:
            capabilities.add(EnvironmentCapability(
                name=IDEFeature.EXTENSION_API.value,
                version="2.0",
                features={"install", "enable", "configure"}
            ))
            
        if self.ide_type in [IDEType.INTELLIJ, IDEType.PYCHARM, IDEType.WEBSTORM]:
            capabilities.add(EnvironmentCapability(
                name=IDEFeature.REFACTORING.value,
                version="2.0",
                features={"rename", "extract", "inline", "move"}
            ))
            
        self._capabilities = capabilities
        
    def _start_monitoring(self):
        """Start IDE monitoring thread"""
        if not self._monitor_thread or not self._monitor_thread.is_alive():
            self._stop_event.clear()
            self._monitor_thread = Thread(target=self._monitor_loop)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
            
    def _monitor_loop(self):
        """IDE monitoring loop (runs in separate thread)"""
        while not self._stop_event.is_set():
            try:
                # This would be implemented by specific IDE adapters
                time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")
                
    async def _add_event(self, event_type: str, **kwargs):
        """Add event to queue"""
        event = IDEEvent(
            event_type=event_type,
            timestamp=time.time(),
            **kwargs
        )
        await self._event_queue.put(event)


@create_adapter_plugin("vscode", {})
class VSCodeAdapter(BaseIDEAdapter):
    """VS Code integration adapter"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ide_type = IDEType.VSCODE
        self.extension_id = config.get("extension_id", "claude-tui-integration")
        self.websocket_url = f"ws://{self.host}:{self.port}/vscode"
        self._websocket: Optional[websocket.WebSocket] = None
        
    async def _establish_connection(self) -> bool:
        """Establish WebSocket connection to VS Code extension"""
        try:
            # Check if VS Code is running
            if not await self._is_vscode_running():
                await self._start_vscode()
                
            # Check if extension is installed
            if not await self._is_extension_installed():
                await self._install_extension()
                
            # Connect via WebSocket
            self._websocket = websocket.create_connection(
                self.websocket_url,
                timeout=10
            )
            
            # Send handshake
            handshake = {
                "type": "handshake",
                "client": "claude-tui",
                "version": "1.0.0"
            }
            self._websocket.send(json.dumps(handshake))
            
            response = json.loads(self._websocket.recv())
            return response.get("status") == "connected"
            
        except Exception as e:
            self.logger.error(f"VS Code connection failed: {e}")
            return False
            
    async def _close_connection(self):
        """Close VS Code connection"""
        if self._websocket:
            try:
                self._websocket.close()
            except:
                pass
            self._websocket = None
            
    async def _send_ide_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send command to VS Code"""
        if not self._websocket:
            raise RuntimeError("Not connected to VS Code")
            
        message = {
            "type": "command",
            "command": command,
            "params": params,
            "id": f"cmd_{int(time.time() * 1000)}"
        }
        
        self._websocket.send(json.dumps(message))
        response = json.loads(self._websocket.recv())
        
        if response.get("error"):
            raise RuntimeError(f"VS Code command failed: {response['error']}")
            
        return response.get("result", {})
        
    async def _is_vscode_running(self) -> bool:
        """Check if VS Code is running"""
        try:
            result = subprocess.run(
                ["code", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
            
    async def _start_vscode(self):
        """Start VS Code with workspace"""
        try:
            cmd = ["code"]
            if self.workspace_path:
                cmd.append(self.workspace_path)
                
            subprocess.Popen(cmd)
            
            # Wait for VS Code to start
            await asyncio.sleep(3)
            
        except Exception as e:
            self.logger.error(f"Failed to start VS Code: {e}")
            
    async def _is_extension_installed(self) -> bool:
        """Check if Claude TUI extension is installed"""
        try:
            result = subprocess.run(
                ["code", "--list-extensions"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return self.extension_id in result.stdout
        except:
            return False
            
    async def _install_extension(self):
        """Install Claude TUI VS Code extension"""
        try:
            # This would install from marketplace or local VSIX
            result = subprocess.run(
                ["code", "--install-extension", self.extension_id],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                self.logger.warning(f"Extension install warning: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Failed to install extension: {e}")


@create_adapter_plugin("intellij", {})
class IntelliJAdapter(BaseIDEAdapter):
    """IntelliJ IDEA integration adapter"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ide_type = IDEType.INTELLIJ
        self.plugin_id = config.get("plugin_id", "com.claude.tui.integration")
        self.http_port = config.get("http_port", 8080)
        self.base_url = f"http://{self.host}:{self.http_port}/api"
        
    async def _establish_connection(self) -> bool:
        """Establish HTTP connection to IntelliJ plugin"""
        try:
            # Check if IntelliJ is running with plugin
            response = requests.get(
                f"{self.base_url}/status",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "ready"
            else:
                # Try to start IntelliJ
                await self._start_intellij()
                await asyncio.sleep(5)
                
                # Retry connection
                response = requests.get(f"{self.base_url}/status", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("status") == "ready"
                    
            return False
            
        except Exception as e:
            self.logger.error(f"IntelliJ connection failed: {e}")
            return False
            
    async def _close_connection(self):
        """Close IntelliJ connection"""
        try:
            requests.post(f"{self.base_url}/disconnect", timeout=5)
        except:
            pass
            
    async def _send_ide_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send command to IntelliJ"""
        try:
            response = requests.post(
                f"{self.base_url}/command",
                json={
                    "command": command,
                    "params": params
                },
                timeout=30
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"IntelliJ command failed: {e}")
            raise
            
    async def _start_intellij(self):
        """Start IntelliJ IDEA"""
        try:
            cmd = ["idea"]
            if self.workspace_path:
                cmd.append(self.workspace_path)
                
            subprocess.Popen(cmd)
            
        except Exception as e:
            self.logger.error(f"Failed to start IntelliJ: {e}")


@create_adapter_plugin("vim", {})
class VimAdapter(BaseIDEAdapter):
    """Vim/Neovim integration adapter"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ide_type = IDEType.VIM if config.get("variant", "vim") == "vim" else IDEType.NEOVIM
        self.socket_path = config.get("socket_path", "/tmp/claude-tui-vim.sock")
        self.vim_script_path = config.get("vim_script_path", "~/.vim/plugin/claude-tui.vim")
        
    async def _establish_connection(self) -> bool:
        """Establish connection to Vim via socket/script"""
        try:
            # Check if Vim is running with our plugin
            if Path(self.socket_path).exists():
                return True
                
            # Install/update Vim script
            await self._install_vim_script()
            
            # Start Vim with plugin
            await self._start_vim()
            
            # Wait for socket
            for _ in range(10):
                if Path(self.socket_path).exists():
                    return True
                await asyncio.sleep(1)
                
            return False
            
        except Exception as e:
            self.logger.error(f"Vim connection failed: {e}")
            return False
            
    async def _close_connection(self):
        """Close Vim connection"""
        try:
            if Path(self.socket_path).exists():
                Path(self.socket_path).unlink()
        except:
            pass
            
    async def _send_ide_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send command to Vim via socket"""
        try:
            # This would use Unix socket communication
            # For now, return a placeholder
            return {"status": "ok", "result": "vim_command_executed"}
            
        except Exception as e:
            self.logger.error(f"Vim command failed: {e}")
            raise
            
    async def _install_vim_script(self):
        """Install Claude TUI Vim script"""
        try:
            script_content = self._generate_vim_script()
            script_path = Path(self.vim_script_path).expanduser()
            script_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(script_path, 'w') as f:
                f.write(script_content)
                
        except Exception as e:
            self.logger.error(f"Failed to install Vim script: {e}")
            
    def _generate_vim_script(self) -> str:
        """Generate Vim script for Claude TUI integration"""
        return f'''
" Claude TUI Integration Plugin
" Auto-generated script for Vim/Neovim integration

if exists('g:claude_tui_loaded')
    finish
endif
let g:claude_tui_loaded = 1

" Socket path for communication
let g:claude_tui_socket = '{self.socket_path}'

" Initialize Claude TUI integration
function! ClaudeTUIInit()
    " Create socket connection
    " This would contain actual Vim script logic
    echo "Claude TUI integration initialized"
endfunction

" Auto-initialize when Vim starts
augroup ClaudeTUI
    autocmd!
    autocmd VimEnter * call ClaudeTUIInit()
augroup END
'''
            
    async def _start_vim(self):
        """Start Vim with workspace"""
        try:
            cmd = ["vim" if self.ide_type == IDEType.VIM else "nvim"]
            if self.workspace_path:
                cmd.append(self.workspace_path)
                
            subprocess.Popen(cmd)
            
        except Exception as e:
            self.logger.error(f"Failed to start Vim: {e}")


class IDEIntelligenceBridge:
    """
    Main bridge class for IDE intelligence integration
    Coordinates between different IDE adapters and provides unified interface
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Registry of active IDE adapters
        self._adapters: Dict[str, BaseIDEAdapter] = {}
        self._ide_configs = config.get("ides", {})
        
        # Intelligence features
        self._auto_completion_cache: Dict[str, List[str]] = {}
        self._refactoring_suggestions: Dict[str, List[Dict[str, Any]]] = {}
        
    async def initialize(self) -> bool:
        """Initialize IDE intelligence bridge"""
        try:
            self.logger.info("Initializing IDE Intelligence Bridge")
            
            # Initialize configured IDEs
            for ide_id, ide_config in self._ide_configs.items():
                await self._initialize_ide(ide_id, ide_config)
                
            self.logger.info("IDE Intelligence Bridge initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize IDE bridge: {e}")
            return False
            
    async def _initialize_ide(self, ide_id: str, config: Dict[str, Any]):
        """Initialize specific IDE adapter"""
        try:
            ide_type = config.get("type", "vscode")
            
            if ide_type == "vscode":
                adapter = VSCodeAdapter(config)
            elif ide_type == "intellij":
                adapter = IntelliJAdapter(config)
            elif ide_type in ["vim", "neovim"]:
                adapter = VimAdapter(config)
            else:
                self.logger.warning(f"Unsupported IDE type: {ide_type}")
                return
                
            if await adapter.initialize():
                self._adapters[ide_id] = adapter
                self.logger.info(f"Initialized IDE adapter: {ide_id}")
            else:
                self.logger.error(f"Failed to initialize IDE adapter: {ide_id}")
                
        except Exception as e:
            self.logger.error(f"Error initializing IDE {ide_id}: {e}")
            
    async def get_intelligent_suggestions(self, ide_id: str, context: str) -> List[str]:
        """Get intelligent code suggestions for specific IDE"""
        try:
            if ide_id not in self._adapters:
                return []
                
            adapter = self._adapters[ide_id]
            result = await adapter.send_command("get_suggestions", {"context": context})
            
            suggestions = result.get("suggestions", [])
            
            # Cache for future use
            self._auto_completion_cache[f"{ide_id}:{context}"] = suggestions
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Failed to get suggestions for {ide_id}: {e}")
            return []
            
    async def perform_refactoring(self, ide_id: str, refactor_type: str, 
                                params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Perform intelligent refactoring in specific IDE"""
        try:
            if ide_id not in self._adapters:
                return None
                
            adapter = self._adapters[ide_id]
            result = await adapter.send_command("refactor", {
                "type": refactor_type,
                **params
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Refactoring failed for {ide_id}: {e}")
            return None
            
    async def sync_across_ides(self, workspace_state: Dict[str, Any]) -> Dict[str, bool]:
        """Synchronize workspace state across all connected IDEs"""
        results = {}
        
        for ide_id, adapter in self._adapters.items():
            try:
                results[ide_id] = await adapter.sync_state(workspace_state)
            except Exception as e:
                self.logger.error(f"Sync failed for {ide_id}: {e}")
                results[ide_id] = False
                
        return results
        
    def get_active_ides(self) -> List[str]:
        """Get list of active IDE connections"""
        return list(self._adapters.keys())
        
    def get_ide_status(self, ide_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific IDE"""
        if ide_id not in self._adapters:
            return None
            
        adapter = self._adapters[ide_id]
        return {
            "ide_id": ide_id,
            "ide_type": adapter.ide_type.value,
            "status": adapter.status.value,
            "capabilities": [cap.name for cap in adapter.capabilities],
            "workspace_path": adapter.workspace_path
        }


if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            "ides": {
                "vscode-main": {
                    "type": "vscode",
                    "workspace_path": "/home/user/project",
                    "port": 9000
                },
                "intellij-main": {
                    "type": "intellij", 
                    "workspace_path": "/home/user/project",
                    "http_port": 8080
                }
            }
        }
        
        bridge = IDEIntelligenceBridge(config)
        await bridge.initialize()
        
        # Get suggestions from VS Code
        suggestions = await bridge.get_intelligent_suggestions(
            "vscode-main", 
            "def process_"
        )
        print(f"VS Code suggestions: {suggestions}")
        
    # Run example
    # asyncio.run(main())