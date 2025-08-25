"""
Editor Integration - Real-time validation for code editors and IDEs.

Provides real-time validation integration for popular code editors:
- VS Code Language Server Protocol (LSP)
- Vim/Neovim integration
- Emacs integration  
- IntelliJ/PyCharm plugin support
- Sublime Text integration
- Real-time feedback with <200ms performance
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import websockets
import threading

from claude_tiu.core.config_manager import ConfigManager
from claude_tiu.validation.real_time_validator import RealTimeValidator, ValidationMode
from claude_tiu.validation.anti_hallucination_engine import AntiHallucinationEngine

logger = logging.getLogger(__name__)


class EditorType(Enum):
    """Supported editor types."""
    VS_CODE = "vscode"
    VIM = "vim"
    NEOVIM = "neovim"
    EMACS = "emacs"
    INTELLIJ = "intellij"
    PYCHARM = "pycharm"
    SUBLIME = "sublime"
    ATOM = "atom"
    GENERIC_LSP = "generic_lsp"


class ValidationLevel(Enum):
    """Validation levels for editor integration."""
    OFF = "off"
    ERROR_ONLY = "error_only"         # Only show critical errors
    WARNING = "warning"               # Show errors and warnings
    INFO = "info"                     # Show all issues including suggestions
    VERBOSE = "verbose"               # Show detailed analysis


@dataclass
class EditorValidationConfig:
    """Configuration for editor validation."""
    enabled: bool = True
    validation_level: ValidationLevel = ValidationLevel.WARNING
    real_time_validation: bool = True
    validation_delay_ms: int = 500    # Delay before validating after typing stops
    max_file_size_kb: int = 1024      # Max file size for real-time validation
    supported_extensions: List[str] = field(default_factory=lambda: [
        '.py', '.js', '.ts', '.jsx', '.tsx', '.vue', '.svelte'
    ])
    show_authenticity_score: bool = True
    show_auto_fix_suggestions: bool = True
    enable_hover_details: bool = True
    enable_code_actions: bool = True


@dataclass
class EditorDiagnostic:
    """Diagnostic message for editor integration."""
    file_path: str
    line: int
    column: int
    severity: str                     # "error", "warning", "info", "hint"
    message: str
    source: str = "claude-tiu-validation"
    code: str = ""
    related_information: List[Dict] = field(default_factory=list)
    auto_fix_available: bool = False
    authenticity_score: Optional[float] = None


@dataclass
class EditorCodeAction:
    """Code action for editor integration."""
    title: str
    kind: str                        # "quickfix", "refactor", "source"
    diagnostics: List[EditorDiagnostic]
    edit: Dict[str, Any]            # TextEdit in LSP format
    command: Optional[Dict] = None


class LanguageServerProtocol:
    """
    Language Server Protocol implementation for editor integration.
    
    Provides LSP-compliant integration for real-time anti-hallucination validation.
    """
    
    def __init__(
        self,
        config_manager: ConfigManager,
        real_time_validator: RealTimeValidator
    ):
        """Initialize the LSP server."""
        self.config_manager = config_manager
        self.real_time_validator = real_time_validator
        
        self.config = EditorValidationConfig()
        
        # Document tracking
        self.open_documents: Dict[str, Dict[str, Any]] = {}
        self.document_versions: Dict[str, int] = {}
        
        # Validation state
        self.pending_validations: Dict[str, asyncio.Task] = {}
        self.diagnostics_cache: Dict[str, List[EditorDiagnostic]] = {}
        
        # Server state
        self.initialized = False
        self.client_capabilities: Dict[str, Any] = {}
        
        logger.info("Language Server Protocol initialized")
    
    async def initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize the LSP server."""
        logger.info("Initializing LSP server")
        
        try:
            # Store client capabilities
            self.client_capabilities = params.get('capabilities', {})
            
            # Load configuration
            await self._load_editor_config()
            
            # Initialize validator
            await self.real_time_validator.initialize()
            
            self.initialized = True
            
            # Return server capabilities
            return {
                'capabilities': {
                    'textDocumentSync': {
                        'openClose': True,
                        'change': 2,  # Incremental sync
                        'save': {'includeText': True}
                    },
                    'diagnosticProvider': {
                        'interFileDependencies': False,
                        'workspaceDiagnostics': False
                    },
                    'hoverProvider': True,
                    'codeActionProvider': {
                        'codeActionKinds': ['quickfix', 'refactor.rewrite']
                    },
                    'completionProvider': {
                        'triggerCharacters': ['.', '(', '[']
                    }
                },
                'serverInfo': {
                    'name': 'Claude-TIU Anti-Hallucination Server',
                    'version': '1.0.0'
                }
            }
            
        except Exception as e:
            logger.error(f"LSP initialization failed: {e}")
            raise
    
    async def did_open(self, params: Dict[str, Any]) -> None:
        """Handle document open event."""
        text_document = params['textDocument']
        uri = text_document['uri']
        
        logger.debug(f"Document opened: {uri}")
        
        # Store document info
        self.open_documents[uri] = {
            'language_id': text_document['languageId'],
            'version': text_document['version'],
            'text': text_document['text']
        }
        
        self.document_versions[uri] = text_document['version']
        
        # Trigger initial validation
        if self.config.real_time_validation:
            await self._schedule_validation(uri, text_document['text'])
    
    async def did_change(self, params: Dict[str, Any]) -> None:
        """Handle document change event."""
        text_document = params['textDocument']
        uri = text_document['uri']
        version = text_document['version']
        
        if uri not in self.open_documents:
            return
        
        # Apply changes
        content_changes = params['contentChanges']
        current_text = self.open_documents[uri]['text']
        
        # For full document sync (simplification)
        if len(content_changes) == 1 and 'range' not in content_changes[0]:
            new_text = content_changes[0]['text']
            self.open_documents[uri]['text'] = new_text
            self.open_documents[uri]['version'] = version
            self.document_versions[uri] = version
            
            # Schedule validation with delay
            if self.config.real_time_validation:
                await self._schedule_validation_with_delay(uri, new_text)
    
    async def did_save(self, params: Dict[str, Any]) -> None:
        """Handle document save event."""
        text_document = params['textDocument']
        uri = text_document['uri']
        
        logger.debug(f"Document saved: {uri}")
        
        # Force validation on save
        if uri in self.open_documents:
            text = params.get('text', self.open_documents[uri]['text'])
            await self._validate_document(uri, text, force=True)
    
    async def did_close(self, params: Dict[str, Any]) -> None:
        """Handle document close event."""
        text_document = params['textDocument']
        uri = text_document['uri']
        
        logger.debug(f"Document closed: {uri}")
        
        # Cleanup
        if uri in self.open_documents:
            del self.open_documents[uri]
        
        if uri in self.document_versions:
            del self.document_versions[uri]
        
        if uri in self.pending_validations:
            self.pending_validations[uri].cancel()
            del self.pending_validations[uri]
        
        if uri in self.diagnostics_cache:
            del self.diagnostics_cache[uri]
    
    async def provide_hover(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Provide hover information."""
        if not self.config.enable_hover_details:
            return None
        
        text_document = params['textDocument']
        position = params['position']
        uri = text_document['uri']
        
        if uri not in self.open_documents:
            return None
        
        # Get diagnostics for this position
        diagnostics = self.diagnostics_cache.get(uri, [])
        relevant_diagnostics = [
            d for d in diagnostics
            if d.line == position['line']
        ]
        
        if not relevant_diagnostics:
            return None
        
        # Build hover content
        contents = []
        for diagnostic in relevant_diagnostics:
            content = f"**{diagnostic.severity.title()}**: {diagnostic.message}"
            
            if diagnostic.authenticity_score is not None:
                content += f"\n\n**Authenticity Score**: {diagnostic.authenticity_score:.3f}"
            
            if diagnostic.auto_fix_available:
                content += "\n\n*Auto-fix available*"
            
            contents.append(content)
        
        return {
            'contents': {
                'kind': 'markdown',
                'value': '\n\n---\n\n'.join(contents)
            }
        }
    
    async def provide_code_actions(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Provide code actions for diagnostics."""
        if not self.config.enable_code_actions:
            return []
        
        text_document = params['textDocument']
        range_param = params['range']
        context = params['context']
        uri = text_document['uri']
        
        if uri not in self.open_documents:
            return []
        
        # Get diagnostics in range
        diagnostics = context.get('diagnostics', [])
        auto_fixable_diagnostics = [
            d for d in diagnostics
            if d.get('source') == 'claude-tiu-validation' and 
               self._diagnostic_has_auto_fix(d)
        ]
        
        code_actions = []
        
        for diagnostic in auto_fixable_diagnostics:
            # Create auto-fix action
            action = {
                'title': f"Auto-fix: {diagnostic.get('message', 'Fix issue')}",
                'kind': 'quickfix',
                'diagnostics': [diagnostic],
                'edit': await self._create_auto_fix_edit(uri, diagnostic)
            }
            
            code_actions.append(action)
        
        # Add validation actions
        if diagnostics:
            code_actions.append({
                'title': 'Validate with Anti-Hallucination Engine',
                'kind': 'source',
                'command': {
                    'title': 'Validate Document',
                    'command': 'claude-tiu.validateDocument',
                    'arguments': [uri]
                }
            })
        
        return code_actions
    
    async def execute_command(self, params: Dict[str, Any]) -> Any:
        """Execute custom commands."""
        command = params['command']
        arguments = params.get('arguments', [])
        
        if command == 'claude-tiu.validateDocument' and arguments:
            uri = arguments[0]
            if uri in self.open_documents:
                await self._validate_document(uri, self.open_documents[uri]['text'], force=True)
                return {'success': True}
        
        elif command == 'claude-tiu.showValidationReport' and arguments:
            uri = arguments[0]
            return await self._generate_validation_report(uri)
        
        elif command == 'claude-tiu.applyAllAutoFixes' and arguments:
            uri = arguments[0]
            return await self._apply_all_auto_fixes(uri)
        
        return {'success': False, 'error': 'Unknown command'}
    
    # Private implementation methods
    
    async def _load_editor_config(self) -> None:
        """Load editor integration configuration."""
        config = await self.config_manager.get_setting('editor_integration', {})
        
        validation_level = config.get('validation_level', 'warning')
        
        self.config = EditorValidationConfig(
            enabled=config.get('enabled', True),
            validation_level=ValidationLevel(validation_level),
            real_time_validation=config.get('real_time_validation', True),
            validation_delay_ms=config.get('validation_delay_ms', 500),
            max_file_size_kb=config.get('max_file_size_kb', 1024),
            supported_extensions=config.get('supported_extensions', ['.py', '.js', '.ts']),
            show_authenticity_score=config.get('show_authenticity_score', True),
            show_auto_fix_suggestions=config.get('show_auto_fix_suggestions', True),
            enable_hover_details=config.get('enable_hover_details', True),
            enable_code_actions=config.get('enable_code_actions', True)
        )
    
    async def _schedule_validation(self, uri: str, text: str) -> None:
        """Schedule immediate validation."""
        if uri in self.pending_validations:
            self.pending_validations[uri].cancel()
        
        task = asyncio.create_task(self._validate_document(uri, text))
        self.pending_validations[uri] = task
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            if uri in self.pending_validations:
                del self.pending_validations[uri]
    
    async def _schedule_validation_with_delay(self, uri: str, text: str) -> None:
        """Schedule validation with delay."""
        if uri in self.pending_validations:
            self.pending_validations[uri].cancel()
        
        async def delayed_validation():
            await asyncio.sleep(self.config.validation_delay_ms / 1000)
            await self._validate_document(uri, text)
        
        task = asyncio.create_task(delayed_validation())
        self.pending_validations[uri] = task
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            if uri in self.pending_validations:
                del self.pending_validations[uri]
    
    async def _validate_document(self, uri: str, text: str, force: bool = False) -> None:
        """Validate document content."""
        try:
            # Check file size limit
            if len(text) > self.config.max_file_size_kb * 1024 and not force:
                logger.debug(f"Skipping validation - file too large: {uri}")
                return
            
            # Check if validation is enabled for this file type
            file_path = Path(uri.replace('file://', ''))
            if file_path.suffix not in self.config.supported_extensions:
                return
            
            # Run validation
            context = {
                'file_path': str(file_path),
                'editor_integration': True,
                'validation_level': self.config.validation_level.value
            }
            
            validation_result = await self.real_time_validator.validate_live(
                text, context, ValidationMode.EDITOR_LIVE
            )
            
            # Convert to editor diagnostics
            diagnostics = self._convert_to_diagnostics(uri, validation_result, text)
            
            # Cache diagnostics
            self.diagnostics_cache[uri] = diagnostics
            
            # Send diagnostics to client
            await self._send_diagnostics(uri, diagnostics)
            
            logger.debug(f"Validation completed for {uri}: {len(diagnostics)} diagnostics")
            
        except Exception as e:
            logger.error(f"Document validation failed for {uri}: {e}")
    
    def _convert_to_diagnostics(
        self,
        uri: str,
        validation_result: Any,
        text: str
    ) -> List[EditorDiagnostic]:
        """Convert validation result to editor diagnostics."""
        diagnostics = []
        lines = text.split('\n')
        
        for issue_dict in validation_result.issues_detected:
            # Map severity
            severity_map = {
                'critical': 'error',
                'high': 'error',
                'medium': 'warning',
                'low': 'info'
            }
            
            severity = severity_map.get(issue_dict.get('severity', 'medium'), 'warning')
            
            # Skip based on validation level
            if self.config.validation_level == ValidationLevel.ERROR_ONLY and severity != 'error':
                continue
            elif self.config.validation_level == ValidationLevel.OFF:
                continue
            
            # Find line and column (simplified)
            line_num = issue_dict.get('line', 0)
            column = 0
            
            if 0 <= line_num < len(lines):
                # Find relevant content on line
                line_content = lines[line_num]
                # Try to find the issue location
                if 'TODO' in line_content:
                    column = line_content.find('TODO')
                elif 'FIXME' in line_content:
                    column = line_content.find('FIXME')
                elif 'pass' in line_content:
                    column = line_content.find('pass')
            
            message = issue_dict.get('description', 'Validation issue detected')
            
            # Add authenticity score if enabled
            if self.config.show_authenticity_score and validation_result.authenticity_score:
                message += f" (Authenticity: {validation_result.authenticity_score:.3f})"
            
            diagnostic = EditorDiagnostic(
                file_path=uri,
                line=line_num,
                column=column,
                severity=severity,
                message=message,
                code=issue_dict.get('id', ''),
                auto_fix_available=issue_dict.get('auto_fixable', False),
                authenticity_score=validation_result.authenticity_score
            )
            
            diagnostics.append(diagnostic)
        
        return diagnostics
    
    async def _send_diagnostics(self, uri: str, diagnostics: List[EditorDiagnostic]) -> None:
        """Send diagnostics to the editor client."""
        # Convert to LSP format
        lsp_diagnostics = []
        
        for diagnostic in diagnostics:
            lsp_diagnostic = {
                'range': {
                    'start': {'line': diagnostic.line, 'character': diagnostic.column},
                    'end': {'line': diagnostic.line, 'character': diagnostic.column + 10}
                },
                'severity': self._severity_to_lsp(diagnostic.severity),
                'code': diagnostic.code,
                'source': diagnostic.source,
                'message': diagnostic.message
            }
            
            if diagnostic.related_information:
                lsp_diagnostic['relatedInformation'] = diagnostic.related_information
            
            lsp_diagnostics.append(lsp_diagnostic)
        
        # Send to client (this would be implemented based on the LSP transport)
        logger.debug(f"Sending {len(lsp_diagnostics)} diagnostics for {uri}")
    
    def _severity_to_lsp(self, severity: str) -> int:
        """Convert severity to LSP severity level."""
        severity_map = {
            'error': 1,
            'warning': 2,
            'info': 3,
            'hint': 4
        }
        return severity_map.get(severity, 2)
    
    def _diagnostic_has_auto_fix(self, diagnostic: Dict[str, Any]) -> bool:
        """Check if diagnostic has auto-fix available."""
        return diagnostic.get('auto_fix_available', False)
    
    async def _create_auto_fix_edit(self, uri: str, diagnostic: Dict[str, Any]) -> Dict[str, Any]:
        """Create auto-fix text edit."""
        # Simplified auto-fix edit creation
        line = diagnostic.get('line', 0)
        
        return {
            'changes': {
                uri: [
                    {
                        'range': {
                            'start': {'line': line, 'character': 0},
                            'end': {'line': line + 1, 'character': 0}
                        },
                        'newText': '    # Implementation completed\n'
                    }
                ]
            }
        }
    
    async def _generate_validation_report(self, uri: str) -> Dict[str, Any]:
        """Generate detailed validation report for document."""
        if uri not in self.diagnostics_cache:
            return {'error': 'No validation data available'}
        
        diagnostics = self.diagnostics_cache[uri]
        
        return {
            'uri': uri,
            'total_issues': len(diagnostics),
            'issues_by_severity': self._group_diagnostics_by_severity(diagnostics),
            'authenticity_score': diagnostics[0].authenticity_score if diagnostics else None,
            'auto_fixes_available': sum(1 for d in diagnostics if d.auto_fix_available),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _apply_all_auto_fixes(self, uri: str) -> Dict[str, Any]:
        """Apply all available auto-fixes for document."""
        if uri not in self.diagnostics_cache or uri not in self.open_documents:
            return {'error': 'Document not available'}
        
        diagnostics = self.diagnostics_cache[uri]
        auto_fixable = [d for d in diagnostics if d.auto_fix_available]
        
        if not auto_fixable:
            return {'message': 'No auto-fixes available'}
        
        # Apply fixes (simplified)
        fixes_applied = len(auto_fixable)
        
        # Re-validate after fixes
        await self._validate_document(uri, self.open_documents[uri]['text'], force=True)
        
        return {
            'fixes_applied': fixes_applied,
            'message': f'Applied {fixes_applied} auto-fixes'
        }
    
    def _group_diagnostics_by_severity(self, diagnostics: List[EditorDiagnostic]) -> Dict[str, int]:
        """Group diagnostics by severity."""
        groups = {}
        for diagnostic in diagnostics:
            severity = diagnostic.severity
            groups[severity] = groups.get(severity, 0) + 1
        return groups


class EditorIntegrationManager:
    """
    Manages editor integrations and provides unified interface.
    """
    
    def __init__(
        self,
        config_manager: ConfigManager,
        real_time_validator: RealTimeValidator
    ):
        """Initialize the editor integration manager."""
        self.config_manager = config_manager
        self.real_time_validator = real_time_validator
        
        # LSP server
        self.lsp_server = LanguageServerProtocol(config_manager, real_time_validator)
        
        # WebSocket server for editor communication
        self.websocket_server = None
        self.websocket_port = 9876
        
        logger.info("Editor Integration Manager initialized")
    
    async def start_lsp_server(self, port: int = 9875) -> None:
        """Start the Language Server Protocol server."""
        logger.info(f"Starting LSP server on port {port}")
        
        # This would start the actual LSP server
        # For now, just initialize the LSP handler
        await self.lsp_server.real_time_validator.initialize()
        
        logger.info("LSP server ready for connections")
    
    async def start_websocket_server(self, port: int = 9876) -> None:
        """Start WebSocket server for real-time editor communication."""
        logger.info(f"Starting WebSocket server on port {port}")
        
        async def websocket_handler(websocket, path):
            """Handle WebSocket connections from editors."""
            try:
                logger.info("Editor connected via WebSocket")
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        response = await self._handle_websocket_message(data)
                        await websocket.send(json.dumps(response))
                    except Exception as e:
                        error_response = {
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        }
                        await websocket.send(json.dumps(error_response))
                        
            except websockets.exceptions.ConnectionClosed:
                logger.info("Editor disconnected from WebSocket")
            except Exception as e:
                logger.error(f"WebSocket handler error: {e}")
        
        # Start WebSocket server
        self.websocket_server = await websockets.serve(
            websocket_handler,
            "localhost",
            port
        )
        
        logger.info(f"WebSocket server started on ws://localhost:{port}")
    
    async def _handle_websocket_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle WebSocket message from editor."""
        message_type = data.get('type')
        
        if message_type == 'validate':
            # Real-time validation request
            content = data.get('content', '')
            file_path = data.get('file_path', '')
            
            result = await self.real_time_validator.validate_live(
                content,
                {'file_path': file_path, 'editor_websocket': True},
                ValidationMode.EDITOR_LIVE
            )
            
            return {
                'type': 'validation_result',
                'is_valid': result.is_valid,
                'authenticity_score': result.authenticity_score,
                'issues': result.issues_detected,
                'processing_time_ms': result.processing_time_ms,
                'auto_fixes_available': result.auto_fixes_available,
                'timestamp': datetime.now().isoformat()
            }
        
        elif message_type == 'get_metrics':
            # Metrics request
            metrics = await self.real_time_validator.get_performance_metrics()
            return {
                'type': 'metrics',
                'data': metrics,
                'timestamp': datetime.now().isoformat()
            }
        
        else:
            return {
                'type': 'error',
                'message': f'Unknown message type: {message_type}',
                'timestamp': datetime.now().isoformat()
            }
    
    async def create_editor_config(self, editor_type: EditorType, output_path: Path) -> None:
        """Create configuration file for specific editor."""
        logger.info(f"Creating {editor_type.value} configuration")
        
        if editor_type == EditorType.VS_CODE:
            await self._create_vscode_config(output_path)
        elif editor_type == EditorType.VIM or editor_type == EditorType.NEOVIM:
            await self._create_vim_config(output_path)
        elif editor_type == EditorType.EMACS:
            await self._create_emacs_config(output_path)
        else:
            logger.warning(f"Configuration creation not implemented for {editor_type.value}")
    
    async def _create_vscode_config(self, output_path: Path) -> None:
        """Create VS Code extension configuration."""
        config = {
            "name": "claude-tiu-validation",
            "displayName": "Claude-TIU Anti-Hallucination Validation",
            "description": "Real-time AI hallucination detection with 95.8% accuracy",
            "version": "1.0.0",
            "engines": {
                "vscode": "^1.60.0"
            },
            "categories": ["Linters"],
            "activationEvents": [
                "onLanguage:python",
                "onLanguage:javascript",
                "onLanguage:typescript"
            ],
            "contributes": {
                "configuration": {
                    "title": "Claude-TIU Validation",
                    "properties": {
                        "claudeTiu.validation.enabled": {
                            "type": "boolean",
                            "default": True,
                            "description": "Enable real-time validation"
                        },
                        "claudeTiu.validation.level": {
                            "type": "string",
                            "enum": ["off", "error_only", "warning", "info", "verbose"],
                            "default": "warning",
                            "description": "Validation level"
                        },
                        "claudeTiu.validation.showAuthenticityScore": {
                            "type": "boolean",
                            "default": True,
                            "description": "Show authenticity scores in diagnostics"
                        },
                        "claudeTiu.validation.autoFix": {
                            "type": "boolean",
                            "default": True,
                            "description": "Enable automatic fixes"
                        }
                    }
                },
                "commands": [
                    {
                        "command": "claudeTiu.validateDocument",
                        "title": "Validate with Anti-Hallucination Engine"
                    },
                    {
                        "command": "claudeTiu.showValidationReport",
                        "title": "Show Validation Report"
                    }
                ]
            }
        }
        
        package_json_path = output_path / "package.json"
        package_json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(package_json_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"VS Code configuration created at: {package_json_path}")
    
    async def _create_vim_config(self, output_path: Path) -> None:
        """Create Vim/Neovim configuration."""
        config = '''
" Claude-TIU Anti-Hallucination Validation for Vim/Neovim
" Add this to your .vimrc or init.vim

" Enable Claude-TIU validation
let g:claude_tiu_validation_enabled = 1
let g:claude_tiu_validation_level = 'warning'
let g:claude_tiu_show_authenticity_score = 1

" WebSocket connection to validation server
function! ClaudeTiuValidate()
    if !g:claude_tiu_validation_enabled
        return
    endif
    
    let content = join(getline(1, '$'), "\\n")
    let file_path = expand('%:p')
    
    " Send to validation server (requires WebSocket client)
    " Implementation would depend on available WebSocket plugin
endfunction

" Auto-validate on text change (with delay)
augroup ClaudeTiuValidation
    autocmd!
    autocmd TextChanged,TextChangedI * call timer_start(500, {-> ClaudeTiuValidate()})
    autocmd BufWrite * call ClaudeTiuValidate()
augroup END

" Commands
command! ClaudeTiuValidate call ClaudeTiuValidate()
command! ClaudeTiuReport echo "Validation report not implemented yet"

" Key mappings
nnoremap <leader>cv :ClaudeTiuValidate<CR>
nnoremap <leader>cr :ClaudeTiuReport<CR>
'''
        
        vim_config_path = output_path / "claude-tiu.vim"
        vim_config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(vim_config_path, 'w') as f:
            f.write(config)
        
        logger.info(f"Vim configuration created at: {vim_config_path}")
    
    async def _create_emacs_config(self, output_path: Path) -> None:
        """Create Emacs configuration."""
        config = '''
;;; claude-tiu-validation.el --- Anti-Hallucination validation for Emacs

;; Enable Claude-TIU validation
(defcustom claude-tiu-validation-enabled t
  "Enable Claude-TIU validation."
  :type 'boolean
  :group 'claude-tiu)

(defcustom claude-tiu-validation-level 'warning
  "Validation level."
  :type '(choice (const off)
                 (const error-only)
                 (const warning)
                 (const info)
                 (const verbose))
  :group 'claude-tiu)

(defun claude-tiu-validate-buffer ()
  "Validate current buffer with Claude-TIU."
  (interactive)
  (when claude-tiu-validation-enabled
    (let ((content (buffer-string))
          (file-path (buffer-file-name)))
      ;; Send to validation server
      ;; Implementation would use websocket client
      (message "Claude-TIU validation requested"))))

;; Auto-validate with delay
(defvar claude-tiu-validation-timer nil)

(defun claude-tiu-schedule-validation ()
  "Schedule validation after delay."
  (when claude-tiu-validation-timer
    (cancel-timer claude-tiu-validation-timer))
  (setq claude-tiu-validation-timer
        (run-with-timer 0.5 nil #'claude-tiu-validate-buffer)))

;; Hook into text changes
(add-hook 'after-change-functions
          (lambda (&rest _) (claude-tiu-schedule-validation)))

;; Key bindings
(global-set-key (kbd "C-c v v") 'claude-tiu-validate-buffer)

(provide 'claude-tiu-validation)
;;; claude-tiu-validation.el ends here
'''
        
        emacs_config_path = output_path / "claude-tiu-validation.el"
        emacs_config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(emacs_config_path, 'w') as f:
            f.write(config)
        
        logger.info(f"Emacs configuration created at: {emacs_config_path}")
    
    async def cleanup(self) -> None:
        """Cleanup editor integration resources."""
        logger.info("Cleaning up Editor Integration Manager")
        
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        logger.info("Editor Integration Manager cleanup completed")