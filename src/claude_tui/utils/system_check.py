"""
System Checker - Environment validation and diagnostics.

Performs comprehensive system checks to ensure all dependencies and 
configurations are properly set up before launching the application.
"""

import asyncio
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

import psutil

logger = logging.getLogger(__name__)


@dataclass
class SystemCheck:
    """Represents a single system check."""
    name: str
    passed: bool
    message: str
    category: str = "general"
    severity: str = "error"  # error, warning, info
    recommendation: Optional[str] = None


@dataclass
class SystemCheckResult:
    """Results from running system checks."""
    all_passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    categories: Dict[str, List[SystemCheck]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    system_info: Dict[str, Any] = field(default_factory=dict)


class SystemChecker:
    """Comprehensive system checker for Claude TUI requirements."""
    
    def __init__(self):
        self.checks: List[SystemCheck] = []
        self.required_python_version = (3, 9)
        self.recommended_memory_gb = 4
        self.required_disk_space_gb = 1
        
    async def run_checks(self) -> SystemCheckResult:
        """Run basic system checks."""
        self.checks.clear()
        
        # Basic checks
        await self._check_python_version()
        await self._check_memory()
        await self._check_disk_space()
        await self._check_environment_variables()
        await self._check_network()
        
        return self._compile_results()
    
    async def run_comprehensive_check(self) -> SystemCheckResult:
        """Run comprehensive system checks including dependencies."""
        self.checks.clear()
        
        # All basic checks
        await self.run_checks()
        
        # Additional comprehensive checks
        await self._check_python_packages()
        await self._check_git()
        await self._check_node_js()
        await self._check_claude_flow()
        await self._check_permissions()
        await self._check_configuration()
        
        return self._compile_results()
    
    async def _check_python_version(self):
        """Check Python version compatibility."""
        current_version = sys.version_info[:2]
        required = self.required_python_version
        
        if current_version >= required:
            self.checks.append(SystemCheck(
                name="Python Version",
                passed=True,
                message=f"Python {current_version[0]}.{current_version[1]} (required: {required[0]}.{required[1]}+)",
                category="runtime"
            ))
        else:
            self.checks.append(SystemCheck(
                name="Python Version", 
                passed=False,
                message=f"Python {current_version[0]}.{current_version[1]} is too old (required: {required[0]}.{required[1]}+)",
                category="runtime",
                recommendation="Upgrade Python to version 3.9 or higher"
            ))
    
    async def _check_memory(self):
        """Check available system memory."""
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            total_gb = memory.total / (1024**3)
            
            if available_gb >= self.recommended_memory_gb:
                self.checks.append(SystemCheck(
                    name="Available Memory",
                    passed=True,
                    message=f"{available_gb:.1f}GB available of {total_gb:.1f}GB total",
                    category="resources"
                ))
            else:
                severity = "warning" if available_gb >= 2 else "error"
                self.checks.append(SystemCheck(
                    name="Available Memory",
                    passed=available_gb >= 2,  # Minimum 2GB
                    message=f"Only {available_gb:.1f}GB available of {total_gb:.1f}GB total (recommended: {self.recommended_memory_gb}GB)",
                    category="resources",
                    severity=severity,
                    recommendation="Close other applications to free up memory or add more RAM"
                ))
        except Exception as e:
            self.checks.append(SystemCheck(
                name="Memory Check",
                passed=False,
                message=f"Failed to check memory: {e}",
                category="resources",
                severity="warning"
            ))
    
    async def _check_disk_space(self):
        """Check available disk space."""
        try:
            current_dir = Path.cwd()
            disk_usage = shutil.disk_usage(current_dir)
            available_gb = disk_usage.free / (1024**3)
            
            if available_gb >= self.required_disk_space_gb:
                self.checks.append(SystemCheck(
                    name="Disk Space",
                    passed=True,
                    message=f"{available_gb:.1f}GB available",
                    category="resources"
                ))
            else:
                self.checks.append(SystemCheck(
                    name="Disk Space",
                    passed=False,
                    message=f"Only {available_gb:.1f}GB available (required: {self.required_disk_space_gb}GB)",
                    category="resources",
                    recommendation="Free up disk space before proceeding"
                ))
        except Exception as e:
            self.checks.append(SystemCheck(
                name="Disk Space Check",
                passed=False,
                message=f"Failed to check disk space: {e}",
                category="resources",
                severity="warning"
            ))
    
    async def _check_environment_variables(self):
        """Check required environment variables."""
        required_vars = ['CLAUDE_CODE_OAUTH_TOKEN']
        optional_vars = ['GITHUB_TOKEN', 'GITHUB_USER', 'GITHUB_REPO']
        
        for var in required_vars:
            value = os.getenv(var)
            if value:
                # Don't log the actual token value
                masked_value = f"{value[:8]}..." if len(value) > 8 else "***"
                self.checks.append(SystemCheck(
                    name=f"Environment Variable: {var}",
                    passed=True,
                    message=f"Set ({masked_value})",
                    category="configuration"
                ))
            else:
                self.checks.append(SystemCheck(
                    name=f"Environment Variable: {var}",
                    passed=False,
                    message="Not set",
                    category="configuration",
                    recommendation=f"Set {var} environment variable for AI functionality"
                ))
        
        for var in optional_vars:
            value = os.getenv(var)
            if value:
                masked_value = f"{value[:8]}..." if len(value) > 8 else "***"
                self.checks.append(SystemCheck(
                    name=f"Optional: {var}",
                    passed=True,
                    message=f"Set ({masked_value})",
                    category="configuration",
                    severity="info"
                ))
            else:
                self.checks.append(SystemCheck(
                    name=f"Optional: {var}",
                    passed=True,  # Optional, so always "passes"
                    message="Not set (optional)",
                    category="configuration", 
                    severity="info",
                    recommendation=f"Set {var} for enhanced GitHub integration"
                ))
    
    async def _check_network(self):
        """Check network connectivity."""
        try:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=5)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                try:
                    async with session.get('https://httpbin.org/status/200') as response:
                        if response.status == 200:
                            self.checks.append(SystemCheck(
                                name="Network Connectivity",
                                passed=True,
                                message="Internet connection available",
                                category="network"
                            ))
                        else:
                            self.checks.append(SystemCheck(
                                name="Network Connectivity",
                                passed=False,
                                message=f"HTTP request failed with status {response.status}",
                                category="network",
                                severity="warning",
                                recommendation="Check internet connection for AI services"
                            ))
                except asyncio.TimeoutError:
                    self.checks.append(SystemCheck(
                        name="Network Connectivity",
                        passed=False,
                        message="Connection timeout",
                        category="network",
                        severity="warning",
                        recommendation="Check internet connection and firewall settings"
                    ))
        except Exception as e:
            self.checks.append(SystemCheck(
                name="Network Check",
                passed=False,
                message=f"Failed to test network: {e}",
                category="network",
                severity="warning"
            ))
    
    async def _check_python_packages(self):
        """Check required Python packages."""
        required_packages = {
            'textual': 'TUI framework',
            'rich': 'Rich text formatting',
            'click': 'CLI interface',
            'pydantic': 'Data validation',
            'aiohttp': 'HTTP client',
            'pyyaml': 'YAML configuration'
        }
        
        for package, description in required_packages.items():
            try:
                __import__(package)
                self.checks.append(SystemCheck(
                    name=f"Package: {package}",
                    passed=True,
                    message=f"Installed ({description})",
                    category="dependencies"
                ))
            except ImportError:
                self.checks.append(SystemCheck(
                    name=f"Package: {package}",
                    passed=False,
                    message=f"Not installed ({description})",
                    category="dependencies",
                    recommendation=f"Install {package}: pip install {package}"
                ))
    
    async def _check_git(self):
        """Check Git installation and configuration."""
        try:
            result = subprocess.run(['git', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                self.checks.append(SystemCheck(
                    name="Git",
                    passed=True,
                    message=f"Installed ({version})",
                    category="tools"
                ))
            else:
                self.checks.append(SystemCheck(
                    name="Git",
                    passed=False,
                    message="Command failed",
                    category="tools",
                    recommendation="Install Git for version control features"
                ))
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.checks.append(SystemCheck(
                name="Git",
                passed=False,
                message="Not found",
                category="tools",
                severity="warning",
                recommendation="Install Git for version control features"
            ))
    
    async def _check_node_js(self):
        """Check Node.js installation for Claude Flow."""
        try:
            result = subprocess.run(['node', '--version'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                self.checks.append(SystemCheck(
                    name="Node.js",
                    passed=True,
                    message=f"Installed ({version})",
                    category="tools"
                ))
            else:
                self.checks.append(SystemCheck(
                    name="Node.js",
                    passed=False,
                    message="Command failed",
                    category="tools",
                    severity="warning",
                    recommendation="Install Node.js 16+ for Claude Flow integration"
                ))
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.checks.append(SystemCheck(
                name="Node.js",
                passed=False,
                message="Not found",
                category="tools",
                severity="warning",
                recommendation="Install Node.js 16+ for Claude Flow integration"
            ))
    
    async def _check_claude_flow(self):
        """Check Claude Flow installation."""
        try:
            result = subprocess.run(['npx', 'claude-flow@alpha', '--version'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.checks.append(SystemCheck(
                    name="Claude Flow",
                    passed=True,
                    message="Available via npx",
                    category="tools"
                ))
            else:
                self.checks.append(SystemCheck(
                    name="Claude Flow",
                    passed=False,
                    message="Command failed",
                    category="tools",
                    severity="warning",
                    recommendation="Claude Flow will be installed automatically when needed"
                ))
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.checks.append(SystemCheck(
                name="Claude Flow",
                passed=False,
                message="Not available",
                category="tools", 
                severity="warning",
                recommendation="Claude Flow will be installed automatically when needed"
            ))
    
    async def _check_permissions(self):
        """Check file system permissions."""
        test_dir = Path.home() / ".claude-tui"
        
        try:
            test_dir.mkdir(exist_ok=True)
            test_file = test_dir / "permission_test.tmp"
            
            # Test write permission
            test_file.write_text("test", encoding='utf-8')
            
            # Test read permission  
            content = test_file.read_text(encoding='utf-8')
            
            # Clean up
            test_file.unlink()
            
            if content == "test":
                self.checks.append(SystemCheck(
                    name="File Permissions",
                    passed=True,
                    message="Read/write permissions OK",
                    category="permissions"
                ))
            else:
                self.checks.append(SystemCheck(
                    name="File Permissions",
                    passed=False,
                    message="File content mismatch",
                    category="permissions",
                    recommendation="Check file system integrity"
                ))
                
        except PermissionError:
            self.checks.append(SystemCheck(
                name="File Permissions",
                passed=False,
                message="Permission denied",
                category="permissions",
                recommendation="Check user permissions for home directory"
            ))
        except Exception as e:
            self.checks.append(SystemCheck(
                name="File Permissions",
                passed=False,
                message=f"Permission test failed: {e}",
                category="permissions",
                severity="warning"
            ))
    
    async def _check_configuration(self):
        """Check configuration file accessibility."""
        config_dir = Path.home() / ".claude-tui"
        config_file = config_dir / "config.yaml"
        
        try:
            if config_file.exists():
                # Try to read existing config
                content = config_file.read_text(encoding='utf-8')
                self.checks.append(SystemCheck(
                    name="Configuration File",
                    passed=True,
                    message=f"Found at {config_file}",
                    category="configuration"
                ))
            else:
                self.checks.append(SystemCheck(
                    name="Configuration File",
                    passed=True,  # It's OK if it doesn't exist
                    message="Will be created on first run",
                    category="configuration",
                    severity="info"
                ))
        except Exception as e:
            self.checks.append(SystemCheck(
                name="Configuration File",
                passed=False,
                message=f"Cannot access config: {e}",
                category="configuration",
                severity="warning",
                recommendation="Check permissions for configuration directory"
            ))
    
    def _compile_results(self) -> SystemCheckResult:
        """Compile all check results into a SystemCheckResult."""
        categories = {}
        errors = []
        warnings = []
        info = []
        recommendations = []
        
        all_passed = True
        
        for check in self.checks:
            # Categorize checks
            if check.category not in categories:
                categories[check.category] = []
            categories[check.category].append(check)
            
            # Collect messages by severity
            if not check.passed:
                all_passed = False
                
                if check.severity == "error":
                    errors.append(check.message)
                else:
                    warnings.append(check.message)
            elif check.severity == "info":
                info.append(check.message)
            
            # Collect recommendations
            if check.recommendation:
                recommendations.append(check.recommendation)
        
        # Gather system information
        system_info = {
            'platform': platform.platform(),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'hostname': platform.node(),
        }
        
        try:
            memory = psutil.virtual_memory()
            system_info.update({
                'total_memory_gb': round(memory.total / (1024**3), 2),
                'available_memory_gb': round(memory.available / (1024**3), 2),
                'cpu_count': psutil.cpu_count(),
                'cpu_percent': psutil.cpu_percent(interval=1)
            })
        except Exception:
            pass  # Skip if psutil data unavailable
        
        return SystemCheckResult(
            all_passed=all_passed,
            errors=errors,
            warnings=warnings,
            info=info,
            categories=categories,
            recommendations=list(set(recommendations)),  # Remove duplicates
            system_info=system_info
        )