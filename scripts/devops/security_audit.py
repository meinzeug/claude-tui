#!/usr/bin/env python3
"""
Automated Security Audit Script for claude-tiu
Performs comprehensive security scanning and generates reports.
"""

import os
import re
import ast
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SecurityFinding:
    """Security audit finding"""
    severity: str  # critical, high, medium, low
    category: str
    description: str
    file_path: str
    line_number: int
    evidence: str
    recommendation: str

class SecurityAuditor:
    """Comprehensive security auditing system"""
    
    def __init__(self, project_path: Path):
        self.project_path = Path(project_path)
        self.findings: List[SecurityFinding] = []
        
        # Security patterns to detect
        self.vulnerability_patterns = {
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']',
                r'sk-[a-zA-Z0-9]{40,}',  # API key pattern
                r'Bearer\s+[a-zA-Z0-9_\-]+',
            ],
            'command_injection': [
                r'subprocess\.(call|run|Popen)\([^)]*shell=True',
                r'os\.system\(',
                r'os\.popen\(',
                r'eval\(',
                r'exec\(',
                r'__import__\(',
            ],
            'path_traversal': [
                r'\.\./|\.\.\\\',
                r'os\.path\.join\([^)]*\.\.',
                r'open\([^)]*\.\.',
                r'Path\([^)]*\.\.',
            ],
            'insecure_random': [
                r'random\.random\(',
                r'random\.choice\(',
                r'random\.randint\(',
            ],
            'sql_injection': [
                r'execute\([^)]*%[sd]',
                r'cursor\.execute\([^)]*\+',
                r'SELECT.*WHERE.*%s',
                r'f".*SELECT.*{.*}"',
            ],
            'xss_vulnerabilities': [
                r'\.innerHTML\s*=',
                r'document\.write\(',
                r'eval\s*\(',
            ],
            'insecure_crypto': [
                r'md5\(',
                r'sha1\(',
                r'DES\(',
                r'RC4\(',
            ]
        }
    
    def run_full_audit(self) -> Dict[str, Any]:
        """Run comprehensive security audit"""
        logger.info("Starting security audit...")
        self.findings = []
        
        # Code analysis
        self._analyze_python_files()
        self._check_dependencies()
        self._audit_configurations()
        self._check_file_permissions()
        self._analyze_network_usage()
        self._check_docker_security()
        self._scan_github_workflows()
        
        # Generate report
        report = self._generate_audit_report()
        logger.info(f"Security audit complete. Found {len(self.findings)} issues.")
        return report
    
    def _analyze_python_files(self):
        """Analyze Python files for security issues"""
        logger.info("Analyzing Python files...")
        python_files = list(self.project_path.rglob("*.py"))
        
        for file_path in python_files:
            try:
                content = file_path.read_text(encoding='utf-8')
                self._scan_file_for_vulnerabilities(file_path, content)
                self._analyze_ast_security(file_path, content)
            except Exception as e:
                self.findings.append(SecurityFinding(
                    severity="medium",
                    category="file_analysis",
                    description=f"Failed to analyze file: {e}",
                    file_path=str(file_path),
                    line_number=0,
                    evidence="",
                    recommendation="Ensure file is readable and valid Python"
                ))
    
    def _scan_file_for_vulnerabilities(self, file_path: Path, content: str):
        """Scan file content for vulnerability patterns"""
        lines = content.split('\n')
        
        for category, patterns in self.vulnerability_patterns.items():
            for pattern in patterns:
                for line_num, line in enumerate(lines, 1):
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        # Skip test files and example patterns
                        if 'test' in str(file_path).lower() or 'example' in str(file_path).lower():
                            continue
                        if 'placeholder' in line.lower() or 'replace-with' in line.lower():
                            continue
                            
                        severity = self._get_severity_for_category(category)
                        self.findings.append(SecurityFinding(
                            severity=severity,
                            category=category,
                            description=f"Potential {category.replace('_', ' ')} detected",
                            file_path=str(file_path.relative_to(self.project_path)),
                            line_number=line_num,
                            evidence=line.strip()[:100],
                            recommendation=self._get_recommendation_for_category(category)
                        ))
    
    def _analyze_ast_security(self, file_path: Path, content: str):
        """AST-based security analysis"""
        try:
            tree = ast.parse(content)
            visitor = SecurityASTVisitor(file_path, self.project_path)
            visitor.visit(tree)
            self.findings.extend(visitor.findings)
        except SyntaxError:
            pass  # File has syntax errors, skip AST analysis
    
    def _check_dependencies(self):
        """Check for vulnerable dependencies"""
        logger.info("Checking dependencies...")
        requirements_files = [
            'requirements.txt', 'requirements-dev.txt', 
            'Pipfile', 'pyproject.toml', 'setup.py'
        ]
        
        for req_file in requirements_files:
            req_path = self.project_path / req_file
            if req_path.exists():
                self._analyze_requirements_file(req_path)
    
    def _analyze_requirements_file(self, req_path: Path):
        """Analyze requirements file for vulnerable packages"""
        # Known vulnerable packages (simplified list - use safety or pip-audit in production)
        vulnerable_packages = {
            'flask': ['0.12.0', '0.12.1', '0.12.2'],
            'django': ['1.11.0', '1.11.1', '1.11.2'],
            'requests': ['2.19.0', '2.19.1'],
            'pillow': ['5.2.0', '5.3.0'],
            'urllib3': ['1.24.1'],
        }
        
        try:
            content = req_path.read_text()
            for line_num, line in enumerate(content.split('\n'), 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    self._check_package_vulnerability(req_path, line, line_num, vulnerable_packages)
        except Exception:
            pass
    
    def _check_package_vulnerability(self, req_path: Path, line: str, line_num: int, 
                                   vulnerable_packages: Dict[str, List[str]]):
        """Check if package version is vulnerable"""
        # Parse package specification
        match = re.match(r'^([a-zA-Z0-9_-]+)([><=!]+)?([\d.]+)?', line)
        if match:
            package_name = match.group(1).lower()
            version = match.group(3)
            
            if package_name in vulnerable_packages and version:
                if version in vulnerable_packages[package_name]:
                    self.findings.append(SecurityFinding(
                        severity="high",
                        category="vulnerable_dependency",
                        description=f"Vulnerable package version: {package_name} {version}",
                        file_path=str(req_path.relative_to(self.project_path)),
                        line_number=line_num,
                        evidence=line,
                        recommendation=f"Update {package_name} to latest version"
                    ))
    
    def _audit_configurations(self):
        """Audit configuration files for security issues"""
        logger.info("Auditing configuration files...")
        config_patterns = [
            '*.json', '*.yaml', '*.yml', '*.ini', '*.cfg', '.env*', '*.toml'
        ]
        
        for pattern in config_patterns:
            for config_file in self.project_path.rglob(pattern):
                if config_file.is_file() and not any(skip in str(config_file) for skip in ['.git', '__pycache__', '.pytest_cache']):
                    self._audit_config_file(config_file)
    
    def _audit_config_file(self, config_file: Path):
        """Audit individual configuration file"""
        try:
            content = config_file.read_text()
            
            # Check for hardcoded secrets
            secret_patterns = [
                r'password.*[:=]\s*["\'][^"\']{8,}["\']',
                r'secret.*[:=]\s*["\'][^"\']{20,}["\']',
                r'key.*[:=]\s*["\'][^"\']{20,}["\']',
                r'token.*[:=]\s*["\'][^"\']{20,}["\']',
            ]
            
            for pattern in secret_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Skip template/placeholder values
                    if any(placeholder in match.group(0).lower() for placeholder in 
                          ['placeholder', 'replace-with', 'your-', 'example', 'xxx']):
                        continue
                        
                    line_num = content[:match.start()].count('\n') + 1
                    self.findings.append(SecurityFinding(
                        severity="critical",
                        category="hardcoded_secrets",
                        description="Potential hardcoded secret in configuration file",
                        file_path=str(config_file.relative_to(self.project_path)),
                        line_number=line_num,
                        evidence=match.group(0)[:50] + "...",
                        recommendation="Use environment variables or secure key management"
                    ))
                    
        except Exception:
            pass
    
    def _check_file_permissions(self):
        """Check file permissions for security issues"""
        logger.info("Checking file permissions...")
        for file_path in self.project_path.rglob("*"):
            if file_path.is_file():
                try:
                    stat = file_path.stat()
                    mode = stat.st_mode
                    
                    # Check for world-writable files
                    if mode & 0o002:
                        self.findings.append(SecurityFinding(
                            severity="medium",
                            category="file_permissions",
                            description="World-writable file detected",
                            file_path=str(file_path.relative_to(self.project_path)),
                            line_number=0,
                            evidence=f"Permissions: {oct(mode)[-3:]}",
                            recommendation="Remove world write permissions"
                        ))
                    
                    # Check for executable config files
                    if file_path.suffix in ['.json', '.yaml', '.yml', '.ini'] and mode & 0o111:
                        self.findings.append(SecurityFinding(
                            severity="low",
                            category="file_permissions",
                            description="Executable configuration file",
                            file_path=str(file_path.relative_to(self.project_path)),
                            line_number=0,
                            evidence=f"Permissions: {oct(mode)[-3:]}",
                            recommendation="Remove execute permissions from config files"
                        ))
                        
                except Exception:
                    pass
    
    def _analyze_network_usage(self):
        """Analyze network-related code for security issues"""
        logger.info("Analyzing network usage...")
        network_patterns = [
            (r'requests\.get\([^)]*verify=False', 'SSL verification disabled'),
            (r'urllib\.request\.urlopen\([^)]*http://', 'Insecure HTTP usage'),
            (r'socket\.socket\(.*SOCK_RAW', 'Raw socket usage'),
            (r'ssl\._create_unverified_context', 'Unverified SSL context'),
        ]
        
        for py_file in self.project_path.rglob("*.py"):
            try:
                content = py_file.read_text()
                for pattern, description in network_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        self.findings.append(SecurityFinding(
                            severity="medium",
                            category="network_security",
                            description=description,
                            file_path=str(py_file.relative_to(self.project_path)),
                            line_number=line_num,
                            evidence=match.group(0)[:50],
                            recommendation="Use secure network communications"
                        ))
            except Exception:
                pass
    
    def _check_docker_security(self):
        """Check Docker configurations for security issues"""
        logger.info("Checking Docker security...")
        dockerfile_path = self.project_path / "Dockerfile"
        
        if dockerfile_path.exists():
            try:
                content = dockerfile_path.read_text()
                lines = content.split('\n')
                
                # Common Docker security issues
                docker_issues = [
                    (r'^FROM.*:latest', 'Using latest tag', 'Use specific version tags'),
                    (r'^USER root', 'Running as root', 'Use non-root user'),
                    (r'ADD\s+http', 'Using ADD with URL', 'Use COPY or curl with verification'),
                    (r'--disable-signature-verification', 'Disabled signature verification', 'Enable signature verification'),
                ]
                
                for line_num, line in enumerate(lines, 1):
                    for pattern, description, recommendation in docker_issues:
                        if re.search(pattern, line, re.IGNORECASE):
                            self.findings.append(SecurityFinding(
                                severity="medium",
                                category="docker_security",
                                description=description,
                                file_path="Dockerfile",
                                line_number=line_num,
                                evidence=line.strip()[:50],
                                recommendation=recommendation
                            ))
            except Exception:
                pass
    
    def _scan_github_workflows(self):
        """Scan GitHub Actions workflows for security issues"""
        logger.info("Scanning GitHub workflows...")
        workflow_dir = self.project_path / ".github" / "workflows"
        
        if workflow_dir.exists():
            for workflow_file in workflow_dir.glob("*.yml"):
                try:
                    content = workflow_file.read_text()
                    
                    # Check for security issues in workflows
                    workflow_issues = [
                        (r'secrets\.[A-Z_]+', 'Secret usage in workflow', 'Review secret usage'),
                        (r'pull_request_target', 'Using pull_request_target', 'Review for security implications'),
                        (r'\$\{\{.*\}\}.*shell', 'Shell injection risk', 'Sanitize inputs in shell commands'),
                    ]
                    
                    lines = content.split('\n')
                    for line_num, line in enumerate(lines, 1):
                        for pattern, description, recommendation in workflow_issues:
                            if re.search(pattern, line, re.IGNORECASE):
                                # Only flag if it looks suspicious
                                if 'github.event' in line or 'github.head_ref' in line:
                                    self.findings.append(SecurityFinding(
                                        severity="medium",
                                        category="workflow_security",
                                        description=description,
                                        file_path=str(workflow_file.relative_to(self.project_path)),
                                        line_number=line_num,
                                        evidence=line.strip()[:50],
                                        recommendation=recommendation
                                    ))
                except Exception:
                    pass
    
    def _get_severity_for_category(self, category: str) -> str:
        """Get severity level for vulnerability category"""
        severity_map = {
            'hardcoded_secrets': 'critical',
            'command_injection': 'critical',
            'sql_injection': 'high',
            'path_traversal': 'high',
            'vulnerable_dependency': 'high',
            'xss_vulnerabilities': 'high',
            'insecure_crypto': 'medium',
            'insecure_random': 'medium',
            'network_security': 'medium',
            'docker_security': 'medium',
            'workflow_security': 'medium',
            'file_permissions': 'low',
        }
        return severity_map.get(category, 'medium')
    
    def _get_recommendation_for_category(self, category: str) -> str:
        """Get recommendation for vulnerability category"""
        recommendations = {
            'hardcoded_secrets': 'Use environment variables or secure key management systems',
            'command_injection': 'Use parameterized commands and input validation',
            'sql_injection': 'Use parameterized queries and ORM',
            'path_traversal': 'Validate and sanitize file paths',
            'insecure_random': 'Use cryptographically secure random functions',
            'vulnerable_dependency': 'Update to latest secure version',
            'xss_vulnerabilities': 'Sanitize user inputs and use safe output methods',
            'insecure_crypto': 'Use modern cryptographic algorithms (SHA-256+, AES)',
            'network_security': 'Use HTTPS and verify SSL certificates',
            'docker_security': 'Follow Docker security best practices',
            'workflow_security': 'Review GitHub Actions security guidelines',
            'file_permissions': 'Set appropriate file permissions (644 for files, 755 for directories)',
        }
        return recommendations.get(category, 'Review code for security implications')
    
    def _generate_audit_report(self) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        # Categorize findings by severity
        by_severity = {'critical': [], 'high': [], 'medium': [], 'low': []}
        by_category = {}
        
        for finding in self.findings:
            by_severity[finding.severity].append(finding)
            if finding.category not in by_category:
                by_category[finding.category] = []
            by_category[finding.category].append(finding)
        
        # Calculate security score
        security_score = self._calculate_security_score(by_severity)
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'project_path': str(self.project_path),
            'total_findings': len(self.findings),
            'security_score': security_score,
            'findings_by_severity': {
                severity: [asdict(f) for f in findings]
                for severity, findings in by_severity.items()
            },
            'findings_by_category': {
                category: len(findings)
                for category, findings in by_category.items()
            },
            'summary': {
                'critical_issues': len(by_severity['critical']),
                'high_issues': len(by_severity['high']),
                'medium_issues': len(by_severity['medium']),
                'low_issues': len(by_severity['low']),
            }
        }
    
    def _calculate_security_score(self, by_severity: Dict[str, List]) -> int:
        """Calculate overall security score (0-100)"""
        base_score = 100
        
        # Deduct points based on severity
        deductions = {
            'critical': 25,
            'high': 10,
            'medium': 5,
            'low': 2
        }
        
        total_deduction = 0
        for severity, findings in by_severity.items():
            total_deduction += len(findings) * deductions[severity]
        
        return max(0, base_score - total_deduction)

class SecurityASTVisitor(ast.NodeVisitor):
    """AST visitor for security analysis"""
    
    def __init__(self, file_path: Path, project_path: Path):
        self.file_path = file_path
        self.project_path = project_path
        self.findings: List[SecurityFinding] = []
    
    def visit_Call(self, node):
        """Visit function calls for security issues"""
        # Check for dangerous function calls
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id == 'subprocess' and node.func.attr in ['call', 'run', 'Popen']:
                    self._check_subprocess_call(node)
        
        elif isinstance(node.func, ast.Name):
            if node.func.id in ['eval', 'exec']:
                self.findings.append(SecurityFinding(
                    severity="critical",
                    category="code_injection",
                    description=f"Dangerous function: {node.func.id}",
                    file_path=str(self.file_path.relative_to(self.project_path)),
                    line_number=node.lineno,
                    evidence=f"{node.func.id}() call",
                    recommendation="Avoid using eval() and exec()"
                ))
        
        self.generic_visit(node)
    
    def _check_subprocess_call(self, node):
        """Check subprocess calls for shell injection"""
        for keyword in node.keywords:
            if keyword.arg == 'shell' and isinstance(keyword.value, ast.Constant):
                if keyword.value.value is True:
                    self.findings.append(SecurityFinding(
                        severity="high",
                        category="command_injection",
                        description="subprocess call with shell=True",
                        file_path=str(self.file_path.relative_to(self.project_path)),
                        line_number=node.lineno,
                        evidence="shell=True",
                        recommendation="Use shell=False and pass commands as list"
                    ))

def main():
    """Main function"""
    project_path = Path(__file__).parent.parent.parent
    
    auditor = SecurityAuditor(project_path)
    report = auditor.run_full_audit()
    
    # Save report
    output_file = project_path / "security-audit-report.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"\n{'='*50}")
    print("SECURITY AUDIT SUMMARY")
    print(f"{'='*50}")
    print(f"Security Score: {report['security_score']}/100")
    print(f"Total Issues: {report['total_findings']}")
    print(f"Critical: {report['summary']['critical_issues']}")
    print(f"High: {report['summary']['high_issues']}")
    print(f"Medium: {report['summary']['medium_issues']}")
    print(f"Low: {report['summary']['low_issues']}")
    print(f"\nDetailed report saved to: {output_file}")
    
    # Exit with error code if critical issues found
    if report['summary']['critical_issues'] > 0:
        print("\n🚨 CRITICAL SECURITY ISSUES FOUND!")
        return 1
    elif report['summary']['high_issues'] > 0:
        print("\n⚠️  HIGH PRIORITY SECURITY ISSUES FOUND!")
        return 1
    else:
        print("\n✅ No critical or high-priority security issues found.")
        return 0

if __name__ == "__main__":
    exit(main())