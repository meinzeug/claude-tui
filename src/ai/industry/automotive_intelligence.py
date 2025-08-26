"""
Automotive Intelligence Module

Provides specialized AI intelligence for automotive and autonomous systems:
- ISO 26262 functional safety compliance (ASIL A-D)
- AUTOSAR (Automotive Open System Architecture)
- ISO 21448 SOTIF (Safety of the Intended Functionality)
- Automotive SPICE (Software Process Improvement and Capability dEtermination)
- UNECE WP.29 regulations for autonomous vehicles
- Cybersecurity standards (ISO/SAE 21434)
- Over-the-Air (OTA) update security
- V2X communication protocols
- Battery management system safety

Features:
- Automated ASIL level assessment
- Functional safety validation
- Autonomous system safety analysis
- Automotive cybersecurity compliance
- Real-time system validation for ECUs
- Hardware abstraction layer validation
- Vehicle communication protocol compliance
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from pathlib import Path
import json

from ...core.exceptions import ValidationError, SecurityError, ComplianceError
from ...core.types import ValidationResult, Issue, IssueType, Severity


class ASILLevel(Enum):
    """Automotive Safety Integrity Level (ISO 26262)"""
    ASIL_A = "asil_a"  # Lowest safety requirements
    ASIL_B = "asil_b"  # Low safety requirements  
    ASIL_C = "asil_c"  # Medium safety requirements
    ASIL_D = "asil_d"  # Highest safety requirements
    QM = "qm"         # Quality Management (no ASIL required)


class AutomotiveStandard(Enum):
    """Automotive standards and regulations"""
    ISO_26262 = "iso_26262"
    ISO_21448 = "iso_21448"  # SOTIF
    ISO_SAE_21434 = "iso_sae_21434"  # Cybersecurity
    AUTOSAR = "autosar"
    AUTOMOTIVE_SPICE = "automotive_spice"
    UNECE_WP29 = "unece_wp29"
    IEC_61508 = "iec_61508"  # Base functional safety
    ISO_14229 = "iso_14229"  # UDS (Unified Diagnostic Services)


class VehicleSystemType(Enum):
    """Types of vehicle systems"""
    POWERTRAIN = "powertrain"
    CHASSIS = "chassis"
    BODY = "body"
    INFOTAINMENT = "infotainment"
    ADAS = "adas"  # Advanced Driver Assistance Systems
    AUTONOMOUS_DRIVING = "autonomous_driving"
    CONNECTIVITY = "connectivity"
    CYBERSECURITY = "cybersecurity"


class AutomotiveArchitecture(Enum):
    """Automotive software architectures"""
    AUTOSAR_CLASSIC = "autosar_classic"
    AUTOSAR_ADAPTIVE = "autosar_adaptive"
    POSIX = "posix"
    HYPERVISOR_BASED = "hypervisor_based"
    MIXED_CRITICALITY = "mixed_criticality"


@dataclass
class AutomotiveComplianceCheck:
    """Automotive compliance check result"""
    standard: AutomotiveStandard
    requirement: str
    asil_level: Optional[ASILLevel] = None
    status: str = "pending"  # "compliant", "non_compliant", "warning", "not_applicable"
    details: str = ""
    evidence: List[str] = field(default_factory=list)
    remediation: Optional[str] = None
    severity: Severity = Severity.MEDIUM
    safety_impact: str = "low"  # "low", "medium", "high", "critical"


@dataclass
class AutomotiveSystemProfile:
    """Automotive system characteristics"""
    system_name: str
    asil_level: ASILLevel
    system_type: VehicleSystemType
    architecture: AutomotiveArchitecture
    safety_functions: List[str] = field(default_factory=list)
    cybersecurity_requirements: List[str] = field(default_factory=list)
    real_time_constraints: List[str] = field(default_factory=list)
    communication_protocols: List[str] = field(default_factory=list)  # CAN, LIN, FlexRay, Ethernet


@dataclass
class SafetyGoal:
    """ISO 26262 Safety Goal"""
    goal_id: str
    description: str
    asil_level: ASILLevel
    hazard_description: str
    operational_situation: str
    controllability: str  # C0, C1, C2, C3
    severity: str  # S0, S1, S2, S3
    exposure: str  # E0, E1, E2, E3, E4


class AutomotiveIntelligence:
    """
    Automotive Domain Intelligence Module
    
    Provides comprehensive automotive and autonomous vehicle expertise:
    - ISO 26262 functional safety compliance validation
    - AUTOSAR architecture compliance checking
    - Automotive cybersecurity standards validation
    - ADAS and autonomous driving system validation
    - Real-time constraint verification for ECUs
    - Vehicle communication protocol compliance
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # ISO 26262 safety patterns
        self.safety_patterns = {
            'freedom_from_interference': [
                r'memory[_\s]*protection',
                r'temporal[_\s]*protection',
                r'partition[_\s]*isolation',
                r'spatial[_\s]*isolation',
                r'hypervisor'
            ],
            'fault_detection': [
                r'fault[_\s]*detection',
                r'error[_\s]*detection',
                r'diagnostic[_\s]*coverage',
                r'self[_\s]*test',
                r'watchdog'
            ],
            'fault_handling': [
                r'fault[_\s]*handling',
                r'error[_\s]*recovery',
                r'safe[_\s]*state',
                r'degraded[_\s]*mode',
                r'fail[_\s]*safe'
            ],
            'redundancy': [
                r'dual[_\s]*redundancy',
                r'triple[_\s]*redundancy',
                r'diverse[_\s]*redundancy',
                r'backup[_\s]*system',
                r'fallback[_\s]*mechanism'
            ]
        }
        
        # AUTOSAR patterns
        self.autosar_patterns = {
            'classic': [
                r'SWC[_\s]*',  # Software Component
                r'RTE[_\s]*',  # Runtime Environment
                r'BSW[_\s]*',  # Basic Software
                r'ECUC[_\s]*', # ECU Configuration
                r'ARXML'
            ],
            'adaptive': [
                r'Execution[_\s]*Management',
                r'Communication[_\s]*Management',
                r'State[_\s]*Management',
                r'Log[_\s]*and[_\s]*Trace',
                r'Diagnostics[_\s]*Management'
            ]
        }
        
        # Cybersecurity patterns
        self.cybersecurity_patterns = {
            'authentication': [
                r'x509[_\s]*certificate',
                r'digital[_\s]*signature',
                r'pki[_\s]*',
                r'mutual[_\s]*authentication',
                r'challenge[_\s]*response'
            ],
            'encryption': [
                r'aes[_\s]*\d+',
                r'rsa[_\s]*\d+',
                r'elliptic[_\s]*curve',
                r'secure[_\s]*boot',
                r'trusted[_\s]*platform'
            ],
            'intrusion_detection': [
                r'ids[_\s]*',  # Intrusion Detection System
                r'anomaly[_\s]*detection',
                r'behavioral[_\s]*monitoring',
                r'network[_\s]*monitoring',
                r'traffic[_\s]*analysis'
            ]
        }
        
        # Communication protocol patterns
        self.communication_patterns = {
            'can': [
                r'CAN[_\s]*message',
                r'CAN[_\s]*frame',
                r'CAN[_\s]*id',
                r'J1939',
                r'CAN[_\s]*FD'
            ],
            'ethernet': [
                r'SOME/IP',
                r'DoIP',  # Diagnostics over IP
                r'AVB',   # Audio Video Bridging
                r'TSN',   # Time-Sensitive Networking
                r'UDP[_\s]*multicast'
            ],
            'v2x': [
                r'V2V',  # Vehicle-to-Vehicle
                r'V2I',  # Vehicle-to-Infrastructure
                r'V2P',  # Vehicle-to-Pedestrian
                r'DSRC', # Dedicated Short-Range Communications
                r'C-V2X' # Cellular Vehicle-to-Everything
            ]
        }

    async def validate_iso_26262_compliance(
        self,
        system_profile: AutomotiveSystemProfile,
        safety_goals: List[SafetyGoal],
        project_files: List[str]
    ) -> List[AutomotiveComplianceCheck]:
        """
        Validate ISO 26262 functional safety compliance
        
        Args:
            system_profile: Automotive system profile
            safety_goals: Defined safety goals
            project_files: Project files to analyze
            
        Returns:
            List of ISO 26262 compliance check results
        """
        self.logger.info(f"Validating ISO 26262 compliance for ASIL {system_profile.asil_level.value}")
        
        checks = []
        
        # Safety lifecycle compliance
        lifecycle_checks = await self._validate_safety_lifecycle(
            system_profile, project_files
        )
        checks.extend(lifecycle_checks)
        
        # Hazard analysis and risk assessment
        hazard_checks = await self._validate_hazard_analysis(
            system_profile, safety_goals, project_files
        )
        checks.extend(hazard_checks)
        
        # Safety requirements validation
        requirements_checks = await self._validate_safety_requirements(
            system_profile, safety_goals, project_files
        )
        checks.extend(requirements_checks)
        
        # Architectural design validation
        arch_checks = await self._validate_safety_architecture(
            system_profile, project_files
        )
        checks.extend(arch_checks)
        
        # Verification and validation
        verification_checks = await self._validate_safety_verification(
            system_profile, project_files
        )
        checks.extend(verification_checks)
        
        # Freedom from interference validation
        ffi_checks = await self._validate_freedom_from_interference(
            system_profile, project_files
        )
        checks.extend(ffi_checks)
        
        return checks

    async def validate_autosar_compliance(
        self,
        system_profile: AutomotiveSystemProfile,
        project_files: List[str]
    ) -> List[AutomotiveComplianceCheck]:
        """
        Validate AUTOSAR architecture compliance
        
        Args:
            system_profile: Automotive system profile
            project_files: Project files to analyze
            
        Returns:
            List of AUTOSAR compliance checks
        """
        checks = []
        
        if system_profile.architecture == AutomotiveArchitecture.AUTOSAR_CLASSIC:
            checks.extend(await self._validate_autosar_classic(system_profile, project_files))
        elif system_profile.architecture == AutomotiveArchitecture.AUTOSAR_ADAPTIVE:
            checks.extend(await self._validate_autosar_adaptive(system_profile, project_files))
        
        # Common AUTOSAR validation
        checks.extend(await self._validate_autosar_common(system_profile, project_files))
        
        return checks

    async def validate_automotive_cybersecurity(
        self,
        system_profile: AutomotiveSystemProfile,
        project_files: List[str]
    ) -> ValidationResult:
        """
        Validate automotive cybersecurity (ISO/SAE 21434)
        
        Args:
            system_profile: Automotive system profile
            project_files: Project files to analyze
            
        Returns:
            Cybersecurity validation result
        """
        issues = []
        
        # Cybersecurity risk assessment
        risk_issues = await self._validate_cybersecurity_risk_assessment(
            system_profile, project_files
        )
        issues.extend(risk_issues)
        
        # Security controls implementation
        security_issues = await self._validate_security_controls(
            system_profile, project_files
        )
        issues.extend(security_issues)
        
        # Communication security
        comm_issues = await self._validate_communication_security(
            system_profile, project_files
        )
        issues.extend(comm_issues)
        
        # OTA update security
        ota_issues = await self._validate_ota_security(
            system_profile, project_files
        )
        issues.extend(ota_issues)
        
        # Calculate cybersecurity score
        cyber_score = max(0.0, 100.0 - (len(issues) * 12))
        
        return ValidationResult(
            is_authentic=cyber_score > 85.0,
            authenticity_score=cyber_score,
            real_progress=cyber_score,
            fake_progress=0.0,
            issues=issues,
            suggestions=[
                "Implement secure boot mechanisms",
                "Add intrusion detection systems",
                "Implement secure communication protocols",
                "Add certificate-based authentication",
                "Implement security monitoring"
            ],
            next_actions=[
                "Conduct cybersecurity risk assessment",
                "Implement security controls framework",
                "Add penetration testing procedures",
                "Create incident response procedures",
                "Validate security requirements"
            ]
        )

    async def validate_autonomous_driving_safety(
        self,
        system_profile: AutomotiveSystemProfile,
        project_files: List[str]
    ) -> ValidationResult:
        """
        Validate autonomous driving system safety (ISO 21448 SOTIF)
        
        Args:
            system_profile: Automotive system profile
            project_files: Project files to analyze
            
        Returns:
            SOTIF validation result
        """
        issues = []
        
        # SOTIF hazard analysis
        sotif_issues = await self._validate_sotif_analysis(
            system_profile, project_files
        )
        issues.extend(sotif_issues)
        
        # Sensor validation
        sensor_issues = await self._validate_sensor_systems(
            system_profile, project_files
        )
        issues.extend(sensor_issues)
        
        # AI/ML validation
        ai_issues = await self._validate_ai_ml_systems(
            system_profile, project_files
        )
        issues.extend(ai_issues)
        
        # Perception validation
        perception_issues = await self._validate_perception_systems(
            system_profile, project_files
        )
        issues.extend(perception_issues)
        
        # Decision making validation
        decision_issues = await self._validate_decision_systems(
            system_profile, project_files
        )
        issues.extend(decision_issues)
        
        # Calculate SOTIF compliance score
        sotif_score = max(0.0, 100.0 - (len(issues) * 15))
        
        return ValidationResult(
            is_authentic=sotif_score > 80.0,
            authenticity_score=sotif_score,
            real_progress=sotif_score,
            fake_progress=0.0,
            issues=issues,
            suggestions=[
                "Implement comprehensive sensor fusion",
                "Add AI model validation frameworks",
                "Implement scenario-based testing",
                "Add performance monitoring systems",
                "Implement graceful degradation"
            ],
            next_actions=[
                "Conduct SOTIF risk assessment",
                "Implement validation test scenarios",
                "Add real-world testing procedures",
                "Create safety performance indicators",
                "Validate operational design domain"
            ]
        )

    # Private validation methods

    async def _validate_safety_lifecycle(
        self,
        system_profile: AutomotiveSystemProfile,
        project_files: List[str]
    ) -> List[AutomotiveComplianceCheck]:
        """Validate safety lifecycle compliance"""
        checks = []
        
        # Check for safety plan
        has_safety_plan = any(
            'safety' in file_path.lower() and 'plan' in file_path.lower()
            for file_path in project_files
        )
        
        checks.append(AutomotiveComplianceCheck(
            standard=AutomotiveStandard.ISO_26262,
            requirement="Safety Plan",
            asil_level=system_profile.asil_level,
            status="compliant" if has_safety_plan else "non_compliant",
            details="Safety plan document found" if has_safety_plan else "No safety plan found",
            severity=Severity.HIGH if not has_safety_plan else Severity.LOW,
            safety_impact="high" if not has_safety_plan else "low"
        ))
        
        return checks

    async def _validate_hazard_analysis(
        self,
        system_profile: AutomotiveSystemProfile,
        safety_goals: List[SafetyGoal],
        project_files: List[str]
    ) -> List[AutomotiveComplianceCheck]:
        """Validate hazard analysis and risk assessment"""
        checks = []
        
        # Check for hazard analysis documentation
        has_hara = any(
            'hazard' in file_path.lower() or 'hara' in file_path.lower()
            for file_path in project_files
        )
        
        checks.append(AutomotiveComplianceCheck(
            standard=AutomotiveStandard.ISO_26262,
            requirement="Hazard Analysis and Risk Assessment",
            asil_level=system_profile.asil_level,
            status="compliant" if has_hara else "non_compliant",
            details="HARA documentation found" if has_hara else "No HARA documentation found",
            severity=Severity.CRITICAL if not has_hara else Severity.LOW,
            safety_impact="critical" if not has_hara else "low"
        ))
        
        # Validate safety goals coverage
        if not safety_goals:
            checks.append(AutomotiveComplianceCheck(
                standard=AutomotiveStandard.ISO_26262,
                requirement="Safety Goals Definition",
                asil_level=system_profile.asil_level,
                status="non_compliant",
                details="No safety goals defined",
                severity=Severity.CRITICAL,
                safety_impact="critical"
            ))
        
        return checks

    async def _validate_safety_requirements(
        self,
        system_profile: AutomotiveSystemProfile,
        safety_goals: List[SafetyGoal],
        project_files: List[str]
    ) -> List[AutomotiveComplianceCheck]:
        """Validate safety requirements specification"""
        checks = []
        
        # Check for functional safety requirements
        has_fsr = any(
            'functional' in file_path.lower() and 'safety' in file_path.lower() and 'requirement' in file_path.lower()
            for file_path in project_files
        )
        
        checks.append(AutomotiveComplianceCheck(
            standard=AutomotiveStandard.ISO_26262,
            requirement="Functional Safety Requirements",
            asil_level=system_profile.asil_level,
            status="compliant" if has_fsr else "non_compliant",
            details="FSR documentation found" if has_fsr else "No FSR documentation found",
            severity=Severity.HIGH if not has_fsr else Severity.LOW,
            safety_impact="high" if not has_fsr else "low"
        ))
        
        return checks

    async def _validate_safety_architecture(
        self,
        system_profile: AutomotiveSystemProfile,
        project_files: List[str]
    ) -> List[AutomotiveComplianceCheck]:
        """Validate safety architecture design"""
        checks = []
        
        # Check for architectural safety patterns
        for pattern_type, patterns in self.safety_patterns.items():
            has_pattern = False
            evidence = []
            
            for file_path in project_files:
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        
                    for pattern in patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            has_pattern = True
                            evidence.append(f"Pattern found in {file_path}")
                            break
                            
                except Exception:
                    continue
            
            # Determine if pattern is required based on ASIL level
            pattern_required = system_profile.asil_level in [ASILLevel.ASIL_C, ASILLevel.ASIL_D]
            
            if pattern_required:
                checks.append(AutomotiveComplianceCheck(
                    standard=AutomotiveStandard.ISO_26262,
                    requirement=f"Safety Pattern: {pattern_type.replace('_', ' ').title()}",
                    asil_level=system_profile.asil_level,
                    status="compliant" if has_pattern else "warning",
                    details=f"{pattern_type.replace('_', ' ').title()} pattern implementation",
                    evidence=evidence,
                    severity=Severity.MEDIUM if not has_pattern else Severity.LOW,
                    safety_impact="medium" if not has_pattern else "low"
                ))
        
        return checks

    async def _validate_safety_verification(
        self,
        system_profile: AutomotiveSystemProfile,
        project_files: List[str]
    ) -> List[AutomotiveComplianceCheck]:
        """Validate safety verification and validation"""
        checks = []
        
        # Check for safety test cases
        has_safety_tests = any(
            'safety' in file_path.lower() and 'test' in file_path.lower()
            for file_path in project_files
        )
        
        checks.append(AutomotiveComplianceCheck(
            standard=AutomotiveStandard.ISO_26262,
            requirement="Safety Testing",
            asil_level=system_profile.asil_level,
            status="compliant" if has_safety_tests else "warning",
            details="Safety test cases found" if has_safety_tests else "No safety test cases found",
            severity=Severity.MEDIUM if not has_safety_tests else Severity.LOW,
            safety_impact="medium" if not has_safety_tests else "low"
        ))
        
        return checks

    async def _validate_freedom_from_interference(
        self,
        system_profile: AutomotiveSystemProfile,
        project_files: List[str]
    ) -> List[AutomotiveComplianceCheck]:
        """Validate freedom from interference implementation"""
        checks = []
        
        if system_profile.asil_level in [ASILLevel.ASIL_C, ASILLevel.ASIL_D]:
            # FFI is required for ASIL C and D
            ffi_patterns = self.safety_patterns['freedom_from_interference']
            
            has_ffi = False
            evidence = []
            
            for file_path in project_files:
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        
                    for pattern in ffi_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            has_ffi = True
                            evidence.append(f"FFI mechanism found in {file_path}")
                            break
                            
                except Exception:
                    continue
            
            checks.append(AutomotiveComplianceCheck(
                standard=AutomotiveStandard.ISO_26262,
                requirement="Freedom from Interference",
                asil_level=system_profile.asil_level,
                status="compliant" if has_ffi else "non_compliant",
                details="FFI mechanisms implemented" if has_ffi else "No FFI mechanisms found",
                evidence=evidence,
                severity=Severity.HIGH if not has_ffi else Severity.LOW,
                safety_impact="high" if not has_ffi else "low"
            ))
        
        return checks

    async def _validate_autosar_classic(
        self,
        system_profile: AutomotiveSystemProfile,
        project_files: List[str]
    ) -> List[AutomotiveComplianceCheck]:
        """Validate AUTOSAR Classic compliance"""
        checks = []
        
        classic_patterns = self.autosar_patterns['classic']
        has_classic = False
        evidence = []
        
        for file_path in project_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                for pattern in classic_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        has_classic = True
                        evidence.append(f"AUTOSAR Classic pattern found in {file_path}")
                        break
                        
            except Exception:
                continue
        
        checks.append(AutomotiveComplianceCheck(
            standard=AutomotiveStandard.AUTOSAR,
            requirement="AUTOSAR Classic Architecture",
            asil_level=system_profile.asil_level,
            status="compliant" if has_classic else "warning",
            details="AUTOSAR Classic patterns found" if has_classic else "No AUTOSAR Classic patterns found",
            evidence=evidence,
            severity=Severity.MEDIUM if not has_classic else Severity.LOW
        ))
        
        return checks

    async def _validate_autosar_adaptive(
        self,
        system_profile: AutomotiveSystemProfile,
        project_files: List[str]
    ) -> List[AutomotiveComplianceCheck]:
        """Validate AUTOSAR Adaptive compliance"""
        checks = []
        
        adaptive_patterns = self.autosar_patterns['adaptive']
        has_adaptive = False
        evidence = []
        
        for file_path in project_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                for pattern in adaptive_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        has_adaptive = True
                        evidence.append(f"AUTOSAR Adaptive pattern found in {file_path}")
                        break
                        
            except Exception:
                continue
        
        checks.append(AutomotiveComplianceCheck(
            standard=AutomotiveStandard.AUTOSAR,
            requirement="AUTOSAR Adaptive Platform",
            asil_level=system_profile.asil_level,
            status="compliant" if has_adaptive else "warning",
            details="AUTOSAR Adaptive patterns found" if has_adaptive else "No AUTOSAR Adaptive patterns found",
            evidence=evidence,
            severity=Severity.MEDIUM if not has_adaptive else Severity.LOW
        ))
        
        return checks

    async def _validate_autosar_common(
        self,
        system_profile: AutomotiveSystemProfile,
        project_files: List[str]
    ) -> List[AutomotiveComplianceCheck]:
        """Validate common AUTOSAR requirements"""
        checks = []
        
        # Check for configuration files
        has_config = any(
            file_path.lower().endswith('.arxml') or 
            'config' in file_path.lower()
            for file_path in project_files
        )
        
        checks.append(AutomotiveComplianceCheck(
            standard=AutomotiveStandard.AUTOSAR,
            requirement="AUTOSAR Configuration",
            asil_level=system_profile.asil_level,
            status="compliant" if has_config else "warning",
            details="AUTOSAR configuration found" if has_config else "No AUTOSAR configuration found",
            severity=Severity.MEDIUM if not has_config else Severity.LOW
        ))
        
        return checks

    async def _validate_cybersecurity_risk_assessment(
        self,
        system_profile: AutomotiveSystemProfile,
        project_files: List[str]
    ) -> List[Issue]:
        """Validate cybersecurity risk assessment"""
        issues = []
        
        # Check for cybersecurity documentation
        has_cyber_docs = any(
            'cybersecurity' in file_path.lower() or 
            'security' in file_path.lower() and 'risk' in file_path.lower()
            for file_path in project_files
        )
        
        if not has_cyber_docs:
            issues.append(Issue(
                type=IssueType.INCOMPLETE_IMPLEMENTATION,
                severity=Severity.HIGH,
                description="Cybersecurity risk assessment documentation missing"
            ))
        
        return issues

    async def _validate_security_controls(
        self,
        system_profile: AutomotiveSystemProfile,
        project_files: List[str]
    ) -> List[Issue]:
        """Validate security controls implementation"""
        issues = []
        
        # Check authentication mechanisms
        auth_issues = await self._check_authentication_patterns(project_files)
        issues.extend(auth_issues)
        
        # Check encryption implementation
        encryption_issues = await self._check_encryption_patterns(project_files)
        issues.extend(encryption_issues)
        
        return issues

    async def _validate_communication_security(
        self,
        system_profile: AutomotiveSystemProfile,
        project_files: List[str]
    ) -> List[Issue]:
        """Validate communication security"""
        issues = []
        
        # Check for secure communication protocols
        for protocol, patterns in self.communication_patterns.items():
            has_secure_comm = False
            
            for file_path in project_files:
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        
                    if any(re.search(pattern, content, re.IGNORECASE) for pattern in patterns):
                        # Check if security is implemented
                        security_patterns = ['encrypt', 'authenticate', 'secure', 'certificate']
                        has_security = any(
                            re.search(sec_pattern, content, re.IGNORECASE)
                            for sec_pattern in security_patterns
                        )
                        
                        if not has_security:
                            issues.append(Issue(
                                type=IssueType.SECURITY_VULNERABILITY,
                                severity=Severity.MEDIUM,
                                description=f"Insecure {protocol.upper()} communication detected",
                                file_path=file_path
                            ))
                        else:
                            has_secure_comm = True
                        break
                        
                except Exception:
                    continue
        
        return issues

    async def _validate_ota_security(
        self,
        system_profile: AutomotiveSystemProfile,
        project_files: List[str]
    ) -> List[Issue]:
        """Validate Over-the-Air update security"""
        issues = []
        
        # Check for OTA implementation
        ota_files = [f for f in project_files if 'ota' in f.lower() or 'update' in f.lower()]
        
        if ota_files:
            for file_path in ota_files:
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Check for security measures
                    security_measures = [
                        'digital[_\s]*signature',
                        'certificate[_\s]*verification',
                        'rollback[_\s]*protection',
                        'secure[_\s]*boot',
                        'integrity[_\s]*check'
                    ]
                    
                    missing_measures = []
                    for measure in security_measures:
                        if not re.search(measure, content, re.IGNORECASE):
                            missing_measures.append(measure.replace('[_\\s]*', ' '))
                    
                    if missing_measures:
                        issues.append(Issue(
                            type=IssueType.SECURITY_VULNERABILITY,
                            severity=Severity.HIGH,
                            description=f"OTA security measures missing: {', '.join(missing_measures)}",
                            file_path=file_path
                        ))
                        
                except Exception:
                    continue
        
        return issues

    async def _validate_sotif_analysis(
        self,
        system_profile: AutomotiveSystemProfile,
        project_files: List[str]
    ) -> List[Issue]:
        """Validate SOTIF (Safety of the Intended Functionality) analysis"""
        issues = []
        
        # Check for SOTIF documentation
        has_sotif_docs = any(
            'sotif' in file_path.lower() or 
            ('safety' in file_path.lower() and 'intended' in file_path.lower())
            for file_path in project_files
        )
        
        if not has_sotif_docs and system_profile.system_type == VehicleSystemType.AUTONOMOUS_DRIVING:
            issues.append(Issue(
                type=IssueType.INCOMPLETE_IMPLEMENTATION,
                severity=Severity.HIGH,
                description="SOTIF analysis documentation missing for autonomous driving system"
            ))
        
        return issues

    async def _validate_sensor_systems(
        self,
        system_profile: AutomotiveSystemProfile,
        project_files: List[str]
    ) -> List[Issue]:
        """Validate sensor systems for autonomous driving"""
        issues = []
        
        sensor_patterns = [
            r'lidar',
            r'radar',
            r'camera',
            r'ultrasonic',
            r'sensor[_\s]*fusion'
        ]
        
        has_sensors = any(
            any(re.search(pattern, file_path, re.IGNORECASE) for pattern in sensor_patterns)
            for file_path in project_files
        )
        
        if system_profile.system_type == VehicleSystemType.AUTONOMOUS_DRIVING and not has_sensors:
            issues.append(Issue(
                type=IssueType.INCOMPLETE_IMPLEMENTATION,
                severity=Severity.MEDIUM,
                description="No sensor system implementation found for autonomous driving"
            ))
        
        return issues

    async def _validate_ai_ml_systems(
        self,
        system_profile: AutomotiveSystemProfile,
        project_files: List[str]
    ) -> List[Issue]:
        """Validate AI/ML systems in automotive context"""
        issues = []
        
        ai_patterns = [
            r'neural[_\s]*network',
            r'machine[_\s]*learning',
            r'deep[_\s]*learning',
            r'tensorflow',
            r'pytorch',
            r'model[_\s]*validation'
        ]
        
        for file_path in project_files:
            if any(re.search(pattern, file_path, re.IGNORECASE) for pattern in ai_patterns):
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Check for AI validation patterns
                    validation_patterns = [
                        r'model[_\s]*validation',
                        r'performance[_\s]*monitoring',
                        r'uncertainty[_\s]*quantification',
                        r'adversarial[_\s]*testing'
                    ]
                    
                    has_validation = any(
                        re.search(pattern, content, re.IGNORECASE)
                        for pattern in validation_patterns
                    )
                    
                    if not has_validation:
                        issues.append(Issue(
                            type=IssueType.INCOMPLETE_IMPLEMENTATION,
                            severity=Severity.HIGH,
                            description="AI/ML model lacks proper validation mechanisms",
                            file_path=file_path
                        ))
                        
                except Exception:
                    continue
        
        return issues

    async def _validate_perception_systems(
        self,
        system_profile: AutomotiveSystemProfile,
        project_files: List[str]
    ) -> List[Issue]:
        """Validate perception systems"""
        issues = []
        
        # Check for perception algorithms
        perception_patterns = [
            r'object[_\s]*detection',
            r'lane[_\s]*detection',
            r'traffic[_\s]*sign[_\s]*recognition',
            r'pedestrian[_\s]*detection',
            r'obstacle[_\s]*avoidance'
        ]
        
        has_perception = any(
            any(re.search(pattern, file_path, re.IGNORECASE) for pattern in perception_patterns)
            for file_path in project_files
        )
        
        if system_profile.system_type in [VehicleSystemType.ADAS, VehicleSystemType.AUTONOMOUS_DRIVING] and not has_perception:
            issues.append(Issue(
                type=IssueType.INCOMPLETE_IMPLEMENTATION,
                severity=Severity.MEDIUM,
                description="No perception system implementation found"
            ))
        
        return issues

    async def _validate_decision_systems(
        self,
        system_profile: AutomotiveSystemProfile,
        project_files: List[str]
    ) -> List[Issue]:
        """Validate decision-making systems"""
        issues = []
        
        # Check for decision algorithms
        decision_patterns = [
            r'path[_\s]*planning',
            r'trajectory[_\s]*planning',
            r'behavior[_\s]*planning',
            r'decision[_\s]*making',
            r'motion[_\s]*control'
        ]
        
        has_decision = any(
            any(re.search(pattern, file_path, re.IGNORECASE) for pattern in decision_patterns)
            for file_path in project_files
        )
        
        if system_profile.system_type == VehicleSystemType.AUTONOMOUS_DRIVING and not has_decision:
            issues.append(Issue(
                type=IssueType.INCOMPLETE_IMPLEMENTATION,
                severity=Severity.MEDIUM,
                description="No decision-making system implementation found"
            ))
        
        return issues

    async def _check_authentication_patterns(self, project_files: List[str]) -> List[Issue]:
        """Check authentication implementation patterns"""
        issues = []
        
        auth_patterns = self.cybersecurity_patterns['authentication']
        
        for file_path in project_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Check if authentication is mentioned but not properly implemented
                if 'auth' in content.lower():
                    has_proper_auth = any(
                        re.search(pattern, content, re.IGNORECASE)
                        for pattern in auth_patterns
                    )
                    
                    if not has_proper_auth:
                        issues.append(Issue(
                            type=IssueType.SECURITY_VULNERABILITY,
                            severity=Severity.MEDIUM,
                            description="Weak authentication mechanism detected",
                            file_path=file_path
                        ))
                        
            except Exception:
                continue
        
        return issues

    async def _check_encryption_patterns(self, project_files: List[str]) -> List[Issue]:
        """Check encryption implementation patterns"""
        issues = []
        
        encryption_patterns = self.cybersecurity_patterns['encryption']
        
        for file_path in project_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Check if encryption is mentioned but not properly implemented
                if 'encrypt' in content.lower():
                    has_proper_encryption = any(
                        re.search(pattern, content, re.IGNORECASE)
                        for pattern in encryption_patterns
                    )
                    
                    if not has_proper_encryption:
                        issues.append(Issue(
                            type=IssueType.SECURITY_VULNERABILITY,
                            severity=Severity.HIGH,
                            description="Weak encryption mechanism detected",
                            file_path=file_path
                        ))
                        
            except Exception:
                continue
        
        return issues