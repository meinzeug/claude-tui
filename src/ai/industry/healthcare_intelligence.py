"""
Healthcare Intelligence Module

Provides specialized AI intelligence for healthcare and medical device development:
- HIPAA compliance validation
- Medical device software standards (IEC 62304)
- FDA 21 CFR Part 820 compliance
- HL7 FHIR integration patterns
- Healthcare data security and privacy
- Clinical workflow optimization
- Medical terminology validation

Features:
- Automated HIPAA compliance checking
- Medical device classification guidance
- Clinical data validation patterns
- Healthcare API security standards
- PHI (Protected Health Information) detection
- Medical coding standards (ICD-10, CPT, SNOMED)
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


class MedicalDeviceClass(Enum):
    """FDA Medical Device Classifications"""
    CLASS_I = "class_i"      # Low risk (tongue depressors, bandages)
    CLASS_II = "class_ii"    # Moderate risk (X-ray machines, wheelchairs)
    CLASS_III = "class_iii"  # High risk (pacemakers, heart valves)


class HIPAADataType(Enum):
    """HIPAA Protected Health Information Categories"""
    NAMES = "names"
    DATES = "dates"
    PHONE_NUMBERS = "phone_numbers"
    FAX_NUMBERS = "fax_numbers"
    EMAIL_ADDRESSES = "email_addresses"
    SSN = "social_security_numbers"
    MEDICAL_RECORD_NUMBERS = "medical_record_numbers"
    HEALTH_PLAN_NUMBERS = "health_plan_numbers"
    ACCOUNT_NUMBERS = "account_numbers"
    CERTIFICATE_NUMBERS = "certificate_numbers"
    VEHICLE_IDENTIFIERS = "vehicle_identifiers"
    DEVICE_IDENTIFIERS = "device_identifiers"
    WEB_URLS = "web_urls"
    IP_ADDRESSES = "ip_addresses"
    BIOMETRIC_IDENTIFIERS = "biometric_identifiers"
    PHOTOS = "full_face_photos"
    UNIQUE_IDENTIFIERS = "unique_identifying_numbers"


@dataclass
class ComplianceCheck:
    """Healthcare compliance check result"""
    standard: str
    requirement: str
    status: str  # "compliant", "non_compliant", "warning", "not_applicable"
    details: str
    evidence: List[str] = field(default_factory=list)
    remediation: Optional[str] = None
    severity: Severity = Severity.MEDIUM


@dataclass
class MedicalDeviceProfile:
    """Medical device software profile"""
    device_class: MedicalDeviceClass
    safety_classification: str  # "A", "B", "C" per IEC 62304
    intended_use: str
    risk_controls: List[str] = field(default_factory=list)
    regulatory_requirements: List[str] = field(default_factory=list)


class HealthcareIntelligence:
    """
    Healthcare Domain Intelligence Module
    
    Provides comprehensive healthcare and medical device development expertise:
    - HIPAA compliance validation and PHI detection
    - FDA medical device software standards
    - Clinical workflow optimization
    - Healthcare data security patterns
    - Medical terminology validation
    - HL7 FHIR integration guidance
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # HIPAA PHI detection patterns
        self.phi_patterns = {
            HIPAADataType.NAMES: [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last name patterns
                r'\bpatient[_\s]*name\b',
                r'\bfull[_\s]*name\b'
            ],
            HIPAADataType.SSN: [
                r'\b\d{3}-\d{2}-\d{4}\b',
                r'\b\d{9}\b',
                r'\bssn\b',
                r'\bsocial[_\s]*security\b'
            ],
            HIPAADataType.DATES: [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',
                r'\b\d{4}-\d{2}-\d{2}\b',
                r'\bdob\b',
                r'\bdate[_\s]*of[_\s]*birth\b'
            ],
            HIPAADataType.PHONE_NUMBERS: [
                r'\b\d{3}-\d{3}-\d{4}\b',
                r'\(\d{3}\)\s*\d{3}-\d{4}',
                r'\bphone\b',
                r'\bmobile\b'
            ],
            HIPAADataType.EMAIL_ADDRESSES: [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            HIPAADataType.MEDICAL_RECORD_NUMBERS: [
                r'\bmrn[:\s]*\d+\b',
                r'\bmedical[_\s]*record[_\s]*number\b',
                r'\bpatient[_\s]*id\b'
            ],
            HIPAADataType.IP_ADDRESSES: [
                r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
            ]
        }
        
        # Medical device standards
        self.device_standards = {
            'IEC_62304': {
                'name': 'Medical Device Software - Software Life Cycle Processes',
                'classes': ['A', 'B', 'C'],
                'requirements': [
                    'software_development_planning',
                    'software_requirements_analysis',
                    'software_architectural_design',
                    'software_detailed_design',
                    'software_implementation',
                    'software_integration_testing',
                    'software_system_testing',
                    'software_release'
                ]
            },
            'ISO_14155': {
                'name': 'Clinical investigation of medical devices for human subjects',
                'requirements': ['clinical_evaluation', 'risk_management', 'clinical_data']
            },
            'ISO_13485': {
                'name': 'Medical devices - Quality management systems',
                'requirements': ['quality_management', 'design_controls', 'risk_management']
            }
        }
        
        # FHIR resource patterns
        self.fhir_resources = {
            'Patient', 'Practitioner', 'Organization', 'Encounter', 
            'Observation', 'DiagnosticReport', 'Medication', 'MedicationRequest',
            'Condition', 'Procedure', 'AllergyIntolerance', 'Immunization',
            'DocumentReference', 'Binary', 'Bundle', 'OperationOutcome'
        }
        
        # Medical coding systems
        self.medical_coding_systems = {
            'ICD_10': 'International Classification of Diseases, 10th Revision',
            'CPT': 'Current Procedural Terminology',
            'SNOMED_CT': 'Systematized Nomenclature of Medicine Clinical Terms',
            'LOINC': 'Logical Observation Identifiers Names and Codes',
            'RxNorm': 'Normalized Names for Clinical Drugs'
        }

    async def validate_hipaa_compliance(
        self,
        file_path: str,
        content: str
    ) -> List[ComplianceCheck]:
        """
        Validate HIPAA compliance for healthcare applications
        
        Args:
            file_path: Path to file being validated
            content: File content to analyze
            
        Returns:
            List of compliance check results
        """
        self.logger.info(f"Validating HIPAA compliance for {file_path}")
        
        checks = []
        
        # Check for PHI exposure
        phi_violations = await self._detect_phi_exposure(content, file_path)
        for violation in phi_violations:
            checks.append(ComplianceCheck(
                standard="HIPAA",
                requirement="PHI Protection",
                status="non_compliant",
                details=violation['description'],
                evidence=violation['evidence'],
                remediation=violation['remediation'],
                severity=Severity.HIGH
            ))
        
        # Check encryption requirements
        encryption_check = await self._validate_encryption_usage(content, file_path)
        checks.append(encryption_check)
        
        # Check access controls
        access_control_check = await self._validate_access_controls(content, file_path)
        checks.append(access_control_check)
        
        # Check audit logging
        audit_check = await self._validate_audit_logging(content, file_path)
        checks.append(audit_check)
        
        # Check data minimization
        data_min_check = await self._validate_data_minimization(content, file_path)
        checks.append(data_min_check)
        
        return checks

    async def validate_medical_device_compliance(
        self,
        device_profile: MedicalDeviceProfile,
        project_files: List[str]
    ) -> List[ComplianceCheck]:
        """
        Validate medical device software compliance (IEC 62304)
        
        Args:
            device_profile: Medical device classification and profile
            project_files: List of project files to analyze
            
        Returns:
            List of compliance check results
        """
        checks = []
        
        # Validate based on device class and safety classification
        if device_profile.safety_classification in ['B', 'C']:
            # Higher safety requirements
            checks.extend(await self._validate_software_architecture(project_files))
            checks.extend(await self._validate_risk_controls(device_profile, project_files))
            
        if device_profile.device_class in [MedicalDeviceClass.CLASS_II, MedicalDeviceClass.CLASS_III]:
            # FDA validation requirements
            checks.extend(await self._validate_fda_requirements(device_profile, project_files))
        
        # Common requirements for all medical devices
        checks.extend(await self._validate_documentation_requirements(project_files))
        checks.extend(await self._validate_testing_requirements(project_files))
        
        return checks

    async def generate_fhir_integration_patterns(
        self,
        use_case: str,
        data_types: List[str]
    ) -> Dict[str, Any]:
        """
        Generate HL7 FHIR integration patterns for healthcare interoperability
        
        Args:
            use_case: Healthcare use case (e.g., 'patient_records', 'lab_results')
            data_types: Required FHIR data types
            
        Returns:
            FHIR integration patterns and code examples
        """
        patterns = {
            'resources': {},
            'interactions': {},
            'security': {},
            'validation': {},
            'code_examples': {}
        }
        
        # Generate resource mappings
        for data_type in data_types:
            if data_type in self.fhir_resources:
                patterns['resources'][data_type] = await self._generate_fhir_resource_pattern(data_type)
        
        # Generate interaction patterns
        patterns['interactions'] = await self._generate_fhir_interactions(use_case, data_types)
        
        # Generate security patterns
        patterns['security'] = await self._generate_fhir_security_patterns(use_case)
        
        # Generate validation patterns
        patterns['validation'] = await self._generate_fhir_validation_patterns(data_types)
        
        # Generate code examples
        patterns['code_examples'] = await self._generate_fhir_code_examples(use_case, data_types)
        
        return patterns

    async def validate_clinical_workflow(
        self,
        workflow_definition: Dict[str, Any],
        file_paths: List[str]
    ) -> ValidationResult:
        """
        Validate clinical workflow implementation for safety and efficiency
        
        Args:
            workflow_definition: Clinical workflow specification
            file_paths: Implementation files to validate
            
        Returns:
            Validation result with clinical workflow assessment
        """
        issues = []
        
        # Validate workflow steps
        workflow_issues = await self._validate_workflow_steps(workflow_definition)
        issues.extend(workflow_issues)
        
        # Validate safety checks
        safety_issues = await self._validate_clinical_safety_checks(workflow_definition, file_paths)
        issues.extend(safety_issues)
        
        # Validate decision points
        decision_issues = await self._validate_clinical_decision_points(workflow_definition)
        issues.extend(decision_issues)
        
        # Validate error handling
        error_handling_issues = await self._validate_clinical_error_handling(file_paths)
        issues.extend(error_handling_issues)
        
        # Calculate authenticity score based on clinical standards adherence
        authenticity_score = max(0.0, 100.0 - (len(issues) * 10))
        
        return ValidationResult(
            is_authentic=authenticity_score > 70.0,
            authenticity_score=authenticity_score,
            real_progress=authenticity_score,
            fake_progress=0.0,
            issues=issues,
            suggestions=[
                "Implement clinical decision support systems",
                "Add comprehensive audit logging",
                "Validate against medical coding standards",
                "Implement patient safety checks"
            ],
            next_actions=[
                "Review clinical workflow with medical professionals",
                "Implement additional safety validations",
                "Add comprehensive error handling",
                "Validate with healthcare standards"
            ]
        )

    async def _detect_phi_exposure(
        self,
        content: str,
        file_path: str
    ) -> List[Dict[str, Any]]:
        """Detect potential PHI exposure in code"""
        violations = []
        
        for phi_type, patterns in self.phi_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    violations.append({
                        'type': phi_type.value,
                        'description': f"Potential {phi_type.value.replace('_', ' ')} exposure detected",
                        'evidence': [f"Line containing: {match.group()}"],
                        'remediation': f"Remove or encrypt {phi_type.value.replace('_', ' ')} data"
                    })
        
        return violations

    async def _validate_encryption_usage(
        self,
        content: str,
        file_path: str
    ) -> ComplianceCheck:
        """Validate encryption usage for PHI protection"""
        
        encryption_patterns = [
            r'encrypt\(',
            r'AES\.encrypt',
            r'crypto\.',
            r'bcrypt',
            r'scrypt',
            r'TLS',
            r'SSL'
        ]
        
        has_encryption = any(
            re.search(pattern, content, re.IGNORECASE) 
            for pattern in encryption_patterns
        )
        
        if has_encryption:
            return ComplianceCheck(
                standard="HIPAA",
                requirement="Encryption of PHI",
                status="compliant",
                details="Encryption mechanisms detected",
                evidence=["Encryption functions found in code"]
            )
        else:
            return ComplianceCheck(
                standard="HIPAA",
                requirement="Encryption of PHI",
                status="non_compliant",
                details="No encryption mechanisms detected for PHI protection",
                remediation="Implement AES-256 encryption for PHI data at rest and in transit",
                severity=Severity.CRITICAL
            )

    async def _validate_access_controls(
        self,
        content: str,
        file_path: str
    ) -> ComplianceCheck:
        """Validate access control implementation"""
        
        access_control_patterns = [
            r'@require_auth',
            r'@login_required',
            r'authenticate',
            r'authorize',
            r'permission',
            r'role.*check',
            r'access.*control'
        ]
        
        has_access_controls = any(
            re.search(pattern, content, re.IGNORECASE)
            for pattern in access_control_patterns
        )
        
        if has_access_controls:
            return ComplianceCheck(
                standard="HIPAA",
                requirement="Access Controls",
                status="compliant",
                details="Access control mechanisms detected",
                evidence=["Authentication/authorization patterns found"]
            )
        else:
            return ComplianceCheck(
                standard="HIPAA",
                requirement="Access Controls",
                status="warning",
                details="Limited access control patterns detected",
                remediation="Implement role-based access controls (RBAC) for PHI access",
                severity=Severity.HIGH
            )

    async def _validate_audit_logging(
        self,
        content: str,
        file_path: str
    ) -> ComplianceCheck:
        """Validate audit logging implementation"""
        
        audit_patterns = [
            r'audit.*log',
            r'log.*access',
            r'logger\.',
            r'logging\.',
            r'track.*activity',
            r'record.*access'
        ]
        
        has_audit_logging = any(
            re.search(pattern, content, re.IGNORECASE)
            for pattern in audit_patterns
        )
        
        if has_audit_logging:
            return ComplianceCheck(
                standard="HIPAA",
                requirement="Audit Logging",
                status="compliant",
                details="Audit logging mechanisms detected",
                evidence=["Logging patterns found in code"]
            )
        else:
            return ComplianceCheck(
                standard="HIPAA",
                requirement="Audit Logging",
                status="non_compliant",
                details="No audit logging detected",
                remediation="Implement comprehensive audit logging for all PHI access",
                severity=Severity.HIGH
            )

    async def _validate_data_minimization(
        self,
        content: str,
        file_path: str
    ) -> ComplianceCheck:
        """Validate data minimization principles"""
        
        # Check for excessive data collection patterns
        excessive_patterns = [
            r'SELECT \*',
            r'get.*all.*data',
            r'fetch.*everything',
            r'.*\.all\(\)'
        ]
        
        has_excessive_collection = any(
            re.search(pattern, content, re.IGNORECASE)
            for pattern in excessive_patterns
        )
        
        if has_excessive_collection:
            return ComplianceCheck(
                standard="HIPAA",
                requirement="Data Minimization",
                status="warning",
                details="Potential excessive data collection detected",
                remediation="Implement specific data queries to collect only necessary PHI",
                severity=Severity.MEDIUM
            )
        else:
            return ComplianceCheck(
                standard="HIPAA",
                requirement="Data Minimization",
                status="compliant",
                details="No excessive data collection patterns detected",
                evidence=["Specific data queries found"]
            )

    async def _validate_software_architecture(
        self,
        project_files: List[str]
    ) -> List[ComplianceCheck]:
        """Validate software architecture for medical devices"""
        checks = []
        
        # Check for modular architecture
        has_modules = any('module' in file_path.lower() for file_path in project_files)
        checks.append(ComplianceCheck(
            standard="IEC 62304",
            requirement="Modular Architecture",
            status="compliant" if has_modules else "warning",
            details="Modular architecture detected" if has_modules else "Consider modular architecture",
            severity=Severity.MEDIUM if not has_modules else Severity.LOW
        ))
        
        # Check for separation of concerns
        has_separation = any(
            any(concern in file_path.lower() for concern in ['controller', 'service', 'model', 'view'])
            for file_path in project_files
        )
        checks.append(ComplianceCheck(
            standard="IEC 62304",
            requirement="Separation of Concerns",
            status="compliant" if has_separation else "warning",
            details="Separation of concerns detected" if has_separation else "Implement clear separation of concerns",
            severity=Severity.MEDIUM if not has_separation else Severity.LOW
        ))
        
        return checks

    async def _validate_risk_controls(
        self,
        device_profile: MedicalDeviceProfile,
        project_files: List[str]
    ) -> List[ComplianceCheck]:
        """Validate risk control implementation"""
        checks = []
        
        for risk_control in device_profile.risk_controls:
            # Check if risk control is implemented
            control_implemented = any(
                risk_control.lower() in file_path.lower()
                for file_path in project_files
            )
            
            checks.append(ComplianceCheck(
                standard="IEC 62304",
                requirement=f"Risk Control: {risk_control}",
                status="compliant" if control_implemented else "non_compliant",
                details=f"Risk control '{risk_control}' implementation status",
                remediation=f"Implement risk control for {risk_control}" if not control_implemented else None,
                severity=Severity.HIGH if not control_implemented else Severity.LOW
            ))
        
        return checks

    async def _validate_fda_requirements(
        self,
        device_profile: MedicalDeviceProfile,
        project_files: List[str]
    ) -> List[ComplianceCheck]:
        """Validate FDA requirements for Class II/III devices"""
        checks = []
        
        # 21 CFR Part 820 requirements
        required_docs = ['design_controls', 'verification', 'validation', 'risk_management']
        
        for doc_type in required_docs:
            has_doc = any(doc_type in file_path.lower() for file_path in project_files)
            checks.append(ComplianceCheck(
                standard="21 CFR Part 820",
                requirement=f"{doc_type.replace('_', ' ').title()} Documentation",
                status="compliant" if has_doc else "non_compliant",
                details=f"{doc_type.replace('_', ' ').title()} documentation status",
                remediation=f"Create {doc_type.replace('_', ' ')} documentation" if not has_doc else None,
                severity=Severity.HIGH if not has_doc else Severity.LOW
            ))
        
        return checks

    async def _validate_documentation_requirements(
        self,
        project_files: List[str]
    ) -> List[ComplianceCheck]:
        """Validate documentation requirements"""
        checks = []
        
        doc_types = ['requirements', 'design', 'testing', 'validation', 'user_manual']
        
        for doc_type in doc_types:
            has_doc = any(
                doc_type in file_path.lower() or 'doc' in file_path.lower()
                for file_path in project_files
            )
            
            checks.append(ComplianceCheck(
                standard="IEC 62304",
                requirement=f"{doc_type.replace('_', ' ').title()} Documentation",
                status="compliant" if has_doc else "warning",
                details=f"{doc_type.replace('_', ' ').title()} documentation status",
                severity=Severity.MEDIUM if not has_doc else Severity.LOW
            ))
        
        return checks

    async def _validate_testing_requirements(
        self,
        project_files: List[str]
    ) -> List[ComplianceCheck]:
        """Validate testing requirements for medical devices"""
        checks = []
        
        test_types = ['unit', 'integration', 'system', 'acceptance', 'performance']
        
        for test_type in test_types:
            has_tests = any(
                f"{test_type}_test" in file_path.lower() or 
                f"test_{test_type}" in file_path.lower() or
                (test_type in file_path.lower() and 'test' in file_path.lower())
                for file_path in project_files
            )
            
            checks.append(ComplianceCheck(
                standard="IEC 62304",
                requirement=f"{test_type.title()} Testing",
                status="compliant" if has_tests else "warning",
                details=f"{test_type.title()} testing implementation status",
                severity=Severity.MEDIUM if not has_tests else Severity.LOW
            ))
        
        return checks

    async def _generate_fhir_resource_pattern(self, resource_type: str) -> Dict[str, Any]:
        """Generate FHIR resource pattern"""
        return {
            'resource_type': resource_type,
            'structure': f"FHIR {resource_type} resource structure",
            'validation_rules': [f"Validate {resource_type} required fields"],
            'security_considerations': [f"Secure {resource_type} data transmission"]
        }

    async def _generate_fhir_interactions(
        self,
        use_case: str,
        data_types: List[str]
    ) -> Dict[str, Any]:
        """Generate FHIR interaction patterns"""
        return {
            'read_operations': [f"GET /{dt}" for dt in data_types],
            'write_operations': [f"POST /{dt}" for dt in data_types],
            'search_operations': [f"GET /{dt}?param=value" for dt in data_types],
            'bundle_operations': ["POST /Bundle"]
        }

    async def _generate_fhir_security_patterns(self, use_case: str) -> Dict[str, Any]:
        """Generate FHIR security patterns"""
        return {
            'authentication': ['OAuth 2.0', 'SMART on FHIR'],
            'authorization': ['RBAC', 'ABAC'],
            'transport_security': ['TLS 1.3', 'Certificate validation'],
            'data_encryption': ['AES-256', 'Field-level encryption']
        }

    async def _generate_fhir_validation_patterns(self, data_types: List[str]) -> Dict[str, Any]:
        """Generate FHIR validation patterns"""
        return {
            'schema_validation': [f"{dt} schema validation" for dt in data_types],
            'business_rules': [f"{dt} business rule validation" for dt in data_types],
            'terminology_validation': ['SNOMED CT', 'ICD-10', 'LOINC']
        }

    async def _generate_fhir_code_examples(
        self,
        use_case: str,
        data_types: List[str]
    ) -> Dict[str, Any]:
        """Generate FHIR code examples"""
        return {
            'patient_creation': '''
            patient = Patient()
            patient.name = [HumanName(family="Doe", given=["John"])]
            patient.gender = "male"
            patient.birthDate = "1990-01-01"
            ''',
            'observation_creation': '''
            observation = Observation()
            observation.status = "final"
            observation.code = CodeableConcept(
                coding=[Coding(system="http://loinc.org", code="29463-7")]
            )
            observation.subject = Reference(reference="Patient/123")
            '''
        }

    async def _validate_workflow_steps(self, workflow_definition: Dict[str, Any]) -> List[Issue]:
        """Validate clinical workflow steps"""
        issues = []
        
        if 'steps' not in workflow_definition:
            issues.append(Issue(
                type=IssueType.INCOMPLETE_IMPLEMENTATION,
                severity=Severity.HIGH,
                description="Workflow steps not defined"
            ))
        
        return issues

    async def _validate_clinical_safety_checks(
        self,
        workflow_definition: Dict[str, Any],
        file_paths: List[str]
    ) -> List[Issue]:
        """Validate clinical safety checks"""
        issues = []
        
        safety_patterns = ['safety_check', 'validate_dosage', 'allergy_check', 'interaction_check']
        
        has_safety_checks = any(
            any(pattern in file_path.lower() for pattern in safety_patterns)
            for file_path in file_paths
        )
        
        if not has_safety_checks:
            issues.append(Issue(
                type=IssueType.INCOMPLETE_IMPLEMENTATION,
                severity=Severity.CRITICAL,
                description="Clinical safety checks not implemented"
            ))
        
        return issues

    async def _validate_clinical_decision_points(
        self,
        workflow_definition: Dict[str, Any]
    ) -> List[Issue]:
        """Validate clinical decision points"""
        issues = []
        
        if 'decision_points' not in workflow_definition:
            issues.append(Issue(
                type=IssueType.INCOMPLETE_IMPLEMENTATION,
                severity=Severity.MEDIUM,
                description="Clinical decision points not defined"
            ))
        
        return issues

    async def _validate_clinical_error_handling(self, file_paths: List[str]) -> List[Issue]:
        """Validate clinical error handling"""
        issues = []
        
        error_handling_patterns = ['try_catch', 'error_handling', 'exception', 'failure_mode']
        
        has_error_handling = any(
            any(pattern in file_path.lower() for pattern in error_handling_patterns)
            for file_path in file_paths
        )
        
        if not has_error_handling:
            issues.append(Issue(
                type=IssueType.INCOMPLETE_IMPLEMENTATION,
                severity=Severity.HIGH,
                description="Clinical error handling not implemented"
            ))
        
        return issues