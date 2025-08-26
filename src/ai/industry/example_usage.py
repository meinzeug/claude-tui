"""
Industry Intelligence Module Usage Examples

This module demonstrates how to use the industry-specific AI intelligence modules
for different domains and compliance requirements.
"""

import asyncio
import logging
from pathlib import Path
from typing import List

from .industry_intelligence_coordinator import IndustryIntelligenceCoordinator, Industry
from .healthcare_intelligence import HealthcareIntelligence, MedicalDeviceClass, MedicalDeviceProfile
from .financial_intelligence import FinancialIntelligence, TradingSystemProfile
from .aerospace_intelligence import AerospaceIntelligence, AviationSystemProfile, DesignAssuranceLevel
from .automotive_intelligence import AutomotiveIntelligence, AutomotiveSystemProfile, ASILLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_healthcare_compliance():
    """Example: Healthcare application HIPAA compliance validation"""
    
    print("\n=== Healthcare Intelligence Example ===")
    
    healthcare_ai = HealthcareIntelligence()
    
    # Example medical device profile
    device_profile = MedicalDeviceProfile(
        device_class=MedicalDeviceClass.CLASS_II,
        safety_classification="B",
        intended_use="Patient monitoring system",
        risk_controls=["alarm_system", "backup_power", "data_integrity"],
        regulatory_requirements=["FDA_510k", "IEC_62304"]
    )
    
    # Example code content for analysis
    sample_code = '''
    import hashlib
    import logging
    from cryptography.fernet import Fernet

    class PatientDataManager:
        def __init__(self):
            self.encryption_key = Fernet.generate_key()
            self.cipher = Fernet(self.encryption_key)
            
        def store_patient_data(self, patient_id, medical_record):
            # Encrypt PHI before storage
            encrypted_data = self.cipher.encrypt(medical_record.encode())
            logging.info(f"Storing encrypted data for patient {patient_id}")
            return encrypted_data
            
        def authenticate_user(self, username, password):
            # Multi-factor authentication required
            return self.verify_credentials(username, password)
    '''
    
    # Validate HIPAA compliance
    compliance_checks = await healthcare_ai.validate_hipaa_compliance(
        "patient_manager.py",
        sample_code
    )
    
    print(f"HIPAA Compliance Checks: {len(compliance_checks)} checks performed")
    for check in compliance_checks:
        print(f"  - {check.requirement}: {check.status}")
        if check.remediation:
            print(f"    Remediation: {check.remediation}")


async def example_financial_compliance():
    """Example: Financial application PCI DSS compliance validation"""
    
    print("\n=== Financial Intelligence Example ===")
    
    financial_ai = FinancialIntelligence()
    
    # Example trading system profile
    trading_profile = TradingSystemProfile(
        system_type="algorithmic",
        asset_classes=["equities", "bonds"],
        latency_requirements="low",
        risk_limits={"daily_var": 100000, "position_limit": 50000}
    )
    
    # Example code content
    sample_code = '''
    import ssl
    import bcrypt
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    class PaymentProcessor:
        def __init__(self):
            self.tls_context = ssl.create_default_context()
            
        def process_payment(self, card_number, amount):
            # Encrypt card data with AES-256
            encrypted_card = self.encrypt_card_data(card_number)
            
            # Log transaction for audit
            logging.info(f"Processing payment: ${amount}")
            
            return self.submit_to_processor(encrypted_card, amount)
            
        def encrypt_card_data(self, card_number):
            # PCI DSS compliant encryption
            key = os.urandom(32)  # AES-256 key
            cipher = Cipher(algorithms.AES(key), modes.CBC(os.urandom(16)))
            encryptor = cipher.encryptor()
            return encryptor.update(card_number.encode()) + encryptor.finalize()
    '''
    
    # Validate PCI DSS compliance
    compliance_checks = await financial_ai.validate_pci_dss_compliance(
        "payment_processor.py",
        sample_code,
        merchant_level="level_1"
    )
    
    print(f"PCI DSS Compliance Checks: {len(compliance_checks)} checks performed")
    for check in compliance_checks:
        print(f"  - {check.requirement}: {check.status} (Risk: {check.risk_level})")


async def example_cross_industry_analysis():
    """Example: Cross-industry compliance analysis"""
    
    print("\n=== Cross-Industry Intelligence Example ===")
    
    coordinator = IndustryIntelligenceCoordinator()
    
    # Simulate project files
    project_files = [
        "src/patient_data.py",
        "src/payment_processing.py", 
        "src/security/encryption.py",
        "src/database/models.py",
        "tests/test_compliance.py"
    ]
    
    # Create sample files for analysis
    sample_files_content = {
        "src/patient_data.py": '''
        # Healthcare data processing
        import hipaa_utils
        from cryptography.fernet import Fernet
        
        class PatientRecord:
            def __init__(self, patient_id, medical_data):
                self.patient_id = patient_id
                self.encrypted_data = self.encrypt_phi(medical_data)
                
            def encrypt_phi(self, data):
                key = Fernet.generate_key()
                f = Fernet(key)
                return f.encrypt(data.encode())
        ''',
        "src/payment_processing.py": '''
        # Financial payment processing
        import pci_dss_utils
        
        class PaymentGateway:
            def process_card_payment(self, card_number, cvv, amount):
                # PCI DSS compliant processing
                encrypted_card = self.tokenize_card(card_number)
                return self.charge_card(encrypted_card, amount)
        ''',
        "src/security/encryption.py": '''
        # Cross-industry encryption utilities
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        import os
        
        class EncryptionManager:
            def __init__(self):
                self.aes_key = os.urandom(32)  # AES-256
                
            def encrypt_sensitive_data(self, data):
                iv = os.urandom(16)
                cipher = Cipher(algorithms.AES(self.aes_key), modes.CBC(iv))
                encryptor = cipher.encryptor()
                return encryptor.update(data.encode()) + encryptor.finalize()
        '''
    }
    
    # Write sample files temporarily
    for file_path, content in sample_files_content.items():
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
    
    try:
        # Detect industry profile
        industry_profile = await coordinator.detect_industry_profile(
            project_files=list(sample_files_content.keys()),
            project_description="Healthcare fintech application with patient payment processing"
        )
        
        print(f"Detected Primary Industry: {industry_profile.primary_industry.value}")
        print(f"Secondary Industries: {[ind.value for ind in industry_profile.secondary_industries]}")
        print(f"Regulatory Scope: {industry_profile.regulatory_scope}")
        
        # Perform cross-industry compliance analysis
        analysis = await coordinator.validate_cross_industry_compliance(
            industry_profile,
            list(sample_files_content.keys())
        )
        
        print(f"\nOverall Compliance Level: {analysis.overall_compliance_level.value}")
        print(f"Compliance Score: {analysis.compliance_score:.1f}%")
        
        print(f"\nUnified Recommendations ({len(analysis.unified_recommendations)}):")
        for i, rec in enumerate(analysis.unified_recommendations[:5], 1):
            print(f"  {i}. {rec}")
        
        if analysis.cross_industry_conflicts:
            print(f"\nCross-Industry Conflicts ({len(analysis.cross_industry_conflicts)}):")
            for conflict in analysis.cross_industry_conflicts:
                print(f"  - {conflict['type']}: {conflict['description']}")
        
        # Generate compliance report
        report = await coordinator.generate_compliance_report(analysis, "summary")
        
        print(f"\nExecutive Summary:")
        summary = report['executive_summary']
        print(f"  Primary Industry: {summary['primary_industry']}")
        print(f"  Compliance Score: {summary['compliance_score']:.1f}%")
        print(f"  Critical Issues: {len(summary['critical_issues'])}")
        
    finally:
        # Cleanup sample files
        for file_path in sample_files_content.keys():
            try:
                Path(file_path).unlink()
                # Remove empty directories
                parent = Path(file_path).parent
                if parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()
            except:
                pass


async def example_best_practices_generation():
    """Example: Generate industry best practices"""
    
    print("\n=== Industry Best Practices Example ===")
    
    coordinator = IndustryIntelligenceCoordinator()
    
    # Create a multi-industry profile
    from .industry_intelligence_coordinator import IndustryProfile
    
    profile = IndustryProfile(
        primary_industry=Industry.HEALTHCARE,
        secondary_industries=[Industry.FINANCIAL],
        regulatory_scope=["HIPAA", "PCI DSS", "GDPR"],
        compliance_level_required="high",
        risk_tolerance="low"
    )
    
    # Generate best practices
    best_practices = await coordinator.generate_industry_best_practices(
        profile,
        focus_areas=["security", "compliance", "data_protection"]
    )
    
    print("Primary Industry Best Practices:")
    primary = best_practices['primary_industry']
    for category, practices in primary.items():
        if practices:
            print(f"  {category.title()}:")
            for practice in practices[:2]:  # Show first 2
                print(f"    - {practice}")
    
    print("\nCross-Industry Best Practices:")
    cross = best_practices['cross_industry']
    for category, practices in cross.items():
        print(f"  {category.title()}:")
        for practice in practices[:2]:  # Show first 2
            print(f"    - {practice}")


async def main():
    """Run all examples"""
    
    print("Industry-Specific AI Intelligence Examples")
    print("=" * 50)
    
    try:
        await example_healthcare_compliance()
        await example_financial_compliance()
        await example_cross_industry_analysis()
        await example_best_practices_generation()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())