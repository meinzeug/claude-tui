#!/usr/bin/env python3
"""
Security Fix: Remove Hardcoded Password
Automatically fixes the hardcoded password vulnerability in init_database.py
"""

import os
import re
import secrets
import shutil
from pathlib import Path
from datetime import datetime

def fix_hardcoded_password():
    """Fix hardcoded password in database initialization script."""
    
    script_file = Path("scripts/init_database.py")
    backup_file = Path(f"scripts/init_database.py.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    if not script_file.exists():
        print(f"❌ File not found: {script_file}")
        return False
    
    print(f"🔒 Fixing hardcoded password vulnerability...")
    
    # Create backup
    shutil.copy2(script_file, backup_file)
    print(f"📋 Backup created: {backup_file}")
    
    # Read current content
    content = script_file.read_text()
    
    # Pattern to find hardcoded password
    password_pattern = r'password\s*=\s*["\']([^"\']+)["\']'
    
    # Find hardcoded passwords
    matches = re.findall(password_pattern, content)
    if matches:
        print(f"🚨 Found {len(matches)} hardcoded password(s): {matches}")
    
    # Fix: Replace hardcoded password with environment variable
    secure_replacement = '''password=os.getenv("DEV_ADMIN_PASSWORD", secrets.token_urlsafe(16))'''
    
    # Replace the specific hardcoded password
    fixed_content = re.sub(
        r'password\s*=\s*["\']DevAdmin123!["\']',
        secure_replacement,
        content
    )
    
    # Add necessary imports if not present
    imports_to_add = []
    if 'import secrets' not in fixed_content:
        imports_to_add.append('import secrets')
    if 'import os' not in fixed_content:
        imports_to_add.append('import os')
    
    if imports_to_add:
        # Add imports after existing imports
        import_section = []
        lines = fixed_content.split('\n')
        insert_position = 0
        
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                insert_position = i + 1
        
        # Insert new imports
        for import_stmt in reversed(imports_to_add):
            lines.insert(insert_position, import_stmt)
        
        fixed_content = '\n'.join(lines)
    
    # Write fixed content
    script_file.write_text(fixed_content)
    
    print(f"✅ Fixed hardcoded password vulnerability")
    print(f"📝 Updated file: {script_file}")
    
    # Generate secure password for environment variable
    secure_password = secrets.token_urlsafe(16)
    
    # Update .env.example file
    env_example_file = Path(".env.example")
    if env_example_file.exists():
        env_content = env_example_file.read_text()
        
        # Add DEV_ADMIN_PASSWORD if not present
        if "DEV_ADMIN_PASSWORD" not in env_content:
            env_content += f"\n# Database Development Admin Password\nDEV_ADMIN_PASSWORD={secure_password}\n"
            env_example_file.write_text(env_content)
            print(f"📝 Added DEV_ADMIN_PASSWORD to .env.example")
    
    # Create/update .env file for development
    env_file = Path(".env")
    if not env_file.exists():
        env_file.write_text(f"# Development Environment Variables\nDEV_ADMIN_PASSWORD={secure_password}\n")
        print(f"📝 Created .env file with secure password")
    else:
        env_content = env_file.read_text()
        if "DEV_ADMIN_PASSWORD" not in env_content:
            env_content += f"\nDEV_ADMIN_PASSWORD={secure_password}\n"
            env_file.write_text(env_content)
            print(f"📝 Added DEV_ADMIN_PASSWORD to .env")
    
    print(f"\n🔐 Security Fix Complete!")
    print(f"📋 Backup: {backup_file}")
    print(f"🔑 Generated secure password: {secure_password}")
    print(f"⚠️  Make sure to set DEV_ADMIN_PASSWORD environment variable in production!")
    
    return True

def verify_fix():
    """Verify that the fix was applied correctly."""
    script_file = Path("scripts/init_database.py")
    
    if not script_file.exists():
        print("❌ Script file not found for verification")
        return False
    
    content = script_file.read_text()
    
    # Check for hardcoded password
    hardcoded_pattern = r'password\s*=\s*["\']DevAdmin123!["\']'
    if re.search(hardcoded_pattern, content):
        print("❌ Hardcoded password still present!")
        return False
    
    # Check for secure replacement
    secure_pattern = r'os\.getenv\s*\(\s*["\']DEV_ADMIN_PASSWORD["\']'
    if not re.search(secure_pattern, content):
        print("❌ Secure password replacement not found!")
        return False
    
    # Check for necessary imports
    if 'import os' not in content:
        print("⚠️ Missing 'import os' statement")
        return False
    
    if 'import secrets' not in content:
        print("⚠️ Missing 'import secrets' statement")
        return False
    
    print("✅ Security fix verified successfully!")
    return True

def main():
    """Main function to fix hardcoded password vulnerability."""
    print("🛡️ Claude-TIU Security Fix: Hardcoded Password Removal")
    print("=" * 60)
    
    if fix_hardcoded_password():
        if verify_fix():
            print("\n✅ Security vulnerability successfully remediated!")
            print("\n📋 Next Steps:")
            print("1. Review the changes in scripts/init_database.py")
            print("2. Test the database initialization")
            print("3. Commit the security fix")
            print("4. Update deployment scripts with environment variables")
            return True
        else:
            print("\n❌ Verification failed. Please check the fix manually.")
            return False
    else:
        print("\n❌ Failed to apply security fix.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)