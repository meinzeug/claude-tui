#!/usr/bin/env python3
"""
Import Standardization Script for Claude-TUI
Fixes all relative imports to absolute 'src.module.component' format
Removes fallback implementations and mock widgets
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

def get_python_files(directory: Path) -> List[Path]:
    """Get all Python files in directory recursively."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ and .git directories
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', '.pytest_cache']]
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    return python_files

def standardize_imports(file_path: Path, src_root: Path) -> bool:
    """Standardize imports in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Get relative path from src root
        rel_path = file_path.relative_to(src_root)
        module_path = str(rel_path.parent).replace(os.sep, '.')
        
        # Fix relative imports to absolute imports
        patterns = [
            # from .module import something -> from src.current.module import something
            (r'from \.([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import', 
             rf'from src.{module_path}.\1 import'),
            # from ..module import something -> from src.parent.module import something
            (r'from \.\.([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import', 
             lambda m: f'from src.{".".join(module_path.split(".")[:-1])}.{m.group(1)} import'),
            # from ...module import something -> from src.grandparent.module import something  
            (r'from \.\.\.([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import', 
             lambda m: f'from src.{".".join(module_path.split(".")[:-2])}.{m.group(1)} import'),
            # import .module -> import src.current.module
            (r'import \.([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)', 
             rf'import src.{module_path}.\1'),
        ]
        
        for pattern, replacement in patterns:
            if callable(replacement):
                content = re.sub(pattern, replacement, content)
            else:
                content = re.sub(pattern, replacement, content)
        
        # Fix claude_tui.* imports to src.claude_tui.*
        content = re.sub(r'from claude_tui\.', 'from src.claude_tui.', content)
        content = re.sub(r'import claude_tui\.', 'import src.claude_tui.', content)
        
        # Remove fallback/mock implementations
        fallback_patterns = [
            # Remove try/except ImportError blocks with fallbacks
            r'try:\s*\n((?:\s{4,}.*\n)*)\s*except ImportError.*?:\s*\n(?:(?:\s{4,}.*\n)*?)(?=\n(?:\S|$))',
            # Remove fallback class definitions
            r'class\s+Fallback\w+.*?(?=\nclass|\nfrom|\nimport|\ndef|\n$)',
            # Remove mock class definitions
            r'class\s+Mock\w+.*?(?=\nclass|\nfrom|\nimport|\ndef|\n$)',
        ]
        
        for pattern in fallback_patterns:
            content = re.sub(pattern, '', content, flags=re.MULTILINE | re.DOTALL)
        
        # Clean up extra newlines
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to standardize imports."""
    if len(sys.argv) > 1:
        src_root = Path(sys.argv[1])
    else:
        src_root = Path('/home/tekkadmin/claude-tui/src')
    
    if not src_root.exists():
        print(f"Source root {src_root} does not exist")
        return 1
    
    print(f"Standardizing imports in {src_root}")
    
    python_files = get_python_files(src_root)
    print(f"Found {len(python_files)} Python files")
    
    modified_count = 0
    for file_path in python_files:
        if standardize_imports(file_path, src_root):
            modified_count += 1
            print(f"Modified: {file_path}")
    
    print(f"\nCompleted: {modified_count} files modified")
    return 0

if __name__ == '__main__':
    sys.exit(main())