"""
Pydantic compatibility layer for v1/v2 compatibility.

This module provides compatibility between Pydantic v1 and v2 by implementing
the v2 field_validator functionality for v1 systems.
"""

import sys
from typing import Any, Callable, Optional, Union

import pydantic
from pydantic import __version__ as pydantic_version


def get_pydantic_major_version() -> int:
    """Get the major version of Pydantic."""
    return int(pydantic_version.split('.')[0])


PYDANTIC_V2 = get_pydantic_major_version() >= 2


if PYDANTIC_V2:
    # Pydantic v2 - use native imports
    from pydantic import field_validator as _field_validator
    from pydantic import BaseModel, Field
    from pydantic.v1 import validator as v1_validator  # For backwards compatibility
    
    def field_validator(*fields, **kwargs):
        """Wrapper for Pydantic v2 field_validator."""
        return _field_validator(*fields, **kwargs)
        
    def validator(*fields, **kwargs):
        """Wrapper for backwards compatibility with v1 validator."""
        return v1_validator(*fields, **kwargs)

else:
    # Pydantic v1 - provide compatibility layer
    from pydantic import BaseModel, Field
    from pydantic import validator as _validator
    
    def field_validator(*fields, mode: str = 'after', **kwargs):
        """
        Compatibility wrapper that maps Pydantic v2 field_validator to v1 validator.
        
        Args:
            *fields: Field names to validate
            mode: Validation mode ('before', 'after'). Default 'after'
            **kwargs: Additional validator arguments
            
        Returns:
            Decorator function for validation methods
        """
        # Map v2 mode to v1 pre/post behavior
        if mode == 'before':
            kwargs['pre'] = True
        elif mode == 'after':
            kwargs['pre'] = False
        else:
            # Default to post-validation for v1
            kwargs['pre'] = False
            
        return _validator(*fields, **kwargs)
    
    def validator(*fields, **kwargs):
        """Native Pydantic v1 validator."""
        return _validator(*fields, **kwargs)


class CompatBaseModel(BaseModel):
    """
    Base model that provides compatibility between Pydantic v1 and v2.
    
    This class ensures that models work consistently across both versions
    by handling configuration differences and providing unified behavior.
    """
    
    if PYDANTIC_V2:
        # Pydantic v2 configuration
        model_config = {
            'use_enum_values': True,
            'validate_assignment': True,
            'arbitrary_types_allowed': True,
        }
    else:
        # Pydantic v1 configuration
        class Config:
            use_enum_values = True
            validate_assignment = True
            arbitrary_types_allowed = True


def create_model_with_compat(model_class):
    """
    Decorator to ensure model compatibility between Pydantic versions.
    
    Args:
        model_class: The Pydantic model class to make compatible
        
    Returns:
        The model class with compatibility fixes applied
    """
    if not PYDANTIC_V2:
        # For v1, ensure proper configuration
        if not hasattr(model_class, 'Config'):
            model_class.Config = type('Config', (), {
                'use_enum_values': True,
                'validate_assignment': True,
                'arbitrary_types_allowed': True,
            })
    
    return model_class


def safe_json_encoders():
    """
    Get JSON encoders that work across Pydantic versions.
    
    Returns:
        Dictionary of JSON encoders compatible with current Pydantic version
    """
    from datetime import datetime
    from uuid import UUID
    
    if PYDANTIC_V2:
        # v2 uses model_config with serialization
        return {
            UUID: str,
            datetime: lambda v: v.isoformat(),
            set: list
        }
    else:
        # v1 uses Config.json_encoders
        return {
            UUID: str,
            datetime: lambda v: v.isoformat(),
            set: list
        }


def get_field_info(*args, **kwargs):
    """
    Get field information compatible with current Pydantic version.
    
    Returns:
        Field info using the appropriate Pydantic version's Field function
    """
    return Field(*args, **kwargs)


# Export compatibility symbols
__all__ = [
    'field_validator',
    'validator', 
    'BaseModel',
    'Field',
    'CompatBaseModel',
    'create_model_with_compat',
    'safe_json_encoders',
    'get_field_info',
    'PYDANTIC_V2',
    'get_pydantic_major_version'
]