"""
Internationalization and localization support for OdorDiff-2.
"""

from .translator import translate, get_supported_languages, set_language
from .odor_descriptors import get_localized_odor_descriptors
from .compliance import check_regional_compliance

__all__ = [
    'translate',
    'get_supported_languages', 
    'set_language',
    'get_localized_odor_descriptors',
    'check_regional_compliance'
]