"""
Multi-language translation system for OdorDiff-2.
"""

from typing import Dict, List, Optional
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Default translations for key terms
DEFAULT_TRANSLATIONS = {
    'en': {
        'molecule': 'molecule',
        'fragrance': 'fragrance',
        'scent': 'scent',
        'safety': 'safety',
        'odor_profile': 'odor profile',
        'generation': 'generation',
        'validation': 'validation',
        'error': 'error',
        'floral': 'floral',
        'citrus': 'citrus', 
        'woody': 'woody',
        'fresh': 'fresh',
        'spicy': 'spicy',
        'sweet': 'sweet',
        'intensity': 'intensity',
        'longevity': 'longevity',
        'sillage': 'projection'
    },
    'es': {
        'molecule': 'molécula',
        'fragrance': 'fragancia',
        'scent': 'aroma',
        'safety': 'seguridad',
        'odor_profile': 'perfil olfativo',
        'generation': 'generación',
        'validation': 'validación',
        'error': 'error',
        'floral': 'floral',
        'citrus': 'cítrico',
        'woody': 'amaderado',
        'fresh': 'fresco',
        'spicy': 'especiado',
        'sweet': 'dulce',
        'intensity': 'intensidad',
        'longevity': 'longevidad',
        'sillage': 'proyección'
    },
    'fr': {
        'molecule': 'molécule',
        'fragrance': 'parfum',
        'scent': 'senteur',
        'safety': 'sécurité',
        'odor_profile': 'profil olfactif',
        'generation': 'génération',
        'validation': 'validation',
        'error': 'erreur',
        'floral': 'floral',
        'citrus': 'agrume',
        'woody': 'boisé',
        'fresh': 'frais',
        'spicy': 'épicé',
        'sweet': 'sucré',
        'intensity': 'intensité',
        'longevity': 'longévité',
        'sillage': 'sillage'
    },
    'de': {
        'molecule': 'Molekül',
        'fragrance': 'Duft',
        'scent': 'Geruch',
        'safety': 'Sicherheit',
        'odor_profile': 'Duftprofil',
        'generation': 'Erzeugung',
        'validation': 'Validierung',
        'error': 'Fehler',
        'floral': 'blumig',
        'citrus': 'zitrusartig',
        'woody': 'holzig',
        'fresh': 'frisch',
        'spicy': 'würzig',
        'sweet': 'süß',
        'intensity': 'Intensität',
        'longevity': 'Haltbarkeit',
        'sillage': 'Sillage'
    },
    'ja': {
        'molecule': '分子',
        'fragrance': '香り',
        'scent': '匂い',
        'safety': '安全性',
        'odor_profile': '香りプロファイル',
        'generation': '生成',
        'validation': '検証',
        'error': 'エラー',
        'floral': '花の香り',
        'citrus': '柑橘系',
        'woody': '木質系',
        'fresh': 'フレッシュ',
        'spicy': 'スパイシー',
        'sweet': '甘い',
        'intensity': '強度',
        'longevity': '持続性',
        'sillage': 'シラージュ'
    },
    'zh': {
        'molecule': '分子',
        'fragrance': '香水',
        'scent': '气味',
        'safety': '安全性',
        'odor_profile': '气味轮廓',
        'generation': '生成',
        'validation': '验证',
        'error': '错误',
        'floral': '花香',
        'citrus': '柑橘',
        'woody': '木质',
        'fresh': '清新',
        'spicy': '辛辣',
        'sweet': '甜美',
        'intensity': '强度',
        'longevity': '持久性',
        'sillage': '扩散力'
    }
}

class Translator:
    """Multi-language translator for OdorDiff-2."""
    
    def __init__(self):
        self.current_language = 'en'
        self.translations = DEFAULT_TRANSLATIONS.copy()
        self.fallback_language = 'en'
        
    def set_language(self, language_code: str) -> bool:
        """Set the current language."""
        if language_code in self.translations:
            self.current_language = language_code
            logger.info(f"Language set to: {language_code}")
            return True
        else:
            logger.warning(f"Unsupported language: {language_code}")
            return False
    
    def translate(self, key: str, language: Optional[str] = None, **kwargs) -> str:
        """Translate a key to the specified language."""
        lang = language or self.current_language
        
        # Get translation from current language
        translation = self.translations.get(lang, {}).get(key)
        
        # Fallback to English if not found
        if translation is None:
            translation = self.translations.get(self.fallback_language, {}).get(key)
        
        # Ultimate fallback to the key itself
        if translation is None:
            translation = key
        
        # Format with kwargs if provided
        try:
            return translation.format(**kwargs) if kwargs else translation
        except KeyError:
            logger.warning(f"Translation formatting failed for key '{key}' with args {kwargs}")
            return translation
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return list(self.translations.keys())
    
    def add_translation(self, language: str, key: str, value: str):
        """Add a new translation."""
        if language not in self.translations:
            self.translations[language] = {}
        self.translations[language][key] = value
        logger.debug(f"Added translation: {language}.{key} = {value}")
    
    def load_translations_from_file(self, file_path: str):
        """Load translations from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_translations = json.load(f)
                
            for lang, translations in file_translations.items():
                if lang not in self.translations:
                    self.translations[lang] = {}
                self.translations[lang].update(translations)
            
            logger.info(f"Loaded translations from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load translations from {file_path}: {e}")

# Global translator instance
_translator = Translator()

def translate(key: str, language: Optional[str] = None, **kwargs) -> str:
    """Translate a key (convenience function)."""
    return _translator.translate(key, language, **kwargs)

def set_language(language_code: str) -> bool:
    """Set the current language (convenience function)."""
    return _translator.set_language(language_code)

def get_supported_languages() -> List[str]:
    """Get supported languages (convenience function)."""
    return _translator.get_supported_languages()

def get_translator() -> Translator:
    """Get the global translator instance."""
    return _translator