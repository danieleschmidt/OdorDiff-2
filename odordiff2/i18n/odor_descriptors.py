"""
Localized odor descriptors for different cultures and regions.
"""

from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

# Culture-specific odor descriptors and associations
CULTURAL_ODOR_DESCRIPTORS = {
    'en': {
        'floral': ['floral', 'flowery', 'rosy', 'jasmine-like', 'lilac', 'violet'],
        'citrus': ['citrus', 'lemony', 'lime-like', 'orange-blossom', 'bergamot', 'grapefruit'],
        'woody': ['woody', 'cedar', 'sandalwood', 'oak', 'pine', 'birch'],
        'fresh': ['fresh', 'clean', 'aquatic', 'marine', 'crisp', 'airy'],
        'spicy': ['spicy', 'peppery', 'cinnamon', 'clove', 'ginger', 'cardamom'],
        'sweet': ['sweet', 'vanilla', 'honey', 'caramel', 'sugar', 'candy'],
        'fruity': ['fruity', 'apple', 'pear', 'berry', 'peach', 'tropical'],
        'herbal': ['herbal', 'lavender', 'rosemary', 'thyme', 'mint', 'sage']
    },
    'es': {
        'floral': ['floral', 'florido', 'rosa', 'jazmín', 'azahar', 'violeta'],
        'citrus': ['cítrico', 'limón', 'lima', 'naranja', 'bergamota', 'pomelo'],
        'woody': ['amaderado', 'cedro', 'sándalo', 'roble', 'pino', 'abedul'],
        'fresh': ['fresco', 'limpio', 'acuático', 'marino', 'cristalino', 'aéreo'],
        'spicy': ['especiado', 'picante', 'canela', 'clavo', 'jengibre', 'cardamomo'],
        'sweet': ['dulce', 'vainilla', 'miel', 'caramelo', 'azúcar', 'golosina'],
        'fruity': ['afrutado', 'manzana', 'pera', 'baya', 'durazno', 'tropical'],
        'herbal': ['herbáceo', 'lavanda', 'romero', 'tomillo', 'menta', 'salvia']
    },
    'fr': {
        'floral': ['floral', 'fleuri', 'rose', 'jasmin', 'fleur d\'oranger', 'violette'],
        'citrus': ['agrume', 'citron', 'lime', 'orange', 'bergamote', 'pamplemousse'],
        'woody': ['boisé', 'cèdre', 'santal', 'chêne', 'pin', 'bouleau'],
        'fresh': ['frais', 'propre', 'aquatique', 'marin', 'cristallin', 'aérien'],
        'spicy': ['épicé', 'poivré', 'cannelle', 'girofle', 'gingembre', 'cardamome'],
        'sweet': ['sucré', 'vanille', 'miel', 'caramel', 'sucre', 'bonbon'],
        'fruity': ['fruité', 'pomme', 'poire', 'baie', 'pêche', 'tropical'],
        'herbal': ['herbacé', 'lavande', 'romarin', 'thym', 'menthe', 'sauge']
    },
    'de': {
        'floral': ['blumig', 'floriert', 'rosig', 'jasmin-artig', 'fliederartig', 'veilchenblau'],
        'citrus': ['zitrusartig', 'zitronig', 'limettenartig', 'orangenblüte', 'bergamotte', 'grapefruit'],
        'woody': ['holzig', 'zeder', 'sandelholz', 'eiche', 'kiefer', 'birke'],
        'fresh': ['frisch', 'sauber', 'wässrig', 'marin', 'kristallin', 'luftig'],
        'spicy': ['würzig', 'pfeffrig', 'zimt', 'nelke', 'ingwer', 'kardamom'],
        'sweet': ['süß', 'vanille', 'honig', 'karamell', 'zucker', 'bonbon'],
        'fruity': ['fruchtig', 'apfel', 'birne', 'beere', 'pfirsich', 'tropisch'],
        'herbal': ['kräuterartig', 'lavendel', 'rosmarin', 'thymian', 'minze', 'salbei']
    },
    'ja': {
        'floral': ['花の香り', '花のような', 'バラの香り', 'ジャスミンのような', 'ライラック', 'すみれ'],
        'citrus': ['柑橘系', 'レモンのような', 'ライムのような', 'オレンジブロッサム', 'ベルガモット', 'グレープフルーツ'],
        'woody': ['木質系', '杉', '白檀', 'オーク', '松', '樺'],
        'fresh': ['フレッシュ', '清潔な', 'アクアティック', 'マリン', 'クリスプ', 'エアリー'],
        'spicy': ['スパイシー', 'ペッパーのような', 'シナモン', 'クローブ', '生姜', 'カルダモン'],
        'sweet': ['甘い', 'バニラ', '蜂蜜', 'キャラメル', '砂糖', 'キャンディ'],
        'fruity': ['フルーティー', 'りんご', '梨', 'ベリー', '桃', 'トロピカル'],
        'herbal': ['ハーブ系', 'ラベンダー', 'ローズマリー', 'タイム', 'ミント', 'セージ']
    },
    'zh': {
        'floral': ['花香', '花卉', '玫瑰香', '茉莉花香', '紫丁香', '紫罗兰'],
        'citrus': ['柑橘', '柠檬香', '青柠香', '橙花', '佛手柑', '西柚'],
        'woody': ['木质', '雪松', '檀香', '橡木', '松木', '桦木'],
        'fresh': ['清新', '清洁', '水生', '海洋', '清脆', '空气感'],
        'spicy': ['辛辣', '胡椒', '肉桂', '丁香', '生姜', '小豆蔻'],
        'sweet': ['甜美', '香草', '蜂蜜', '焦糖', '糖', '糖果'],
        'fruity': ['果香', '苹果', '梨', '浆果', '桃子', '热带水果'],
        'herbal': ['草本', '薰衣草', '迷迭香', '百里香', '薄荷', '鼠尾草']
    }
}

# Regional preferences and cultural associations
REGIONAL_PREFERENCES = {
    'en': {
        'popular_notes': ['fresh', 'citrus', 'floral', 'woody'],
        'avoided_notes': [],
        'cultural_associations': {
            'rose': 'romance, femininity',
            'lavender': 'calm, relaxation',
            'vanilla': 'comfort, warmth'
        }
    },
    'es': {
        'popular_notes': ['floral', 'fruity', 'sweet', 'spicy'],
        'avoided_notes': [],
        'cultural_associations': {
            'rosa': 'romanticismo, feminidad',
            'lavanda': 'calma, relajación',
            'vainilla': 'comodidad, calidez'
        }
    },
    'fr': {
        'popular_notes': ['floral', 'woody', 'fresh', 'sophisticated'],
        'avoided_notes': [],
        'cultural_associations': {
            'rose': 'élégance, romantisme',
            'lavande': 'provence, détente',
            'vanille': 'douceur, gourmandise'
        }
    },
    'de': {
        'popular_notes': ['woody', 'fresh', 'herbal', 'clean'],
        'avoided_notes': ['overly_sweet'],
        'cultural_associations': {
            'rose': 'romantik, weiblichkeit',
            'lavendel': 'entspannung, natur',
            'vanille': 'gemütlichkeit, wärme'
        }
    },
    'ja': {
        'popular_notes': ['fresh', 'clean', 'subtle', 'natural'],
        'avoided_notes': ['too_strong', 'overpowering'],
        'cultural_associations': {
            'さくら': '春、美しさ、はかなさ',
            'ゆず': '冬、清潔、日本らしさ',
            'ひのき': '癒し、自然、和'
        }
    },
    'zh': {
        'popular_notes': ['floral', 'fresh', 'sweet', 'elegant'],
        'avoided_notes': ['too_bold'],
        'cultural_associations': {
            '茉莉': '纯洁，优雅，女性气质',
            '檀香': '宁静，冥想，传统',
            '桂花': '秋天，团圆，甜蜜'
        }
    }
}

def get_localized_odor_descriptors(category: str, language: str = 'en') -> List[str]:
    """Get localized odor descriptors for a category."""
    descriptors = CULTURAL_ODOR_DESCRIPTORS.get(language, {})
    return descriptors.get(category, CULTURAL_ODOR_DESCRIPTORS.get('en', {}).get(category, []))

def get_cultural_preferences(language: str = 'en') -> Dict:
    """Get cultural preferences for fragrances."""
    return REGIONAL_PREFERENCES.get(language, REGIONAL_PREFERENCES.get('en', {}))

def adapt_odor_description_to_culture(odor_notes: List[str], target_language: str = 'en') -> List[str]:
    """Adapt odor description to cultural context."""
    adapted_notes = []
    preferences = get_cultural_preferences(target_language)
    avoided_notes = preferences.get('avoided_notes', [])
    
    for note in odor_notes:
        # Skip culturally avoided notes
        if note in avoided_notes:
            continue
        
        # Find localized equivalent
        localized_descriptors = get_localized_odor_descriptors(note, target_language)
        if localized_descriptors:
            adapted_notes.append(localized_descriptors[0])  # Use primary descriptor
        else:
            adapted_notes.append(note)  # Keep original if no translation
    
    return adapted_notes

def get_cultural_fragrance_recommendations(preferences: Dict, language: str = 'en') -> List[Dict]:
    """Get culturally appropriate fragrance recommendations."""
    cultural_prefs = get_cultural_preferences(language)
    popular_notes = cultural_prefs.get('popular_notes', [])
    
    recommendations = []
    
    # Match user preferences with cultural preferences
    for note in popular_notes:
        if note in preferences.get('liked_notes', []):
            descriptors = get_localized_odor_descriptors(note, language)
            recommendations.append({
                'note_category': note,
                'descriptors': descriptors,
                'cultural_fit': 'high'
            })
    
    return recommendations