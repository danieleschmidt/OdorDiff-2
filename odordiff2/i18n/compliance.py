"""
Regional compliance and regulatory checks for fragrances.
"""

from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ComplianceResult:
    """Result of regulatory compliance check."""
    is_compliant: bool
    region: str
    violations: List[str]
    warnings: List[str]
    restrictions: List[str]
    max_concentration: Optional[float] = None

# IFRA (International Fragrance Association) restrictions by region
IFRA_RESTRICTIONS = {
    'EU': {
        'benzyl_salicylate': {'max_concentration': 0.03, 'restriction_type': 'allergen'},
        'linalool': {'max_concentration': 0.01, 'restriction_type': 'allergen'},
        'limonene': {'max_concentration': 0.01, 'restriction_type': 'allergen'},
        'citronellol': {'max_concentration': 0.01, 'restriction_type': 'allergen'},
        'geraniol': {'max_concentration': 0.01, 'restriction_type': 'allergen'},
        'eugenol': {'max_concentration': 0.01, 'restriction_type': 'allergen'},
        'oakmoss': {'max_concentration': 0.001, 'restriction_type': 'banned_ingredient'},
        'treemoss': {'max_concentration': 0.001, 'restriction_type': 'banned_ingredient'}
    },
    'US': {
        'benzyl_salicylate': {'max_concentration': 0.05, 'restriction_type': 'limited'},
        'linalool': {'max_concentration': 0.02, 'restriction_type': 'limited'},
        'limonene': {'max_concentration': 0.02, 'restriction_type': 'limited'},
        'phthalates': {'max_concentration': 0.0, 'restriction_type': 'banned'},
        'diethyl_phthalate': {'max_concentration': 0.1, 'restriction_type': 'limited'}
    },
    'CA': {  # Canada
        'benzyl_salicylate': {'max_concentration': 0.04, 'restriction_type': 'limited'},
        'linalool': {'max_concentration': 0.015, 'restriction_type': 'allergen'},
        'limonene': {'max_concentration': 0.015, 'restriction_type': 'allergen'},
        'microplastics': {'max_concentration': 0.0, 'restriction_type': 'banned'}
    },
    'JP': {  # Japan
        'formaldehyde': {'max_concentration': 0.001, 'restriction_type': 'toxic'},
        'benzyl_salicylate': {'max_concentration': 0.02, 'restriction_type': 'limited'},
        'artificial_musks': {'max_concentration': 0.05, 'restriction_type': 'limited'}
    },
    'CN': {  # China
        'benzyl_salicylate': {'max_concentration': 0.02, 'restriction_type': 'limited'},
        'heavy_metals': {'max_concentration': 0.0001, 'restriction_type': 'toxic'},
        'prohibited_colorants': {'max_concentration': 0.0, 'restriction_type': 'banned'}
    }
}

# Regional labeling requirements
LABELING_REQUIREMENTS = {
    'EU': {
        'allergen_declaration_threshold': 0.001,  # 0.1% in rinse-off, 0.01% in leave-on
        'required_warnings': ['potential allergens must be declared'],
        'language_requirements': ['local language required'],
        'ingredient_listing': 'INCI names required'
    },
    'US': {
        'allergen_declaration_threshold': None,  # No specific threshold
        'required_warnings': ['for external use only'],
        'language_requirements': ['English required'],
        'ingredient_listing': 'ingredients must be listed in descending order'
    },
    'CA': {
        'allergen_declaration_threshold': 0.001,
        'required_warnings': ['potential allergens must be declared'],
        'language_requirements': ['English and French required'],
        'ingredient_listing': 'INCI names required'
    },
    'JP': {
        'allergen_declaration_threshold': 0.001,
        'required_warnings': ['usage instructions in Japanese'],
        'language_requirements': ['Japanese required'],
        'ingredient_listing': 'Japanese ingredient names required'
    },
    'CN': {
        'allergen_declaration_threshold': 0.001,
        'required_warnings': ['import registration required'],
        'language_requirements': ['Simplified Chinese required'],
        'ingredient_listing': 'Chinese names required'
    }
}

# Marketing and cultural restrictions
CULTURAL_RESTRICTIONS = {
    'EU': {
        'animal_testing': 'prohibited',
        'sustainability_claims': 'must be substantiated',
        'natural_claims': 'regulated'
    },
    'US': {
        'animal_testing': 'not required but not prohibited',
        'organic_claims': 'USDA regulated',
        'cruelty_free_claims': 'voluntary'
    },
    'CA': {
        'animal_testing': 'prohibited for cosmetics',
        'natural_claims': 'regulated',
        'bilingual_labeling': 'required'
    },
    'JP': {
        'animal_testing': 'not prohibited but discouraged',
        'traditional_ingredients': 'respected',
        'minimalist_formulation': 'preferred'
    },
    'CN': {
        'animal_testing': 'may be required for imports',
        'traditional_medicine_claims': 'regulated',
        'import_permits': 'required'
    }
}

class RegionalComplianceChecker:
    """Check fragrance compliance with regional regulations."""
    
    def __init__(self):
        self.restrictions = IFRA_RESTRICTIONS
        self.labeling_requirements = LABELING_REQUIREMENTS
        self.cultural_restrictions = CULTURAL_RESTRICTIONS
    
    def check_molecule_compliance(self, molecule_name: str, concentration: float, 
                                region: str = 'EU') -> ComplianceResult:
        """Check if a molecule complies with regional regulations."""
        violations = []
        warnings = []
        restrictions = []
        max_allowed_concentration = None
        
        regional_restrictions = self.restrictions.get(region, {})
        
        # Check specific molecule restrictions
        if molecule_name.lower() in regional_restrictions:
            restriction = regional_restrictions[molecule_name.lower()]
            max_allowed = restriction['max_concentration']
            restriction_type = restriction['restriction_type']
            
            max_allowed_concentration = max_allowed
            
            if concentration > max_allowed:
                if restriction_type == 'banned_ingredient':
                    violations.append(f"{molecule_name} is banned in {region}")
                elif restriction_type == 'allergen':
                    violations.append(f"{molecule_name} exceeds allergen limit ({concentration:.3f} > {max_allowed:.3f})")
                else:
                    violations.append(f"{molecule_name} exceeds concentration limit in {region}")
            
            if restriction_type == 'allergen' and concentration > 0.001:
                restrictions.append(f"{molecule_name} must be declared as allergen")
        
        # General safety checks
        if concentration > 0.1:  # 10% concentration
            warnings.append(f"High concentration of {molecule_name} - consider safety assessment")
        
        # Check labeling requirements
        labeling_reqs = self.labeling_requirements.get(region, {})
        if labeling_reqs.get('allergen_declaration_threshold'):
            threshold = labeling_reqs['allergen_declaration_threshold']
            if concentration > threshold:
                restrictions.append(f"{molecule_name} must be declared on label (>{threshold*100:.2f}%)")
        
        is_compliant = len(violations) == 0
        
        return ComplianceResult(
            is_compliant=is_compliant,
            region=region,
            violations=violations,
            warnings=warnings,
            restrictions=restrictions,
            max_concentration=max_allowed_concentration
        )
    
    def check_formula_compliance(self, formula: Dict, region: str = 'EU') -> ComplianceResult:
        """Check entire fragrance formula for compliance."""
        all_violations = []
        all_warnings = []
        all_restrictions = []
        
        # Check individual components
        for ingredient, concentration in formula.items():
            result = self.check_molecule_compliance(ingredient, concentration, region)
            all_violations.extend(result.violations)
            all_warnings.extend(result.warnings)
            all_restrictions.extend(result.restrictions)
        
        # Check total allergen content
        total_allergens = self._calculate_total_allergens(formula, region)
        if total_allergens > 0.05:  # 5% total allergens
            all_warnings.append(f"High total allergen content: {total_allergens*100:.1f}%")
        
        # Regional specific checks
        if region == 'EU':
            # Check for banned UV filters
            uv_filters = ['benzophenone-3', 'octinoxate']
            for uv_filter in uv_filters:
                if uv_filter in formula:
                    all_violations.append(f"{uv_filter} is restricted in EU cosmetics")
        
        elif region == 'CN':
            # Check import requirements
            all_restrictions.append("Import registration and animal testing may be required")
        
        is_compliant = len(all_violations) == 0
        
        return ComplianceResult(
            is_compliant=is_compliant,
            region=region,
            violations=all_violations,
            warnings=all_warnings,
            restrictions=all_restrictions
        )
    
    def _calculate_total_allergens(self, formula: Dict, region: str) -> float:
        """Calculate total allergen content in formula."""
        allergens = ['linalool', 'limonene', 'citronellol', 'geraniol', 'eugenol', 
                    'benzyl_salicylate', 'benzyl_benzoate', 'farnesol']
        
        total = 0.0
        for ingredient, concentration in formula.items():
            if ingredient.lower() in allergens:
                total += concentration
        
        return total
    
    def get_regional_guidelines(self, region: str) -> Dict:
        """Get regulatory guidelines for a region."""
        guidelines = {
            'restrictions': self.restrictions.get(region, {}),
            'labeling': self.labeling_requirements.get(region, {}),
            'cultural': self.cultural_restrictions.get(region, {})
        }
        
        return guidelines

# Global compliance checker instance
_compliance_checker = RegionalComplianceChecker()

def check_regional_compliance(formula: Dict, region: str = 'EU') -> ComplianceResult:
    """Check formula compliance (convenience function)."""
    return _compliance_checker.check_formula_compliance(formula, region)

def check_molecule_compliance(molecule_name: str, concentration: float, 
                            region: str = 'EU') -> ComplianceResult:
    """Check single molecule compliance (convenience function)."""
    return _compliance_checker.check_molecule_compliance(molecule_name, concentration, region)

def get_supported_regions() -> List[str]:
    """Get list of supported regulatory regions."""
    return list(IFRA_RESTRICTIONS.keys())

def get_regional_guidelines(region: str) -> Dict:
    """Get regulatory guidelines for region (convenience function)."""
    return _compliance_checker.get_regional_guidelines(region)