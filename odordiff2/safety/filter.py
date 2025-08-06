"""
Safety filtering system for generated molecules.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from ..models.molecule import Molecule, SafetyReport


class SafetyFilter:
    """
    Comprehensive safety assessment system for generated molecules.
    Filters out toxic, harmful, or non-compliant molecules.
    """
    
    def __init__(
        self, 
        toxicity_threshold: float = 0.1,
        irritant_check: bool = True,
        eco_threshold: float = 0.3,
        ifra_compliance: bool = True
    ):
        self.toxicity_threshold = toxicity_threshold
        self.irritant_check = irritant_check
        self.eco_threshold = eco_threshold
        self.ifra_compliance = ifra_compliance
        
        # Load known toxic/allergenic patterns
        self._toxic_patterns = self._load_toxic_patterns()
        self._allergenic_patterns = self._load_allergenic_patterns()
        self._banned_substructures = self._load_banned_substructures()
        
    def _load_toxic_patterns(self) -> List[str]:
        """Load known toxic SMARTS patterns."""
        return [
            # Heavy metals and toxic elements
            "[As]", "[Hg]", "[Pb]", "[Cd]", "[Be]",
            # Highly reactive/toxic functional groups
            "[N+](=O)[O-]",  # Nitro groups (some)
            "C(=O)Cl",       # Acid chlorides
            "[Si](Cl)(Cl)Cl", # Silicon tetrachloride
            "S(=O)(=O)Cl",   # Sulfonyl chlorides
            # Known carcinogenic patterns
            "c1ccc2c(c1)ccc3c2cccc3",  # Anthracene derivatives (some)
            "C=CC(=O)N",     # Acrylamide-like
        ]
    
    def _load_allergenic_patterns(self) -> List[str]:
        """Load known allergenic/sensitizing SMARTS patterns."""
        return [
            # Common fragrance allergens from EU regulation
            "CC(C)=CCc1ccc(C)cc1C",    # Limonene derivatives
            "CCc1ccc(C(C)C)cc1",       # p-Cymene derivatives  
            "CC(=CCO)C",               # Linalool derivatives
            "CC1=CCC(CC1)C(C)C",       # Terpinene derivatives
            "c1ccc(cc1)C=O",           # Benzaldehyde derivatives (some)
        ]
    
    def _load_banned_substructures(self) -> List[str]:
        """Load internationally banned/restricted substructures."""
        return [
            # IFRA prohibited materials
            "[Br]c1ccccc1[Br]",       # Dibrominated aromatics
            "C(Cl)(Cl)Cl",            # Chloroform-like
            "CCl4",                   # Carbon tetrachloride
            "c1ccc2c(c1)oc3ccccc23",  # Furocoumarins (photosensitizers)
            # Endocrine disruptors
            "c1ccc(cc1)c2ccc(cc2)O",  # Some phenolic compounds
        ]
    
    def assess_molecule(self, molecule: Molecule) -> SafetyReport:
        """
        Perform comprehensive safety assessment of a molecule.
        
        Args:
            molecule: Molecule object to assess
            
        Returns:
            SafetyReport with detailed safety analysis
        """
        if not molecule.is_valid:
            return SafetyReport(
                toxicity=1.0,
                skin_sensitizer=True,
                eco_score=1.0,
                ifra_compliant=False,
                regulatory_flags=[{"region": "ALL", "status": "INVALID_STRUCTURE"}]
            )
        
        # Run all safety checks
        toxicity_score = self._assess_toxicity(molecule)
        sensitizer_risk = self._check_skin_sensitization(molecule)
        eco_impact = self._assess_environmental_impact(molecule)
        ifra_status = self._check_ifra_compliance(molecule)
        regulatory_flags = self._check_regulatory_compliance(molecule)
        
        return SafetyReport(
            toxicity=toxicity_score,
            skin_sensitizer=sensitizer_risk,
            eco_score=eco_impact,
            ifra_compliant=ifra_status,
            regulatory_flags=regulatory_flags
        )
    
    def _assess_toxicity(self, molecule: Molecule) -> float:
        """Assess toxicity risk using structure-based rules."""
        mol = molecule.mol
        if not mol:
            return 1.0
            
        toxicity_score = 0.0
        
        # Check against toxic patterns
        for pattern in self._toxic_patterns:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                toxicity_score += 0.3
                
        # Check molecular weight (very large molecules may be problematic)
        mw = molecule.get_property('molecular_weight') or 0
        if mw > 500:
            toxicity_score += 0.1
        if mw > 1000:
            toxicity_score += 0.3
            
        # Check LogP (extreme values may indicate bioaccumulation)
        logp = molecule.get_property('logP') or 0
        if abs(logp) > 6:
            toxicity_score += 0.2
            
        # Simple heuristic: too many heteroatoms may indicate reactivity
        heavy_atoms = mol.GetNumHeavyAtoms()
        hetero_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() not in ['C', 'H'])
        if heavy_atoms > 0 and hetero_atoms / heavy_atoms > 0.5:
            toxicity_score += 0.1
            
        return min(toxicity_score, 1.0)
    
    def _check_skin_sensitization(self, molecule: Molecule) -> bool:
        """Check for skin sensitization potential."""
        mol = molecule.mol
        if not mol:
            return True
            
        # Check against known allergenic patterns
        for pattern in self._allergenic_patterns:
            try:
                if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    return True
            except:
                continue
                
        # Additional heuristics
        # Presence of certain functional groups
        aldehyde_pattern = Chem.MolFromSmarts("C=O")  # Simplified aldehyde check
        if mol.HasSubstructMatch(aldehyde_pattern):
            # Count aldehydes - some are sensitizers
            matches = mol.GetSubstructMatches(aldehyde_pattern)
            if len(matches) > 2:  # Multiple aldehydes increase risk
                return True
                
        return False
    
    def _assess_environmental_impact(self, molecule: Molecule) -> float:
        """Assess environmental impact/biodegradability."""
        mol = molecule.mol
        if not mol:
            return 1.0
            
        eco_score = 0.0
        
        # Highly halogenated compounds are often persistent
        halogen_count = sum(1 for atom in mol.GetAtoms() 
                          if atom.GetSymbol() in ['F', 'Cl', 'Br', 'I'])
        if halogen_count > 3:
            eco_score += 0.4
        elif halogen_count > 1:
            eco_score += 0.2
            
        # Aromatic rings can be persistent
        aromatic_rings = molecule.get_property('aromatic_rings') or 0
        if aromatic_rings > 3:
            eco_score += 0.3
        elif aromatic_rings > 2:
            eco_score += 0.1
            
        # Very high LogP indicates bioaccumulation potential
        logp = molecule.get_property('logP') or 0
        if logp > 5:
            eco_score += 0.4
        elif logp > 3:
            eco_score += 0.2
            
        return min(eco_score, 1.0)
    
    def _check_ifra_compliance(self, molecule: Molecule) -> bool:
        """Check compliance with IFRA standards."""
        mol = molecule.mol
        if not mol:
            return False
            
        # Check against banned substructures
        for pattern in self._banned_substructures:
            try:
                if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    return False
            except:
                continue
                
        # Additional IFRA-like checks
        # Molecular weight limit for fragrances
        mw = molecule.get_property('molecular_weight') or 0
        if mw > 1000:  # Very large molecules not typical for fragrances
            return False
            
        return True
    
    def _check_regulatory_compliance(self, molecule: Molecule) -> List[Dict[str, str]]:
        """Check compliance with various regulatory frameworks."""
        flags = []
        mol = molecule.mol
        
        if not mol:
            flags.append({"region": "ALL", "status": "INVALID_STRUCTURE"})
            return flags
            
        # EU cosmetics regulation
        if self._check_skin_sensitization(molecule):
            flags.append({"region": "EU", "status": "POTENTIAL_ALLERGEN"})
            
        # US EPA concerns
        eco_score = self._assess_environmental_impact(molecule)
        if eco_score > 0.5:
            flags.append({"region": "US_EPA", "status": "ENVIRONMENTAL_CONCERN"})
            
        # General toxicity flags
        toxicity = self._assess_toxicity(molecule)
        if toxicity > 0.5:
            flags.append({"region": "ALL", "status": "HIGH_TOXICITY_RISK"})
            
        return flags
    
    def filter_molecules(self, molecules: List[Molecule]) -> Tuple[List[Molecule], List[SafetyReport]]:
        """
        Filter a list of molecules, keeping only safe ones.
        
        Args:
            molecules: List of molecules to filter
            
        Returns:
            Tuple of (safe_molecules, all_safety_reports)
        """
        safe_molecules = []
        safety_reports = []
        
        for molecule in molecules:
            report = self.assess_molecule(molecule)
            safety_reports.append(report)
            
            # Apply filtering criteria
            passes_safety = (
                report.toxicity <= self.toxicity_threshold and
                report.eco_score <= self.eco_threshold and
                report.ifra_compliant and
                (not self.irritant_check or not report.skin_sensitizer)
            )
            
            if passes_safety:
                molecule.safety_score = 1.0 - report.toxicity  # Convert to safety score
                safe_molecules.append(molecule)
                
        return safe_molecules, safety_reports
    
    def get_safety_summary(self, molecules: List[Molecule]) -> Dict[str, Any]:
        """Get summary statistics of safety assessment."""
        if not molecules:
            return {}
            
        _, reports = self.filter_molecules(molecules)
        
        return {
            'total_molecules': len(molecules),
            'safe_molecules': len([r for r in reports if r.toxicity <= self.toxicity_threshold]),
            'average_toxicity': np.mean([r.toxicity for r in reports]),
            'average_eco_score': np.mean([r.eco_score for r in reports]),
            'sensitizer_rate': np.mean([r.skin_sensitizer for r in reports]),
            'ifra_compliance_rate': np.mean([r.ifra_compliant for r in reports]),
            'regulatory_flags': sum(len(r.regulatory_flags) for r in reports)
        }