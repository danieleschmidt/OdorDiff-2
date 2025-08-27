import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors
from ..models.molecule import Molecule
from ..safety.filter import SafetyFilter
from ..utils.validation import validate_smiles

logger = logging.getLogger(__name__)

class RegulatoryFramework(Enum):
    """Supported regulatory frameworks."""
    FDA_US = "fda_us"
    EMA_EU = "ema_eu"
    PMDA_JAPAN = "pmda_japan"
    NMPA_CHINA = "nmpa_china"
    TGA_AUSTRALIA = "tga_australia"
    HEALTH_CANADA = "health_canada"
    IFRA_COSMETIC = "ifra_cosmetic"
    REACH_EU = "reach_eu"
    TSCA_US = "tsca_us"
    ICH_GLOBAL = "ich_global"

class ComplianceStatus(Enum):
    """Compliance status levels."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    NEEDS_REVIEW = "needs_review"
    TESTING_REQUIRED = "testing_required"
    DOCUMENTATION_NEEDED = "documentation_needed"
    CONDITIONAL_APPROVAL = "conditional_approval"

@dataclass
class RegulatoryRequirement:
    """Individual regulatory requirement."""
    id: str
    framework: RegulatoryFramework
    title: str
    description: str
    molecular_criteria: Dict[str, Any]
    documentation_required: List[str]
    testing_protocols: List[str]
    approval_timeline_days: int
    priority_level: int  # 1 (highest) to 5 (lowest)
    auto_checkable: bool = True
    
@dataclass 
class ComplianceReport:
    """Comprehensive compliance assessment report."""
    molecule_id: str
    smiles: str
    timestamp: datetime
    overall_status: ComplianceStatus
    framework_results: Dict[RegulatoryFramework, Dict[str, Any]]
    required_actions: List[str]
    estimated_approval_timeline: timedelta
    estimated_cost: float
    risk_assessment: Dict[str, float]
    recommendations: List[str]
    generated_documents: List[str] = field(default_factory=list)
    
class RegulatoryKnowledgeBase:
    """Comprehensive regulatory knowledge base."""
    
    def __init__(self):
        self.requirements = self._initialize_requirements()
        self.molecular_databases = self._load_molecular_databases()
        self.precedent_cases = self._load_precedent_cases()
        self.regulatory_contacts = self._load_regulatory_contacts()
        
        logger.info(f"Initialized regulatory knowledge base with {len(self.requirements)} requirements")
    
    def _initialize_requirements(self) -> Dict[str, RegulatoryRequirement]:
        """Initialize regulatory requirements database."""
        
        requirements = {}
        
        # FDA Drug Development Requirements
        requirements['fda_drug_safety'] = RegulatoryRequirement(
            id='fda_drug_safety',
            framework=RegulatoryFramework.FDA_US,
            title='FDA Drug Safety Assessment',
            description='Basic safety requirements for pharmaceutical compounds',
            molecular_criteria={
                'molecular_weight': {'min': 150, 'max': 800},
                'logp': {'min': -2, 'max': 5},
                'hbd': {'max': 5},  # Hydrogen bond donors
                'hba': {'max': 10},  # Hydrogen bond acceptors
                'rotatable_bonds': {'max': 10},
                'tpsa': {'max': 140},  # Topological polar surface area
                'alerts': ['reactive_groups', 'toxicophores']
            },
            documentation_required=[
                'nonclinical_pharmacology',
                'nonclinical_pharmacokinetics', 
                'nonclinical_toxicology',
                'chemistry_manufacturing_controls',
                'environmental_assessment'
            ],
            testing_protocols=[
                'in_vitro_genotoxicity',
                'acute_toxicity_study',
                'repeat_dose_toxicity',
                'reproductive_toxicity',
                'carcinogenicity_assessment'
            ],
            approval_timeline_days=1095,  # ~3 years
            priority_level=1
        )
        
        # EMA Drug Development Requirements
        requirements['ema_drug_safety'] = RegulatoryRequirement(
            id='ema_drug_safety',
            framework=RegulatoryFramework.EMA_EU,
            title='EMA Drug Safety Assessment',
            description='European Medicines Agency safety requirements',
            molecular_criteria={
                'molecular_weight': {'min': 150, 'max': 800},
                'logp': {'min': -2, 'max': 5},
                'pains_alerts': 0,  # Pan-assay interference compounds
                'brenk_alerts': 0,  # Toxic and reactive groups
                'lipinski_violations': {'max': 1}
            },
            documentation_required=[
                'quality_dossier',
                'nonclinical_overview',
                'clinical_overview',
                'risk_management_plan',
                'environmental_risk_assessment'
            ],
            testing_protocols=[
                'ich_s2_genotoxicity',
                'ich_s7a_safety_pharmacology',
                'ich_s4_carcinogenicity',
                'ich_s5_reproductive_toxicity'
            ],
            approval_timeline_days=400,  # Centralized procedure
            priority_level=1
        )
        
        # IFRA Cosmetic/Fragrance Requirements
        requirements['ifra_fragrance'] = RegulatoryRequirement(
            id='ifra_fragrance',
            framework=RegulatoryFramework.IFRA_COSMETIC,
            title='IFRA Fragrance Safety Standards',
            description='International Fragrance Association safety standards',
            molecular_criteria={
                'skin_sensitization': {'max': 0.1},  # EC3 value
                'photo_toxicity': {'max': 0.001},
                'irritation_potential': {'max': 0.05},
                'allergenicity': 'low',
                'biodegradability': {'min': 0.6},
                'vapor_pressure': {'min': 0.001, 'max': 100},  # Pa at 25°C
                'restricted_substances': []  # Must not contain
            },
            documentation_required=[
                'safety_assessment',
                'exposure_assessment',
                'toxicological_profile',
                'environmental_profile',
                'ifra_certificate'
            ],
            testing_protocols=[
                'oecd_442c_skin_sensitization',
                'oecd_432_eye_irritation',
                'oecd_439_skin_irritation',
                'oecd_301_biodegradability'
            ],
            approval_timeline_days=180,
            priority_level=2
        )
        
        # REACH EU Chemical Registration
        requirements['reach_registration'] = RegulatoryRequirement(
            id='reach_registration',
            framework=RegulatoryFramework.REACH_EU,
            title='REACH Chemical Registration',
            description='EU REACH regulation for chemical substances',
            molecular_criteria={
                'production_volume': {'min': 1000},  # kg/year
                'pbt_assessment': 'required',  # Persistent, Bioaccumulative, Toxic
                'cmr_classification': 'none',  # Carcinogenic, Mutagenic, Reproductive toxin
                'endocrine_disruption': 'low_concern'
            },
            documentation_required=[
                'chemical_safety_report',
                'registration_dossier',
                'exposure_scenarios',
                'sds_safety_data_sheet',
                'tonnage_information'
            ],
            testing_protocols=[
                'oecd_471_bacterial_mutation',
                'oecd_473_chromosomal_aberration',
                'oecd_408_repeated_dose_90day',
                'oecd_443_extended_reproduction'
            ],
            approval_timeline_days=365,
            priority_level=1
        )
        
        return requirements
    
    def _load_molecular_databases(self) -> Dict[str, List[str]]:
        """Load databases of regulated molecular structures."""
        
        # In practice, these would be loaded from external databases
        databases = {
            'fda_approved_drugs': [
                'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
                'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
                'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'  # Caffeine
            ],
            'ema_approved_drugs': [
                'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
                'CC(=O)OC1=CC=CC=C1C(=O)O'  # Aspirin
            ],
            'ifra_restricted': [
                'CC1=CC=C2C(=C1)C=CC(=C2O)C=O',  # Cinnamaldehyde (restricted)
                'COC1=CC=C(C=C1)C=O'  # p-Anisaldehyde (limited use)
            ],
            'reach_registered': [
                'CCO',  # Ethanol
                'CC(C)O',  # Isopropanol
                'C1=CC=CC=C1'  # Benzene (registered with restrictions)
            ]
        }
        
        return databases
    
    def _load_precedent_cases(self) -> List[Dict[str, Any]]:
        """Load historical precedent cases for regulatory decisions."""
        
        precedents = [
            {
                'case_id': 'FDA_2023_001',
                'molecule_smiles': 'CC1=CC=C(C=C1)C(C)C(=O)O',
                'framework': RegulatoryFramework.FDA_US,
                'decision': ComplianceStatus.COMPLIANT,
                'timeline_days': 1200,
                'key_factors': ['established_safety_profile', 'similar_approved_compounds'],
                'documentation_gaps': [],
                'lessons_learned': 'Strong nonclinical package accelerated review'
            },
            {
                'case_id': 'EMA_2023_015',
                'molecule_smiles': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
                'framework': RegulatoryFramework.EMA_EU,
                'decision': ComplianceStatus.NEEDS_REVIEW,
                'timeline_days': 450,
                'key_factors': ['novel_mechanism', 'limited_safety_data'],
                'documentation_gaps': ['long_term_toxicity', 'drug_interactions'],
                'lessons_learned': 'Additional studies required for novel compounds'
            }
        ]
        
        return precedents
    
    def _load_regulatory_contacts(self) -> Dict[RegulatoryFramework, Dict[str, str]]:
        """Load regulatory authority contact information."""
        
        contacts = {
            RegulatoryFramework.FDA_US: {
                'pre_submission': 'https://www.fda.gov/drugs/types-applications/pre-submission-basic-information',
                'guidance_documents': 'https://www.fda.gov/drugs/guidance-compliance-regulatory-information',
                'contact_email': 'druginfo@fda.hhs.gov'
            },
            RegulatoryFramework.EMA_EU: {
                'pre_submission': 'https://www.ema.europa.eu/en/human-regulatory/research-development/scientific-advice-protocol-assistance',
                'guidance_documents': 'https://www.ema.europa.eu/en/human-regulatory/research-development/scientific-guidelines',
                'contact_email': 'info@ema.europa.eu'
            },
            RegulatoryFramework.IFRA_COSMETIC: {
                'standards': 'https://ifrafragrance.org/safe-use/library',
                'contact_email': 'info@ifrafragrance.org'
            }
        }
        
        return contacts

class MolecularPropertyCalculator:
    """Calculate molecular properties for regulatory assessment."""
    
    def __init__(self):
        logger.info("Initialized molecular property calculator")
    
    def calculate_all_properties(self, smiles: str) -> Dict[str, Any]:
        """Calculate all regulatory-relevant molecular properties."""
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        properties = {}
        
        # Basic molecular descriptors
        properties['molecular_weight'] = Descriptors.MolWt(mol)
        properties['logp'] = Descriptors.MolLogP(mol)
        properties['hbd'] = Descriptors.NumHDonors(mol)
        properties['hba'] = Descriptors.NumHAcceptors(mol)
        properties['rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
        properties['tpsa'] = Descriptors.TPSA(mol)
        properties['heavy_atoms'] = mol.GetNumHeavyAtoms()
        
        # Drug-likeness properties
        properties['lipinski_violations'] = self._count_lipinski_violations(mol)
        properties['qed'] = self._calculate_qed(mol)
        
        # Safety-related properties
        properties['pains_alerts'] = self._check_pains(mol)
        properties['brenk_alerts'] = self._check_brenk_filters(mol)
        properties['toxicophores'] = self._identify_toxicophores(mol)
        
        # Fragrance-specific properties
        if self._is_potential_fragrance(mol):
            properties.update(self._calculate_fragrance_properties(mol))
        
        return properties
    
    def _count_lipinski_violations(self, mol) -> int:
        """Count Lipinski rule of 5 violations."""
        violations = 0
        
        if Descriptors.MolWt(mol) > 500:
            violations += 1
        if Descriptors.MolLogP(mol) > 5:
            violations += 1
        if Descriptors.NumHDonors(mol) > 5:
            violations += 1
        if Descriptors.NumHAcceptors(mol) > 10:
            violations += 1
        
        return violations
    
    def _calculate_qed(self, mol) -> float:
        """Calculate Quantitative Estimate of Drug-likeness."""
        # Simplified QED calculation
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        
        # QED components (simplified)
        mw_score = 1.0 if 150 <= mw <= 500 else 0.5
        logp_score = 1.0 if -2 <= logp <= 5 else 0.5
        hbd_score = 1.0 if hbd <= 5 else 0.5
        hba_score = 1.0 if hba <= 10 else 0.5
        
        qed = (mw_score * logp_score * hbd_score * hba_score) ** 0.25
        return qed
    
    def _check_pains(self, mol) -> int:
        """Check for Pan-Assay Interference Compounds (PAINS)."""
        # Simplified PAINS filter - in practice would use full PAINS database
        pains_patterns = [
            'C1=CC=CC=C1',  # Simple benzene (flagged for demo)
            'C=O',  # Aldehyde groups
            'N=O'   # Nitroso groups
        ]
        
        alerts = 0
        smiles = Chem.MolToSmiles(mol)
        
        for pattern in pains_patterns:
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                alerts += 1
        
        return alerts
    
    def _check_brenk_filters(self, mol) -> int:
        """Check for toxic and reactive functional groups (Brenk filters)."""
        # Simplified Brenk filter patterns
        toxic_patterns = [
            '[N+](=O)[O-]',  # Nitro groups
            'C(=O)OO',       # Peroxides
            'N=N',           # Azo compounds
            'C#N'            # Nitriles
        ]
        
        alerts = 0
        
        for pattern in toxic_patterns:
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                alerts += 1
        
        return alerts
    
    def _identify_toxicophores(self, mol) -> List[str]:
        """Identify known toxicophoric groups."""
        toxicophores = []
        
        toxicophore_patterns = {
            'aromatic_amine': 'c-[NX3]',
            'alkyl_halide': '[CH2X4][ClX1,BrX1,IX1]',
            'epoxide': 'C1OC1',
            'aldehyde': '[CX3H1](=O)',
            'quinone': 'C1=CC(=O)C=CC1=O'
        }
        
        for name, smarts in toxicophore_patterns.items():
            pattern_mol = Chem.MolFromSmarts(smarts)
            if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                toxicophores.append(name)
        
        return toxicophores
    
    def _is_potential_fragrance(self, mol) -> bool:
        """Determine if molecule could be a fragrance compound."""
        # Simple heuristics for fragrance compounds
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        
        # Fragrances typically have MW 100-300 and moderate lipophilicity
        return 100 <= mw <= 300 and 1 <= logp <= 5
    
    def _calculate_fragrance_properties(self, mol) -> Dict[str, Any]:
        """Calculate fragrance-specific properties."""
        return {
            'vapor_pressure_est': self._estimate_vapor_pressure(mol),
            'skin_permeability_est': self._estimate_skin_permeability(mol),
            'odor_threshold_est': self._estimate_odor_threshold(mol),
            'volatility_class': self._classify_volatility(mol)
        }
    
    def _estimate_vapor_pressure(self, mol) -> float:
        """Estimate vapor pressure (simplified model)."""
        # Very simplified estimation based on molecular weight and structure
        mw = Descriptors.MolWt(mol)
        return max(0.001, 1000 / mw)  # Rough inverse relationship
    
    def _estimate_skin_permeability(self, mol) -> float:
        """Estimate skin permeability coefficient."""
        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        
        # Simplified model based on Potts & Guy equation
        log_kp = -2.7 + 0.71 * logp - 0.0061 * mw
        return 10 ** log_kp
    
    def _estimate_odor_threshold(self, mol) -> float:
        """Estimate olfactory detection threshold."""
        # Placeholder estimation - would use QSAR models in practice
        return np.random.uniform(0.001, 10.0)  # mg/m³
    
    def _classify_volatility(self, mol) -> str:
        """Classify fragrance volatility (top/middle/base note)."""
        mw = Descriptors.MolWt(mol)
        
        if mw < 150:
            return 'top_note'  # High volatility
        elif mw < 250:
            return 'middle_note'  # Medium volatility
        else:
            return 'base_note'  # Low volatility

class RegulatoryAI:
    """AI-powered regulatory compliance and approval system."""
    
    def __init__(self):
        self.knowledge_base = RegulatoryKnowledgeBase()
        self.property_calculator = MolecularPropertyCalculator()
        self.compliance_history = []
        
        logger.info("RegulatoryAI system initialized")
    
    async def assess_compliance(self, smiles: str, target_frameworks: List[RegulatoryFramework],
                              intended_use: str = "pharmaceutical") -> ComplianceReport:
        """Comprehensive regulatory compliance assessment."""
        
        if not validate_smiles(smiles):
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        logger.info(f"Assessing compliance for {smiles} across {len(target_frameworks)} frameworks")
        
        # Calculate molecular properties
        properties = self.property_calculator.calculate_all_properties(smiles)
        
        # Assess each regulatory framework
        framework_results = {}
        
        for framework in target_frameworks:
            framework_assessment = await self._assess_single_framework(
                smiles, properties, framework, intended_use
            )
            framework_results[framework] = framework_assessment
        
        # Determine overall compliance status
        overall_status = self._determine_overall_status(framework_results)
        
        # Generate recommendations and required actions
        required_actions = self._generate_required_actions(framework_results)
        recommendations = self._generate_recommendations(smiles, properties, framework_results)
        
        # Estimate timelines and costs
        timeline = self._estimate_approval_timeline(framework_results)
        cost = self._estimate_approval_cost(framework_results, intended_use)
        
        # Risk assessment
        risk_assessment = self._perform_risk_assessment(properties, framework_results)
        
        # Create comprehensive report
        report = ComplianceReport(
            molecule_id=f"mol_{hash(smiles) % 100000:05d}",
            smiles=smiles,
            timestamp=datetime.now(),
            overall_status=overall_status,
            framework_results=framework_results,
            required_actions=required_actions,
            estimated_approval_timeline=timeline,
            estimated_cost=cost,
            risk_assessment=risk_assessment,
            recommendations=recommendations
        )
        
        self.compliance_history.append(report)
        
        return report
    
    async def _assess_single_framework(self, smiles: str, properties: Dict[str, Any],
                                     framework: RegulatoryFramework, intended_use: str) -> Dict[str, Any]:
        """Assess compliance against a single regulatory framework."""
        
        # Find relevant requirements
        relevant_requirements = self._find_relevant_requirements(framework, intended_use)
        
        assessment = {
            'framework': framework,
            'requirements_checked': len(relevant_requirements),
            'molecular_compliance': {},
            'documentation_status': {},
            'testing_status': {},
            'compliance_score': 0.0,
            'critical_issues': [],
            'warnings': [],
            'precedent_cases': []
        }
        
        total_score = 0
        max_score = 0
        
        for req in relevant_requirements:
            req_assessment = self._check_requirement_compliance(properties, req)
            assessment['molecular_compliance'][req.id] = req_assessment
            
            total_score += req_assessment['score']
            max_score += req_assessment['max_score']
            
            if req_assessment['status'] == 'critical_failure':
                assessment['critical_issues'].append(req_assessment['issue'])
            elif req_assessment['status'] == 'warning':
                assessment['warnings'].append(req_assessment['issue'])
        
        assessment['compliance_score'] = total_score / max_score if max_score > 0 else 0
        
        # Check for precedent cases
        assessment['precedent_cases'] = self._find_precedent_cases(smiles, framework)
        
        return assessment
    
    def _find_relevant_requirements(self, framework: RegulatoryFramework, 
                                  intended_use: str) -> List[RegulatoryRequirement]:
        """Find requirements relevant to framework and intended use."""
        
        relevant = []
        
        for req_id, req in self.knowledge_base.requirements.items():
            if req.framework == framework:
                # Check if requirement applies to intended use
                if intended_use in ['pharmaceutical', 'drug'] and 'drug' in req_id:
                    relevant.append(req)
                elif intended_use in ['cosmetic', 'fragrance'] and 'fragrance' in req_id:
                    relevant.append(req)
                elif intended_use == 'chemical' and 'registration' in req_id:
                    relevant.append(req)
        
        return relevant
    
    def _check_requirement_compliance(self, properties: Dict[str, Any], 
                                    requirement: RegulatoryRequirement) -> Dict[str, Any]:
        """Check compliance against a specific requirement."""
        
        assessment = {
            'requirement_id': requirement.id,
            'status': 'compliant',
            'score': 0,
            'max_score': 100,
            'details': {},
            'issue': None
        }
        
        score = 100  # Start with full score
        
        # Check molecular criteria
        for criterion, limits in requirement.molecular_criteria.items():
            if criterion in properties:
                prop_value = properties[criterion]
                
                if isinstance(limits, dict):
                    # Range check
                    if 'min' in limits and prop_value < limits['min']:
                        score -= 20
                        assessment['details'][criterion] = f"Below minimum: {prop_value} < {limits['min']}"
                        if score <= 0:
                            assessment['status'] = 'critical_failure'
                            assessment['issue'] = f"{criterion} below critical minimum"
                    
                    if 'max' in limits and prop_value > limits['max']:
                        score -= 20
                        assessment['details'][criterion] = f"Above maximum: {prop_value} > {limits['max']}"
                        if score <= 20:
                            assessment['status'] = 'critical_failure'
                            assessment['issue'] = f"{criterion} above critical maximum"
                
                elif isinstance(limits, (int, float)):
                    # Exact value or threshold
                    if prop_value != limits:
                        score -= 10
                        assessment['details'][criterion] = f"Does not meet requirement: {prop_value} != {limits}"
                        if score <= 50:
                            assessment['status'] = 'warning'
                            assessment['issue'] = f"{criterion} does not meet specification"
        
        assessment['score'] = max(0, score)
        
        return assessment
    
    def _find_precedent_cases(self, smiles: str, framework: RegulatoryFramework) -> List[Dict[str, Any]]:
        """Find similar precedent cases."""
        
        precedents = []
        
        for case in self.knowledge_base.precedent_cases:
            if case['framework'] == framework:
                # Calculate structural similarity (simplified)
                similarity = self._calculate_structural_similarity(smiles, case['molecule_smiles'])
                
                if similarity > 0.7:  # High similarity threshold
                    precedents.append({
                        'case_id': case['case_id'],
                        'similarity': similarity,
                        'decision': case['decision'],
                        'timeline_days': case['timeline_days'],
                        'key_factors': case['key_factors'],
                        'lessons_learned': case['lessons_learned']
                    })
        
        return sorted(precedents, key=lambda x: x['similarity'], reverse=True)[:3]
    
    def _calculate_structural_similarity(self, smiles1: str, smiles2: str) -> float:
        """Calculate structural similarity between molecules."""
        # Simplified similarity calculation
        # In practice, would use Tanimoto coefficient with molecular fingerprints
        
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        # Simple molecular weight similarity as proxy
        mw1 = Descriptors.MolWt(mol1)
        mw2 = Descriptors.MolWt(mol2)
        
        mw_similarity = 1 - abs(mw1 - mw2) / max(mw1, mw2)
        
        # Add structural features similarity
        features1 = [mol1.GetNumHeavyAtoms(), mol1.GetNumRings()]
        features2 = [mol2.GetNumHeavyAtoms(), mol2.GetNumRings()]
        
        feature_similarity = 1 - sum(abs(f1 - f2) for f1, f2 in zip(features1, features2)) / 10
        
        return (mw_similarity + feature_similarity) / 2
    
    def _determine_overall_status(self, framework_results: Dict[RegulatoryFramework, Dict[str, Any]]) -> ComplianceStatus:
        """Determine overall compliance status across all frameworks."""
        
        if not framework_results:
            return ComplianceStatus.NEEDS_REVIEW
        
        critical_issues = any(result['critical_issues'] for result in framework_results.values())
        warnings = any(result['warnings'] for result in framework_results.values())
        
        avg_compliance = np.mean([result['compliance_score'] for result in framework_results.values()])
        
        if critical_issues:
            return ComplianceStatus.NON_COMPLIANT
        elif avg_compliance < 0.5:
            return ComplianceStatus.TESTING_REQUIRED
        elif warnings or avg_compliance < 0.8:
            return ComplianceStatus.NEEDS_REVIEW
        else:
            return ComplianceStatus.COMPLIANT
    
    def _generate_required_actions(self, framework_results: Dict[RegulatoryFramework, Dict[str, Any]]) -> List[str]:
        """Generate list of required actions for compliance."""
        
        actions = []
        
        for framework, result in framework_results.items():
            if result['critical_issues']:
                actions.extend([
                    f"Address critical issue in {framework.value}: {issue}"
                    for issue in result['critical_issues']
                ])
            
            if result['warnings']:
                actions.extend([
                    f"Resolve warning for {framework.value}: {warning}"
                    for warning in result['warnings']
                ])
            
            if result['compliance_score'] < 0.8:
                actions.append(f"Improve compliance score for {framework.value} (current: {result['compliance_score']:.1%})")
        
        return list(set(actions))  # Remove duplicates
    
    def _generate_recommendations(self, smiles: str, properties: Dict[str, Any],
                                framework_results: Dict[RegulatoryFramework, Dict[str, Any]]) -> List[str]:
        """Generate AI-powered recommendations for regulatory success."""
        
        recommendations = []
        
        # Molecular optimization recommendations
        if properties.get('molecular_weight', 0) > 500:
            recommendations.append("Consider reducing molecular weight for better drug-likeness")
        
        if properties.get('lipinski_violations', 0) > 1:
            recommendations.append("Optimize molecular structure to reduce Lipinski violations")
        
        if properties.get('pains_alerts', 0) > 0:
            recommendations.append("Remove or modify PAINS-flagged substructures")
        
        # Framework-specific recommendations
        for framework, result in framework_results.items():
            if result['precedent_cases']:
                best_precedent = result['precedent_cases'][0]
                recommendations.append(
                    f"Consider lessons from similar case {best_precedent['case_id']}: {best_precedent['lessons_learned']}"
                )
        
        # Documentation recommendations
        if any('documentation' in action for action in self._generate_required_actions(framework_results)):
            recommendations.append("Begin early preparation of regulatory documentation packages")
        
        # Testing recommendations
        low_compliance = [f for f, r in framework_results.items() if r['compliance_score'] < 0.6]
        if low_compliance:
            recommendations.append("Initiate early safety testing to address potential compliance gaps")
        
        return recommendations
    
    def _estimate_approval_timeline(self, framework_results: Dict[RegulatoryFramework, Dict[str, Any]]) -> timedelta:
        """Estimate total approval timeline across frameworks."""
        
        if not framework_results:
            return timedelta(days=365)
        
        # Find the longest timeline among frameworks
        max_days = 0
        
        for framework in framework_results.keys():
            for req in self.knowledge_base.requirements.values():
                if req.framework == framework:
                    max_days = max(max_days, req.approval_timeline_days)
        
        # Add delays for compliance issues
        for result in framework_results.values():
            if result['critical_issues']:
                max_days += 180  # 6 months delay for critical issues
            elif result['warnings']:
                max_days += 60   # 2 months delay for warnings
        
        return timedelta(days=max_days)
    
    def _estimate_approval_cost(self, framework_results: Dict[RegulatoryFramework, Dict[str, Any]],
                              intended_use: str) -> float:
        """Estimate total approval cost."""
        
        base_costs = {
            RegulatoryFramework.FDA_US: 2_000_000,      # $2M for new drug
            RegulatoryFramework.EMA_EU: 1_500_000,      # $1.5M for EMA
            RegulatoryFramework.IFRA_COSMETIC: 50_000,   # $50K for fragrance
            RegulatoryFramework.REACH_EU: 200_000        # $200K for REACH
        }
        
        total_cost = 0
        
        for framework in framework_results.keys():
            base_cost = base_costs.get(framework, 100_000)
            
            # Add costs for compliance issues
            result = framework_results[framework]
            if result['critical_issues']:
                base_cost *= 1.5  # 50% increase for critical issues
            elif result['warnings']:
                base_cost *= 1.2  # 20% increase for warnings
            
            total_cost += base_cost
        
        return total_cost
    
    def _perform_risk_assessment(self, properties: Dict[str, Any],
                               framework_results: Dict[RegulatoryFramework, Dict[str, Any]]) -> Dict[str, float]:
        """Perform comprehensive risk assessment."""
        
        risks = {
            'regulatory_failure_risk': 0.0,
            'timeline_delay_risk': 0.0,
            'cost_overrun_risk': 0.0,
            'safety_concern_risk': 0.0,
            'market_access_risk': 0.0
        }
        
        # Regulatory failure risk
        avg_compliance = np.mean([r['compliance_score'] for r in framework_results.values()])
        risks['regulatory_failure_risk'] = 1 - avg_compliance
        
        # Timeline delay risk
        has_critical_issues = any(r['critical_issues'] for r in framework_results.values())
        risks['timeline_delay_risk'] = 0.8 if has_critical_issues else 0.3
        
        # Cost overrun risk
        risks['cost_overrun_risk'] = 0.6 if has_critical_issues else 0.2
        
        # Safety concern risk
        toxicophores = len(properties.get('toxicophores', []))
        risks['safety_concern_risk'] = min(1.0, toxicophores * 0.3)
        
        # Market access risk
        non_compliant_frameworks = sum(1 for r in framework_results.values() if r['critical_issues'])
        risks['market_access_risk'] = non_compliant_frameworks / len(framework_results)
        
        return risks
    
    async def generate_regulatory_documents(self, compliance_report: ComplianceReport) -> Dict[str, str]:
        """Generate regulatory submission documents using AI."""
        
        documents = {}
        
        # Generate safety assessment document
        documents['safety_assessment'] = self._generate_safety_assessment(compliance_report)
        
        # Generate regulatory strategy document
        documents['regulatory_strategy'] = self._generate_regulatory_strategy(compliance_report)
        
        # Generate compliance summary
        documents['compliance_summary'] = self._generate_compliance_summary(compliance_report)
        
        # Update compliance report with generated documents
        compliance_report.generated_documents = list(documents.keys())
        
        return documents
    
    def _generate_safety_assessment(self, report: ComplianceReport) -> str:
        """Generate comprehensive safety assessment document."""
        
        doc = f"""
COMPREHENSIVE SAFETY ASSESSMENT

Molecule ID: {report.molecule_id}
SMILES: {report.smiles}
Assessment Date: {report.timestamp.strftime('%Y-%m-%d')}

EXECUTIVE SUMMARY
Overall Safety Status: {report.overall_status.value.upper()}

RISK ASSESSMENT
"""
        
        for risk_type, risk_value in report.risk_assessment.items():
            risk_level = "HIGH" if risk_value > 0.7 else "MEDIUM" if risk_value > 0.3 else "LOW"
            doc += f"{risk_type.replace('_', ' ').title()}: {risk_level} ({risk_value:.1%})\n"
        
        doc += "\nREGULATORY FRAMEWORK ANALYSIS\n"
        
        for framework, results in report.framework_results.items():
            doc += f"\n{framework.value.upper()}:\n"
            doc += f"  Compliance Score: {results['compliance_score']:.1%}\n"
            
            if results['critical_issues']:
                doc += f"  Critical Issues: {', '.join(results['critical_issues'])}\n"
            
            if results['warnings']:
                doc += f"  Warnings: {', '.join(results['warnings'])}\n"
        
        doc += "\nRECOMMENDATIONS\n"
        for i, rec in enumerate(report.recommendations, 1):
            doc += f"{i}. {rec}\n"
        
        return doc
    
    def _generate_regulatory_strategy(self, report: ComplianceReport) -> str:
        """Generate regulatory strategy document."""
        
        doc = f"""
REGULATORY STRATEGY DOCUMENT

Molecule: {report.smiles}
Strategy Date: {report.timestamp.strftime('%Y-%m-%d')}

STRATEGIC OVERVIEW
Estimated Approval Timeline: {report.estimated_approval_timeline.days} days
Estimated Total Cost: ${report.estimated_cost:,.0f}

PHASED APPROACH
"""
        
        # Phase 1: Address Critical Issues
        critical_actions = [a for a in report.required_actions if 'critical' in a.lower()]
        if critical_actions:
            doc += "\nPHASE 1: Address Critical Compliance Issues (0-6 months)\n"
            for action in critical_actions:
                doc += f"- {action}\n"
        
        # Phase 2: Resolve Warnings
        warning_actions = [a for a in report.required_actions if 'warning' in a.lower()]
        if warning_actions:
            doc += "\nPHASE 2: Resolve Regulatory Warnings (6-12 months)\n"
            for action in warning_actions:
                doc += f"- {action}\n"
        
        # Phase 3: Final Submission
        doc += "\nPHASE 3: Final Regulatory Submission (12+ months)\n"
        doc += "- Prepare comprehensive dossier\n"
        doc += "- Conduct regulatory authority meetings\n"
        doc += "- Submit applications\n"
        doc += "- Respond to regulatory queries\n"
        
        return doc
    
    def _generate_compliance_summary(self, report: ComplianceReport) -> str:
        """Generate executive compliance summary."""
        
        compliant_frameworks = sum(1 for r in report.framework_results.values() 
                                 if not r['critical_issues'] and r['compliance_score'] > 0.8)
        total_frameworks = len(report.framework_results)
        
        doc = f"""
COMPLIANCE EXECUTIVE SUMMARY

Molecule: {report.smiles}
Overall Status: {report.overall_status.value.upper()}
Compliance Rate: {compliant_frameworks}/{total_frameworks} frameworks

KEY METRICS
- Average Compliance Score: {np.mean([r['compliance_score'] for r in report.framework_results.values()]):.1%}
- Regulatory Failure Risk: {report.risk_assessment.get('regulatory_failure_risk', 0):.1%}
- Timeline Delay Risk: {report.risk_assessment.get('timeline_delay_risk', 0):.1%}

IMMEDIATE ACTIONS REQUIRED
"""
        
        for i, action in enumerate(report.required_actions[:5], 1):
            doc += f"{i}. {action}\n"
        
        if len(report.required_actions) > 5:
            doc += f"... and {len(report.required_actions) - 5} additional actions\n"
        
        return doc
    
    def get_regulatory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive regulatory assessment statistics."""
        
        if not self.compliance_history:
            return {'status': 'No assessments completed yet'}
        
        stats = {
            'total_assessments': len(self.compliance_history),
            'status_distribution': {},
            'framework_statistics': {},
            'average_metrics': {},
            'success_rate': 0.0
        }
        
        # Status distribution
        for report in self.compliance_history:
            status = report.overall_status.value
            stats['status_distribution'][status] = stats['status_distribution'].get(status, 0) + 1
        
        # Framework statistics
        all_frameworks = set()
        for report in self.compliance_history:
            all_frameworks.update(report.framework_results.keys())
        
        for framework in all_frameworks:
            framework_scores = []
            for report in self.compliance_history:
                if framework in report.framework_results:
                    framework_scores.append(report.framework_results[framework]['compliance_score'])
            
            stats['framework_statistics'][framework.value] = {
                'assessments': len(framework_scores),
                'average_score': np.mean(framework_scores) if framework_scores else 0,
                'success_rate': sum(1 for s in framework_scores if s > 0.8) / len(framework_scores) if framework_scores else 0
            }
        
        # Average metrics
        all_costs = [r.estimated_cost for r in self.compliance_history]
        all_timelines = [r.estimated_approval_timeline.days for r in self.compliance_history]
        
        stats['average_metrics'] = {
            'cost': np.mean(all_costs) if all_costs else 0,
            'timeline_days': np.mean(all_timelines) if all_timelines else 0,
            'regulatory_failure_risk': np.mean([r.risk_assessment.get('regulatory_failure_risk', 0) for r in self.compliance_history])
        }
        
        # Overall success rate
        successful = sum(1 for r in self.compliance_history if r.overall_status == ComplianceStatus.COMPLIANT)
        stats['success_rate'] = successful / len(self.compliance_history)
        
        return stats

# Demo and testing functions
async def demo_regulatory_ai():
    """Demonstrate regulatory AI capabilities."""
    
    print("⚖️ Initializing Regulatory AI System...")
    regulatory_ai = RegulatoryAI()
    
    # Test molecules
    test_molecules = {
        'Ibuprofen': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
        'Novel Compound': 'CCN1C=NC2=C1C(=O)N(C(=O)N2C)CC',
        'Fragrance Molecule': 'CC(=O)OC1=CC=C(C=C1)CC=C'
    }
    
    frameworks_to_test = {
        'Ibuprofen': [RegulatoryFramework.FDA_US, RegulatoryFramework.EMA_EU],
        'Novel Compound': [RegulatoryFramework.FDA_US, RegulatoryFramework.REACH_EU],
        'Fragrance Molecule': [RegulatoryFramework.IFRA_COSMETIC, RegulatoryFramework.REACH_EU]
    }
    
    print("\n📊 Running comprehensive regulatory assessments...")
    
    for name, smiles in test_molecules.items():
        print(f"\n\n=== ASSESSING {name.upper()} ===")
        print(f"SMILES: {smiles}")
        
        try:
            # Perform compliance assessment
            report = await regulatory_ai.assess_compliance(
                smiles=smiles,
                target_frameworks=frameworks_to_test[name],
                intended_use='pharmaceutical' if name != 'Fragrance Molecule' else 'cosmetic'
            )
            
            print(f"\nOVERALL STATUS: {report.overall_status.value.upper()}")
            print(f"Timeline: {report.estimated_approval_timeline.days} days")
            print(f"Estimated Cost: ${report.estimated_cost:,.0f}")
            
            # Framework results
            print("\nFRAMEWORK COMPLIANCE:")
            for framework, results in report.framework_results.items():
                print(f"  {framework.value}: {results['compliance_score']:.1%}")
                if results['critical_issues']:
                    print(f"    Critical Issues: {', '.join(results['critical_issues'])}")
                if results['warnings']:
                    print(f"    Warnings: {', '.join(results['warnings'])}")
            
            # Risk assessment
            print("\nRISK ASSESSMENT:")
            for risk, value in report.risk_assessment.items():
                risk_level = "HIGH" if value > 0.7 else "MEDIUM" if value > 0.3 else "LOW"
                print(f"  {risk.replace('_', ' ').title()}: {risk_level} ({value:.1%})")
            
            # Top recommendations
            print("\nTOP RECOMMENDATIONS:")
            for i, rec in enumerate(report.recommendations[:3], 1):
                print(f"  {i}. {rec}")
            
            # Generate regulatory documents
            print("\n📄 Generating regulatory documents...")
            documents = await regulatory_ai.generate_regulatory_documents(report)
            
            for doc_type in documents.keys():
                print(f"  ✓ Generated: {doc_type.replace('_', ' ').title()}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n\n📊 REGULATORY AI STATISTICS")
    stats = regulatory_ai.get_regulatory_statistics()
    
    print(f"Total Assessments: {stats['total_assessments']}")
    print(f"Overall Success Rate: {stats['success_rate']:.1%}")
    print(f"Average Approval Cost: ${stats['average_metrics']['cost']:,.0f}")
    print(f"Average Timeline: {stats['average_metrics']['timeline_days']:.0f} days")
    
    print("\nFramework Performance:")
    for framework, framework_stats in stats['framework_statistics'].items():
        print(f"  {framework}: {framework_stats['success_rate']:.1%} success rate")
    
    print("\n✅ Regulatory AI demonstration complete!")

if __name__ == "__main__":
    asyncio.run(demo_regulatory_ai())
