"""
Synthesis route planning and feasibility assessment.
"""

from typing import List, Dict, Any, Optional, Tuple
import random
import json
from dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from ..models.molecule import Molecule, SynthesisRoute
from ..utils.logging import get_logger, log_function_call
from ..utils.validation import InputValidator

logger = get_logger(__name__)


@dataclass
class ReactionStep:
    """Represents a single synthetic transformation."""
    reaction_type: str
    reactants: List[str]  # SMILES
    products: List[str]   # SMILES
    reagents: List[str]   # Required reagents
    conditions: Dict[str, Any]  # Temperature, solvent, etc.
    yield_estimate: float  # 0-1
    difficulty: int  # 1-5 scale
    cost_factor: float  # Relative cost multiplier
    

class SynthesisPlanner:
    """
    Advanced synthesis planning system for generated molecules.
    Plans synthetic routes and estimates feasibility.
    """
    
    def __init__(self):
        self.logger = logger
        
        # Load reaction templates and commercial starting materials
        self.reaction_templates = self._load_reaction_templates()
        self.commercial_materials = self._load_commercial_materials()
        self.retrosynthesis_rules = self._load_retrosynthesis_rules()
        
        # Cost and availability data
        self.material_costs = self._load_material_costs()
        
    def _load_reaction_templates(self) -> List[Dict[str, Any]]:
        """Load common organic reaction templates."""
        return [
            {
                'name': 'Friedel-Crafts Acylation',
                'pattern': '[c:1][H:2]>>[c:1][C:2](=O)[R]',
                'conditions': {'temperature': 25, 'solvent': 'DCM', 'catalyst': 'AlCl3'},
                'yield_range': (0.6, 0.85),
                'difficulty': 2,
                'cost_factor': 1.2
            },
            {
                'name': 'Grignard Addition',
                'pattern': '[C:1](=O)>>[C:1]([OH])([R])',
                'conditions': {'temperature': -78, 'solvent': 'THF', 'reagent': 'RMgBr'},
                'yield_range': (0.7, 0.9),
                'difficulty': 3,
                'cost_factor': 1.5
            },
            {
                'name': 'Williamson Ether Synthesis',
                'pattern': '[OH:1]>>[O:1][R]',
                'conditions': {'temperature': 80, 'solvent': 'DMF', 'base': 'K2CO3'},
                'yield_range': (0.65, 0.8),
                'difficulty': 2,
                'cost_factor': 1.1
            },
            {
                'name': 'Aldol Condensation',
                'pattern': '[C:1](=O)[CH2:2]>>[C:1](=O)[CH:2]=[CH][R]',
                'conditions': {'temperature': 0, 'solvent': 'EtOH', 'base': 'NaOH'},
                'yield_range': (0.5, 0.75),
                'difficulty': 3,
                'cost_factor': 1.3
            },
            {
                'name': 'Reduction with NaBH4',
                'pattern': '[C:1](=O)>>[C:1][OH]',
                'conditions': {'temperature': 0, 'solvent': 'MeOH', 'reagent': 'NaBH4'},
                'yield_range': (0.8, 0.95),
                'difficulty': 1,
                'cost_factor': 1.0
            },
            {
                'name': 'Oxidation with PCC',
                'pattern': '[C:1][OH]>>[C:1]=O',
                'conditions': {'temperature': 25, 'solvent': 'DCM', 'reagent': 'PCC'},
                'yield_range': (0.7, 0.85),
                'difficulty': 2,
                'cost_factor': 1.4
            },
            {
                'name': 'Suzuki Coupling',
                'pattern': '[c:1][Br].[c:2][B]([OH])([OH])>>[c:1][c:2]',
                'conditions': {'temperature': 80, 'solvent': 'toluene', 'catalyst': 'Pd(PPh3)4'},
                'yield_range': (0.75, 0.9),
                'difficulty': 4,
                'cost_factor': 2.5
            },
            {
                'name': 'Fischer Esterification',
                'pattern': '[C:1](=O)[OH].[OH:2][R]>>[C:1](=O)[O:2][R]',
                'conditions': {'temperature': 65, 'solvent': 'none', 'catalyst': 'H2SO4'},
                'yield_range': (0.6, 0.8),
                'difficulty': 1,
                'cost_factor': 0.9
            }
        ]
    
    def _load_commercial_materials(self) -> List[Dict[str, Any]]:
        """Load database of commercially available starting materials."""
        return [
            {'smiles': 'c1ccccc1', 'name': 'benzene', 'price_per_g': 0.15, 'availability': 'high'},
            {'smiles': 'c1ccc(cc1)C=O', 'name': 'benzaldehyde', 'price_per_g': 0.25, 'availability': 'high'},
            {'smiles': 'c1ccc(cc1)CO', 'name': 'benzyl alcohol', 'price_per_g': 0.35, 'availability': 'high'},
            {'smiles': 'CC(=O)c1ccccc1', 'name': 'acetophenone', 'price_per_g': 0.45, 'availability': 'high'},
            {'smiles': 'c1ccc(cc1)O', 'name': 'phenol', 'price_per_g': 0.20, 'availability': 'high'},
            {'smiles': 'CCO', 'name': 'ethanol', 'price_per_g': 0.05, 'availability': 'high'},
            {'smiles': 'CC(C)O', 'name': '2-propanol', 'price_per_g': 0.08, 'availability': 'high'},
            {'smiles': 'CC(C)=O', 'name': 'acetone', 'price_per_g': 0.06, 'availability': 'high'},
            {'smiles': 'CC(=O)O', 'name': 'acetic acid', 'price_per_g': 0.12, 'availability': 'high'},
            {'smiles': 'C=CCO', 'name': 'allyl alcohol', 'price_per_g': 0.85, 'availability': 'medium'},
            {'smiles': 'CC(C)=CCO', 'name': 'prenol', 'price_per_g': 2.50, 'availability': 'medium'},
            {'smiles': 'COc1ccccc1', 'name': 'anisole', 'price_per_g': 0.65, 'availability': 'medium'},
            {'smiles': 'c1ccc2c(c1)cccc2', 'name': 'naphthalene', 'price_per_g': 0.40, 'availability': 'high'},
            {'smiles': 'Cc1ccccc1', 'name': 'toluene', 'price_per_g': 0.10, 'availability': 'high'},
            {'smiles': 'Cc1ccc(cc1)C', 'name': 'p-xylene', 'price_per_g': 0.18, 'availability': 'high'}
        ]
    
    def _load_retrosynthesis_rules(self) -> List[Dict[str, Any]]:
        """Load retrosynthesis disconnection rules."""
        return [
            {
                'name': 'Ketone -> Alcohol + Oxidation',
                'product_pattern': '[C:1](=O)',
                'precursor_pattern': '[C:1][OH]',
                'reaction': 'Oxidation with PCC',
                'priority': 0.8
            },
            {
                'name': 'Ester -> Acid + Alcohol',
                'product_pattern': '[C:1](=O)[O:2][R:3]',
                'precursor_pattern': '[C:1](=O)[OH].[OH:2][R:3]',
                'reaction': 'Fischer Esterification',
                'priority': 0.9
            },
            {
                'name': 'Ether -> Alcohol + Alkyl Halide',
                'product_pattern': '[O:1]([R:2])[R:3]',
                'precursor_pattern': '[OH:1][R:2].[Br][R:3]',
                'reaction': 'Williamson Ether Synthesis',
                'priority': 0.7
            },
            {
                'name': 'Aromatic Ketone -> ArH + Acid Chloride',
                'product_pattern': '[c:1][C:2](=O)',
                'precursor_pattern': '[c:1][H].[C:2](=O)[Cl]',
                'reaction': 'Friedel-Crafts Acylation',
                'priority': 0.85
            }
        ]
    
    def _load_material_costs(self) -> Dict[str, float]:
        """Load material cost data."""
        return {
            'standard_reagents': 1.0,
            'organometallics': 2.5,
            'catalysts': 5.0,
            'specialty_chemicals': 3.0,
            'solvents': 0.5
        }
    
    @log_function_call(logger)
    def suggest_synthesis_routes(
        self,
        target_molecule: Molecule,
        starting_materials: str = 'commercial',
        max_steps: int = 5,
        green_chemistry: bool = False,
        max_routes: int = 5
    ) -> List[SynthesisRoute]:
        """
        Suggest synthesis routes for a target molecule.
        
        Args:
            target_molecule: Target molecule to synthesize
            starting_materials: 'commercial' or 'any'
            max_steps: Maximum number of synthetic steps
            green_chemistry: Prefer environmentally friendly routes
            max_routes: Maximum number of routes to return
            
        Returns:
            List of suggested synthesis routes
        """
        if not target_molecule.is_valid:
            logger.error("Invalid target molecule for synthesis planning")
            return []
            
        target_smiles = target_molecule.smiles
        logger.info(f"Planning synthesis routes for: {target_smiles}")
        
        # Check if target is already commercially available
        if self._is_commercial_available(target_smiles):
            logger.info("Target molecule is commercially available")
            return [SynthesisRoute(
                steps=[{
                    'reaction_type': 'Commercial Purchase',
                    'reactants': [],
                    'products': [target_smiles],
                    'reagents': [],
                    'conditions': {},
                    'yield_estimate': 1.0,
                    'difficulty': 0,
                    'cost_factor': 1.0
                }],
                score=1.0,
                total_yield=1.0,
                cost_estimate=self._estimate_purchase_cost(target_smiles)
            )]
        
        # Generate synthesis routes using retrosynthesis
        routes = self._generate_retrosynthetic_routes(
            target_smiles, max_steps, max_routes
        )
        
        # Score and rank routes
        scored_routes = []
        for route in routes:
            score = self._score_synthesis_route(route, green_chemistry)
            if score > 0.1:  # Filter very low-scoring routes
                scored_routes.append(SynthesisRoute(
                    steps=route,
                    score=score,
                    total_yield=self._calculate_total_yield(route),
                    cost_estimate=self._estimate_route_cost(route)
                ))
        
        # Sort by score
        scored_routes.sort(key=lambda r: r.score, reverse=True)
        
        logger.info(f"Generated {len(scored_routes)} viable synthesis routes")
        return scored_routes[:max_routes]
    
    def _is_commercial_available(self, smiles: str) -> bool:
        """Check if molecule is commercially available."""
        canonical_smiles = self._canonicalize_smiles(smiles)
        return any(
            self._canonicalize_smiles(mat['smiles']) == canonical_smiles 
            for mat in self.commercial_materials
        )
    
    def _canonicalize_smiles(self, smiles: str) -> str:
        """Convert SMILES to canonical form."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return Chem.MolToSmiles(mol)
        except:
            pass
        return smiles
    
    def _generate_retrosynthetic_routes(
        self, 
        target_smiles: str, 
        max_steps: int, 
        max_routes: int
    ) -> List[List[ReactionStep]]:
        """Generate retrosynthetic routes using disconnection rules."""
        routes = []
        
        # For simplicity, generate template-based routes
        # Real implementation would use advanced retrosynthesis algorithms
        
        for i in range(min(max_routes, 3)):  # Generate up to 3 different approaches
            route = self._generate_single_route(target_smiles, max_steps, approach=i)
            if route and len(route) <= max_steps:
                routes.append(route)
        
        return routes
    
    def _generate_single_route(self, target_smiles: str, max_steps: int, approach: int = 0) -> List[ReactionStep]:
        """Generate a single synthesis route."""
        route = []
        current_molecule = target_smiles
        
        for step in range(max_steps):
            # Find applicable disconnections
            possible_steps = self._find_disconnections(current_molecule, approach)
            
            if not possible_steps:
                break
                
            # Choose best disconnection
            chosen_step = max(possible_steps, key=lambda s: s['priority'])
            
            # Create reaction step
            reaction_step = ReactionStep(
                reaction_type=chosen_step['reaction'],
                reactants=chosen_step['precursors'],
                products=[current_molecule],
                reagents=chosen_step.get('reagents', []),
                conditions=chosen_step.get('conditions', {}),
                yield_estimate=chosen_step.get('yield', 0.75),
                difficulty=chosen_step.get('difficulty', 2),
                cost_factor=chosen_step.get('cost_factor', 1.0)
            )
            
            route.append(reaction_step)
            
            # Check if we've reached commercial materials
            if all(self._is_commercial_available(sm) for sm in chosen_step['precursors']):
                break
                
            # Continue with first precursor for next iteration
            current_molecule = chosen_step['precursors'][0] if chosen_step['precursors'] else target_smiles
        
        # Reverse route (we built it backwards)
        return list(reversed(route))
    
    def _find_disconnections(self, smiles: str, approach: int = 0) -> List[Dict[str, Any]]:
        """Find possible retrosynthetic disconnections."""
        disconnections = []
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return []
        except:
            return []
        
        # Pattern-based disconnections (simplified)
        if approach == 0:  # Functional group approach
            disconnections.extend(self._functional_group_disconnections(smiles))
        elif approach == 1:  # Ring formation approach
            disconnections.extend(self._ring_formation_disconnections(smiles))
        else:  # Carbon-carbon bond approach
            disconnections.extend(self._cc_bond_disconnections(smiles))
            
        return disconnections
    
    def _functional_group_disconnections(self, smiles: str) -> List[Dict[str, Any]]:
        """Find disconnections based on functional groups."""
        disconnections = []
        
        # Ketone -> Alcohol + Oxidation
        if '=O' in smiles and 'c' in smiles:  # Aromatic ketone
            precursor = smiles.replace('(=O)', 'O')  # Simplified
            disconnections.append({
                'reaction': 'Oxidation with PCC',
                'precursors': [precursor],
                'reagents': ['PCC', 'DCM'],
                'conditions': {'temperature': 25, 'solvent': 'DCM'},
                'yield': 0.8,
                'difficulty': 2,
                'cost_factor': 1.4,
                'priority': 0.8
            })
        
        # Ester -> Acid + Alcohol (simplified pattern matching)
        if 'C(=O)O' in smiles and smiles.count('C') > 2:
            acid = smiles.split('O')[0] + 'O'  # Very simplified
            alcohol = 'O' + 'O'.join(smiles.split('O')[1:])  # Very simplified
            disconnections.append({
                'reaction': 'Fischer Esterification',
                'precursors': [acid, alcohol],
                'reagents': ['H2SO4'],
                'conditions': {'temperature': 65, 'catalyst': 'H2SO4'},
                'yield': 0.7,
                'difficulty': 1,
                'cost_factor': 0.9,
                'priority': 0.9
            })
        
        return disconnections
    
    def _ring_formation_disconnections(self, smiles: str) -> List[Dict[str, Any]]:
        """Find disconnections involving ring formation."""
        disconnections = []
        
        # Simple aromatic synthesis from substituted benzenes
        if 'c1ccc' in smiles:
            # Assume Friedel-Crafts acylation possibility
            base_benzene = 'c1ccccc1'
            if smiles != base_benzene:
                disconnections.append({
                    'reaction': 'Friedel-Crafts Acylation',
                    'precursors': [base_benzene, 'CC(=O)Cl'],  # Simplified
                    'reagents': ['AlCl3'],
                    'conditions': {'temperature': 25, 'solvent': 'DCM', 'catalyst': 'AlCl3'},
                    'yield': 0.75,
                    'difficulty': 2,
                    'cost_factor': 1.2,
                    'priority': 0.7
                })
        
        return disconnections
    
    def _cc_bond_disconnections(self, smiles: str) -> List[Dict[str, Any]]:
        """Find carbon-carbon bond disconnections."""
        disconnections = []
        
        # Look for potential Suzuki coupling sites
        if 'c1ccc' in smiles and smiles.count('c') > 6:  # Biaryl compound
            # Simplified biaryl disconnection
            fragment1 = 'c1ccc(cc1)Br'
            fragment2 = 'c1ccc(cc1)B(O)O'
            disconnections.append({
                'reaction': 'Suzuki Coupling',
                'precursors': [fragment1, fragment2],
                'reagents': ['Pd(PPh3)4', 'K2CO3'],
                'conditions': {'temperature': 80, 'solvent': 'toluene', 'catalyst': 'Pd(PPh3)4'},
                'yield': 0.85,
                'difficulty': 4,
                'cost_factor': 2.5,
                'priority': 0.6
            })
        
        return disconnections
    
    def _score_synthesis_route(self, route: List[ReactionStep], green_chemistry: bool = False) -> float:
        """Score a synthesis route based on multiple factors."""
        if not route:
            return 0.0
        
        # Base score
        score = 1.0
        
        # Penalize long routes
        length_penalty = max(0.1, 1.0 - 0.15 * (len(route) - 1))
        score *= length_penalty
        
        # Factor in individual step difficulties
        avg_difficulty = sum(step.difficulty for step in route) / len(route)
        difficulty_factor = max(0.2, 1.0 - (avg_difficulty - 1) * 0.2)
        score *= difficulty_factor
        
        # Factor in yields
        total_yield = self._calculate_total_yield(route)
        score *= total_yield
        
        # Factor in costs
        cost_factor = sum(step.cost_factor for step in route) / len(route)
        cost_penalty = max(0.5, 2.0 - cost_factor)
        score *= cost_penalty
        
        # Green chemistry bonus
        if green_chemistry:
            green_bonus = self._assess_green_chemistry(route)
            score *= (1.0 + green_bonus * 0.2)
        
        return max(0.0, min(1.0, score))
    
    def _calculate_total_yield(self, route: List[ReactionStep]) -> float:
        """Calculate total yield of synthesis route."""
        if not route:
            return 0.0
        
        total_yield = 1.0
        for step in route:
            total_yield *= step.yield_estimate
        
        return total_yield
    
    def _estimate_route_cost(self, route: List[ReactionStep]) -> float:
        """Estimate total cost of synthesis route ($/g)."""
        if not route:
            return 0.0
        
        base_cost = 10.0  # Base cost per gram
        
        for step in route:
            # Material costs
            base_cost += len(step.reagents) * 5.0
            base_cost *= step.cost_factor
            
            # Labor and equipment costs
            base_cost += step.difficulty * 15.0
        
        return round(base_cost, 2)
    
    def _estimate_purchase_cost(self, smiles: str) -> float:
        """Estimate cost if purchasing commercially."""
        for material in self.commercial_materials:
            if self._canonicalize_smiles(material['smiles']) == self._canonicalize_smiles(smiles):
                return material['price_per_g'] * 1000  # Convert to $/kg
        return 100.0  # Default cost
    
    def _assess_green_chemistry(self, route: List[ReactionStep]) -> float:
        """Assess how green/sustainable a synthesis route is."""
        green_score = 0.0
        
        for step in route:
            # Prefer aqueous solvents
            if step.conditions.get('solvent') in ['H2O', 'EtOH', 'none']:
                green_score += 0.2
            elif step.conditions.get('solvent') in ['DCM', 'CHCl3', 'benzene']:
                green_score -= 0.3
                
            # Prefer mild conditions
            temp = step.conditions.get('temperature', 25)
            if temp <= 50:
                green_score += 0.1
            elif temp > 100:
                green_score -= 0.2
                
            # Penalize toxic reagents
            toxic_reagents = ['AlCl3', 'BF3', 'HF', 'Br2', 'Cl2']
            if any(reagent in step.reagents for reagent in toxic_reagents):
                green_score -= 0.3
        
        return max(-1.0, min(1.0, green_score))
    
    @log_function_call(logger)
    def estimate_synthesizability(self, molecule: Molecule) -> float:
        """
        Estimate how synthesizable a molecule is.
        
        Args:
            molecule: Molecule to assess
            
        Returns:
            Synthesizability score (0-1)
        """
        if not molecule.is_valid:
            return 0.0
            
        # Check commercial availability first
        if self._is_commercial_available(molecule.smiles):
            return 1.0
        
        # Generate a few synthesis routes and take the best score
        try:
            routes = self.suggest_synthesis_routes(
                molecule, 
                max_steps=4, 
                max_routes=3
            )
            
            if not routes:
                return 0.2  # Some baseline synthesizability
            
            # Return the best route score
            return routes[0].score
            
        except Exception as e:
            logger.error(f"Error estimating synthesizability: {e}")
            return 0.3  # Conservative estimate