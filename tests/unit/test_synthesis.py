"""
Unit tests for synthesis route planning and feasibility assessment.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from odordiff2.core.synthesis import SynthesisPlanner, ReactionStep
from odordiff2.models.molecule import Molecule, SynthesisRoute


class TestReactionStep:
    """Test ReactionStep dataclass."""
    
    def test_reaction_step_creation(self):
        """Test creating a ReactionStep instance."""
        step = ReactionStep(
            reaction_type="Friedel-Crafts Acylation",
            reactants=["c1ccccc1", "CC(=O)Cl"],
            products=["CC(=O)c1ccccc1"],
            reagents=["AlCl3"],
            conditions={"temperature": 25, "solvent": "DCM"},
            yield_estimate=0.8,
            difficulty=2,
            cost_factor=1.2
        )
        
        assert step.reaction_type == "Friedel-Crafts Acylation"
        assert step.reactants == ["c1ccccc1", "CC(=O)Cl"]
        assert step.products == ["CC(=O)c1ccccc1"]
        assert step.reagents == ["AlCl3"]
        assert step.conditions == {"temperature": 25, "solvent": "DCM"}
        assert step.yield_estimate == 0.8
        assert step.difficulty == 2
        assert step.cost_factor == 1.2
    
    def test_reaction_step_defaults(self):
        """Test ReactionStep with minimal parameters."""
        step = ReactionStep(
            reaction_type="Test Reaction",
            reactants=["A"],
            products=["B"],
            reagents=[],
            conditions={},
            yield_estimate=0.5,
            difficulty=1,
            cost_factor=1.0
        )
        
        assert step.reaction_type == "Test Reaction"
        assert step.reactants == ["A"]
        assert step.products == ["B"]
        assert step.reagents == []
        assert step.conditions == {}


class TestSynthesisPlanner:
    """Test SynthesisPlanner class."""
    
    @pytest.fixture
    def planner(self):
        """Create SynthesisPlanner instance for testing."""
        return SynthesisPlanner()
    
    @pytest.fixture
    def sample_molecule(self):
        """Create a sample molecule for testing."""
        mol = Molecule("CC(=O)c1ccccc1", confidence=0.9)
        mol.is_valid = True
        return mol
    
    @pytest.fixture
    def invalid_molecule(self):
        """Create an invalid molecule for testing."""
        mol = Molecule("invalid_smiles", confidence=0.5)
        mol.is_valid = False
        return mol
    
    def test_planner_initialization(self, planner):
        """Test SynthesisPlanner initialization."""
        assert hasattr(planner, 'reaction_templates')
        assert hasattr(planner, 'commercial_materials')
        assert hasattr(planner, 'retrosynthesis_rules')
        assert hasattr(planner, 'material_costs')
        
        assert len(planner.reaction_templates) > 0
        assert len(planner.commercial_materials) > 0
        assert len(planner.retrosynthesis_rules) > 0
        assert len(planner.material_costs) > 0
    
    def test_load_reaction_templates(self, planner):
        """Test loading of reaction templates."""
        templates = planner.reaction_templates
        
        # Check we have expected reactions
        reaction_names = [t['name'] for t in templates]
        expected_reactions = [
            'Friedel-Crafts Acylation',
            'Grignard Addition',
            'Williamson Ether Synthesis',
            'Aldol Condensation',
            'Reduction with NaBH4',
            'Oxidation with PCC',
            'Suzuki Coupling',
            'Fischer Esterification'
        ]
        
        for reaction in expected_reactions:
            assert reaction in reaction_names
        
        # Check template structure
        template = templates[0]
        assert 'name' in template
        assert 'pattern' in template
        assert 'conditions' in template
        assert 'yield_range' in template
        assert 'difficulty' in template
        assert 'cost_factor' in template
    
    def test_load_commercial_materials(self, planner):
        """Test loading of commercial materials."""
        materials = planner.commercial_materials
        
        # Check we have common materials
        smiles_list = [m['smiles'] for m in materials]
        expected_materials = ['c1ccccc1', 'CCO', 'CC(C)O', 'CC(=O)O']
        
        for material in expected_materials:
            assert material in smiles_list
        
        # Check material structure
        material = materials[0]
        assert 'smiles' in material
        assert 'name' in material
        assert 'price_per_g' in material
        assert 'availability' in material
    
    def test_canonicalize_smiles(self, planner):
        """Test SMILES canonicalization."""
        # Test valid SMILES
        canonical = planner._canonicalize_smiles("c1ccccc1")
        assert canonical == "c1ccccc1"
        
        # Test invalid SMILES should return original
        invalid = planner._canonicalize_smiles("invalid_smiles")
        assert invalid == "invalid_smiles"
    
    def test_is_commercial_available(self, planner):
        """Test checking commercial availability."""
        # Test available material
        assert planner._is_commercial_available("c1ccccc1")  # benzene
        assert planner._is_commercial_available("CCO")  # ethanol
        
        # Test unavailable material
        assert not planner._is_commercial_available("very_complex_molecule")
    
    def test_suggest_synthesis_routes_invalid_molecule(self, planner, invalid_molecule):
        """Test synthesis route suggestion for invalid molecule."""
        routes = planner.suggest_synthesis_routes(invalid_molecule)
        assert routes == []
    
    def test_suggest_synthesis_routes_commercial_available(self, planner):
        """Test synthesis routes for commercially available molecule."""
        mol = Molecule("c1ccccc1", confidence=0.9)  # benzene
        mol.is_valid = True
        
        routes = planner.suggest_synthesis_routes(mol)
        
        assert len(routes) == 1
        assert routes[0].steps[0]['reaction_type'] == 'Commercial Purchase'
        assert routes[0].score == 1.0
        assert routes[0].total_yield == 1.0
    
    def test_suggest_synthesis_routes_synthetic_target(self, planner, sample_molecule):
        """Test synthesis route suggestion for synthetic target."""
        routes = planner.suggest_synthesis_routes(sample_molecule, max_routes=3)
        
        assert isinstance(routes, list)
        assert len(routes) <= 3
        
        if routes:  # If routes were generated
            route = routes[0]
            assert isinstance(route, SynthesisRoute)
            assert hasattr(route, 'steps')
            assert hasattr(route, 'score')
            assert hasattr(route, 'total_yield')
            assert hasattr(route, 'cost_estimate')
            assert 0 <= route.score <= 1
            assert 0 <= route.total_yield <= 1
            assert route.cost_estimate >= 0
    
    def test_generate_retrosynthetic_routes(self, planner):
        """Test retrosynthetic route generation."""
        routes = planner._generate_retrosynthetic_routes("CC(=O)c1ccccc1", 3, 2)
        
        assert isinstance(routes, list)
        assert len(routes) <= 2
        
        if routes:
            route = routes[0]
            assert isinstance(route, list)
            assert len(route) <= 3
            
            if route:
                step = route[0]
                assert isinstance(step, ReactionStep)
    
    def test_find_disconnections(self, planner):
        """Test finding disconnections."""
        # Test functional group approach
        disconnections = planner._find_disconnections("CC(=O)c1ccccc1", approach=0)
        assert isinstance(disconnections, list)
        
        # Test ring formation approach
        disconnections = planner._find_disconnections("c1ccc(cc1)C", approach=1)
        assert isinstance(disconnections, list)
        
        # Test C-C bond approach
        disconnections = planner._find_disconnections("c1ccc(cc1)c2ccccc2", approach=2)
        assert isinstance(disconnections, list)
    
    def test_functional_group_disconnections(self, planner):
        """Test functional group-based disconnections."""
        # Test ketone disconnection
        disconnections = planner._functional_group_disconnections("CC(=O)c1ccccc1")
        
        assert isinstance(disconnections, list)
        
        if disconnections:
            disconnection = disconnections[0]
            assert 'reaction' in disconnection
            assert 'precursors' in disconnection
            assert 'reagents' in disconnection
            assert 'priority' in disconnection
    
    def test_ring_formation_disconnections(self, planner):
        """Test ring formation disconnections."""
        disconnections = planner._ring_formation_disconnections("c1ccc(cc1)C(=O)C")
        
        assert isinstance(disconnections, list)
        
        if disconnections:
            disconnection = disconnections[0]
            assert 'reaction' in disconnection
            assert 'precursors' in disconnection
            assert 'priority' in disconnection
    
    def test_cc_bond_disconnections(self, planner):
        """Test carbon-carbon bond disconnections."""
        # Test biaryl compound
        disconnections = planner._cc_bond_disconnections("c1ccc(cc1)c2ccccc2")
        
        assert isinstance(disconnections, list)
        
        if disconnections:
            disconnection = disconnections[0]
            assert 'reaction' in disconnection
            assert 'precursors' in disconnection
            assert 'priority' in disconnection
    
    def test_calculate_total_yield(self, planner):
        """Test total yield calculation."""
        # Empty route
        assert planner._calculate_total_yield([]) == 0.0
        
        # Single step
        step = ReactionStep(
            reaction_type="Test",
            reactants=["A"],
            products=["B"],
            reagents=[],
            conditions={},
            yield_estimate=0.8,
            difficulty=1,
            cost_factor=1.0
        )
        assert planner._calculate_total_yield([step]) == 0.8
        
        # Multiple steps
        step2 = ReactionStep(
            reaction_type="Test2",
            reactants=["B"],
            products=["C"],
            reagents=[],
            conditions={},
            yield_estimate=0.9,
            difficulty=1,
            cost_factor=1.0
        )
        assert planner._calculate_total_yield([step, step2]) == 0.72  # 0.8 * 0.9
    
    def test_estimate_route_cost(self, planner):
        """Test route cost estimation."""
        # Empty route
        assert planner._estimate_route_cost([]) == 0.0
        
        # Single step
        step = ReactionStep(
            reaction_type="Test",
            reactants=["A"],
            products=["B"],
            reagents=["Reagent1", "Reagent2"],
            conditions={},
            yield_estimate=0.8,
            difficulty=2,
            cost_factor=1.5
        )
        
        cost = planner._estimate_route_cost([step])
        assert cost > 0
        assert isinstance(cost, float)
    
    def test_estimate_purchase_cost(self, planner):
        """Test purchase cost estimation."""
        # Commercial material
        cost = planner._estimate_purchase_cost("c1ccccc1")  # benzene
        assert cost > 0
        
        # Non-commercial material
        cost = planner._estimate_purchase_cost("very_complex_molecule")
        assert cost == 100.0  # default cost
    
    def test_score_synthesis_route(self, planner):
        """Test synthesis route scoring."""
        # Empty route
        assert planner._score_synthesis_route([]) == 0.0
        
        # Good route
        step = ReactionStep(
            reaction_type="Test",
            reactants=["A"],
            products=["B"],
            reagents=["NaBH4"],
            conditions={"temperature": 25, "solvent": "EtOH"},
            yield_estimate=0.9,
            difficulty=1,
            cost_factor=1.0
        )
        
        score = planner._score_synthesis_route([step])
        assert 0 <= score <= 1
        
        # Test green chemistry bonus
        green_score = planner._score_synthesis_route([step], green_chemistry=True)
        assert green_score >= score  # Should be same or higher with green bonus
    
    def test_assess_green_chemistry(self, planner):
        """Test green chemistry assessment."""
        # Green step
        green_step = ReactionStep(
            reaction_type="Green",
            reactants=["A"],
            products=["B"],
            reagents=["NaBH4"],
            conditions={"temperature": 25, "solvent": "H2O"},
            yield_estimate=0.8,
            difficulty=1,
            cost_factor=1.0
        )
        
        green_score = planner._assess_green_chemistry([green_step])
        assert green_score > 0
        
        # Non-green step
        toxic_step = ReactionStep(
            reaction_type="Toxic",
            reactants=["A"],
            products=["B"],
            reagents=["AlCl3", "Br2"],
            conditions={"temperature": 150, "solvent": "DCM"},
            yield_estimate=0.8,
            difficulty=1,
            cost_factor=1.0
        )
        
        toxic_score = planner._assess_green_chemistry([toxic_step])
        assert toxic_score < 0
    
    def test_estimate_synthesizability_invalid_molecule(self, planner, invalid_molecule):
        """Test synthesizability estimation for invalid molecule."""
        score = planner.estimate_synthesizability(invalid_molecule)
        assert score == 0.0
    
    def test_estimate_synthesizability_commercial_molecule(self, planner):
        """Test synthesizability for commercially available molecule."""
        mol = Molecule("c1ccccc1", confidence=0.9)
        mol.is_valid = True
        
        score = planner.estimate_synthesizability(mol)
        assert score == 1.0
    
    def test_estimate_synthesizability_synthetic_molecule(self, planner, sample_molecule):
        """Test synthesizability for synthetic molecule."""
        score = planner.estimate_synthesizability(sample_molecule)
        assert 0 <= score <= 1
    
    @patch('odordiff2.core.synthesis.logger')
    def test_error_handling_in_synthesizability(self, mock_logger, planner, sample_molecule):
        """Test error handling in synthesizability estimation."""
        # Mock an exception in suggest_synthesis_routes
        with patch.object(planner, 'suggest_synthesis_routes', side_effect=Exception("Test error")):
            score = planner.estimate_synthesizability(sample_molecule)
            
            assert score == 0.3  # Conservative estimate
            mock_logger.error.assert_called_once()
    
    def test_synthesis_routes_with_parameters(self, planner, sample_molecule):
        """Test synthesis route suggestion with different parameters."""
        # Test with different max_steps
        routes_short = planner.suggest_synthesis_routes(
            sample_molecule, max_steps=2, max_routes=2
        )
        routes_long = planner.suggest_synthesis_routes(
            sample_molecule, max_steps=6, max_routes=2
        )
        
        assert isinstance(routes_short, list)
        assert isinstance(routes_long, list)
        
        # Test with green chemistry preference
        routes_green = planner.suggest_synthesis_routes(
            sample_molecule, green_chemistry=True, max_routes=3
        )
        
        assert isinstance(routes_green, list)
        
        # Test with different starting materials
        routes_any = planner.suggest_synthesis_routes(
            sample_molecule, starting_materials='any', max_routes=2
        )
        
        assert isinstance(routes_any, list)
    
    def test_route_filtering_and_scoring(self, planner):
        """Test that low-scoring routes are filtered out."""
        # Create a mock route with very low yield
        mock_route = [
            ReactionStep(
                reaction_type="Poor",
                reactants=["A"],
                products=["B"],
                reagents=["expensive_reagent"],
                conditions={"temperature": 200, "solvent": "toxic"},
                yield_estimate=0.01,  # Very low yield
                difficulty=5,  # Very difficult
                cost_factor=10.0  # Very expensive
            )
        ]
        
        score = planner._score_synthesis_route(mock_route)
        
        # Very low-scoring routes should get filtered
        assert score <= 0.1
    
    def test_material_costs_structure(self, planner):
        """Test material costs data structure."""
        costs = planner.material_costs
        
        expected_categories = [
            'standard_reagents',
            'organometallics', 
            'catalysts',
            'specialty_chemicals',
            'solvents'
        ]
        
        for category in expected_categories:
            assert category in costs
            assert isinstance(costs[category], (int, float))
            assert costs[category] > 0
    
    def test_retrosynthesis_rules_structure(self, planner):
        """Test retrosynthesis rules data structure."""
        rules = planner.retrosynthesis_rules
        
        assert len(rules) > 0
        
        for rule in rules:
            assert 'name' in rule
            assert 'product_pattern' in rule
            assert 'precursor_pattern' in rule
            assert 'reaction' in rule
            assert 'priority' in rule
            assert 0 <= rule['priority'] <= 1


class TestSynthesisRouteIntegration:
    """Integration tests for synthesis planning."""
    
    def test_full_synthesis_planning_workflow(self):
        """Test complete synthesis planning workflow."""
        planner = SynthesisPlanner()
        
        # Create a complex molecule
        mol = Molecule("CC(=O)c1ccc(cc1)OC", confidence=0.8)
        mol.is_valid = True
        
        # Plan synthesis routes
        routes = planner.suggest_synthesis_routes(
            mol,
            max_steps=4,
            green_chemistry=True,
            max_routes=3
        )
        
        assert isinstance(routes, list)
        
        if routes:
            # Check route quality
            best_route = routes[0]
            assert isinstance(best_route, SynthesisRoute)
            assert best_route.score > 0
            assert 0 <= best_route.total_yield <= 1
            assert best_route.cost_estimate > 0
            
            # Verify steps structure
            if best_route.steps:
                for step in best_route.steps:
                    if isinstance(step, ReactionStep):
                        assert hasattr(step, 'reaction_type')
                        assert hasattr(step, 'reactants')
                        assert hasattr(step, 'products')
                    elif isinstance(step, dict):
                        assert 'reaction_type' in step
        
        # Test synthesizability estimation
        synth_score = planner.estimate_synthesizability(mol)
        assert 0 <= synth_score <= 1
    
    @pytest.mark.parametrize("smiles,expected_available", [
        ("c1ccccc1", True),  # benzene - commercial
        ("CCO", True),       # ethanol - commercial  
        ("CC(C)O", True),    # isopropanol - commercial
        ("very_complex_unreal_molecule", False)  # not available
    ])
    def test_commercial_availability_check(self, smiles, expected_available):
        """Test commercial availability for various molecules."""
        planner = SynthesisPlanner()
        
        is_available = planner._is_commercial_available(smiles)
        assert is_available == expected_available