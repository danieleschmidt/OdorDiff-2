#!/usr/bin/env python3
"""
Generation 1 Basic Functionality Test
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_basic_functionality():
    """Test core OdorDiff2 functionality"""
    print('=== GENERATION 1: BASIC FUNCTIONALITY TEST ===')
    
    try:
        import odordiff2
        from odordiff2 import OdorDiffusion, SafetyFilter, Molecule
        print(f'✓ Imports successful - OdorDiff2 v{odordiff2.__version__}')
    except Exception as e:
        print(f'✗ Import failed: {e}')
        return False

    try:
        # Test core model initialization
        print('Testing OdorDiffusion model...')
        model = OdorDiffusion.from_pretrained('odordiff2-safe-v1', device='cpu')
        print('✓ Model initialized successfully')
        
        # Test molecule generation
        print('Testing molecule generation...')
        molecules = model.generate(
            prompt='fresh citrus with floral undertones',
            num_molecules=3,
            safety_filter=None,
            synthesizability_min=0.5
        )
        print(f'✓ Generated {len(molecules)} molecules')

        # Test molecule properties
        for i, mol in enumerate(molecules[:2]):
            print(f'  Molecule {i+1}: {mol.smiles}')
            print(f'    Odor: {mol.odor_profile}')
            print(f'    Safety: {mol.safety_score:.2f}, Synthesis: {mol.synth_score:.2f}')
            print(f'    Cost: ${mol.estimated_cost:.2f}/kg')

        # Test safety filter
        print('Testing SafetyFilter...')
        safety = SafetyFilter(toxicity_threshold=0.1)
        safe_molecules, unsafe = safety.filter_molecules(molecules)
        print(f'✓ Safety filter processed: {len(safe_molecules)} safe, {len(unsafe)} filtered')

        # Test fragrance design
        print('Testing fragrance design...')
        fragrance = model.design_fragrance(
            base_notes="sandalwood, amber",
            heart_notes="jasmine, rose", 
            top_notes="bergamot, lemon",
            style="elegant, modern"
        )
        formula = fragrance.to_perfume_formula(concentration='eau_de_parfum')
        print(f'✓ Fragrance formula created: {formula["fragrance_oil_percent"]:.1f}% oil concentration')

        print('=== GENERATION 1 TESTS PASSED ===')
        return True
        
    except Exception as e:
        print(f'✗ Test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_basic_functionality()
    sys.exit(0 if success else 1)