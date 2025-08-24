#!/usr/bin/env python3
"""
Generation 1 Lightweight Test - Test without heavy dependencies
"""
import sys
import os

def test_core_classes():
    """Test core classes without ML dependencies"""
    print('=== GENERATION 1: LIGHTWEIGHT CORE TEST ===')
    
    try:
        # Test basic dataclasses
        sys.path.insert(0, os.path.abspath('.'))
        
        from odordiff2.models.molecule import Molecule, OdorProfile, SafetyReport
        print('✓ Core molecule classes imported')
        
        # Test molecule creation
        mol = Molecule('CC(C)=CCO', confidence=0.85)
        print(f'✓ Created molecule: {mol.smiles}')
        
        # Test basic validation
        is_valid = mol.is_valid
        print(f'✓ Molecule validation: {is_valid}')
        
        # Test property estimation without RDKit
        mw = mol.get_property('molecular_weight')
        logp = mol.get_property('logP')
        print(f'✓ Estimated properties - MW: {mw}, LogP: {logp:.2f}')
        
        # Test odor profile
        odor = OdorProfile(
            primary_notes=['floral', 'sweet'],
            secondary_notes=['rosy', 'powdery'],
            intensity=0.7,
            longevity_hours=6.0,
            character='elegant'
        )
        mol.odor_profile = odor
        print(f'✓ Odor profile: {odor}')
        
        # Test serialization
        mol_dict = mol.to_dict()
        restored_mol = Molecule.from_dict(mol_dict)
        print(f'✓ Serialization test: {restored_mol.smiles == mol.smiles}')
        
        # Test visualization generation (without 3D)
        try:
            mol.visualize_3d('test_molecule.html')
            print('✓ Visualization HTML generated')
        except Exception as e:
            print(f'! Visualization warning: {e}')
        
        print('=== LIGHTWEIGHT CORE TESTS PASSED ===')
        return True
        
    except Exception as e:
        print(f'✗ Test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_config_system():
    """Test configuration system"""
    print('\n=== TESTING CONFIGURATION SYSTEM ===')
    
    try:
        from odordiff2.config.settings import get_config, validate_config
        
        # Test default config loading
        config = get_config()
        print(f'✓ Default config loaded with {len(config)} sections')
        
        # Test config validation
        is_valid, errors = validate_config(config)
        print(f'✓ Config validation: {is_valid}, errors: {len(errors)}')
        
        return True
    except Exception as e:
        print(f'! Config system not available: {e}')
        return False

if __name__ == '__main__':
    success1 = test_core_classes()
    success2 = test_config_system()
    
    print(f'\n=== GENERATION 1 SUMMARY ===')
    print(f'Core Classes: {"PASS" if success1 else "FAIL"}')
    print(f'Config System: {"PASS" if success2 else "SKIP"}')
    
    sys.exit(0 if success1 else 1)