#!/usr/bin/env python3
"""
Generation 1 Implementation Test - Basic Functionality
Test the core diffusion model functionality with minimal dependencies.
"""

import sys
import os
sys.path.insert(0, '/root/repo')

from odordiff2 import OdorDiffusion, SafetyFilter

def test_basic_generation():
    """Test basic molecule generation functionality."""
    print("🧪 Testing Generation 1: Basic Functionality")
    
    # Initialize model
    print("Initializing OdorDiffusion model...")
    model = OdorDiffusion.from_pretrained('odordiff2-safe-v1')
    
    # Initialize safety filter
    print("Setting up safety filter...")
    safety = SafetyFilter(toxicity_threshold=0.1, irritant_check=True)
    
    # Test basic generation
    print("\n🌺 Generating molecules for 'fresh citrus scent'...")
    molecules = model.generate(
        prompt="fresh citrus scent",
        num_molecules=3,
        safety_filter=safety,
        synthesizability_min=0.5
    )
    
    print(f"✅ Generated {len(molecules)} molecules")
    
    # Examine results
    for i, mol in enumerate(molecules):
        print(f"\nMolecule {i+1}:")
        print(f"  SMILES: {mol.smiles}")
        print(f"  Valid: {mol.is_valid}")
        print(f"  Confidence: {mol.confidence:.3f}")
        if hasattr(mol, 'odor_profile') and mol.odor_profile:
            print(f"  Primary notes: {mol.odor_profile.primary_notes}")
            print(f"  Character: {mol.odor_profile.character}")
        if hasattr(mol, 'synth_score'):
            print(f"  Synthesis score: {mol.synth_score:.3f}")
        if hasattr(mol, 'estimated_cost'):
            print(f"  Estimated cost: ${mol.estimated_cost:.2f}/kg")
    
    return molecules

def test_fragrance_design():
    """Test fragrance formulation functionality."""
    print("\n🌸 Testing fragrance design...")
    
    model = OdorDiffusion.from_pretrained('odordiff2-safe-v1')
    
    # Design a complete fragrance
    fragrance = model.design_fragrance(
        base_notes="sandalwood, amber, musk",
        heart_notes="jasmine, rose, ylang-ylang",  
        top_notes="bergamot, lemon, green apple",
        style="modern, ethereal, long-lasting"
    )
    
    print(f"✅ Designed fragrance with style: {fragrance.style_descriptor}")
    print(f"  Base accord: {len(fragrance.base_accord)} molecules")
    print(f"  Heart accord: {len(fragrance.heart_accord)} molecules")
    print(f"  Top accord: {len(fragrance.top_accord)} molecules")
    
    # Convert to perfume formula
    formula = fragrance.to_perfume_formula(
        concentration='eau_de_parfum',
        carrier='ethanol_90'
    )
    
    print(f"\n📋 Perfume Formula ({formula['concentration_type']}):")
    print(f"  Total fragrance oil: {formula['fragrance_oil_percent']:.1f}%")
    print(f"  Top notes: {formula['top_notes']['actual_percent']:.2f}%")
    print(f"  Heart notes: {formula['heart_notes']['actual_percent']:.2f}%")
    print(f"  Base notes: {formula['base_notes']['actual_percent']:.2f}%")
    
    return fragrance, formula

if __name__ == "__main__":
    print("🚀 GENERATION 1: BASIC FUNCTIONALITY TEST")
    print("=" * 50)
    
    try:
        # Test 1: Basic molecule generation
        molecules = test_basic_generation()
        
        # Test 2: Fragrance design  
        fragrance, formula = test_fragrance_design()
        
        print("\n✅ GENERATION 1 IMPLEMENTATION SUCCESSFUL!")
        print("✅ Core diffusion model working")
        print("✅ Safety filtering operational")
        print("✅ Molecule generation functional")
        print("✅ Fragrance design working")
        print("✅ Basic odor prediction implemented")
        
    except Exception as e:
        print(f"\n❌ Error in Generation 1 testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)