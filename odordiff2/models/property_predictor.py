"""
Advanced molecular property prediction models for OdorDiff-2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import pickle

logger = logging.getLogger(__name__)


@dataclass
class MolecularProperties:
    """Comprehensive molecular properties."""
    molecular_weight: float
    logp: float
    hbd: int  # Hydrogen bond donors
    hba: int  # Hydrogen bond acceptors
    tpsa: float  # Topological polar surface area
    rotatable_bonds: int
    aromatic_rings: int
    heteroatoms: int
    formal_charge: int
    heavy_atoms: int
    complexity_score: float
    drug_likeness: float
    synthesizability: float
    stability_score: float
    volatility: float  # Important for fragrances
    polarity: float
    dipole_moment: float


class MolecularPropertyPredictor(nn.Module):
    """Neural network for comprehensive molecular property prediction."""
    
    def __init__(self, 
                 input_dim: int = 2048,
                 hidden_dims: List[int] = [1024, 512, 256],
                 output_dim: int = 17,  # Number of properties
                 dropout: float = 0.2,
                 use_batch_norm: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Property scalers (will be fitted during training)
        self.property_scalers = {}
        
    def forward(self, fingerprints: torch.Tensor) -> torch.Tensor:
        """Predict molecular properties from fingerprints."""
        return self.network(fingerprints)
    
    def predict_properties(self, fingerprints: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict and parse individual properties."""
        predictions = self.forward(fingerprints)
        
        property_names = [
            'molecular_weight', 'logp', 'hbd', 'hba', 'tpsa',
            'rotatable_bonds', 'aromatic_rings', 'heteroatoms', 
            'formal_charge', 'heavy_atoms', 'complexity_score',
            'drug_likeness', 'synthesizability', 'stability_score',
            'volatility', 'polarity', 'dipole_moment'
        ]
        
        property_dict = {}
        for i, prop_name in enumerate(property_names):
            property_dict[prop_name] = predictions[:, i]
        
        return property_dict


class OdorDescriptorPredictor(nn.Module):
    """Neural network for odor descriptor prediction."""
    
    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dims: List[int] = [512, 256, 128],
                 num_descriptors: int = 50,  # Number of odor descriptors
                 dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Multi-task output heads
        self.intensity_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.longevity_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.sillage_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.descriptor_head = nn.Sequential(
            nn.Linear(prev_dim, num_descriptors),
            nn.Sigmoid()
        )
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Odor descriptor vocabulary
        self.odor_descriptors = [
            'floral', 'rose', 'jasmine', 'lavender', 'lily', 'violet',
            'citrus', 'lemon', 'orange', 'bergamot', 'lime', 'grapefruit',
            'woody', 'sandalwood', 'cedar', 'pine', 'oak', 'bamboo',
            'fresh', 'clean', 'aquatic', 'marine', 'ozonic', 'cucumber',
            'fruity', 'apple', 'peach', 'berry', 'tropical', 'melon',
            'spicy', 'pepper', 'cinnamon', 'clove', 'cardamom', 'ginger',
            'sweet', 'vanilla', 'honey', 'caramel', 'sugar', 'candy',
            'green', 'grass', 'leaf', 'herbal', 'mint', 'basil',
            'musky', 'animalic', 'powdery', 'aldehydic'
        ]
    
    def forward(self, fingerprints: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict odor properties."""
        shared_features = self.shared_layers(fingerprints)
        
        return {
            'intensity': self.intensity_head(shared_features).squeeze(-1),
            'longevity': self.longevity_head(shared_features).squeeze(-1),
            'sillage': self.sillage_head(shared_features).squeeze(-1),
            'descriptors': self.descriptor_head(shared_features)
        }
    
    def interpret_descriptors(self, descriptor_probs: torch.Tensor, threshold: float = 0.5) -> List[List[str]]:
        """Interpret descriptor probabilities as text labels."""
        batch_size = descriptor_probs.size(0)
        interpreted = []
        
        for i in range(batch_size):
            active_descriptors = []
            probs = descriptor_probs[i]
            
            for j, prob in enumerate(probs):
                if prob > threshold and j < len(self.odor_descriptors):
                    active_descriptors.append(self.odor_descriptors[j])
            
            interpreted.append(active_descriptors)
        
        return interpreted


class SynthesizabilityPredictor(nn.Module):
    """Neural network for synthesis difficulty prediction."""
    
    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Multi-output for different synthesis aspects
        self.difficulty_head = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        )
        
        self.cost_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()  # Cost should be positive
        )
        
        self.steps_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()  # Number of steps should be positive
        )
        
        self.shared_layers = nn.Sequential(*layers)
    
    def forward(self, fingerprints: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict synthesis properties."""
        shared_features = self.shared_layers(fingerprints)
        
        return {
            'difficulty': self.difficulty_head(shared_features).squeeze(-1),  # 0-1, lower is easier
            'estimated_cost': self.cost_head(shared_features).squeeze(-1),    # Cost per gram
            'estimated_steps': self.steps_head(shared_features).squeeze(-1)   # Number of synthesis steps
        }


class SafetyPredictor(nn.Module):
    """Neural network for safety assessment."""
    
    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Safety prediction heads
        self.toxicity_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.irritation_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.sensitization_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.environmental_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, fingerprints: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict safety properties."""
        shared_features = self.shared_layers(fingerprints)
        
        return {
            'toxicity_risk': self.toxicity_head(shared_features).squeeze(-1),
            'irritation_risk': self.irritation_head(shared_features).squeeze(-1),
            'sensitization_risk': self.sensitization_head(shared_features).squeeze(-1),
            'environmental_risk': self.environmental_head(shared_features).squeeze(-1)
        }


class RDKitPropertyCalculator:
    """Calculate molecular properties using RDKit."""
    
    @staticmethod
    def calculate_properties(smiles: str) -> Optional[MolecularProperties]:
        """Calculate comprehensive molecular properties."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Basic properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            aromatic_rings = Descriptors.NumAromaticRings(mol)
            heteroatoms = Descriptors.NumHeteroatoms(mol)
            formal_charge = Chem.rdmolops.GetFormalCharge(mol)
            heavy_atoms = Descriptors.HeavyAtomCount(mol)
            
            # Complexity score (simplified)
            complexity_score = (
                aromatic_rings * 0.2 +
                rotatable_bonds * 0.1 +
                heteroatoms * 0.15 +
                (mw / 100) * 0.1
            ) / 4.0
            
            # Drug-likeness (Lipinski's rule)
            drug_likeness = 1.0
            if mw > 500:
                drug_likeness -= 0.25
            if logp > 5:
                drug_likeness -= 0.25
            if hbd > 5:
                drug_likeness -= 0.25
            if hba > 10:
                drug_likeness -= 0.25
            drug_likeness = max(0.0, drug_likeness)
            
            # Synthesizability (heuristic)
            synth_score = 1.0 - min(1.0, complexity_score)
            
            # Stability score (heuristic)
            stability_score = 1.0
            if aromatic_rings == 0 and rotatable_bonds > 8:
                stability_score -= 0.2
            if formal_charge != 0:
                stability_score -= 0.1
            stability_score = max(0.0, stability_score)
            
            # Volatility (inverse of molecular weight, important for fragrances)
            volatility = max(0.0, 1.0 - (mw / 300.0))
            
            # Polarity (based on TPSA)
            polarity = min(1.0, tpsa / 100.0)
            
            # Estimated dipole moment (heuristic)
            dipole_moment = polarity * (1.0 + heteroatoms * 0.1)
            
            return MolecularProperties(
                molecular_weight=mw,
                logp=logp,
                hbd=hbd,
                hba=hba,
                tpsa=tpsa,
                rotatable_bonds=rotatable_bonds,
                aromatic_rings=aromatic_rings,
                heteroatoms=heteroatoms,
                formal_charge=formal_charge,
                heavy_atoms=heavy_atoms,
                complexity_score=complexity_score,
                drug_likeness=drug_likeness,
                synthesizability=synth_score,
                stability_score=stability_score,
                volatility=volatility,
                polarity=polarity,
                dipole_moment=dipole_moment
            )
        
        except Exception as e:
            logger.warning(f"Error calculating properties for {smiles}: {e}")
            return None


class ComprehensivePropertyPredictor:
    """Comprehensive molecular property predictor combining neural and traditional methods."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        
        # Initialize neural predictors
        self.property_predictor = MolecularPropertyPredictor().to(device)
        self.odor_predictor = OdorDescriptorPredictor().to(device)
        self.synthesis_predictor = SynthesizabilityPredictor().to(device)
        self.safety_predictor = SafetyPredictor().to(device)
        
        # Traditional calculator
        self.rdkit_calculator = RDKitPropertyCalculator()
        
        # State
        self.is_trained = False
    
    def predict_all_properties(self, 
                              smiles: str, 
                              fingerprint: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Predict all molecular properties."""
        results = {
            'smiles': smiles,
            'valid': False,
            'rdkit_properties': None,
            'neural_predictions': {}
        }
        
        # Calculate RDKit properties
        rdkit_props = self.rdkit_calculator.calculate_properties(smiles)
        if rdkit_props:
            results['valid'] = True
            results['rdkit_properties'] = rdkit_props
        
        # Neural predictions (if fingerprint provided and models trained)
        if fingerprint is not None and self.is_trained:
            fingerprint = fingerprint.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Molecular properties
                mol_props = self.property_predictor.predict_properties(fingerprint)
                results['neural_predictions']['molecular'] = {
                    k: v.item() for k, v in mol_props.items()
                }
                
                # Odor properties
                odor_props = self.odor_predictor(fingerprint)
                results['neural_predictions']['odor'] = {
                    'intensity': odor_props['intensity'].item(),
                    'longevity': odor_props['longevity'].item(),
                    'sillage': odor_props['sillage'].item(),
                    'descriptors': self.odor_predictor.interpret_descriptors(
                        odor_props['descriptors']
                    )[0]
                }
                
                # Synthesis properties
                synth_props = self.synthesis_predictor(fingerprint)
                results['neural_predictions']['synthesis'] = {
                    k: v.item() for k, v in synth_props.items()
                }
                
                # Safety properties
                safety_props = self.safety_predictor(fingerprint)
                results['neural_predictions']['safety'] = {
                    k: v.item() for k, v in safety_props.items()
                }
        
        return results
    
    def get_fingerprint(self, smiles: str) -> Optional[torch.Tensor]:
        """Calculate molecular fingerprint."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=2048
            )
            return torch.tensor(np.array(fingerprint), dtype=torch.float32)
        
        except Exception as e:
            logger.warning(f"Error calculating fingerprint for {smiles}: {e}")
            return None
    
    def save_models(self, save_dir: str):
        """Save all trained models."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(self.property_predictor.state_dict(), 
                  os.path.join(save_dir, 'property_predictor.pt'))
        torch.save(self.odor_predictor.state_dict(), 
                  os.path.join(save_dir, 'odor_predictor.pt'))
        torch.save(self.synthesis_predictor.state_dict(), 
                  os.path.join(save_dir, 'synthesis_predictor.pt'))
        torch.save(self.safety_predictor.state_dict(), 
                  os.path.join(save_dir, 'safety_predictor.pt'))
        
        logger.info(f"Models saved to {save_dir}")
    
    def load_models(self, save_dir: str):
        """Load trained models."""
        import os
        
        if os.path.exists(os.path.join(save_dir, 'property_predictor.pt')):
            self.property_predictor.load_state_dict(
                torch.load(os.path.join(save_dir, 'property_predictor.pt'), 
                          map_location=self.device)
            )
            self.odor_predictor.load_state_dict(
                torch.load(os.path.join(save_dir, 'odor_predictor.pt'), 
                          map_location=self.device)
            )
            self.synthesis_predictor.load_state_dict(
                torch.load(os.path.join(save_dir, 'synthesis_predictor.pt'), 
                          map_location=self.device)
            )
            self.safety_predictor.load_state_dict(
                torch.load(os.path.join(save_dir, 'safety_predictor.pt'), 
                          map_location=self.device)
            )
            
            self.is_trained = True
            logger.info(f"Models loaded from {save_dir}")
        else:
            logger.warning(f"No trained models found in {save_dir}")


# Example usage and testing
if __name__ == "__main__":
    # Test property calculation
    predictor = ComprehensivePropertyPredictor()
    
    test_smiles = "CC(C)=CCO"  # Linalool
    fingerprint = predictor.get_fingerprint(test_smiles)
    
    if fingerprint is not None:
        results = predictor.predict_all_properties(test_smiles, fingerprint)
        
        print("SMILES:", results['smiles'])
        print("Valid:", results['valid'])
        
        if results['rdkit_properties']:
            props = results['rdkit_properties']
            print(f"Molecular Weight: {props.molecular_weight:.2f}")
            print(f"LogP: {props.logp:.2f}")
            print(f"Volatility: {props.volatility:.2f}")
    else:
        print(f"Invalid SMILES: {test_smiles}")