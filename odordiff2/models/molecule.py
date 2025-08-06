"""
Molecule data structures and representations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


@dataclass
class OdorProfile:
    """Represents the odor characteristics of a molecule."""
    primary_notes: List[str] = field(default_factory=list)
    secondary_notes: List[str] = field(default_factory=list)
    intensity: float = 0.0  # 0-1 scale
    longevity_hours: float = 0.0
    sillage: float = 0.0  # projection/diffusion strength
    character: str = ""
    
    def __str__(self) -> str:
        notes = ", ".join(self.primary_notes[:3])
        return f"{notes} (intensity: {self.intensity:.2f})"


@dataclass
class SynthesisRoute:
    """Represents a synthetic route to produce a molecule."""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 0.0  # feasibility score 0-1
    total_yield: float = 0.0
    cost_estimate: float = 0.0  # $/g
    
    
@dataclass 
class SafetyReport:
    """Safety assessment results for a molecule."""
    toxicity: float = 0.0  # 0 = safe, 1 = toxic
    skin_sensitizer: bool = False
    eco_score: float = 0.0  # environmental impact 0-1
    ifra_compliant: bool = True
    regulatory_flags: List[Dict[str, str]] = field(default_factory=list)


class Molecule:
    """Represents a generated scent molecule with properties and predictions."""
    
    def __init__(self, smiles: str, confidence: float = 1.0):
        self.smiles = smiles
        self.confidence = confidence
        self._mol = None
        self._properties = {}
        
        # Core attributes
        self.odor_profile = OdorProfile()
        self.safety_score = 0.0
        self.synth_score = 0.0
        self.estimated_cost = 0.0
        
    @property
    def mol(self) -> Optional[Chem.Mol]:
        """Get RDKit molecule object."""
        if self._mol is None and self.smiles:
            self._mol = Chem.MolFromSmiles(self.smiles)
        return self._mol
    
    @property
    def is_valid(self) -> bool:
        """Check if molecule is chemically valid."""
        return self.mol is not None
    
    def get_property(self, name: str) -> Optional[float]:
        """Get cached molecular property."""
        if name not in self._properties and self.mol:
            self._compute_properties()
        return self._properties.get(name)
    
    def _compute_properties(self):
        """Compute and cache molecular properties."""
        if not self.mol:
            return
            
        self._properties.update({
            'molecular_weight': Descriptors.MolWt(self.mol),
            'logP': Descriptors.MolLogP(self.mol),
            'tpsa': Descriptors.TPSA(self.mol),
            'hbd': rdMolDescriptors.CalcNumHBD(self.mol),
            'hba': rdMolDescriptors.CalcNumHBA(self.mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(self.mol),
            'aromatic_rings': Descriptors.NumAromaticRings(self.mol)
        })
    
    def get_molecular_fingerprint(self) -> Optional[np.ndarray]:
        """Get Morgan fingerprint for similarity calculations."""
        if not self.mol:
            return None
        from rdkit.Chem import rdMolDescriptors
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(self.mol, 2, nBits=2048)
        return np.array(fp)
    
    def calculate_similarity(self, other: 'Molecule') -> float:
        """Calculate Tanimoto similarity to another molecule."""
        fp1 = self.get_molecular_fingerprint()
        fp2 = other.get_molecular_fingerprint()
        
        if fp1 is None or fp2 is None:
            return 0.0
            
        from rdkit import DataStructs
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    
    def visualize_3d(self, save_path: str = "molecule.html"):
        """Generate 3D visualization HTML."""
        if not self.mol:
            raise ValueError("Invalid molecule for visualization")
            
        # Basic 3D visualization using py3Dmol
        try:
            import py3Dmol
            from rdkit.Chem import rdDepictor
            from rdkit.Chem.rdDepictor import Compute2DCoords
            
            # Generate 2D coordinates as fallback
            Compute2DCoords(self.mol)
            
            # Create simple HTML visualization
            html_content = f"""
            <html>
            <head><title>Molecule Visualization</title></head>
            <body>
                <h2>Molecule: {self.smiles}</h2>
                <div id="viewer" style="width:600px;height:400px;"></div>
                <script src="https://3dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
                <script>
                let viewer = $3Dmol.createViewer("viewer");
                viewer.addModel("{self.smiles}", "smi");
                viewer.setStyle({{}}, {{stick:{{}}}});
                viewer.zoomTo();
                viewer.render();
                </script>
                <p><strong>Odor Profile:</strong> {self.odor_profile}</p>
                <p><strong>Safety Score:</strong> {self.safety_score:.2f}</p>
                <p><strong>Synthesis Score:</strong> {self.synth_score:.2f}</p>
            </body>
            </html>
            """
            
            with open(save_path, 'w') as f:
                f.write(html_content)
                
        except ImportError:
            # Fallback to simple text representation
            with open(save_path, 'w') as f:
                f.write(f"Molecule: {self.smiles}\nOdor: {self.odor_profile}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert molecule to dictionary representation."""
        return {
            'smiles': self.smiles,
            'confidence': self.confidence,
            'odor_profile': {
                'primary_notes': self.odor_profile.primary_notes,
                'secondary_notes': self.odor_profile.secondary_notes,
                'intensity': self.odor_profile.intensity,
                'longevity_hours': self.odor_profile.longevity_hours,
                'sillage': self.odor_profile.sillage,
                'character': self.odor_profile.character
            },
            'safety_score': self.safety_score,
            'synth_score': self.synth_score,
            'estimated_cost': self.estimated_cost,
            'properties': self._properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Molecule':
        """Create molecule from dictionary representation."""
        mol = cls(data['smiles'], data.get('confidence', 1.0))
        
        # Restore odor profile
        odor_data = data.get('odor_profile', {})
        mol.odor_profile = OdorProfile(
            primary_notes=odor_data.get('primary_notes', []),
            secondary_notes=odor_data.get('secondary_notes', []),
            intensity=odor_data.get('intensity', 0.0),
            longevity_hours=odor_data.get('longevity_hours', 0.0),
            sillage=odor_data.get('sillage', 0.0),
            character=odor_data.get('character', '')
        )
        
        # Restore other attributes
        mol.safety_score = data.get('safety_score', 0.0)
        mol.synth_score = data.get('synth_score', 0.0) 
        mol.estimated_cost = data.get('estimated_cost', 0.0)
        mol._properties = data.get('properties', {})
        
        return mol
    
    def __repr__(self) -> str:
        return f"Molecule(smiles='{self.smiles}', odor='{self.odor_profile}', safety={self.safety_score:.2f})"