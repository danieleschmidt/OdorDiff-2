"""
Molecule data structures and representations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Optional RDKit import with fallback
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    Descriptors = None
    rdMolDescriptors = None


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
    def mol(self):
        """Get RDKit molecule object."""
        if not RDKIT_AVAILABLE:
            return None
        if self._mol is None and self.smiles:
            self._mol = Chem.MolFromSmiles(self.smiles)
        return self._mol
    
    @property
    def is_valid(self) -> bool:
        """Check if molecule is chemically valid."""
        if not RDKIT_AVAILABLE:
            # Basic SMILES validation without RDKit
            return self._basic_smiles_validation()
        return self.mol is not None
    
    def _basic_smiles_validation(self) -> bool:
        """Basic SMILES validation without RDKit."""
        if not self.smiles or len(self.smiles) < 2:
            return False
        # Check for basic chemistry characters
        valid_chars = set('CNOSPFClBrI()[]=#-+1234567890@Hncos')
        return all(c in valid_chars for c in self.smiles)
    
    def get_property(self, name: str) -> Optional[float]:
        """Get cached molecular property."""
        if name not in self._properties:
            if RDKIT_AVAILABLE and self.mol:
                self._compute_properties()
            else:
                self._estimate_properties()
        return self._properties.get(name)
    
    def _compute_properties(self):
        """Compute and cache molecular properties using RDKit."""
        if not RDKIT_AVAILABLE or not self.mol:
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
    
    def _estimate_properties(self):
        """Estimate properties without RDKit (simplified)."""
        if not self.smiles:
            return
            
        # Very basic property estimation from SMILES string
        carbon_count = self.smiles.count('C') + self.smiles.count('c')
        oxygen_count = self.smiles.count('O') + self.smiles.count('o')
        nitrogen_count = self.smiles.count('N') + self.smiles.count('n')
        aromatic_count = self.smiles.count('c')
        
        # Rough molecular weight estimation
        estimated_mw = carbon_count * 12 + oxygen_count * 16 + nitrogen_count * 14 + 50
        
        self._properties.update({
            'molecular_weight': estimated_mw,
            'logP': min(5.0, carbon_count * 0.3 - oxygen_count * 0.5),
            'tpsa': oxygen_count * 20 + nitrogen_count * 15,
            'hbd': min(5, oxygen_count // 2 + nitrogen_count // 3),
            'hba': oxygen_count + nitrogen_count,
            'rotatable_bonds': max(0, carbon_count - aromatic_count - 2),
            'aromatic_rings': max(0, aromatic_count // 6)
        })
    
    def get_molecular_fingerprint(self) -> Optional[np.ndarray]:
        """Get Morgan fingerprint for similarity calculations."""
        if not RDKIT_AVAILABLE or not self.mol:
            return self._simple_fingerprint()
        
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(self.mol, 2, nBits=2048)
        return np.array(fp)
    
    def _simple_fingerprint(self) -> Optional[np.ndarray]:
        """Simple fingerprint based on SMILES string."""
        if not self.smiles:
            return None
        # Create a basic fingerprint from character counts
        features = np.zeros(64)
        features[0] = len(self.smiles) % 64
        features[1] = self.smiles.count('C') % 64
        features[2] = self.smiles.count('c') % 64
        features[3] = self.smiles.count('O') % 64
        features[4] = self.smiles.count('N') % 64
        features[5] = self.smiles.count('=') % 64
        return features
    
    def calculate_similarity(self, other: 'Molecule') -> float:
        """Calculate similarity to another molecule."""
        fp1 = self.get_molecular_fingerprint()
        fp2 = other.get_molecular_fingerprint()
        
        if fp1 is None or fp2 is None:
            return 0.0
        
        if RDKIT_AVAILABLE:
            from rdkit import DataStructs
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        else:
            # Simple cosine similarity for basic fingerprints
            dot_product = np.dot(fp1, fp2)
            norm1 = np.linalg.norm(fp1)
            norm2 = np.linalg.norm(fp2)
            return dot_product / (norm1 * norm2 + 1e-10)
    
    def visualize_3d(self, save_path: str = "molecule.html"):
        """Generate 3D visualization HTML."""
        if not self.is_valid:
            raise ValueError("Invalid molecule for visualization")
            
        # Create simple HTML visualization
        html_content = f"""
        <html>
        <head><title>Molecule Visualization</title></head>
        <body>
            <h2>Molecule: {self.smiles}</h2>
            <div id="viewer" style="width:600px;height:400px;border:1px solid #ccc;"></div>
            <script src="https://3dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
            <script>
            let viewer = $3Dmol.createViewer("viewer");
            viewer.addModel("{self.smiles}", "smi");
            viewer.setStyle({{}}, {{stick:{{}}}});
            viewer.zoomTo();
            viewer.render();
            </script>
            <div style="margin-top: 20px;">
                <p><strong>Odor Profile:</strong> {self.odor_profile}</p>
                <p><strong>Safety Score:</strong> {self.safety_score:.2f}</p>
                <p><strong>Synthesis Score:</strong> {self.synth_score:.2f}</p>
                <p><strong>Molecular Weight:</strong> {self.get_property('molecular_weight') or 'N/A'}</p>
                <p><strong>LogP:</strong> {self.get_property('logP') or 'N/A'}</p>
            </div>
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(html_content)
    
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