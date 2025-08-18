"""
OdorDiff-2: Safe Text-to-Scent Molecule Diffusion

A state-of-the-art AI system for generating novel scent molecules from text descriptions,
with integrated safety filters and synthesizability scoring.
"""

# Core imports with error handling for optional dependencies
try:
    from .core.diffusion import OdorDiffusion
except ImportError as e:
    print(f"Warning: Could not import OdorDiffusion: {e}")
    OdorDiffusion = None

try:
    from .safety.filter import SafetyFilter
except ImportError as e:
    print(f"Warning: Could not import SafetyFilter: {e}")
    SafetyFilter = None

try:
    from .models.molecule import Molecule, OdorProfile
except ImportError as e:
    print(f"Warning: Could not import Molecule/OdorProfile: {e}")
    Molecule = OdorProfile = None

try:
    from .core.synthesis import SynthesisPlanner
except ImportError as e:
    print(f"Warning: Could not import SynthesisPlanner (likely missing RDKit): {e}")
    SynthesisPlanner = None

try:
    from .visualization.viewer import MoleculeViewer
except ImportError as e:
    print(f"Warning: Could not import MoleculeViewer: {e}")
    MoleculeViewer = None

__version__ = "1.0.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.ai"

__all__ = [
    "OdorDiffusion",
    "SafetyFilter", 
    "Molecule",
    "OdorProfile",
    "SynthesisPlanner",
    "MoleculeViewer"
]