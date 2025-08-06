"""
OdorDiff-2: Safe Text-to-Scent Molecule Diffusion

A state-of-the-art AI system for generating novel scent molecules from text descriptions,
with integrated safety filters and synthesizability scoring.
"""

from .core.diffusion import OdorDiffusion
from .safety.filter import SafetyFilter
from .models.molecule import Molecule, OdorProfile
from .core.synthesis import SynthesisPlanner
from .visualization.viewer import MoleculeViewer

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