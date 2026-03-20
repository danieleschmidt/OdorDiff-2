"""Novelty checking for generated molecules using Tanimoto similarity."""

from __future__ import annotations

import numpy as np
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .molecule import MoleculeRepresentation


def _make_reference_fingerprint(seed: int) -> np.ndarray:
    """Generate a deterministic binary reference fingerprint."""
    rng = np.random.default_rng(seed=seed)
    fp = (rng.random(128) > 0.5).astype(float)
    return fp


# Hardcoded reference set: 20 reference molecule fingerprints
REFERENCE_SET: List[np.ndarray] = [
    _make_reference_fingerprint(seed=100 + i) for i in range(20)
]


class NoveltyChecker:
    """Checks novelty of generated molecules against a reference set.
    
    Uses Tanimoto (Jaccard) similarity on binarized fingerprints to
    determine if a molecule is novel compared to known molecules.
    """

    def __init__(self, reference_set: List[np.ndarray] = None):
        """Initialize with reference fingerprints.
        
        Args:
            reference_set: List of reference fingerprint arrays. 
                           Defaults to the built-in REFERENCE_SET of 20 molecules.
        """
        self.reference_set = reference_set if reference_set is not None else REFERENCE_SET

    @staticmethod
    def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Compute Tanimoto (Jaccard) similarity between two binary fingerprints.
        
        Tanimoto = |intersection| / |union|
        
        Args:
            fp1: First fingerprint array (binarized).
            fp2: Second fingerprint array (binarized).
            
        Returns:
            Similarity score in [0, 1].
        """
        # Binarize
        b1 = (fp1 > 0).astype(float)
        b2 = (fp2 > 0).astype(float)

        intersection = float(np.sum(b1 * b2))
        union = float(np.sum((b1 + b2) > 0))

        if union == 0:
            return 0.0
        return intersection / union

    def max_similarity(self, molecule: "MoleculeRepresentation") -> float:
        """Compute the maximum Tanimoto similarity to any reference molecule.
        
        Args:
            molecule: MoleculeRepresentation to check.
            
        Returns:
            Maximum Tanimoto similarity score in [0, 1].
        """
        fp = molecule.to_fingerprint()
        if len(self.reference_set) == 0:
            return 0.0
        similarities = [
            self.tanimoto_similarity(fp, ref_fp)
            for ref_fp in self.reference_set
        ]
        return float(max(similarities))

    def is_novel(
        self,
        molecule: "MoleculeRepresentation",
        threshold: float = 0.6,
    ) -> bool:
        """Check if a molecule is novel compared to the reference set.
        
        A molecule is considered novel if its maximum Tanimoto similarity
        to any reference molecule is below the threshold.
        
        Args:
            molecule: MoleculeRepresentation to check.
            threshold: Maximum similarity allowed (exclusive). Default 0.6.
            
        Returns:
            True if the molecule is novel (max similarity < threshold).
        """
        return self.max_similarity(molecule) < threshold

    def __repr__(self) -> str:
        return (
            f"NoveltyChecker(n_references={len(self.reference_set)})"
        )
