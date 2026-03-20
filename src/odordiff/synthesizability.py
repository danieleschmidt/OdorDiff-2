"""Synthesizability scoring for generated molecules."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .molecule import MoleculeRepresentation

# Penalty constants
RING_PENALTY = 0.15   # per ring
BRANCH_PENALTY = 0.10  # per branching point (atom with >2 bonds)


class SynthesizabilityScorer:
    """Proxy SA (Synthetic Accessibility) score for molecules.
    
    Scores molecules on a 0-1 scale where higher scores indicate
    greater synthesizability. Penalizes rings, branching, and
    high atom count complexity.
    """

    def score(self, molecule: "MoleculeRepresentation") -> float:
        """Score synthesizability of a molecule.
        
        Args:
            molecule: MoleculeRepresentation to score.
            
        Returns:
            Float in [0, 1] where 1.0 = highly synthesizable.
        """
        n_atoms = molecule.num_atoms
        n_rings = molecule.num_rings
        n_branches = self._count_branching_points(molecule)

        # Ring penalty
        ring_penalty = RING_PENALTY * n_rings

        # Branching penalty
        branch_penalty = BRANCH_PENALTY * n_branches

        # Complexity penalty based on atom count
        # Molecules with fewer atoms are generally easier to synthesize
        complexity_penalty = self._complexity_penalty(n_atoms)

        # Start from 1.0 and subtract penalties
        sa_score = 1.0 - ring_penalty - branch_penalty - complexity_penalty

        # Clamp to [0, 1]
        return float(np.clip(sa_score, 0.0, 1.0))

    def _count_branching_points(self, molecule: "MoleculeRepresentation") -> int:
        """Count atoms with more than 2 bonds (branching points).
        
        Args:
            molecule: MoleculeRepresentation to analyze.
            
        Returns:
            Number of branching atoms.
        """
        n = molecule.num_atoms
        count = 0
        for i in range(n):
            degree = int(np.sum(molecule.bonds[i] > 0))
            if degree > 2:
                count += 1
        return count

    def _complexity_penalty(self, n_atoms: int) -> float:
        """Compute complexity penalty based on atom count.
        
        Args:
            n_atoms: Number of atoms.
            
        Returns:
            Complexity penalty in [0, ~0.5].
        """
        # Penalty grows logarithmically with atom count
        # 1-5 atoms: minimal penalty (~0.0)
        # 10 atoms: ~0.1
        # 30 atoms: ~0.2
        # 50 atoms: ~0.25
        if n_atoms <= 1:
            return 0.0
        return min(0.1 * np.log(n_atoms) / np.log(10), 0.5)

    def __repr__(self) -> str:
        return (
            f"SynthesizabilityScorer("
            f"ring_penalty={RING_PENALTY}, "
            f"branch_penalty={BRANCH_PENALTY})"
        )
