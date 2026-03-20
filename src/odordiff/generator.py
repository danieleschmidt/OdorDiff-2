"""High-level OdorDiff generator combining all components."""

from __future__ import annotations

import numpy as np
from typing import List, Optional

from .molecule import MoleculeRepresentation
from .text_conditioner import TextConditioner
from .diffusion import DiffusionSampler
from .synthesizability import SynthesizabilityScorer
from .novelty import NoveltyChecker


class OdorDiffGenerator:
    """End-to-end olfactory molecule generator.
    
    Combines text conditioning, diffusion sampling, synthesizability
    scoring, and novelty checking to generate novel, synthesizable
    molecules matching a given olfactory description.
    """

    def __init__(self, seed: int = 42):
        """Initialize the generator with all components.
        
        Args:
            seed: Random seed for reproducible generation.
        """
        self.seed = seed
        self.conditioner = TextConditioner()
        self.sampler = DiffusionSampler(seed=seed)
        self.scorer = SynthesizabilityScorer()
        self.novelty_checker = NoveltyChecker()
        self._rng = np.random.default_rng(seed=seed)

    def _latent_to_molecule(
        self,
        latent: np.ndarray,
        molecule_seed: int,
    ) -> MoleculeRepresentation:
        """Convert a latent vector to a MoleculeRepresentation.
        
        Args:
            latent: Latent vector from the diffusion sampler.
            molecule_seed: Seed for deterministic construction.
            
        Returns:
            A MoleculeRepresentation.
        """
        params = self.sampler.latent_to_molecule_params(latent)
        n_atoms = params["n_atoms"]
        atom_types = params["atom_types"]
        bond_density = params["bond_density"]

        # Build adjacency matrix
        rng = np.random.default_rng(seed=molecule_seed)
        bonds = np.zeros((n_atoms, n_atoms), dtype=float)

        # First ensure the molecule is connected (chain backbone)
        for i in range(n_atoms - 1):
            bonds[i, i + 1] = 1.0
            bonds[i + 1, i] = 1.0

        # Add a few extra bonds based on bond_density (kept sparse to preserve synthesizability)
        for i in range(n_atoms):
            for j in range(i + 2, n_atoms):
                if rng.random() < bond_density * 0.04:
                    bonds[i, j] = 1.0
                    bonds[j, i] = 1.0

        return MoleculeRepresentation(atoms=atom_types, bonds=bonds)

    def generate(
        self,
        text: str,
        n_molecules: int = 5,
        min_synthesizability: float = 0.5,
        require_novel: bool = True,
        max_attempts: int = 100,
    ) -> List[MoleculeRepresentation]:
        """Generate molecules matching the given olfactory description.
        
        Args:
            text: Olfactory text description (e.g., "fresh citrus").
            n_molecules: Number of molecules to generate.
            min_synthesizability: Minimum SA score (0-1) to accept.
            require_novel: If True, only return novel molecules.
            max_attempts: Maximum sampling attempts before giving up.
            
        Returns:
            List of MoleculeRepresentation objects passing all filters.
        """
        # Encode text condition
        condition = self.conditioner.encode(text)

        results: List[MoleculeRepresentation] = []
        attempt = 0

        while len(results) < n_molecules and attempt < max_attempts:
            # Sample from diffusion model
            sample_seed = int(self._rng.integers(0, 2**31))
            latent = self.sampler.sample(
                condition=condition,
                guidance_scale=1.5,
                n_steps=50,
                seed=sample_seed,
            )

            # Convert latent to molecule
            molecule = self._latent_to_molecule(latent, molecule_seed=sample_seed)

            # Filter by synthesizability
            sa_score = self.scorer.score(molecule)
            if sa_score < min_synthesizability:
                attempt += 1
                continue

            # Filter by novelty
            if require_novel and not self.novelty_checker.is_novel(molecule):
                attempt += 1
                continue

            results.append(molecule)
            attempt += 1

        return results

    def __repr__(self) -> str:
        return f"OdorDiffGenerator(seed={self.seed})"
