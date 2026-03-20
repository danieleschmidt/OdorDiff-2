"""Molecule representation for olfactory molecule generation."""

from __future__ import annotations

import numpy as np
from typing import List


class MoleculeRepresentation:
    """Represents a molecule using atomic numbers and bond adjacency matrix.
    
    Attributes:
        atoms: List of atomic numbers for each atom.
        bonds: Numpy adjacency matrix representing bonds.
    """

    def __init__(self, atoms: List[int], bonds: np.ndarray):
        """Initialize a molecule representation.
        
        Args:
            atoms: List of atomic numbers (e.g., [6, 6, 8] for C-C-O).
            bonds: Square adjacency matrix of shape (n_atoms, n_atoms).
        """
        self.atoms = atoms
        self.bonds = np.array(bonds, dtype=float)
        if self.bonds.shape != (len(atoms), len(atoms)):
            raise ValueError(
                f"bonds shape {self.bonds.shape} must match "
                f"({len(atoms)}, {len(atoms)})"
            )

    def to_fingerprint(self) -> np.ndarray:
        """Convert molecule to a 128-dimensional fingerprint vector.
        
        The fingerprint encodes atom counts, bond counts, ring information,
        branching, and atom-type histogram features.
        
        Returns:
            128-dimensional numpy array.
        """
        fp = np.zeros(128, dtype=float)

        # Features 0-9: atom-type histogram (normalized counts for Z=1..10)
        for z in self.atoms:
            idx = min(z - 1, 9)
            if 0 <= idx < 10:
                fp[idx] += 1.0
        if len(self.atoms) > 0:
            fp[0:10] /= max(len(self.atoms), 1)

        # Feature 10: num_atoms (normalized)
        fp[10] = min(self.num_atoms / 50.0, 1.0)

        # Feature 11: num_bonds (normalized)
        fp[11] = min(self.num_bonds / 100.0, 1.0)

        # Feature 12: num_rings (normalized)
        fp[12] = min(self.num_rings / 10.0, 1.0)

        # Feature 13: branching_factor
        fp[13] = self.branching_factor

        # Features 14-63: adjacency-based hash features
        n = len(self.atoms)
        if n > 0:
            for i in range(n):
                for j in range(i + 1, n):
                    if self.bonds[i, j] > 0:
                        z_i = self.atoms[i]
                        z_j = self.atoms[j]
                        idx = ((z_i * 31 + z_j) % 50) + 14
                        fp[idx] += self.bonds[i, j]

        # Normalize features 14-63
        max_val = fp[14:64].max()
        if max_val > 0:
            fp[14:64] /= max_val

        # Features 64-127: degree distribution and ring-closure features
        degrees = np.array([
            int(np.sum(self.bonds[i] > 0)) for i in range(n)
        ], dtype=float) if n > 0 else np.array([])

        for k, deg in enumerate(degrees[:32]):
            fp[64 + k] = min(deg / 4.0, 1.0)

        # Additional ring-encoding: flag atoms in rings
        in_ring = self._atoms_in_rings()
        for k, flag in enumerate(in_ring[:32]):
            fp[96 + k] = float(flag)

        return fp

    @property
    def num_atoms(self) -> int:
        """Number of atoms in the molecule."""
        return len(self.atoms)

    @property
    def num_bonds(self) -> int:
        """Number of bonds (upper triangle of adjacency matrix)."""
        return int(np.sum(self.bonds > 0) // 2)

    @property
    def num_rings(self) -> int:
        """Number of rings detected via simple cycle detection (DFS)."""
        return len(self._find_rings())

    @property
    def branching_factor(self) -> float:
        """Ratio of atoms with more than 2 bonds (branching atoms)."""
        n = len(self.atoms)
        if n == 0:
            return 0.0
        branching = sum(
            1 for i in range(n)
            if int(np.sum(self.bonds[i] > 0)) > 2
        )
        return branching / n

    def _find_rings(self) -> list:
        """Find fundamental cycles using DFS back-edge detection.
        
        Returns:
            List of cycles (each cycle is a list of atom indices).
        """
        n = len(self.atoms)
        if n == 0:
            return []

        visited = [False] * n
        in_stack = [False] * n
        parent = [-1] * n
        rings = []

        def dfs(v: int):
            visited[v] = True
            in_stack[v] = True
            neighbors = [u for u in range(n) if self.bonds[v, u] > 0]
            for u in neighbors:
                if not visited[u]:
                    parent[u] = v
                    dfs(u)
                elif in_stack[u] and u != parent[v]:
                    # Found a back edge: reconstruct cycle
                    cycle = [v]
                    cur = v
                    while cur != u:
                        cur = parent[cur]
                        if cur == -1:
                            break
                        cycle.append(cur)
                    rings.append(cycle)
            in_stack[v] = False

        for start in range(n):
            if not visited[start]:
                dfs(start)

        return rings

    def _atoms_in_rings(self) -> list:
        """Returns a boolean list indicating which atoms participate in rings."""
        n = len(self.atoms)
        in_ring = [False] * n
        for ring in self._find_rings():
            for atom in ring:
                if 0 <= atom < n:
                    in_ring[atom] = True
        return in_ring

    def __repr__(self) -> str:
        return (
            f"MoleculeRepresentation(atoms={self.atoms}, "
            f"num_bonds={self.num_bonds}, num_rings={self.num_rings})"
        )
