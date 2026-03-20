"""Diffusion sampler with classifier-free guidance for molecule generation."""

from __future__ import annotations

import numpy as np
from typing import Optional


class DiffusionSampler:
    """DDPM-style diffusion sampler with classifier-free guidance.
    
    Generates molecule latent vectors conditioned on text embeddings.
    Uses a simple linear schedule for the noise process and a learned
    denoising function approximated by linear projection.
    
    Classifier-free guidance mixes conditional and unconditional predictions
    to steer generation toward the desired olfactory descriptor.
    """

    LATENT_DIM = 128

    def __init__(self, seed: int = 0):
        """Initialize the diffusion sampler.
        
        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed=seed)

        # Condition projection: 64-dim text embed -> 128-dim latent
        self._cond_proj = self.rng.standard_normal((self.LATENT_DIM, 64)) * 0.1

        # Unconditional embedding (null condition for CFG)
        self._null_embed = np.zeros(64, dtype=float)

        # Simple denoising weight matrix
        self._denoise_W = self.rng.standard_normal(
            (self.LATENT_DIM, self.LATENT_DIM)
        ) * 0.01
        # Ensure it's close to identity (stable denoising)
        self._denoise_W += np.eye(self.LATENT_DIM) * 0.9

    def _noise_schedule(self, n_steps: int) -> np.ndarray:
        """Linear noise schedule returning beta values.
        
        Args:
            n_steps: Number of diffusion steps.
            
        Returns:
            Array of beta values of shape (n_steps,).
        """
        beta_start = 1e-4
        beta_end = 0.02
        return np.linspace(beta_start, beta_end, n_steps)

    def _denoise_step(
        self,
        x: np.ndarray,
        t: int,
        n_steps: int,
        condition: np.ndarray,
        guidance_scale: float,
    ) -> np.ndarray:
        """Single denoising step with classifier-free guidance.
        
        Args:
            x: Current noisy latent of shape (latent_dim,).
            t: Current timestep (counting down from n_steps to 0).
            n_steps: Total number of steps.
            condition: Conditioning embedding of shape (64,).
            guidance_scale: CFG guidance scale (1.0 = no guidance, >1 = stronger).
            
        Returns:
            Denoised latent of shape (latent_dim,).
        """
        betas = self._noise_schedule(n_steps)
        beta_t = betas[t]
        alpha_t = 1.0 - beta_t

        # Conditional prediction
        cond_signal = self._cond_proj @ condition
        eps_cond = self._denoise_W @ x - cond_signal * beta_t

        # Unconditional prediction (null condition)
        null_signal = self._cond_proj @ self._null_embed
        eps_uncond = self._denoise_W @ x - null_signal * beta_t

        # Classifier-free guidance: mix conditional and unconditional
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        # DDPM reverse step
        x_denoised = (x - (1 - alpha_t) ** 0.5 * eps) / (alpha_t ** 0.5 + 1e-8)

        # Add a small amount of noise for t > 0
        if t > 0:
            noise = self.rng.standard_normal(self.LATENT_DIM)
            x_denoised += (beta_t ** 0.5) * noise * 0.1

        return x_denoised

    def sample(
        self,
        condition: np.ndarray,
        guidance_scale: float = 1.5,
        n_steps: int = 50,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Sample a molecule latent vector from the diffusion model.
        
        Args:
            condition: Conditioning embedding of shape (64,).
            guidance_scale: CFG guidance scale. 1.0 = no guidance, higher = stronger.
            n_steps: Number of denoising steps.
            seed: Optional per-sample seed override.
            
        Returns:
            Sampled latent vector of shape (latent_dim,).
        """
        rng = np.random.default_rng(seed=seed) if seed is not None else self.rng

        # Start from pure noise
        x = rng.standard_normal(self.LATENT_DIM)

        # Reverse diffusion (denoising loop)
        for t in range(n_steps - 1, -1, -1):
            x = self._denoise_step(x, t, n_steps, condition, guidance_scale)

        return x

    def latent_to_molecule_params(self, latent: np.ndarray) -> dict:
        """Convert a latent vector to molecule construction parameters.
        
        Args:
            latent: Latent vector of shape (latent_dim,).
            
        Returns:
            Dict with 'n_atoms', 'atom_types', 'bond_density'.
        """
        # Number of atoms: sigmoid scaling from latent
        n_atoms = int(5 + 25 * (1 / (1 + np.exp(-latent[0]))))

        # Atom types: use latent values to pick from common organic atoms
        # C=6, H=1, O=8, N=7, S=16
        organic_atoms = [6, 6, 6, 6, 6, 6, 1, 1, 8, 7, 16]
        atom_types = []
        for i in range(n_atoms):
            idx = int(abs(latent[i % len(latent)] * 100)) % len(organic_atoms)
            atom_types.append(organic_atoms[idx])

        # Bond density: how connected the molecule is
        bond_density = float(0.2 + 0.4 * (1 / (1 + np.exp(-latent[1]))))

        return {
            "n_atoms": n_atoms,
            "atom_types": atom_types,
            "bond_density": bond_density,
        }

    def __repr__(self) -> str:
        return (
            f"DiffusionSampler(latent_dim={self.LATENT_DIM})"
        )
