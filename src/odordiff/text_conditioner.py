"""Text conditioning for olfactory descriptor embeddings."""

from __future__ import annotations

import numpy as np
from typing import Dict

# Descriptor embeddings: each descriptor maps to a 64-dim vector.
# Generated as deterministic pseudo-random vectors seeded by descriptor index.
_DESCRIPTOR_NAMES = [
    "fresh", "citrus", "floral", "woody", "musky",
    "sweet", "spicy", "earthy", "marine", "green",
]

def _make_descriptor_embeddings() -> Dict[str, np.ndarray]:
    """Create deterministic 64-dim embeddings for each descriptor."""
    embeddings = {}
    for i, name in enumerate(_DESCRIPTOR_NAMES):
        rng = np.random.default_rng(seed=42 + i)
        vec = rng.standard_normal(64)
        # Normalize to unit sphere
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        embeddings[name] = vec
    return embeddings


DESCRIPTOR_EMBEDDINGS: Dict[str, np.ndarray] = _make_descriptor_embeddings()


class TextConditioner:
    """Encodes olfactory text descriptions into conditioning vectors.
    
    Supports encoding free-text smell descriptions into 64-dimensional
    vectors by detecting known olfactory descriptors and computing
    weighted combinations of their embeddings.
    """

    def __init__(self):
        self.embeddings = DESCRIPTOR_EMBEDDINGS

    def _detect_descriptors(self, text: str) -> Dict[str, float]:
        """Detect olfactory descriptors in text and return confidence scores.
        
        Args:
            text: Input text description.
            
        Returns:
            Dict mapping descriptor names to confidence scores (0-1).
        """
        text_lower = text.lower()
        scores: Dict[str, float] = {}

        for descriptor in _DESCRIPTOR_NAMES:
            if descriptor in text_lower:
                # Base confidence for exact match
                scores[descriptor] = 1.0
            else:
                # Partial/semantic match via keyword synonyms
                synonyms = {
                    "fresh": ["clean", "cool", "airy", "crisp", "light"],
                    "citrus": ["lemon", "orange", "lime", "grapefruit", "bergamot"],
                    "floral": ["rose", "jasmine", "lavender", "flower", "bloom", "petal"],
                    "woody": ["wood", "cedar", "sandalwood", "pine", "bark", "timber"],
                    "musky": ["musk", "animalic", "sensual", "warm", "amber"],
                    "sweet": ["vanilla", "honey", "sugar", "caramel", "candy"],
                    "spicy": ["pepper", "cinnamon", "clove", "ginger", "spice", "hot"],
                    "earthy": ["earth", "soil", "moss", "mushroom", "petrichor", "damp"],
                    "marine": ["ocean", "sea", "aquatic", "water", "coastal", "salt"],
                    "green": ["grass", "leaf", "herb", "cut", "vegetal", "plant"],
                }
                for syn in synonyms.get(descriptor, []):
                    if syn in text_lower:
                        scores[descriptor] = max(scores.get(descriptor, 0.0), 0.7)

        return scores

    def encode(self, text: str) -> np.ndarray:
        """Encode text description into a 64-dimensional conditioning vector.
        
        Args:
            text: Olfactory text description (e.g., "fresh citrus").
            
        Returns:
            64-dimensional numpy array.
        """
        scores = self._detect_descriptors(text)

        if not scores:
            # Return zero vector if no descriptors detected
            return np.zeros(64, dtype=float)

        # Weighted sum of descriptor embeddings
        result = np.zeros(64, dtype=float)
        total_weight = 0.0
        for descriptor, confidence in scores.items():
            result += confidence * self.embeddings[descriptor]
            total_weight += confidence

        if total_weight > 0:
            result /= total_weight

        # Normalize to unit sphere
        norm = np.linalg.norm(result)
        if norm > 1e-8:
            result /= norm

        return result

    def multi_label_encode(self, text: str) -> Dict[str, float]:
        """Encode text and return per-descriptor confidence scores.
        
        Args:
            text: Olfactory text description.
            
        Returns:
            Dict mapping descriptor names to confidence scores (0.0-1.0).
            All descriptors are returned, with 0.0 for undetected ones.
        """
        scores = self._detect_descriptors(text)
        # Return all descriptors with 0.0 for undetected
        return {d: scores.get(d, 0.0) for d in _DESCRIPTOR_NAMES}

    def __repr__(self) -> str:
        return f"TextConditioner(descriptors={_DESCRIPTOR_NAMES})"
