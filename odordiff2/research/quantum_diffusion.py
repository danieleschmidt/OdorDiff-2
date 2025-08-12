"""
Quantum-informed molecular diffusion model incorporating vibrational theory.

This module implements a novel approach to molecular generation that incorporates
quantum mechanical vibrational theory to better predict and generate molecules
with specific olfactory properties.

References:
- Turin, L. (1996). A spectroscopic mechanism for primary olfactory reception.
- Franco, M.I. et al. (2011). Molecular vibration-sensing component in Drosophila melanogaster olfaction.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import math

from ..core.diffusion import OdorDiffusion, MolecularDecoder
from ..models.molecule import Molecule
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VibrationalSignature:
    """Represents the vibrational signature of a molecule."""
    frequencies: np.ndarray  # Vibrational frequencies (cm^-1)
    intensities: np.ndarray  # IR intensities
    raman_activities: np.ndarray  # Raman activities
    olfactory_relevance: float  # Relevance to olfaction [0-1]


class QuantumVibrationalEncoder(nn.Module):
    """
    Encodes molecular vibrational signatures into latent space.
    
    Uses quantum harmonic oscillator principles to encode vibrational
    modes that are relevant to olfactory perception.
    """
    
    def __init__(self, max_frequencies: int = 100, hidden_dim: int = 256):
        super().__init__()
        self.max_frequencies = max_frequencies
        self.hidden_dim = hidden_dim
        
        # Frequency embedding using quantum harmonic oscillator basis
        self.frequency_embedding = nn.Linear(max_frequencies, hidden_dim)
        self.intensity_projection = nn.Linear(max_frequencies, hidden_dim)
        
        # Attention mechanism for olfactory-relevant frequencies
        self.olfactory_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, vibrational_signatures: List[VibrationalSignature]) -> torch.Tensor:
        """Encode vibrational signatures to latent vectors."""
        batch_size = len(vibrational_signatures)
        device = next(self.parameters()).device
        
        # Prepare frequency and intensity tensors
        freq_tensors = []
        intensity_tensors = []
        
        for signature in vibrational_signatures:
            # Pad or truncate to max_frequencies
            freqs = signature.frequencies[:self.max_frequencies]
            intens = signature.intensities[:self.max_frequencies]
            
            if len(freqs) < self.max_frequencies:
                freqs = np.pad(freqs, (0, self.max_frequencies - len(freqs)))
                intens = np.pad(intens, (0, self.max_frequencies - len(intens)))
            
            freq_tensors.append(torch.tensor(freqs, dtype=torch.float32))
            intensity_tensors.append(torch.tensor(intens, dtype=torch.float32))
        
        frequencies = torch.stack(freq_tensors).to(device)  # (B, max_freq)
        intensities = torch.stack(intensity_tensors).to(device)  # (B, max_freq)
        
        # Apply quantum harmonic oscillator transformation
        freq_encoded = self._quantum_harmonic_transform(frequencies)
        freq_embedded = self.frequency_embedding(freq_encoded)  # (B, hidden_dim)
        
        intensity_embedded = self.intensity_projection(intensities)  # (B, hidden_dim)
        
        # Combine frequency and intensity information
        combined = freq_embedded + intensity_embedded  # (B, hidden_dim)
        combined = combined.unsqueeze(1)  # (B, 1, hidden_dim)
        
        # Apply attention for olfactory relevance
        attended, _ = self.olfactory_attention(combined, combined, combined)
        attended = attended.squeeze(1)  # (B, hidden_dim)
        
        # Final projection
        output = self.output_proj(attended)
        
        return output
    
    def _quantum_harmonic_transform(self, frequencies: torch.Tensor) -> torch.Tensor:
        """
        Apply quantum harmonic oscillator transformation to frequencies.
        
        Uses the energy levels E_n = ħω(n + 1/2) to transform raw frequencies
        into a quantum-mechanically informed representation.
        """
        # Convert cm^-1 to energy units (simplified)
        hbar_c = 1.986e-23  # Reduced Planck constant * speed of light (J⋅cm)
        
        # Quantum energy levels for each frequency
        quantum_energies = hbar_c * frequencies  # Simplified energy conversion
        
        # Apply harmonic oscillator wavefunctions (Hermite polynomials)
        # Using first few energy levels as features
        n_levels = 5
        quantum_features = []
        
        for n in range(n_levels):
            # Simplified Hermite polynomial features
            level_features = torch.exp(-quantum_energies / 2) * torch.pow(quantum_energies, n)
            quantum_features.append(level_features)
        
        # Pad or truncate to match max_frequencies
        quantum_tensor = torch.stack(quantum_features, dim=-1)  # (B, max_freq, n_levels)
        quantum_flat = quantum_tensor.reshape(quantum_tensor.shape[0], -1)  # (B, max_freq * n_levels)
        
        # Project to max_frequencies dimension
        if quantum_flat.shape[1] > self.max_frequencies:
            quantum_flat = quantum_flat[:, :self.max_frequencies]
        elif quantum_flat.shape[1] < self.max_frequencies:
            padding = torch.zeros(quantum_flat.shape[0], 
                                self.max_frequencies - quantum_flat.shape[1], 
                                device=quantum_flat.device)
            quantum_flat = torch.cat([quantum_flat, padding], dim=1)
        
        return quantum_flat


class QuantumInformedDecoder(MolecularDecoder):
    """
    Molecular decoder that incorporates quantum vibrational information.
    """
    
    def __init__(self, latent_dim: int = 512, vocab_size: int = 100, quantum_dim: int = 256):
        super().__init__(latent_dim, vocab_size)
        self.quantum_dim = quantum_dim
        
        # Quantum vibrational guidance
        self.vibrational_guidance = nn.Sequential(
            nn.Linear(quantum_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.Tanh()
        )
        
        # Modified decoder with quantum conditioning
        self.quantum_conditioned_decoder = nn.LSTM(
            512 + quantum_dim, 256, num_layers=2, batch_first=True
        )
        
    def forward_with_quantum(
        self, 
        latent: torch.Tensor, 
        quantum_signature: torch.Tensor,
        max_length: int = 100
    ) -> torch.Tensor:
        """Decode with quantum vibrational guidance."""
        batch_size = latent.size(0)
        device = latent.device
        
        # Apply quantum guidance to latent representation
        quantum_guidance = self.vibrational_guidance(quantum_signature)
        guided_latent = latent + quantum_guidance
        
        # Initialize sequence
        sequence = torch.full((batch_size, 1), self.vocab['START'], device=device)
        outputs = []
        
        # Project latent with quantum conditioning
        latent_hidden = self.latent_proj(guided_latent).unsqueeze(1)
        quantum_hidden = quantum_signature.unsqueeze(1)
        
        for _ in range(max_length):
            # Embed current sequence
            embedded = self.embedding(sequence)
            
            # Concatenate with quantum-conditioned latent
            conditioned = torch.cat([
                embedded, 
                latent_hidden.expand(-1, embedded.size(1), -1),
                quantum_hidden.expand(-1, embedded.size(1), -1)
            ], dim=-1)
            
            # Decode with quantum conditioning
            output, _ = self.quantum_conditioned_decoder(conditioned)
            logits = self.output_proj(output[:, -1:, :])
            
            # Sample next token
            next_token = torch.multinomial(F.softmax(logits.squeeze(1), dim=-1), 1)
            sequence = torch.cat([sequence, next_token], dim=1)
            outputs.append(logits.squeeze(1))
            
            if (next_token == self.vocab['END']).all():
                break
        
        return torch.stack(outputs, dim=1)


class QuantumInformedDiffusion(OdorDiffusion):
    """
    Advanced diffusion model incorporating quantum vibrational theory for 
    more accurate olfactory property prediction and molecule generation.
    """
    
    def __init__(self, device: str = "cpu", enable_quantum: bool = True):
        super().__init__(device)
        self.enable_quantum = enable_quantum
        
        if enable_quantum:
            # Replace decoder with quantum-informed version
            self.vibrational_encoder = QuantumVibrationalEncoder()
            self.quantum_decoder = QuantumInformedDecoder()
            
            # Move to device
            self.vibrational_encoder.to(device)
            self.quantum_decoder.to(device)
            
            # Vibrational database for known molecules
            self._vibrational_database = self._init_vibrational_database()
            
        logger.info(f"QuantumInformedDiffusion initialized (quantum_enabled={enable_quantum})")
    
    def _init_vibrational_database(self) -> Dict[str, VibrationalSignature]:
        """Initialize database of known molecular vibrational signatures."""
        return {
            # Linalool (floral, lavender)
            'CC(C)=CCO': VibrationalSignature(
                frequencies=np.array([3340, 2970, 2920, 1640, 1450, 1380, 1100, 890]),
                intensities=np.array([0.8, 0.9, 0.7, 0.6, 0.5, 0.4, 0.7, 0.3]),
                raman_activities=np.array([0.3, 0.8, 0.6, 0.9, 0.4, 0.2, 0.5, 0.1]),
                olfactory_relevance=0.95
            ),
            # Limonene (citrus)
            'CC1=CCC(CC1)C(C)C': VibrationalSignature(
                frequencies=np.array([2970, 2920, 1640, 1450, 1380, 890, 800]),
                intensities=np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]),
                raman_activities=np.array([0.7, 0.6, 0.8, 0.5, 0.3, 0.2, 0.1]),
                olfactory_relevance=0.90
            ),
            # Benzaldehyde (almond)
            'c1ccc(cc1)C=O': VibrationalSignature(
                frequencies=np.array([3050, 1700, 1600, 1580, 1450, 1200, 750]),
                intensities=np.array([0.6, 1.0, 0.8, 0.7, 0.5, 0.6, 0.4]),
                raman_activities=np.array([0.4, 0.3, 0.9, 0.8, 0.4, 0.2, 0.6]),
                olfactory_relevance=0.85
            ),
        }
    
    def generate_with_quantum(
        self,
        prompt: str,
        target_vibrational_profile: Optional[VibrationalSignature] = None,
        num_molecules: int = 5,
        quantum_weight: float = 0.3,
        **kwargs
    ) -> List[Molecule]:
        """
        Generate molecules using quantum-informed diffusion.
        
        Args:
            prompt: Text description of desired scent
            target_vibrational_profile: Target vibrational signature
            num_molecules: Number of molecules to generate
            quantum_weight: Weight of quantum guidance [0-1]
            
        Returns:
            List of quantum-optimized molecules
        """
        if not self.enable_quantum:
            logger.warning("Quantum features disabled, falling back to standard generation")
            return self.generate(prompt, num_molecules, **kwargs)
        
        logger.info(f"Generating {num_molecules} molecules with quantum guidance")
        
        # If no target profile provided, predict from prompt
        if target_vibrational_profile is None:
            target_vibrational_profile = self._predict_target_vibrational_profile(prompt)
        
        # Generate base molecules
        base_molecules = self._template_based_generation(prompt, num_molecules * 2)
        
        # Score molecules using quantum-informed metrics
        quantum_scored_molecules = []
        for molecule in base_molecules:
            try:
                # Predict vibrational signature
                predicted_signature = self._predict_vibrational_signature(molecule)
                
                # Calculate quantum similarity to target
                quantum_similarity = self._calculate_vibrational_similarity(
                    predicted_signature, target_vibrational_profile
                )
                
                # Update molecule with quantum information
                molecule.quantum_similarity = quantum_similarity
                molecule.vibrational_signature = predicted_signature
                molecule.confidence = (
                    molecule.confidence * (1 - quantum_weight) + 
                    quantum_similarity * quantum_weight
                )
                
                quantum_scored_molecules.append(molecule)
                
            except Exception as e:
                logger.warning(f"Failed quantum scoring for molecule {molecule.smiles}: {e}")
                quantum_scored_molecules.append(molecule)
        
        # Sort by quantum-enhanced confidence
        quantum_scored_molecules.sort(key=lambda m: m.confidence, reverse=True)
        
        # Apply enhanced odor prediction using quantum information
        for molecule in quantum_scored_molecules[:num_molecules]:
            molecule.odor_profile = self._predict_quantum_enhanced_odor(molecule, prompt)
        
        logger.info("Quantum-informed generation completed")
        return quantum_scored_molecules[:num_molecules]
    
    def _predict_target_vibrational_profile(self, prompt: str) -> VibrationalSignature:
        """Predict target vibrational profile from text prompt."""
        prompt_lower = prompt.lower()
        
        # Map common scent descriptors to vibrational characteristics
        if any(word in prompt_lower for word in ['floral', 'flower', 'rose', 'lavender']):
            # Floral molecules typically have C=C stretches, O-H stretches
            frequencies = np.array([3300, 1650, 1450, 1100, 800])
            intensities = np.array([0.8, 0.7, 0.5, 0.6, 0.3])
            olfactory_relevance = 0.9
            
        elif any(word in prompt_lower for word in ['citrus', 'lemon', 'fresh']):
            # Citrus molecules have characteristic terpene vibrations
            frequencies = np.array([2970, 1640, 890, 800])
            intensities = np.array([0.9, 0.7, 0.4, 0.3])
            olfactory_relevance = 0.85
            
        elif any(word in prompt_lower for word in ['sweet', 'vanilla', 'almond']):
            # Sweet molecules often have C=O stretches
            frequencies = np.array([3050, 1700, 1600, 1200])
            intensities = np.array([0.6, 1.0, 0.8, 0.6])
            olfactory_relevance = 0.8
            
        else:
            # Default profile
            frequencies = np.array([3000, 1500, 1000])
            intensities = np.array([0.7, 0.6, 0.4])
            olfactory_relevance = 0.7
        
        # Generate corresponding Raman activities (simplified)
        raman_activities = intensities * np.random.uniform(0.3, 0.8, len(intensities))
        
        return VibrationalSignature(
            frequencies=frequencies,
            intensities=intensities,
            raman_activities=raman_activities,
            olfactory_relevance=olfactory_relevance
        )
    
    def _predict_vibrational_signature(self, molecule: Molecule) -> VibrationalSignature:
        """Predict vibrational signature for a molecule."""
        if not molecule.is_valid:
            return VibrationalSignature(
                frequencies=np.array([]),
                intensities=np.array([]),
                raman_activities=np.array([]),
                olfactory_relevance=0.0
            )
        
        # Check if we have experimental data
        if molecule.smiles in self._vibrational_database:
            return self._vibrational_database[molecule.smiles]
        
        # Predict based on functional groups (simplified)
        frequencies = []
        intensities = []
        
        smiles = molecule.smiles.upper()
        
        # Functional group analysis
        if 'O' in smiles and 'H' in smiles:  # Likely O-H stretch
            frequencies.extend([3200, 3400])
            intensities.extend([0.8, 0.6])
        
        if 'C=C' in smiles or '=' in smiles:  # C=C stretch
            frequencies.extend([1640, 1620])
            intensities.extend([0.7, 0.5])
        
        if 'C=O' in smiles:  # C=O stretch
            frequencies.extend([1700, 1720])
            intensities.extend([1.0, 0.9])
        
        if 'C1' in smiles or 'c1' in smiles:  # Aromatic C=C
            frequencies.extend([1600, 1580])
            intensities.extend([0.8, 0.6])
        
        # Add alkyl C-H stretches
        frequencies.extend([2970, 2920, 2850])
        intensities.extend([0.9, 0.8, 0.7])
        
        # Convert to arrays
        frequencies = np.array(frequencies)
        intensities = np.array(intensities)
        
        # Generate Raman activities
        raman_activities = intensities * np.random.uniform(0.2, 0.7, len(intensities))
        
        # Estimate olfactory relevance based on molecular properties
        mw = molecule.get_property('molecular_weight') or 150
        olfactory_relevance = min(1.0, max(0.3, (300 - mw) / 200))  # Prefer MW 100-300
        
        return VibrationalSignature(
            frequencies=frequencies,
            intensities=intensities,
            raman_activities=raman_activities,
            olfactory_relevance=olfactory_relevance
        )
    
    def _calculate_vibrational_similarity(
        self, 
        signature1: VibrationalSignature, 
        signature2: VibrationalSignature
    ) -> float:
        """Calculate similarity between two vibrational signatures."""
        if len(signature1.frequencies) == 0 or len(signature2.frequencies) == 0:
            return 0.0
        
        # Create frequency-intensity histograms
        freq_range = (500, 4000)  # Typical IR range
        bins = 50
        
        hist1 = self._create_vibrational_histogram(signature1, freq_range, bins)
        hist2 = self._create_vibrational_histogram(signature2, freq_range, bins)
        
        # Calculate normalized cross-correlation
        correlation = np.corrcoef(hist1, hist2)[0, 1]
        
        # Handle NaN case
        if np.isnan(correlation):
            correlation = 0.0
        
        # Weight by olfactory relevance
        relevance_weight = (signature1.olfactory_relevance + signature2.olfactory_relevance) / 2
        
        similarity = abs(correlation) * relevance_weight
        
        return min(1.0, max(0.0, similarity))
    
    def _create_vibrational_histogram(
        self, 
        signature: VibrationalSignature, 
        freq_range: Tuple[float, float], 
        bins: int
    ) -> np.ndarray:
        """Create histogram representation of vibrational signature."""
        freq_min, freq_max = freq_range
        histogram = np.zeros(bins)
        
        for freq, intensity in zip(signature.frequencies, signature.intensities):
            if freq_min <= freq <= freq_max:
                bin_idx = int((freq - freq_min) / (freq_max - freq_min) * (bins - 1))
                histogram[bin_idx] += intensity
        
        # Normalize
        if histogram.sum() > 0:
            histogram = histogram / histogram.sum()
        
        return histogram
    
    def _predict_quantum_enhanced_odor(self, molecule: Molecule, prompt: str):
        """Predict odor using quantum vibrational information."""
        # Start with base prediction
        base_profile = self._predict_odor(molecule, prompt)
        
        # Enhance with quantum information
        if hasattr(molecule, 'vibrational_signature'):
            signature = molecule.vibrational_signature
            
            # Analyze dominant vibrational modes
            if len(signature.frequencies) > 0:
                dominant_freq = signature.frequencies[np.argmax(signature.intensities)]
                
                # Map frequency ranges to olfactory characteristics
                if 1600 <= dominant_freq <= 1700:  # C=O region
                    base_profile.primary_notes.append('sweet')
                    base_profile.character += ', aldehyde-like'
                elif 1640 <= dominant_freq <= 1680:  # C=C region
                    base_profile.primary_notes.append('fresh')
                    base_profile.character += ', green'
                elif 3200 <= dominant_freq <= 3400:  # O-H region
                    base_profile.primary_notes.append('clean')
                    base_profile.character += ', alcoholic'
                
                # Adjust intensity based on vibrational relevance
                base_profile.intensity *= signature.olfactory_relevance
        
        return base_profile
    
    def benchmark_quantum_accuracy(
        self, 
        test_molecules: List[Tuple[str, str, VibrationalSignature]]
    ) -> Dict[str, float]:
        """
        Benchmark quantum-informed predictions against known data.
        
        Args:
            test_molecules: List of (SMILES, odor_description, vibrational_signature)
            
        Returns:
            Benchmark metrics
        """
        logger.info(f"Benchmarking quantum accuracy on {len(test_molecules)} molecules")
        
        predictions = []
        actuals = []
        vibrational_similarities = []
        
        for smiles, odor_desc, actual_signature in test_molecules:
            try:
                # Create molecule
                molecule = Molecule(smiles)
                
                # Predict vibrational signature
                predicted_signature = self._predict_vibrational_signature(molecule)
                
                # Calculate similarity
                similarity = self._calculate_vibrational_similarity(
                    predicted_signature, actual_signature
                )
                vibrational_similarities.append(similarity)
                
                # Generate with quantum guidance
                quantum_molecules = self.generate_with_quantum(
                    odor_desc, actual_signature, num_molecules=1
                )
                
                if quantum_molecules:
                    predictions.append(quantum_molecules[0].smiles)
                    actuals.append(smiles)
                
            except Exception as e:
                logger.warning(f"Failed benchmark for {smiles}: {e}")
        
        # Calculate metrics
        vibrational_accuracy = np.mean(vibrational_similarities)
        structural_similarity = self._calculate_structural_similarity(predictions, actuals)
        
        metrics = {
            'vibrational_prediction_accuracy': vibrational_accuracy,
            'structural_similarity': structural_similarity,
            'quantum_enhancement_factor': vibrational_accuracy / max(0.1, structural_similarity),
            'sample_size': len(test_molecules)
        }
        
        logger.info(f"Quantum benchmark completed: {metrics}")
        return metrics
    
    def _calculate_structural_similarity(self, predictions: List[str], actuals: List[str]) -> float:
        """Calculate structural similarity between predicted and actual molecules."""
        if not predictions or not actuals:
            return 0.0
        
        similarities = []
        for pred, actual in zip(predictions, actuals):
            # Simplified structural similarity (would use Tanimoto in real implementation)
            similarity = len(set(pred) & set(actual)) / len(set(pred) | set(actual))
            similarities.append(similarity)
        
        return np.mean(similarities)