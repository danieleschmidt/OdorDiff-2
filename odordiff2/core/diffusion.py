"""
Core diffusion model for text-to-molecule generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from transformers import AutoTokenizer, AutoModel
import random

from ..models.molecule import Molecule, OdorProfile
from ..safety.filter import SafetyFilter


class TextEncoder(nn.Module):
    """Encode text descriptions into latent space."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
    def forward(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to embeddings."""
        inputs = self.tokenizer(
            texts, return_tensors="pt", 
            padding=True, truncation=True, max_length=512
        )
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return F.normalize(embeddings, p=2, dim=1)


class MolecularDecoder(nn.Module):
    """Decode latent representations to molecular SMILES."""
    
    def __init__(self, latent_dim: int = 512, vocab_size: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        
        # SMILES vocabulary (simplified)
        self.vocab = {
            'PAD': 0, 'START': 1, 'END': 2,
            'C': 3, 'c': 4, 'N': 5, 'n': 6, 'O': 7, 'o': 8,
            'S': 9, 's': 10, 'P': 11, 'F': 12, 'Cl': 13, 'Br': 14,
            '(': 15, ')': 16, '[': 17, ']': 18, '=': 19, '#': 20,
            '-': 21, '+': 22, '1': 23, '2': 24, '3': 25, '4': 26,
            '5': 27, '6': 28, '7': 29, '8': 30, '9': 31, '0': 32,
            '@': 33, 'H': 34
        }
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Decoder architecture
        self.embedding = nn.Embedding(vocab_size, 256)
        self.latent_proj = nn.Linear(latent_dim, 256)
        self.decoder = nn.LSTM(512, 256, num_layers=2, batch_first=True)
        self.output_proj = nn.Linear(256, vocab_size)
        
    def forward(self, latent: torch.Tensor, max_length: int = 100) -> torch.Tensor:
        """Decode latent to SMILES sequence."""
        batch_size = latent.size(0)
        device = latent.device
        
        # Initialize with START token
        sequence = torch.full((batch_size, 1), self.vocab['START'], device=device)
        outputs = []
        
        # Project latent to hidden space
        latent_hidden = self.latent_proj(latent).unsqueeze(1)  # (B, 1, 256)
        
        for _ in range(max_length):
            # Embed current sequence
            embedded = self.embedding(sequence)  # (B, seq_len, 256)
            
            # Concatenate with latent conditioning
            conditioned = torch.cat([embedded, latent_hidden.expand(-1, embedded.size(1), -1)], dim=-1)
            
            # Decode
            output, _ = self.decoder(conditioned)
            logits = self.output_proj(output[:, -1:, :])  # Take last timestep
            
            # Sample next token
            next_token = torch.multinomial(F.softmax(logits.squeeze(1), dim=-1), 1)
            sequence = torch.cat([sequence, next_token], dim=1)
            outputs.append(logits.squeeze(1))
            
            # Stop if END token
            if (next_token == self.vocab['END']).all():
                break
                
        return torch.stack(outputs, dim=1)
    
    def decode_to_smiles(self, sequence: torch.Tensor) -> List[str]:
        """Convert token sequences back to SMILES strings."""
        smiles_list = []
        
        for seq in sequence:
            tokens = []
            for token_id in seq:
                token = self.reverse_vocab.get(token_id.item(), '')
                if token == 'END':
                    break
                elif token not in ['PAD', 'START']:
                    tokens.append(token)
            smiles_list.append(''.join(tokens))
            
        return smiles_list


class SimpleDiffusionScheduler:
    """Simplified diffusion noise scheduler."""
    
    def __init__(self, num_steps: int = 100, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def add_noise(self, x: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to clean data."""
        noise = torch.randn_like(x)
        alpha_cumprod_t = self.alpha_cumprod[timesteps].view(-1, 1)
        
        noisy_x = torch.sqrt(alpha_cumprod_t) * x + torch.sqrt(1 - alpha_cumprod_t) * noise
        return noisy_x, noise
        
    def denoise_step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor) -> torch.Tensor:
        """Single denoising step."""
        alpha_t = self.alphas[timestep]
        alpha_cumprod_t = self.alpha_cumprod[timestep]
        
        pred_original = (sample - torch.sqrt(1 - alpha_cumprod_t) * model_output) / torch.sqrt(alpha_cumprod_t)
        pred_prev = torch.sqrt(alpha_t) * pred_original + torch.sqrt(1 - alpha_t) * model_output
        
        return pred_prev


class OdorDiffusion:
    """
    Main class for text-to-scent molecule generation using diffusion models.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        
        # Core components
        self.text_encoder = TextEncoder()
        self.molecular_decoder = MolecularDecoder()
        self.scheduler = SimpleDiffusionScheduler()
        
        # Move to device
        self.text_encoder.to(device)
        self.molecular_decoder.to(device)
        
        # Odor prediction models (simplified)
        self._odor_predictor = self._init_odor_predictor()
        
        # Known fragrance molecules for similarity-based generation
        self._fragrance_database = self._init_fragrance_database()
        
    def _init_odor_predictor(self) -> Dict[str, Any]:
        """Initialize odor prediction model (simplified rule-based)."""
        return {
            'floral_patterns': ['C=CCO', 'c1ccccc1CO', 'CC(C)=CCO'],
            'citrus_patterns': ['CC(C)=CC', 'C=C(C)C', 'CC1=CC(=O)CC(C)(C)O1'],
            'woody_patterns': ['CC(C)(C)c1ccccc1', 'c1ccc2c(c1)cccc2C'],
            'fresh_patterns': ['CCO', 'CC(C)O', 'C=CCO'],
            'spicy_patterns': ['COc1ccccc1', 'c1ccc(cc1)C=O', 'CC(=O)c1ccccc1']
        }
    
    def _init_fragrance_database(self) -> List[Dict[str, Any]]:
        """Initialize database of known fragrance molecules."""
        return [
            {'smiles': 'CC(C)=CCO', 'name': 'linalool', 'odor': 'floral, lavender'},
            {'smiles': 'CC(C)=CCCC(C)=CCO', 'name': 'geraniol', 'odor': 'rose, geranium'},
            {'smiles': 'CC1=CCC(CC1)C(C)C', 'name': 'limonene', 'odor': 'citrus, orange'},
            {'smiles': 'c1ccc(cc1)C=O', 'name': 'benzaldehyde', 'odor': 'almond, cherry'},
            {'smiles': 'CCc1ccc(cc1)C(C)C', 'name': 'p-cymene', 'odor': 'fresh, citrus'},
            {'smiles': 'CC(C)(C)c1ccccc1O', 'name': 'BHT', 'odor': 'phenolic, medicinal'},
            {'smiles': 'COc1ccc(cc1)C=O', 'name': 'anisaldehyde', 'odor': 'sweet, vanilla'},
            {'smiles': 'CC(=O)c1ccccc1', 'name': 'acetophenone', 'odor': 'floral, hawthorne'}
        ]
    
    @classmethod
    def from_pretrained(cls, model_name: str, device: str = "cpu") -> 'OdorDiffusion':
        """Load pretrained model (placeholder - would load actual weights)."""
        print(f"Loading pretrained model: {model_name}")
        model = cls(device=device)
        # In real implementation, would load trained weights here
        return model
        
    def generate(
        self, 
        prompt: str,
        num_molecules: int = 5,
        safety_filter: Optional[SafetyFilter] = None,
        synthesizability_min: float = 0.0,
        **kwargs
    ) -> List[Molecule]:
        """
        Generate scent molecules from text description.
        
        Args:
            prompt: Text description of desired scent
            num_molecules: Number of molecules to generate
            safety_filter: Optional safety filter
            synthesizability_min: Minimum synthesis feasibility score
            
        Returns:
            List of generated molecules
        """
        print(f"Generating {num_molecules} molecules for: '{prompt}'")
        
        # For now, use template-based generation (would be neural in real implementation)
        molecules = self._template_based_generation(prompt, num_molecules)
        
        # Predict odor profiles
        for molecule in molecules:
            molecule.odor_profile = self._predict_odor(molecule, prompt)
            molecule.synth_score = self._estimate_synthesizability(molecule)
            molecule.estimated_cost = self._estimate_cost(molecule)
        
        # Apply safety filtering if provided
        if safety_filter:
            safe_molecules, _ = safety_filter.filter_molecules(molecules)
            molecules = safe_molecules
            
        # Filter by synthesizability
        if synthesizability_min > 0:
            molecules = [m for m in molecules if m.synth_score >= synthesizability_min]
            
        # Sort by confidence/quality
        molecules.sort(key=lambda m: m.confidence, reverse=True)
        
        return molecules[:num_molecules]
    
    def _template_based_generation(self, prompt: str, num_molecules: int) -> List[Molecule]:
        """Template-based molecule generation (placeholder for neural generation)."""
        molecules = []
        
        # Analyze prompt for scent categories
        prompt_lower = prompt.lower()
        
        # Select base templates based on prompt
        templates = []
        if any(word in prompt_lower for word in ['floral', 'flower', 'rose', 'jasmine', 'lavender']):
            templates.extend(['CC(C)=CCO', 'CCCC(C)=CCO', 'CC(C)(O)CCO'])
        if any(word in prompt_lower for word in ['citrus', 'lemon', 'orange', 'lime', 'bergamot']):
            templates.extend(['CC(C)=CC', 'C=C(C)CCC=C(C)C', 'CC1=CCC(CC1)C(C)C'])
        if any(word in prompt_lower for word in ['woody', 'cedar', 'sandalwood', 'pine']):
            templates.extend(['CC(C)(C)c1ccccc1', 'c1ccc2c(c1)cccc2', 'CC12CCC3C(C1)CCC4=CC(=O)CCC34C2'])
        if any(word in prompt_lower for word in ['fresh', 'clean', 'aquatic', 'marine', 'ozone']):
            templates.extend(['CCO', 'CCCO', 'CC(C)O', 'C=CCO'])
        if any(word in prompt_lower for word in ['spicy', 'pepper', 'cinnamon', 'clove']):
            templates.extend(['COc1ccccc1', 'c1ccc(cc1)C=O', 'CC(=O)c1ccccc1'])
        if any(word in prompt_lower for word in ['sweet', 'vanilla', 'caramel', 'honey']):
            templates.extend(['COc1ccc(cc1)C=O', 'CC(=O)OC', 'CC(C)(C)c1ccc(cc1)O'])
            
        # Default templates if no match
        if not templates:
            templates = ['CC(C)=CCO', 'CC(C)=CC', 'c1ccccc1C=O']
            
        # Generate variations
        for i in range(num_molecules):
            if i < len(templates):
                base_smiles = templates[i]
            else:
                base_smiles = random.choice(templates)
                
            # Add some random variation (simplified)
            variation = self._add_molecular_variation(base_smiles)
            confidence = random.uniform(0.7, 0.95)
            
            molecules.append(Molecule(variation, confidence))
            
        return molecules
    
    def _add_molecular_variation(self, smiles: str) -> str:
        """Add small structural variations to base molecule."""
        # Simple variations (in real implementation would be more sophisticated)
        variations = [
            smiles,  # Original
            smiles.replace('C', 'CC', 1) if 'C' in smiles else smiles,  # Add methyl
            smiles.replace('CCO', 'CCCO') if 'CCO' in smiles else smiles,  # Extend chain
            smiles.replace('C=C', 'CC') if 'C=C' in smiles else smiles,  # Saturate
        ]
        
        return random.choice([v for v in variations if v != smiles] or [smiles])
    
    def _predict_odor(self, molecule: Molecule, prompt: str) -> OdorProfile:
        """Predict odor profile based on structure and prompt."""
        if not molecule.is_valid:
            return OdorProfile()
            
        # Pattern-based odor prediction (simplified)
        primary_notes = []
        secondary_notes = []
        character = ""
        
        smiles = molecule.smiles.lower()
        
        # Check against known patterns
        for category, patterns in self._odor_predictor.items():
            for pattern in patterns:
                if pattern.lower() in smiles:
                    if category == 'floral_patterns':
                        primary_notes.extend(['floral', 'sweet'])
                        secondary_notes.extend(['rosy', 'powdery'])
                        character = "elegant, feminine"
                    elif category == 'citrus_patterns':
                        primary_notes.extend(['citrus', 'fresh'])
                        secondary_notes.extend(['zesty', 'bright'])
                        character = "energizing, uplifting"
                    elif category == 'woody_patterns':
                        primary_notes.extend(['woody', 'warm'])
                        secondary_notes.extend(['cedar', 'dry'])
                        character = "sophisticated, masculine"
                    elif category == 'fresh_patterns':
                        primary_notes.extend(['fresh', 'clean'])
                        secondary_notes.extend(['aquatic', 'ozonic'])
                        character = "modern, crisp"
                    elif category == 'spicy_patterns':
                        primary_notes.extend(['spicy', 'warm'])
                        secondary_notes.extend(['peppery', 'aromatic'])
                        character = "exotic, intense"
                        
        # Remove duplicates and limit
        primary_notes = list(set(primary_notes))[:3]
        secondary_notes = list(set(secondary_notes))[:3]
        
        # Estimate intensity and longevity based on molecular properties
        mw = molecule.get_property('molecular_weight') or 150
        intensity = min(1.0, (mw - 100) / 200)  # Heavier = more intense
        longevity_hours = max(1.0, min(12.0, mw / 20))  # Heavier = longer lasting
        sillage = random.uniform(0.3, 0.8)
        
        return OdorProfile(
            primary_notes=primary_notes,
            secondary_notes=secondary_notes,
            intensity=intensity,
            longevity_hours=longevity_hours,
            sillage=sillage,
            character=character or "pleasant, balanced"
        )
    
    def _estimate_synthesizability(self, molecule: Molecule) -> float:
        """Estimate how easy the molecule is to synthesize."""
        if not molecule.is_valid:
            return 0.0
            
        # Simple heuristics (real implementation would use retrosynthesis prediction)
        score = 0.8  # Base score
        
        # Complexity penalties
        mw = molecule.get_property('molecular_weight') or 0
        if mw > 300:
            score -= 0.2
        if mw > 500:
            score -= 0.3
            
        # Ring complexity
        aromatic_rings = molecule.get_property('aromatic_rings') or 0
        if aromatic_rings > 2:
            score -= 0.1
            
        # Heteroatom bonus (many common fragrance precursors have O, N)
        mol = molecule.mol
        if mol:
            hetero_count = sum(1 for atom in mol.GetAtoms() 
                             if atom.GetSymbol() in ['O', 'N', 'S'])
            if 1 <= hetero_count <= 3:
                score += 0.1
                
        return max(0.0, min(1.0, score))
    
    def _estimate_cost(self, molecule: Molecule) -> float:
        """Estimate production cost per kg."""
        if not molecule.is_valid:
            return 1000.0
            
        # Base cost estimation (simplified)
        mw = molecule.get_property('molecular_weight') or 150
        synth_score = molecule.synth_score or 0.5
        
        # Base cost increases with complexity
        base_cost = 50 + (mw - 100) * 0.5
        
        # Synthesis difficulty multiplier
        difficulty_multiplier = 1.0 / max(0.1, synth_score)
        
        # Economies of scale (random factor for market demand)
        scale_factor = random.uniform(0.8, 2.0)
        
        estimated_cost = base_cost * difficulty_multiplier * scale_factor
        
        return round(estimated_cost, 2)
    
    def design_fragrance(
        self,
        base_notes: str,
        heart_notes: str,
        top_notes: str,
        style: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> 'FragranceFormulation':
        """Design a complete fragrance with multiple accords."""
        print(f"Designing fragrance with style: {style}")
        
        # Generate molecules for each note category
        base_molecules = self.generate(f"{base_notes} {style}", num_molecules=3)
        heart_molecules = self.generate(f"{heart_notes} {style}", num_molecules=3)
        top_molecules = self.generate(f"{top_notes} {style}", num_molecules=3)
        
        return FragranceFormulation(
            base_accord=base_molecules,
            heart_accord=heart_molecules,
            top_accord=top_molecules,
            style_descriptor=style
        )


class FragranceFormulation:
    """Represents a complete fragrance formulation."""
    
    def __init__(
        self,
        base_accord: List[Molecule],
        heart_accord: List[Molecule],
        top_accord: List[Molecule],
        style_descriptor: str
    ):
        self.base_accord = base_accord
        self.heart_accord = heart_accord
        self.top_accord = top_accord
        self.style_descriptor = style_descriptor
        
    def to_perfume_formula(self, concentration: str = 'eau_de_parfum', carrier: str = 'ethanol_90') -> Dict[str, Any]:
        """Convert to perfume formula with concentrations."""
        concentration_levels = {
            'parfum': 0.25,
            'eau_de_parfum': 0.18,
            'eau_de_toilette': 0.12,
            'eau_de_cologne': 0.06
        }
        
        fragrance_oil_percent = concentration_levels.get(concentration, 0.15)
        
        # Typical pyramid proportions
        top_proportion = 0.3
        heart_proportion = 0.5
        base_proportion = 0.2
        
        formula = {
            'concentration_type': concentration,
            'fragrance_oil_percent': fragrance_oil_percent * 100,
            'carrier': carrier,
            'top_notes': {
                'molecules': [m.to_dict() for m in self.top_accord],
                'proportion_of_fragrance': top_proportion,
                'actual_percent': fragrance_oil_percent * top_proportion * 100
            },
            'heart_notes': {
                'molecules': [m.to_dict() for m in self.heart_accord],
                'proportion_of_fragrance': heart_proportion,
                'actual_percent': fragrance_oil_percent * heart_proportion * 100
            },
            'base_notes': {
                'molecules': [m.to_dict() for m in self.base_accord],
                'proportion_of_fragrance': base_proportion,
                'actual_percent': fragrance_oil_percent * base_proportion * 100
            },
            'style': self.style_descriptor
        }
        
        return formula