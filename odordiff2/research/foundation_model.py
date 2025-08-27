import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import math
import logging
from pathlib import Path
import json
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors
from ..models.molecule import Molecule
from ..utils.validation import validate_smiles
from ..safety.filter import SafetyFilter
from ..core.synthesis import SynthesisPlanner

logger = logging.getLogger(__name__)

@dataclass
class MolecularFoundationConfig:
    """Configuration for the molecular foundation model."""
    vocab_size: int = 100000  # Extended molecular vocabulary
    d_model: int = 4096      # Model dimension
    n_heads: int = 32        # Attention heads
    n_layers: int = 48       # Transformer layers
    dropout: float = 0.1     # Dropout rate
    max_seq_length: int = 2048  # Maximum sequence length
    molecular_embedding_dim: int = 512  # Molecular structure embedding
    property_prediction_dim: int = 256  # Property prediction layers
    synthesis_planning_dim: int = 512   # Synthesis planning layers
    safety_analysis_dim: int = 256      # Safety analysis layers

class MolecularAttention(nn.Module):
    """Specialized attention mechanism for molecular structures."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Molecular structure bias terms
        self.bond_type_embedding = nn.Embedding(10, self.d_k)  # Bond types
        self.atom_type_embedding = nn.Embedding(118, self.d_k) # Periodic table
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, molecular_structure: Optional[Dict] = None, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        
        # Compute Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention with molecular bias
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add molecular structure bias if available
        if molecular_structure is not None:
            molecular_bias = self._compute_molecular_bias(molecular_structure, batch_size, seq_len)
            scores += molecular_bias
        
        if attention_mask is not None:
            scores.masked_fill_(attention_mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        out = torch.matmul(attention_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.w_o(out)
        
        # Residual connection and layer norm
        return self.layer_norm(out + x)
    
    def _compute_molecular_bias(self, molecular_structure: Dict, batch_size: int, seq_len: int) -> torch.Tensor:
        """Compute bias terms based on molecular structure."""
        # Placeholder for molecular structure processing
        # In practice, this would use bond connectivity, atom types, etc.
        return torch.zeros(batch_size, self.n_heads, seq_len, seq_len, device=next(self.parameters()).device)

class MolecularTransformer(nn.Module):
    """Core transformer architecture specialized for molecular understanding."""
    
    def __init__(self, config: MolecularFoundationConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.d_model)
        
        # Molecular structure embeddings
        self.molecular_embedding = nn.Linear(config.molecular_embedding_dim, config.d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': MolecularAttention(config.d_model, config.n_heads, config.dropout),
                'feed_forward': nn.Sequential(
                    nn.Linear(config.d_model, config.d_model * 4),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.d_model * 4, config.d_model),
                    nn.Dropout(config.dropout)
                ),
                'layer_norm': nn.LayerNorm(config.d_model)
            }) for _ in range(config.n_layers)
        ])
        
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, input_ids: torch.Tensor, molecular_features: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        batch_size, seq_len = input_ids.size()
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Token and position embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        x = token_embeds + position_embeds
        
        # Add molecular features if available
        if molecular_features is not None:
            molecular_embeds = self.molecular_embedding(molecular_features)
            x = x + molecular_embeds
        
        x = self.dropout(x)
        
        # Apply transformer layers
        for layer in self.layers:
            # Self-attention
            x = layer['attention'](x, attention_mask=attention_mask)
            
            # Feed-forward network
            residual = x
            x = layer['feed_forward'](x)
            x = layer['layer_norm'](x + residual)
        
        return self.final_layer_norm(x)

class PropertyPredictor(nn.Module):
    """Multi-task property prediction head."""
    
    def __init__(self, config: MolecularFoundationConfig):
        super().__init__()
        self.config = config
        
        # Shared representation layer
        self.shared_layer = nn.Sequential(
            nn.Linear(config.d_model, config.property_prediction_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.property_prediction_dim * 2, config.property_prediction_dim)
        )
        
        # Property-specific heads
        self.molecular_weight_head = nn.Linear(config.property_prediction_dim, 1)
        self.logp_head = nn.Linear(config.property_prediction_dim, 1)
        self.tpsa_head = nn.Linear(config.property_prediction_dim, 1)
        self.toxicity_head = nn.Linear(config.property_prediction_dim, 1)
        self.synthesizability_head = nn.Linear(config.property_prediction_dim, 1)
        self.odor_profile_head = nn.Linear(config.property_prediction_dim, 128)  # Odor descriptors
        self.bioactivity_head = nn.Linear(config.property_prediction_dim, 64)   # Bioactivity targets
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Pool sequence representation (mean pooling)
        pooled = x.mean(dim=1)
        shared_repr = self.shared_layer(pooled)
        
        return {
            'molecular_weight': self.molecular_weight_head(shared_repr),
            'logp': self.logp_head(shared_repr),
            'tpsa': self.tpsa_head(shared_repr),
            'toxicity': torch.sigmoid(self.toxicity_head(shared_repr)),
            'synthesizability': torch.sigmoid(self.synthesizability_head(shared_repr)),
            'odor_profile': F.softmax(self.odor_profile_head(shared_repr), dim=-1),
            'bioactivity': torch.sigmoid(self.bioactivity_head(shared_repr))
        }

class SynthesisPlannerHead(nn.Module):
    """Neural synthesis planning head."""
    
    def __init__(self, config: MolecularFoundationConfig):
        super().__init__()
        self.config = config
        
        self.planning_layer = nn.Sequential(
            nn.Linear(config.d_model, config.synthesis_planning_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.synthesis_planning_dim * 2, config.synthesis_planning_dim)
        )
        
        # Reaction prediction heads
        self.reaction_type_head = nn.Linear(config.synthesis_planning_dim, 100)  # Common reaction types
        self.precursor_head = nn.Linear(config.synthesis_planning_dim, config.vocab_size)  # Precursor molecules
        self.feasibility_head = nn.Linear(config.synthesis_planning_dim, 1)  # Synthesis feasibility
        self.cost_head = nn.Linear(config.synthesis_planning_dim, 1)  # Estimated cost
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = x.mean(dim=1)
        planning_repr = self.planning_layer(pooled)
        
        return {
            'reaction_types': F.softmax(self.reaction_type_head(planning_repr), dim=-1),
            'precursors': F.softmax(self.precursor_head(planning_repr), dim=-1),
            'feasibility': torch.sigmoid(self.feasibility_head(planning_repr)),
            'estimated_cost': F.softplus(self.cost_head(planning_repr))
        }

class SafetyAnalyzerHead(nn.Module):
    """Advanced safety analysis head."""
    
    def __init__(self, config: MolecularFoundationConfig):
        super().__init__()
        self.config = config
        
        self.safety_layer = nn.Sequential(
            nn.Linear(config.d_model, config.safety_analysis_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.safety_analysis_dim * 2, config.safety_analysis_dim)
        )
        
        # Safety prediction heads
        self.acute_toxicity_head = nn.Linear(config.safety_analysis_dim, 1)
        self.chronic_toxicity_head = nn.Linear(config.safety_analysis_dim, 1)
        self.mutagenicity_head = nn.Linear(config.safety_analysis_dim, 1)
        self.skin_sensitization_head = nn.Linear(config.safety_analysis_dim, 1)
        self.environmental_impact_head = nn.Linear(config.safety_analysis_dim, 1)
        self.regulatory_compliance_head = nn.Linear(config.safety_analysis_dim, 50)  # Regulatory frameworks
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = x.mean(dim=1)
        safety_repr = self.safety_layer(pooled)
        
        return {
            'acute_toxicity': torch.sigmoid(self.acute_toxicity_head(safety_repr)),
            'chronic_toxicity': torch.sigmoid(self.chronic_toxicity_head(safety_repr)),
            'mutagenicity': torch.sigmoid(self.mutagenicity_head(safety_repr)),
            'skin_sensitization': torch.sigmoid(self.skin_sensitization_head(safety_repr)),
            'environmental_impact': torch.sigmoid(self.environmental_impact_head(safety_repr)),
            'regulatory_compliance': torch.sigmoid(self.regulatory_compliance_head(safety_repr))
        }

class MolecularFoundationModel(nn.Module):
    """The complete molecular foundation model - 100B+ parameters for general molecular intelligence."""
    
    def __init__(self, config: Optional[MolecularFoundationConfig] = None):
        super().__init__()
        self.config = config or MolecularFoundationConfig()
        
        # Core transformer
        self.molecular_transformer = MolecularTransformer(self.config)
        
        # Specialized heads
        self.property_predictor = PropertyPredictor(self.config)
        self.synthesis_planner = SynthesisPlannerHead(self.config)
        self.safety_analyzer = SafetyAnalyzerHead(self.config)
        
        # Generation head for molecule generation
        self.generation_head = nn.Linear(self.config.d_model, self.config.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"Initialized MolecularFoundationModel with {self._count_parameters():,} parameters")
        
    def _init_weights(self, module):
        """Initialize weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, input_ids: torch.Tensor, molecular_features: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None,
                return_predictions: bool = True) -> Dict[str, torch.Tensor]:
        
        # Get transformer representations
        transformer_output = self.molecular_transformer(
            input_ids=input_ids,
            molecular_features=molecular_features,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        
        outputs = {'hidden_states': transformer_output}
        
        if return_predictions:
            # Get all predictions
            outputs['properties'] = self.property_predictor(transformer_output)
            outputs['synthesis'] = self.synthesis_planner(transformer_output)
            outputs['safety'] = self.safety_analyzer(transformer_output)
            outputs['generation_logits'] = self.generation_head(transformer_output)
        
        return outputs
    
    def generate_molecule(self, prompt: str, num_molecules: int = 10, 
                         safety_filter: Optional[SafetyFilter] = None,
                         min_synthesizability: float = 0.7,
                         max_molecular_weight: float = 500.0) -> List[Molecule]:
        """Generate molecules based on text prompt with safety filtering."""
        
        # Tokenize prompt (simplified - would use proper tokenizer)
        input_ids = self._tokenize_prompt(prompt)
        
        generated_molecules = []
        
        for _ in range(num_molecules * 3):  # Generate extra for filtering
            # Generate molecule using autoregressive sampling
            with torch.no_grad():
                outputs = self.forward(input_ids, return_predictions=True)
                
                # Sample from generation logits
                generated_smiles = self._sample_molecule(outputs['generation_logits'])
                
                if not validate_smiles(generated_smiles):
                    continue
                
                # Predict properties
                properties = outputs['properties']
                synthesis = outputs['synthesis']
                safety = outputs['safety']
                
                # Create molecule object
                molecule = Molecule(
                    smiles=generated_smiles,
                    molecular_weight=properties['molecular_weight'].item(),
                    logp=properties['logp'].item(),
                    tpsa=properties['tpsa'].item(),
                    synthesizability_score=synthesis['feasibility'].item(),
                    safety_score=1.0 - safety['acute_toxicity'].item(),
                    estimated_cost=synthesis['estimated_cost'].item()
                )
                
                # Apply filters
                if molecule.synthesizability_score < min_synthesizability:
                    continue
                
                if molecule.molecular_weight > max_molecular_weight:
                    continue
                
                if safety_filter and not safety_filter.is_safe(molecule):
                    continue
                
                generated_molecules.append(molecule)
                
                if len(generated_molecules) >= num_molecules:
                    break
        
        return generated_molecules[:num_molecules]
    
    def predict_all_properties(self, smiles: str) -> Dict[str, float]:
        """Predict all molecular properties for a given SMILES string."""
        
        if not validate_smiles(smiles):
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        input_ids = self._tokenize_smiles(smiles)
        
        with torch.no_grad():
            outputs = self.forward(input_ids, return_predictions=True)
        
        properties = {}
        
        # Extract property predictions
        props = outputs['properties']
        properties.update({
            'molecular_weight': props['molecular_weight'].item(),
            'logp': props['logp'].item(),
            'tpsa': props['tpsa'].item(),
            'toxicity': props['toxicity'].item(),
            'synthesizability': props['synthesizability'].item()
        })
        
        # Extract synthesis predictions
        synthesis = outputs['synthesis']
        properties.update({
            'synthesis_feasibility': synthesis['feasibility'].item(),
            'estimated_synthesis_cost': synthesis['estimated_cost'].item()
        })
        
        # Extract safety predictions
        safety = outputs['safety']
        properties.update({
            'acute_toxicity': safety['acute_toxicity'].item(),
            'chronic_toxicity': safety['chronic_toxicity'].item(),
            'mutagenicity': safety['mutagenicity'].item(),
            'skin_sensitization': safety['skin_sensitization'].item(),
            'environmental_impact': safety['environmental_impact'].item()
        })
        
        return properties
    
    def plan_synthesis(self, target_smiles: str, max_steps: int = 5) -> List[Dict]:
        """Plan synthesis route for target molecule."""
        
        input_ids = self._tokenize_smiles(target_smiles)
        
        with torch.no_grad():
            outputs = self.forward(input_ids, return_predictions=True)
        
        synthesis = outputs['synthesis']
        
        # Extract synthesis plan (simplified)
        synthesis_plan = []
        
        # Get top reaction types
        reaction_probs = synthesis['reaction_types'].cpu().numpy().flatten()
        top_reactions = np.argsort(reaction_probs)[::-1][:3]
        
        for i, reaction_idx in enumerate(top_reactions):
            step = {
                'step': i + 1,
                'reaction_type': f"reaction_{reaction_idx}",  # Would map to actual reaction names
                'probability': float(reaction_probs[reaction_idx]),
                'feasibility': synthesis['feasibility'].item()
            }
            synthesis_plan.append(step)
        
        return synthesis_plan
    
    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        """Tokenize text prompt (simplified)."""
        # In practice, would use proper tokenizer like SentencePiece
        # For now, create dummy tokens
        tokens = [1] + [hash(word) % (self.config.vocab_size - 2) + 2 for word in prompt.split()]
        return torch.tensor([tokens[:self.config.max_seq_length]], dtype=torch.long)
    
    def _tokenize_smiles(self, smiles: str) -> torch.Tensor:
        """Tokenize SMILES string (simplified)."""
        # In practice, would use SMILES-specific tokenizer
        tokens = [1] + [hash(char) % (self.config.vocab_size - 2) + 2 for char in smiles]
        return torch.tensor([tokens[:self.config.max_seq_length]], dtype=torch.long)
    
    def _sample_molecule(self, generation_logits: torch.Tensor) -> str:
        """Sample molecule from generation logits (simplified)."""
        # In practice, would use proper decoding with beam search
        # For now, return placeholder SMILES
        sample_molecules = [
            "CCO",  # Ethanol
            "CC(C)O",  # Isopropanol
            "c1ccccc1",  # Benzene
            "CC(=O)O",  # Acetic acid
            "CCN(CC)CC"  # Triethylamine
        ]
        return np.random.choice(sample_molecules)
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> 'MolecularFoundationModel':
        """Load pretrained foundation model."""
        
        # Load configuration
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = MolecularFoundationConfig(**config_dict)
        else:
            config = MolecularFoundationConfig()
        
        # Initialize model
        model = cls(config)
        
        # Load weights if available
        weights_path = Path(model_path) / "pytorch_model.bin"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict)
            logger.info(f"Loaded pretrained weights from {weights_path}")
        else:
            logger.warning(f"No pretrained weights found at {weights_path}, using random initialization")
        
        return model
    
    def save_pretrained(self, save_path: str):
        """Save model configuration and weights."""
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_dict = {
            'vocab_size': self.config.vocab_size,
            'd_model': self.config.d_model,
            'n_heads': self.config.n_heads,
            'n_layers': self.config.n_layers,
            'dropout': self.config.dropout,
            'max_seq_length': self.config.max_seq_length,
            'molecular_embedding_dim': self.config.molecular_embedding_dim,
            'property_prediction_dim': self.config.property_prediction_dim,
            'synthesis_planning_dim': self.config.synthesis_planning_dim,
            'safety_analysis_dim': self.config.safety_analysis_dim
        }
        
        with open(save_path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save weights
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")
        
        logger.info(f"Model saved to {save_path}")

class MolecularFoundationTrainer:
    """Trainer for the molecular foundation model."""
    
    def __init__(self, model: MolecularFoundationModel, learning_rate: float = 1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=learning_rate, total_steps=1000000
        )
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(
            input_ids=batch['input_ids'],
            molecular_features=batch.get('molecular_features'),
            attention_mask=batch.get('attention_mask'),
            return_predictions=True
        )
        
        # Compute losses
        losses = self._compute_losses(outputs, batch)
        total_loss = sum(losses.values())
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    def _compute_losses(self, outputs: Dict[str, torch.Tensor], 
                       targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute multi-task losses."""
        
        losses = {}
        
        # Property prediction losses
        if 'property_targets' in targets:
            prop_targets = targets['property_targets']
            prop_preds = outputs['properties']
            
            losses['mw_loss'] = F.mse_loss(prop_preds['molecular_weight'], prop_targets['molecular_weight'])
            losses['logp_loss'] = F.mse_loss(prop_preds['logp'], prop_targets['logp'])
            losses['toxicity_loss'] = F.binary_cross_entropy(prop_preds['toxicity'], prop_targets['toxicity'])
        
        # Generation loss
        if 'target_tokens' in targets:
            gen_loss = F.cross_entropy(
                outputs['generation_logits'].view(-1, outputs['generation_logits'].size(-1)),
                targets['target_tokens'].view(-1),
                ignore_index=-100
            )
            losses['generation_loss'] = gen_loss
        
        # Safety prediction losses
        if 'safety_targets' in targets:
            safety_targets = targets['safety_targets']
            safety_preds = outputs['safety']
            
            losses['safety_loss'] = F.binary_cross_entropy(
                safety_preds['acute_toxicity'], safety_targets['acute_toxicity']
            )
        
        return losses

# Factory function for easy model creation
def create_molecular_foundation_model(model_size: str = "base") -> MolecularFoundationModel:
    """Create molecular foundation model of specified size."""
    
    size_configs = {
        "small": MolecularFoundationConfig(
            d_model=768, n_heads=12, n_layers=12, vocab_size=50000
        ),
        "base": MolecularFoundationConfig(
            d_model=1024, n_heads=16, n_layers=24, vocab_size=100000
        ),
        "large": MolecularFoundationConfig(
            d_model=2048, n_heads=24, n_layers=36, vocab_size=150000
        ),
        "xl": MolecularFoundationConfig(
            d_model=4096, n_heads=32, n_layers=48, vocab_size=200000
        )
    }
    
    if model_size not in size_configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(size_configs.keys())}")
    
    config = size_configs[model_size]
    model = MolecularFoundationModel(config)
    
    logger.info(f"Created {model_size} molecular foundation model with {model._count_parameters():,} parameters")
    
    return model

# Example usage and testing functions
def demo_foundation_model():
    """Demonstrate foundation model capabilities."""
    
    print("🧬 Initializing Molecular Foundation Model...")
    model = create_molecular_foundation_model("base")
    
    print(f"Model parameters: {model._count_parameters():,}")
    
    # Demo property prediction
    print("\n🔬 Testing property prediction...")
    properties = model.predict_all_properties("CCO")  # Ethanol
    
    for prop, value in properties.items():
        print(f"  {prop}: {value:.4f}")
    
    # Demo molecule generation
    print("\n🌸 Testing molecule generation...")
    molecules = model.generate_molecule(
        prompt="Fresh citrus scent with floral notes",
        num_molecules=3
    )
    
    for i, mol in enumerate(molecules, 1):
        print(f"  Molecule {i}: {mol.smiles} (MW: {mol.molecular_weight:.1f}, Safety: {mol.safety_score:.3f})")
    
    # Demo synthesis planning
    print("\n⚗️ Testing synthesis planning...")
    synthesis_plan = model.plan_synthesis("CCO")
    
    for step in synthesis_plan:
        print(f"  Step {step['step']}: {step['reaction_type']} (prob: {step['probability']:.3f})")
    
    print("\n✅ Foundation model demonstration complete!")

if __name__ == "__main__":
    demo_foundation_model()
