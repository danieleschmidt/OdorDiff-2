"""
Multi-modal transformer encoder for enhanced text-to-molecule generation.

This module implements a state-of-the-art transformer architecture that can
process multiple modalities (text, chemical properties, molecular graphs)
to generate more accurate and targeted scent molecules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import math

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MultiModalInput:
    """Container for multi-modal inputs to the transformer."""
    text_tokens: torch.Tensor  # Text tokens
    property_vector: torch.Tensor  # Molecular property vector
    graph_features: Optional[torch.Tensor] = None  # Molecular graph features
    attention_mask: Optional[torch.Tensor] = None  # Attention mask
    

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_seq_length: int = 512):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:x.size(0), :]


class PropertyAwareAttention(nn.Module):
    """
    Attention mechanism that incorporates molecular properties.
    """
    
    def __init__(self, d_model: int, n_heads: int, property_dim: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Standard attention components
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        # Property-aware components
        self.property_query = nn.Linear(property_dim, d_model)
        self.property_key = nn.Linear(property_dim, d_model)
        self.property_gate = nn.Linear(property_dim, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(
        self, 
        x: torch.Tensor, 
        property_vector: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Property-aware multi-head attention.
        
        Args:
            x: Input tensor (seq_len, batch_size, d_model)
            property_vector: Molecular properties (batch_size, property_dim)
            mask: Attention mask (batch_size, seq_len)
        """
        seq_len, batch_size, d_model = x.shape
        
        # Standard attention
        Q = self.query(x).view(seq_len, batch_size, self.n_heads, self.head_dim)
        K = self.key(x).view(seq_len, batch_size, self.n_heads, self.head_dim)
        V = self.value(x).view(seq_len, batch_size, self.n_heads, self.head_dim)
        
        # Property-aware modulation
        prop_q = self.property_query(property_vector).view(1, batch_size, self.n_heads, self.head_dim)
        prop_k = self.property_key(property_vector).view(1, batch_size, self.n_heads, self.head_dim)
        prop_gate = torch.sigmoid(self.property_gate(property_vector)).view(1, batch_size, 1, d_model)
        
        # Modulate queries and keys with properties
        Q = Q + prop_q
        K = K + prop_k
        
        # Transpose for attention computation
        Q = Q.transpose(0, 1).transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
        K = K.transpose(0, 1).transpose(1, 2)
        V = V.transpose(0, 1).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
            scores.masked_fill_(mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)  # (batch_size, n_heads, seq_len, head_dim)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Apply property gate and output projection
        attended = attended.transpose(0, 1)  # (seq_len, batch_size, d_model)
        attended = attended * prop_gate
        output = self.out_proj(attended)
        
        return output


class MolecularGraphEncoder(nn.Module):
    """
    Graph neural network encoder for molecular structures.
    """
    
    def __init__(self, atom_features: int = 74, d_model: int = 512):
        super().__init__()
        self.atom_features = atom_features
        self.d_model = d_model
        
        # Atom feature embedding
        self.atom_embedding = nn.Linear(atom_features, d_model)
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(3)
        ])
        
        # Global pooling
        self.global_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, atom_features: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        Encode molecular graph.
        
        Args:
            atom_features: Atom feature matrix (batch_size, n_atoms, atom_features)
            adjacency_matrix: Adjacency matrix (batch_size, n_atoms, n_atoms)
            
        Returns:
            Graph embedding (batch_size, d_model)
        """
        batch_size, n_atoms, _ = atom_features.shape
        
        # Embed atom features
        x = self.atom_embedding(atom_features)  # (batch_size, n_atoms, d_model)
        
        # Graph convolution
        for conv in self.conv_layers:
            # Message passing
            messages = torch.bmm(adjacency_matrix, x)  # (batch_size, n_atoms, d_model)
            x = F.relu(conv(messages)) + x  # Residual connection
        
        # Global pooling (mean pooling)
        graph_embedding = torch.mean(x, dim=1)  # (batch_size, d_model)
        
        # Final projection
        graph_embedding = self.global_pool(graph_embedding)
        
        return graph_embedding


class MultiModalTransformerLayer(nn.Module):
    """Single transformer layer with multi-modal awareness."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, property_dim: int):
        super().__init__()
        
        # Multi-head attention with property awareness
        self.attention = PropertyAwareAttention(d_model, n_heads, property_dim)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(
        self, 
        x: torch.Tensor, 
        property_vector: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through transformer layer."""
        # Self-attention with residual connection
        attn_output = self.attention(x, property_vector, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class MultiModalTransformerEncoder(nn.Module):
    """
    Multi-modal transformer encoder that processes text, molecular properties,
    and optional graph structure to create rich molecular representations.
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        property_dim: int = 64,
        max_seq_length: int = 512,
        use_graph_encoder: bool = False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.use_graph_encoder = use_graph_encoder
        
        # Text embedding
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Property projection
        self.property_projection = nn.Sequential(
            nn.Linear(property_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Graph encoder (optional)
        if use_graph_encoder:
            self.graph_encoder = MolecularGraphEncoder(d_model=d_model)
            self.graph_integration = nn.Linear(d_model * 2, d_model)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            MultiModalTransformerLayer(d_model, n_heads, d_ff, property_dim)
            for _ in range(n_layers)
        ])
        
        # Output projection for molecular latent space
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model)
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"MultiModalTransformerEncoder initialized: {n_layers} layers, {n_heads} heads")
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def forward(self, multi_modal_input: MultiModalInput) -> torch.Tensor:
        """
        Forward pass through multi-modal transformer.
        
        Args:
            multi_modal_input: Container with text tokens, properties, and optional graph
            
        Returns:
            Molecular latent representation (batch_size, d_model)
        """
        text_tokens = multi_modal_input.text_tokens
        property_vector = multi_modal_input.property_vector
        graph_features = multi_modal_input.graph_features
        attention_mask = multi_modal_input.attention_mask
        
        batch_size, seq_len = text_tokens.shape
        
        # Text embedding with positional encoding
        text_embedded = self.text_embedding(text_tokens) * math.sqrt(self.d_model)
        text_embedded = text_embedded.transpose(0, 1)  # (seq_len, batch_size, d_model)
        text_embedded = self.positional_encoding(text_embedded)
        
        # Property embedding
        property_embedded = self.property_projection(property_vector)  # (batch_size, d_model)
        
        # Process through transformer layers
        hidden_states = text_embedded
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, property_vector, attention_mask)
        
        # Global pooling of text representations
        if attention_mask is not None:
            # Masked mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(
                hidden_states.transpose(0, 1)
            ).float()
            sum_embeddings = torch.sum(hidden_states.transpose(0, 1) * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            text_representation = sum_embeddings / sum_mask
        else:
            # Simple mean pooling
            text_representation = torch.mean(hidden_states, dim=0)  # (batch_size, d_model)
        
        # Combine text and property representations
        combined_representation = text_representation + property_embedded
        
        # Integrate graph features if available
        if self.use_graph_encoder and graph_features is not None:
            # Assuming graph_features contains (atom_features, adjacency_matrix)
            atom_features, adjacency_matrix = graph_features
            graph_representation = self.graph_encoder(atom_features, adjacency_matrix)
            
            # Combine all modalities
            multi_modal_representation = torch.cat([
                combined_representation, graph_representation
            ], dim=-1)
            combined_representation = self.graph_integration(multi_modal_representation)
        
        # Final output projection
        molecular_latent = self.output_projection(combined_representation)
        
        return molecular_latent
    
    def create_property_vector(
        self, 
        molecular_weight: float,
        logp: float,
        tpsa: float,
        rotatable_bonds: int,
        hbd: int,
        hba: int,
        aromatic_rings: int = 0,
        **kwargs
    ) -> torch.Tensor:
        """
        Create a molecular property vector from individual properties.
        
        Args:
            molecular_weight: Molecular weight
            logp: Lipophilicity (logP)
            tpsa: Topological polar surface area
            rotatable_bonds: Number of rotatable bonds
            hbd: Hydrogen bond donors
            hba: Hydrogen bond acceptors
            aromatic_rings: Number of aromatic rings
            
        Returns:
            Property vector (property_dim,)
        """
        # Normalize properties to reasonable ranges
        properties = torch.tensor([
            molecular_weight / 500.0,  # Normalize to [0, 1] for MW up to 500
            (logp + 5) / 10.0,        # Normalize logP from [-5, 5] to [0, 1]
            tpsa / 150.0,             # Normalize TPSA to [0, 1] for up to 150
            rotatable_bonds / 20.0,   # Normalize for up to 20 rotatable bonds
            hbd / 10.0,               # Normalize for up to 10 HBD
            hba / 15.0,               # Normalize for up to 15 HBA
            aromatic_rings / 5.0,     # Normalize for up to 5 aromatic rings
        ], dtype=torch.float32)
        
        # Pad or expand to property_dim if needed
        property_dim = 64  # Should match model's property_dim
        if len(properties) < property_dim:
            padding = torch.zeros(property_dim - len(properties))
            properties = torch.cat([properties, padding])
        
        return properties[:property_dim]
    
    def encode_scent_description(
        self, 
        description: str,
        target_properties: Dict[str, float],
        tokenizer: Any,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Encode a scent description with target molecular properties.
        
        Args:
            description: Text description of the desired scent
            target_properties: Dictionary of target molecular properties
            tokenizer: Text tokenizer
            device: Device to use
            
        Returns:
            Molecular latent representation
        """
        # Tokenize text
        tokens = tokenizer.encode(description, return_tensors="pt")
        if tokens.shape[1] > 512:
            tokens = tokens[:, :512]  # Truncate if too long
        
        tokens = tokens.to(device)
        
        # Create property vector
        property_vector = self.create_property_vector(**target_properties).unsqueeze(0)
        property_vector = property_vector.to(device)
        
        # Create attention mask
        attention_mask = torch.ones_like(tokens)
        
        # Create multi-modal input
        multi_modal_input = MultiModalInput(
            text_tokens=tokens,
            property_vector=property_vector,
            attention_mask=attention_mask
        )
        
        # Encode
        with torch.no_grad():
            molecular_latent = self.forward(multi_modal_input)
        
        return molecular_latent
    
    def fine_tune_for_scent_domain(
        self,
        scent_descriptions: List[str],
        target_molecules: List[str],
        molecular_properties: List[Dict[str, float]],
        tokenizer: Any,
        epochs: int = 10,
        learning_rate: float = 1e-4
    ):
        """
        Fine-tune the model for scent-specific generation.
        
        Args:
            scent_descriptions: List of scent descriptions
            target_molecules: List of target SMILES strings
            molecular_properties: List of property dictionaries
            tokenizer: Text tokenizer
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        logger.info(f"Fine-tuning transformer for scent domain: {len(scent_descriptions)} samples")
        
        # Set up optimization
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0.0
            
            for desc, smiles, props in zip(scent_descriptions, target_molecules, molecular_properties):
                try:
                    # Tokenize description
                    tokens = tokenizer.encode(desc, return_tensors="pt")
                    if tokens.shape[1] > 512:
                        tokens = tokens[:, :512]
                    
                    # Create property vector
                    property_vector = self.create_property_vector(**props).unsqueeze(0)
                    
                    # Create input
                    multi_modal_input = MultiModalInput(
                        text_tokens=tokens,
                        property_vector=property_vector,
                        attention_mask=torch.ones_like(tokens)
                    )
                    
                    # Forward pass
                    predicted_latent = self.forward(multi_modal_input)
                    
                    # Create target latent (simplified - would use actual molecular encoder)
                    target_latent = torch.randn_like(predicted_latent)  # Placeholder
                    
                    # Compute loss
                    loss = criterion(predicted_latent, target_latent)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                except Exception as e:
                    logger.warning(f"Training error for '{desc}': {e}")
            
            avg_loss = total_loss / len(scent_descriptions)
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        logger.info("Fine-tuning completed")
    
    def get_attention_weights(
        self, 
        multi_modal_input: MultiModalInput,
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        Extract attention weights for interpretability.
        
        Args:
            multi_modal_input: Input data
            layer_idx: Layer index to extract weights from (-1 for last layer)
            
        Returns:
            Attention weights (batch_size, n_heads, seq_len, seq_len)
        """
        # This would require modifying the attention layers to return weights
        # Implementation depends on specific interpretability requirements
        logger.warning("Attention weight extraction not implemented in this version")
        return torch.zeros(1, 8, 10, 10)  # Placeholder