"""
Neural diffusion model trainer for OdorDiff-2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import json
import random
import wandb
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for diffusion training."""
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 100
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    save_every: int = 10
    eval_every: int = 5
    checkpoint_dir: str = "checkpoints"
    use_wandb: bool = False
    scheduler_type: str = "cosine"  # linear, cosine, polynomial


class NeuralOdorPredictor(nn.Module):
    """Neural network for odor property prediction."""
    
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, output_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 4, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, molecular_fingerprints: torch.Tensor) -> torch.Tensor:
        return self.network(molecular_fingerprints)


class EnhancedMolecularDecoder(nn.Module):
    """Enhanced molecular decoder with attention and better architecture."""
    
    def __init__(self, latent_dim: int = 512, vocab_size: int = 100, hidden_dim: int = 256, 
                 num_layers: int = 3, dropout: float = 0.1, use_attention: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Enhanced SMILES vocabulary
        self.vocab = {
            'PAD': 0, 'START': 1, 'END': 2, 'UNK': 3,
            'C': 4, 'c': 5, 'N': 6, 'n': 7, 'O': 8, 'o': 9,
            'S': 10, 's': 11, 'P': 12, 'F': 13, 'Cl': 14, 'Br': 15, 'I': 16,
            '(': 17, ')': 18, '[': 19, ']': 20, '=': 21, '#': 22, '-': 23, '+': 24,
            '1': 25, '2': 26, '3': 27, '4': 28, '5': 29, '6': 30, '7': 31, '8': 32, '9': 33, '0': 34,
            '@': 35, 'H': 36, '%': 37, '/': 38, '\\': 39, '.': 40
        }
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Architecture components
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.positional_encoding = self._create_positional_encoding(512, hidden_dim)
        
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Transformer decoder layers
        if use_attention:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        else:
            self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, 
                                 batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, vocab_size)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def _create_positional_encoding(self, max_seq_len: int, d_model: int) -> torch.Tensor:
        """Create positional encoding for transformer."""
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, latent: torch.Tensor, target_sequence: Optional[torch.Tensor] = None, 
                max_length: int = 100, temperature: float = 1.0) -> torch.Tensor:
        """Forward pass with teacher forcing during training."""
        batch_size = latent.size(0)
        device = latent.device
        
        # Project latent
        memory = self.latent_proj(latent).unsqueeze(1)  # (B, 1, hidden_dim)
        
        if target_sequence is not None:  # Training mode with teacher forcing
            # Embed target sequence
            embedded = self.embedding(target_sequence[:, :-1])  # Remove last token
            seq_len = embedded.size(1)
            
            # Add positional encoding
            pos_enc = self.positional_encoding[:, :seq_len, :].to(device)
            embedded = embedded + pos_enc
            embedded = self.dropout(embedded)
            
            if self.use_attention:
                # Create causal mask
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
                output = self.decoder(embedded, memory, tgt_mask=tgt_mask)
            else:
                # Concatenate with memory for each timestep
                memory_expanded = memory.expand(-1, seq_len, -1)
                decoder_input = torch.cat([embedded, memory_expanded], dim=-1)
                output, _ = self.decoder(decoder_input)
            
            logits = self.output_proj(output)
            return logits
        
        else:  # Inference mode
            sequence = torch.full((batch_size, 1), self.vocab['START'], device=device)
            outputs = []
            
            hidden = None
            
            for step in range(max_length):
                # Embed current sequence
                embedded = self.embedding(sequence[:, -1:])
                pos_enc = self.positional_encoding[:, step:step+1, :].to(device)
                embedded = embedded + pos_enc
                embedded = self.dropout(embedded)
                
                if self.use_attention:
                    # Use only the last token for inference
                    output = self.decoder(embedded, memory)
                else:
                    # Concatenate with memory
                    decoder_input = torch.cat([embedded, memory], dim=-1)
                    output, hidden = self.decoder(decoder_input, hidden)
                
                logits = self.output_proj(output)
                
                # Apply temperature scaling
                logits = logits / temperature
                
                # Sample next token
                probs = F.softmax(logits.squeeze(1), dim=-1)
                next_token = torch.multinomial(probs, 1)
                sequence = torch.cat([sequence, next_token], dim=1)
                outputs.append(logits.squeeze(1))
                
                # Stop if all sequences have END token
                if (next_token == self.vocab['END']).all():
                    break
            
            return torch.stack(outputs, dim=1) if outputs else torch.zeros(batch_size, 0, self.vocab_size, device=device)
    
    def decode_to_smiles(self, sequence: torch.Tensor) -> List[str]:
        """Convert token sequences back to SMILES strings."""
        smiles_list = []
        
        for seq in sequence:
            tokens = []
            for token_id in seq:
                if isinstance(token_id, torch.Tensor):
                    token_id = token_id.item()
                    
                token = self.reverse_vocab.get(token_id, 'UNK')
                if token == 'END':
                    break
                elif token not in ['PAD', 'START', 'UNK']:
                    tokens.append(token)
            
            smiles_list.append(''.join(tokens))
        
        return smiles_list


class DiffusionTrainer:
    """Enhanced trainer for diffusion models with comprehensive features."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if requested
        if config.use_wandb:
            wandb.init(project="odordiff2-training", config=vars(config))
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
    def setup_model_and_optimizers(self, text_encoder_dim: int = 384):
        """Initialize model components and optimizers."""
        
        # Initialize enhanced models
        self.molecular_decoder = EnhancedMolecularDecoder(
            latent_dim=text_encoder_dim,
            vocab_size=len(EnhancedMolecularDecoder().vocab),
            hidden_dim=512,
            num_layers=4,
            use_attention=True
        ).to(self.device)
        
        self.odor_predictor = NeuralOdorPredictor(
            input_dim=2048,
            hidden_dim=512,
            output_dim=128
        ).to(self.device)
        
        # Setup optimizers with different learning rates
        self.decoder_optimizer = torch.optim.AdamW(
            self.molecular_decoder.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.odor_optimizer = torch.optim.AdamW(
            self.odor_predictor.parameters(),
            lr=self.config.learning_rate * 0.5,  # Lower LR for odor predictor
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Setup learning rate schedulers
        if self.config.scheduler_type == "cosine":
            self.decoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.decoder_optimizer, T_max=self.config.epochs
            )
            self.odor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.odor_optimizer, T_max=self.config.epochs
            )
        elif self.config.scheduler_type == "linear":
            self.decoder_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.decoder_optimizer, start_factor=1.0, end_factor=0.1, total_iters=self.config.epochs
            )
            self.odor_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.odor_optimizer, start_factor=1.0, end_factor=0.1, total_iters=self.config.epochs
            )
        
        logger.info(f"Model initialized on {self.device}")
        logger.info(f"Molecular decoder parameters: {sum(p.numel() for p in self.molecular_decoder.parameters()):,}")
        logger.info(f"Odor predictor parameters: {sum(p.numel() for p in self.odor_predictor.parameters()):,}")
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.molecular_decoder.train()
        self.odor_predictor.train()
        
        total_decoder_loss = 0.0
        total_odor_loss = 0.0
        total_samples = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            text_embeddings = batch['text_embeddings'].to(self.device)
            smiles_sequences = batch['smiles_tokens'].to(self.device)
            odor_properties = batch['odor_properties'].to(self.device)
            molecular_fingerprints = batch['molecular_fingerprints'].to(self.device)
            
            batch_size = text_embeddings.size(0)
            total_samples += batch_size
            
            # Train molecular decoder
            self.decoder_optimizer.zero_grad()
            
            # Forward pass with teacher forcing
            decoder_output = self.molecular_decoder(text_embeddings, smiles_sequences)
            
            # Compute sequence loss (cross-entropy)
            decoder_loss = F.cross_entropy(
                decoder_output.view(-1, decoder_output.size(-1)),
                smiles_sequences[:, 1:].contiguous().view(-1),  # Target is shifted by 1
                ignore_index=0,  # Ignore padding tokens
                label_smoothing=0.1
            )
            
            decoder_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.molecular_decoder.parameters(), self.config.grad_clip_norm)
            self.decoder_optimizer.step()
            
            # Train odor predictor
            self.odor_optimizer.zero_grad()
            
            predicted_odors = self.odor_predictor(molecular_fingerprints)
            odor_loss = F.mse_loss(predicted_odors, odor_properties)
            
            # Add L1 regularization for sparsity
            l1_reg = sum(torch.norm(param, 1) for param in self.odor_predictor.parameters())
            odor_loss += 1e-5 * l1_reg
            
            odor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.odor_predictor.parameters(), self.config.grad_clip_norm)
            self.odor_optimizer.step()
            
            # Update metrics
            total_decoder_loss += decoder_loss.item() * batch_size
            total_odor_loss += odor_loss.item() * batch_size
            
            # Update progress bar
            progress_bar.set_postfix({
                'decoder_loss': f"{decoder_loss.item():.4f}",
                'odor_loss': f"{odor_loss.item():.4f}",
                'lr': f"{self.decoder_optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log to wandb if enabled
            if self.config.use_wandb and batch_idx % 100 == 0:
                wandb.log({
                    'train/decoder_loss': decoder_loss.item(),
                    'train/odor_loss': odor_loss.item(),
                    'train/learning_rate': self.decoder_optimizer.param_groups[0]['lr'],
                    'train/step': self.step
                })
            
            self.step += 1
        
        # Compute average losses
        avg_decoder_loss = total_decoder_loss / total_samples
        avg_odor_loss = total_odor_loss / total_samples
        
        return {
            'decoder_loss': avg_decoder_loss,
            'odor_loss': avg_odor_loss,
            'total_loss': avg_decoder_loss + avg_odor_loss
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.molecular_decoder.eval()
        self.odor_predictor.eval()
        
        total_decoder_loss = 0.0
        total_odor_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                text_embeddings = batch['text_embeddings'].to(self.device)
                smiles_sequences = batch['smiles_tokens'].to(self.device)
                odor_properties = batch['odor_properties'].to(self.device)
                molecular_fingerprints = batch['molecular_fingerprints'].to(self.device)
                
                batch_size = text_embeddings.size(0)
                total_samples += batch_size
                
                # Decoder validation
                decoder_output = self.molecular_decoder(text_embeddings, smiles_sequences)
                decoder_loss = F.cross_entropy(
                    decoder_output.view(-1, decoder_output.size(-1)),
                    smiles_sequences[:, 1:].contiguous().view(-1),
                    ignore_index=0
                )
                
                # Odor predictor validation
                predicted_odors = self.odor_predictor(molecular_fingerprints)
                odor_loss = F.mse_loss(predicted_odors, odor_properties)
                
                total_decoder_loss += decoder_loss.item() * batch_size
                total_odor_loss += odor_loss.item() * batch_size
        
        avg_decoder_loss = total_decoder_loss / total_samples
        avg_odor_loss = total_odor_loss / total_samples
        
        return {
            'val_decoder_loss': avg_decoder_loss,
            'val_odor_loss': avg_odor_loss,
            'val_total_loss': avg_decoder_loss + avg_odor_loss
        }
    
    def train(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None):
        """Main training loop."""
        logger.info(f"Starting training for {self.config.epochs} epochs")
        
        training_history = []
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_dataloader)
            
            # Update schedulers
            self.decoder_scheduler.step()
            self.odor_scheduler.step()
            
            # Validation
            val_metrics = {}
            if val_dataloader and epoch % self.config.eval_every == 0:
                val_metrics = self.validate(val_dataloader)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            training_history.append(epoch_metrics)
            
            # Log epoch results
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs} completed:")
            for key, value in epoch_metrics.items():
                logger.info(f"  {key}: {value:.4f}")
            
            # Log to wandb
            if self.config.use_wandb:
                wandb.log({**epoch_metrics, 'epoch': epoch})
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                is_best = val_metrics.get('val_total_loss', train_metrics['total_loss']) < self.best_loss
                if is_best:
                    self.best_loss = val_metrics.get('val_total_loss', train_metrics['total_loss'])
                
                self.save_checkpoint(
                    epoch=epoch,
                    metrics=epoch_metrics,
                    is_best=is_best
                )
        
        logger.info("Training completed!")
        return training_history
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'step': self.step,
            'molecular_decoder_state_dict': self.molecular_decoder.state_dict(),
            'odor_predictor_state_dict': self.odor_predictor.state_dict(),
            'decoder_optimizer_state_dict': self.decoder_optimizer.state_dict(),
            'odor_optimizer_state_dict': self.odor_optimizer.state_dict(),
            'decoder_scheduler_state_dict': self.decoder_scheduler.state_dict(),
            'odor_scheduler_state_dict': self.odor_scheduler.state_dict(),
            'config': vars(self.config),
            'metrics': metrics,
            'best_loss': self.best_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved at epoch {epoch + 1}")
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.molecular_decoder.load_state_dict(checkpoint['molecular_decoder_state_dict'])
        self.odor_predictor.load_state_dict(checkpoint['odor_predictor_state_dict'])
        
        if 'decoder_optimizer_state_dict' in checkpoint:
            self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
            self.odor_optimizer.load_state_dict(checkpoint['odor_optimizer_state_dict'])
            self.decoder_scheduler.load_state_dict(checkpoint['decoder_scheduler_state_dict'])
            self.odor_scheduler.load_state_dict(checkpoint['odor_scheduler_state_dict'])
        
        self.epoch = checkpoint.get('epoch', 0)
        self.step = checkpoint.get('step', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.epoch + 1}, step {self.step}")