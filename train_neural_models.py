#!/usr/bin/env python3
"""
End-to-end training script for OdorDiff-2 neural models.
"""

import argparse
import logging
import torch
from pathlib import Path
import json
import wandb
from typing import Optional
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from odordiff2.training.diffusion_trainer import DiffusionTrainer, TrainingConfig
from odordiff2.training.dataset_generator import (
    SyntheticDatasetGenerator, FragranceDataset, DatasetConfig,
    create_dataloaders
)
from odordiff2.models.property_predictor import ComprehensivePropertyPredictor
from odordiff2.core.diffusion import OdorDiffusion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_datasets(args) -> tuple:
    """Setup training and validation datasets."""
    logger.info("Setting up datasets...")
    
    dataset_config = DatasetConfig(
        max_smiles_length=args.max_smiles_length,
        fingerprint_size=args.fingerprint_size,
        odor_property_dim=args.odor_property_dim,
        train_split=args.train_split,
        val_split=args.val_split
    )
    
    # Generate synthetic dataset if needed
    if not Path(args.data_dir).exists() or args.regenerate_data:
        logger.info("Generating synthetic dataset...")
        generator = SyntheticDatasetGenerator(dataset_config)
        
        # Generate full dataset
        dataset = generator.generate_synthetic_dataset(
            num_samples=args.num_samples,
            save_path=Path(args.data_dir) / "full_dataset.json"
        )
        
        # Create train/val/test splits
        train_data, val_data, test_data = generator.create_train_val_test_split(
            dataset, args.data_dir
        )
        
        logger.info(f"Dataset created: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    # Load datasets
    train_dataset = FragranceDataset(
        data_path=Path(args.data_dir) / "train.json",
        config=dataset_config
    )
    
    val_dataset = FragranceDataset(
        data_path=Path(args.data_dir) / "val.json", 
        config=dataset_config
    )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    return train_loader, val_loader, dataset_config


def train_diffusion_model(args, train_loader, val_loader):
    """Train the main diffusion model."""
    logger.info("Starting diffusion model training...")
    
    # Training configuration
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        save_every=args.save_every,
        eval_every=args.eval_every,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.use_wandb,
        scheduler_type=args.scheduler_type
    )
    
    # Initialize trainer
    trainer = DiffusionTrainer(training_config)
    trainer.setup_model_and_optimizers()
    
    # Resume from checkpoint if specified
    if args.resume_checkpoint:
        trainer.load_checkpoint(args.resume_checkpoint)
        logger.info(f"Resumed training from {args.resume_checkpoint}")
    
    # Train model
    training_history = trainer.train(train_loader, val_loader)
    
    # Save final model
    trainer.save_checkpoint(
        epoch=args.epochs - 1,
        metrics=training_history[-1] if training_history else {},
        is_best=True
    )
    
    logger.info("Diffusion model training completed!")
    return trainer


def train_property_predictors(args, train_loader, val_loader):
    """Train molecular property prediction models."""
    logger.info("Training property prediction models...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    property_predictor = ComprehensivePropertyPredictor(device)
    
    # Setup optimizers for property predictors
    property_optimizer = torch.optim.AdamW(
        property_predictor.property_predictor.parameters(),
        lr=args.learning_rate * 0.5,
        weight_decay=args.weight_decay
    )
    
    odor_optimizer = torch.optim.AdamW(
        property_predictor.odor_predictor.parameters(),
        lr=args.learning_rate * 0.5,
        weight_decay=args.weight_decay
    )
    
    synthesis_optimizer = torch.optim.AdamW(
        property_predictor.synthesis_predictor.parameters(),
        lr=args.learning_rate * 0.5,
        weight_decay=args.weight_decay
    )
    
    safety_optimizer = torch.optim.AdamW(
        property_predictor.safety_predictor.parameters(),
        lr=args.learning_rate * 0.5,
        weight_decay=args.weight_decay
    )
    
    # Training loop for property predictors
    epochs = args.epochs // 2  # Train for fewer epochs
    
    for epoch in range(epochs):
        logger.info(f"Property prediction training epoch {epoch + 1}/{epochs}")
        
        property_predictor.property_predictor.train()
        property_predictor.odor_predictor.train()
        property_predictor.synthesis_predictor.train()
        property_predictor.safety_predictor.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            fingerprints = batch['molecular_fingerprints'].to(device)
            odor_properties = batch['odor_properties'].to(device)
            
            # Train property predictor
            property_optimizer.zero_grad()
            mol_props = property_predictor.property_predictor(fingerprints)
            # Create synthetic targets for molecular properties (in real scenario, use actual data)
            synthetic_mol_targets = torch.rand_like(mol_props) 
            prop_loss = torch.nn.functional.mse_loss(mol_props, synthetic_mol_targets)
            prop_loss.backward()
            property_optimizer.step()
            
            # Train odor predictor
            odor_optimizer.zero_grad()
            odor_preds = property_predictor.odor_predictor(fingerprints)
            odor_loss = torch.nn.functional.mse_loss(
                odor_preds['intensity'], odor_properties[:, 0]
            ) + torch.nn.functional.mse_loss(
                odor_preds['longevity'], odor_properties[:, 1]
            ) + torch.nn.functional.mse_loss(
                odor_preds['sillage'], odor_properties[:, 2]
            )
            odor_loss.backward()
            odor_optimizer.step()
            
            # Train synthesis predictor
            synthesis_optimizer.zero_grad()
            synth_preds = property_predictor.synthesis_predictor(fingerprints)
            # Create synthetic synthesis targets
            synth_targets = {\n                'difficulty': torch.rand(fingerprints.size(0)).to(device),\n                'estimated_cost': torch.rand(fingerprints.size(0)).to(device) * 100,\n                'estimated_steps': torch.rand(fingerprints.size(0)).to(device) * 10\n            }
            synth_loss = sum([
                torch.nn.functional.mse_loss(synth_preds[key], synth_targets[key])
                for key in synth_targets.keys()
            ])
            synth_loss.backward()
            synthesis_optimizer.step()
            
            # Train safety predictor
            safety_optimizer.zero_grad()
            safety_preds = property_predictor.safety_predictor(fingerprints)
            # Create synthetic safety targets
            safety_targets = {
                key: torch.rand(fingerprints.size(0)).to(device) * 0.3  # Low risk targets
                for key in safety_preds.keys()
            }
            safety_loss = sum([
                torch.nn.functional.mse_loss(safety_preds[key], safety_targets[key])
                for key in safety_targets.keys()
            ])
            safety_loss.backward()
            safety_optimizer.step()
            
            batch_total_loss = prop_loss.item() + odor_loss.item() + synth_loss.item() + safety_loss.item()
            total_loss += batch_total_loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
        
        if args.use_wandb:
            wandb.log({
                'property_training/epoch': epoch,
                'property_training/avg_loss': avg_loss
            })
    
    # Mark as trained and save
    property_predictor.is_trained = True
    property_predictor.save_models(Path(args.checkpoint_dir) / "property_models")
    
    logger.info("Property prediction model training completed!")
    return property_predictor


def create_integrated_model(args, diffusion_trainer, property_predictor):
    """Create integrated OdorDiffusion model with all components."""
    logger.info("Creating integrated model...")
    
    # Create OdorDiffusion model
    model = OdorDiffusion(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        use_neural_generation=True
    )
    
    # Replace components with trained versions
    model.molecular_decoder = diffusion_trainer.molecular_decoder
    model.odor_predictor = property_predictor.odor_predictor
    model.is_trained = True
    
    # Save integrated model
    model_path = Path(args.checkpoint_dir) / "odordiff2_complete.pt"
    torch.save({
        'molecular_decoder': model.molecular_decoder.state_dict(),
        'text_encoder': model.text_encoder.state_dict(),
        'odor_predictor': model.odor_predictor.state_dict(),
        'scheduler': model.scheduler,
        'is_trained': True,
        'training_config': vars(TrainingConfig())
    }, model_path)
    
    logger.info(f"Integrated model saved to {model_path}")
    return model


def evaluate_model(model, val_loader, args):
    """Evaluate the trained model."""
    logger.info("Evaluating trained model...")
    
    # Generate sample molecules
    test_prompts = [
        "Fresh floral scent with rose and jasmine",
        "Woody cedar with vanilla undertones",
        "Citrus bergamot with lemon zest",
        "Spicy cinnamon with warm cloves",
        "Aquatic marine with sea salt"
    ]
    
    results = []
    for prompt in test_prompts:
        logger.info(f"Testing prompt: '{prompt}'")
        
        molecules = model.generate(
            prompt=prompt,
            num_molecules=3,
            temperature=0.8,
            use_neural=True
        )
        
        for i, mol in enumerate(molecules):
            result = {
                'prompt': prompt,
                'molecule_id': i,
                'smiles': mol.smiles,
                'valid': mol.is_valid,
                'confidence': mol.confidence,
                'synth_score': mol.synth_score,
                'estimated_cost': mol.estimated_cost,
                'odor_profile': {
                    'primary_notes': mol.odor_profile.primary_notes if mol.odor_profile else [],
                    'intensity': mol.odor_profile.intensity if mol.odor_profile else 0,
                    'longevity_hours': mol.odor_profile.longevity_hours if mol.odor_profile else 0
                }
            }
            results.append(result)
            
            logger.info(f"  Generated: {mol.smiles} (valid: {mol.is_valid}, conf: {mol.confidence:.2f})")
    
    # Save evaluation results
    eval_path = Path(args.checkpoint_dir) / "evaluation_results.json"
    with open(eval_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {eval_path}")


def main():
    parser = argparse.ArgumentParser(description="Train OdorDiff-2 neural models")
    
    # Dataset arguments
    parser.add_argument("--data-dir", type=str, default="data/fragrance_dataset",
                       help="Directory for dataset files")
    parser.add_argument("--num-samples", type=int, default=10000,
                       help="Number of synthetic samples to generate")
    parser.add_argument("--regenerate-data", action="store_true",
                       help="Regenerate synthetic dataset")
    parser.add_argument("--max-smiles-length", type=int, default=100,
                       help="Maximum SMILES sequence length")
    parser.add_argument("--fingerprint-size", type=int, default=2048,
                       help="Molecular fingerprint size")
    parser.add_argument("--odor-property-dim", type=int, default=128,
                       help="Odor property vector dimension")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0,
                       help="Gradient clipping norm")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                       help="Number of warmup steps")
    parser.add_argument("--scheduler-type", type=str, default="cosine",
                       choices=["linear", "cosine", "polynomial"],
                       help="Learning rate scheduler type")
    
    # Model arguments
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--resume-checkpoint", type=str,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--save-every", type=int, default=10,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--eval-every", type=int, default=5,
                       help="Evaluate every N epochs")
    
    # Data split arguments
    parser.add_argument("--train-split", type=float, default=0.8,
                       help="Training data split ratio")
    parser.add_argument("--val-split", type=float, default=0.1,
                       help="Validation data split ratio")
    
    # System arguments
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--use-wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--wandb-project", type=str, default="odordiff2",
                       help="Weights & Biases project name")
    
    # Training mode arguments
    parser.add_argument("--train-diffusion", action="store_true", default=True,
                       help="Train diffusion model")
    parser.add_argument("--train-properties", action="store_true", default=True,
                       help="Train property prediction models")
    parser.add_argument("--evaluate", action="store_true", default=True,
                       help="Evaluate trained model")
    
    args = parser.parse_args()
    
    # Create output directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize Weights & Biases if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"odordiff2-training-{args.epochs}epochs"
        )
    
    try:
        # Setup datasets
        train_loader, val_loader, dataset_config = setup_datasets(args)
        
        # Train diffusion model
        diffusion_trainer = None
        if args.train_diffusion:
            diffusion_trainer = train_diffusion_model(args, train_loader, val_loader)
        
        # Train property predictors
        property_predictor = None
        if args.train_properties:
            property_predictor = train_property_predictors(args, train_loader, val_loader)
        
        # Create integrated model
        if diffusion_trainer and property_predictor:
            integrated_model = create_integrated_model(args, diffusion_trainer, property_predictor)
            
            # Evaluate model
            if args.evaluate:
                evaluate_model(integrated_model, val_loader, args)
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        if args.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()