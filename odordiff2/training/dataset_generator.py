"""
Dataset generation and management for OdorDiff-2 training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import random
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import json
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from tqdm import tqdm
import requests
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass 
class DatasetConfig:
    """Configuration for dataset generation."""
    max_smiles_length: int = 100
    fingerprint_size: int = 2048
    odor_property_dim: int = 128
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    min_molecular_weight: float = 50.0
    max_molecular_weight: float = 500.0
    include_stereochemistry: bool = True


class FragranceDataset(Dataset):
    """Dataset class for fragrance molecules and descriptions."""
    
    def __init__(self, 
                 data_path: Optional[str] = None,
                 config: Optional[DatasetConfig] = None,
                 text_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        
        self.config = config or DatasetConfig()
        self.text_encoder_model = text_encoder_model
        
        # Initialize text encoder
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_model)
        self.text_encoder = AutoModel.from_pretrained(text_encoder_model)
        
        # Initialize SMILES vocabulary
        self.smiles_vocab = self._build_smiles_vocabulary()
        
        # Load or generate data
        if data_path and Path(data_path).exists():
            self.data = self._load_data(data_path)
        else:
            logger.warning("No data path provided or file not found. Use generate_synthetic_data() to create dataset.")
            self.data = []
    
    def _build_smiles_vocabulary(self) -> Dict[str, int]:
        """Build comprehensive SMILES vocabulary."""
        vocab = {
            'PAD': 0, 'START': 1, 'END': 2, 'UNK': 3,
            # Atoms
            'C': 4, 'c': 5, 'N': 6, 'n': 7, 'O': 8, 'o': 9,
            'S': 10, 's': 11, 'P': 12, 'B': 13, 'F': 14, 'Cl': 15, 'Br': 16, 'I': 17,
            # Bonds and structure
            '(': 18, ')': 19, '[': 20, ']': 21, '=': 22, '#': 23, '-': 24, '+': 25,
            # Numbers
            '1': 26, '2': 27, '3': 28, '4': 29, '5': 30, '6': 31, '7': 32, '8': 33, '9': 34, '0': 35,
            # Special characters
            '@': 36, 'H': 37, '%': 38, '/': 39, '\\': 40, '.': 41, ':': 42
        }
        return vocab
    
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load data from file."""
        logger.info(f"Loading data from {data_path}")
        
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data = json.load(f)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            data = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        logger.info(f"Loaded {len(data)} samples")
        return data
    
    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Encode text descriptions to embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt",
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings
    
    def tokenize_smiles(self, smiles: str) -> List[int]:
        """Tokenize SMILES string to integer sequence."""
        tokens = [self.smiles_vocab['START']]
        
        for char in smiles:
            token_id = self.smiles_vocab.get(char, self.smiles_vocab['UNK'])
            tokens.append(token_id)
        
        tokens.append(self.smiles_vocab['END'])
        
        # Pad or truncate
        if len(tokens) > self.config.max_smiles_length:
            tokens = tokens[:self.config.max_smiles_length]
            tokens[-1] = self.smiles_vocab['END']  # Ensure END token
        else:
            tokens.extend([self.smiles_vocab['PAD']] * (self.config.max_smiles_length - len(tokens)))
        
        return tokens
    
    def compute_molecular_fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        """Compute molecular fingerprint from SMILES."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Compute Morgan fingerprint
            fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=self.config.fingerprint_size
            )
            
            return np.array(fingerprint, dtype=np.float32)
        
        except Exception as e:
            logger.warning(f"Error computing fingerprint for {smiles}: {e}")
            return None
    
    def extract_odor_properties(self, odor_description: str, smiles: str) -> np.ndarray:
        """Extract odor properties from description and molecular structure."""
        # This would use trained models in production
        # For now, create synthetic properties based on keywords
        
        properties = np.zeros(self.config.odor_property_dim, dtype=np.float32)
        
        # Basic property extraction based on keywords
        description_lower = odor_description.lower()
        
        # Intensity (first component)
        if any(word in description_lower for word in ['strong', 'intense', 'powerful']):
            properties[0] = 0.8 + 0.2 * random.random()
        elif any(word in description_lower for word in ['light', 'subtle', 'delicate']):
            properties[0] = 0.1 + 0.3 * random.random()
        else:
            properties[0] = 0.4 + 0.4 * random.random()
        
        # Longevity (second component)
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mw = Descriptors.MolWt(mol)
                # Heavier molecules typically last longer
                properties[1] = min(1.0, mw / 300.0) * (0.5 + 0.5 * random.random())
        except:
            properties[1] = 0.5 * random.random()
        
        # Sillage (third component)
        if any(word in description_lower for word in ['projecting', 'diffusive', 'radiant']):
            properties[2] = 0.7 + 0.3 * random.random()
        else:
            properties[2] = 0.3 + 0.4 * random.random()
        
        # Fill remaining dimensions with correlated noise
        for i in range(3, self.config.odor_property_dim):
            base_value = properties[i % 3]  # Correlate with basic properties
            noise = 0.2 * (random.random() - 0.5)  # Add noise
            properties[i] = max(0, min(1, base_value + noise))
        
        return properties
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = self.data[idx]
        
        # Extract components
        description = sample['description']
        smiles = sample['smiles']
        
        # Process text
        text_embedding = self.encode_text(description).squeeze(0)
        
        # Process SMILES
        smiles_tokens = torch.tensor(self.tokenize_smiles(smiles), dtype=torch.long)
        
        # Compute fingerprint
        fingerprint = self.compute_molecular_fingerprint(smiles)
        if fingerprint is None:
            fingerprint = np.random.rand(self.config.fingerprint_size).astype(np.float32)
        
        # Extract odor properties
        odor_properties = self.extract_odor_properties(
            sample.get('odor_description', description), 
            smiles
        )
        
        return {
            'text_embeddings': text_embedding,
            'smiles_tokens': smiles_tokens,
            'molecular_fingerprints': torch.tensor(fingerprint, dtype=torch.float32),
            'odor_properties': torch.tensor(odor_properties, dtype=torch.float32),
            'description': description,
            'smiles': smiles
        }


class SyntheticDatasetGenerator:
    """Generator for synthetic fragrance training data."""
    
    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        
        # Predefined fragrance categories and templates
        self.fragrance_categories = {
            'floral': {
                'descriptions': [
                    'delicate rose petals with morning dew',
                    'jasmine flowers blooming at sunset',
                    'lavender fields swaying in summer breeze',
                    'peony and lily white bouquet',
                    'violet leaves with soft iris heart',
                    'magnolia blossoms in spring garden',
                    'tuberose and gardenia night blooms',
                    'freesia with soft powdery finish'
                ],
                'smiles_templates': [
                    'CC(C)=CCO',  # Linalool-like
                    'CC(C)=CCCC(C)=CCO',  # Geraniol-like
                    'CC(C)(O)CCO',  # Diols
                    'c1ccc(cc1)CO',  # Benzyl alcohol derivatives
                    'CC(=O)OCC(C)=C'  # Acetates
                ]
            },
            'citrus': {
                'descriptions': [
                    'fresh lemon zest with bergamot sparkle',
                    'orange grove in morning sunshine',
                    'lime and grapefruit juice blend',
                    'mandarin peel with yuzu brightness',
                    'bitter orange with petitgrain leaves',
                    'blood orange with spicy notes',
                    'lemon verbena with green freshness'
                ],
                'smiles_templates': [
                    'CC(C)=CC',  # Limonene-like
                    'C=C(C)CCC=C(C)C',  # Monoterpenes
                    'CC1=CCC(CC1)C(C)C',  # p-Cymene-like
                    'CC(C)C1=CC=C(C=C1)C',  # Terpinene-like
                    'C1=CC(=CC=C1CC=C)O'  # Eugenol-like
                ]
            },
            'woody': {
                'descriptions': [
                    'sandalwood heart with cedar base',
                    'aged oak barrel with vanilla hints',
                    'pine forest after rain shower',
                    'driftwood warming in ocean breeze',
                    'patchouli earth with amber glow',
                    'cedar chest with rose undertones',
                    'birch tar with smoky leather'
                ],
                'smiles_templates': [
                    'CC(C)(C)c1ccccc1',  # tert-Butylbenzene-like
                    'c1ccc2c(c1)cccc2',  # Naphthalene-like
                    'CC12CCC3C(C1)CCC4=CCCC34C2',  # Steroid-like
                    'COc1ccc2c(c1)ccc(c2)C',  # Methylated aromatics
                    'CC1=CC(=CC=C1)C(C)(C)C'  # Substituted aromatics
                ]
            },
            'fresh': {
                'descriptions': [
                    'ocean breeze with salt crystal',
                    'morning dew on grass blades',
                    'clean laundry drying in wind',
                    'cucumber water with mint leaves',
                    'rain on hot concrete pavement',
                    'alpine air with snow crystals',
                    'waterfall mist in forest clearing'
                ],
                'smiles_templates': [
                    'CCO',  # Simple alcohols
                    'CCCO',  # Propanol
                    'CC(C)O',  # Isopropanol
                    'C=CCO',  # Allyl alcohol
                    'CCCCO',  # Butanol
                    'CC(O)CO',  # Glycols
                    'OCCCO'  # Propylene glycol
                ]
            },
            'spicy': {
                'descriptions': [
                    'black pepper with cardamom warmth',
                    'cinnamon bark with clove spice',
                    'nutmeg powder with allspice berry',
                    'ginger root with pink peppercorn',
                    'star anise with fennel seeds',
                    'paprika heat with cayenne fire',
                    'sichuan pepper with long pepper'
                ],
                'smiles_templates': [
                    'COc1ccccc1',  # Anisole-like
                    'c1ccc(cc1)C=O',  # Benzaldehyde-like
                    'CC(=O)c1ccccc1',  # Acetophenone-like
                    'C=CCc1ccccc1',  # Allyl benzene-like
                    'COc1cc(cc(c1O)OC)C=O'  # Vanillin-like
                ]
            },
            'fruity': {
                'descriptions': [
                    'ripe peach with apricot sweetness',
                    'red apple with pear freshness',
                    'tropical pineapple with mango',
                    'berry medley with blackcurrant',
                    'grape cluster with fig richness',
                    'cherry blossom with plum juice',
                    'melon slice with watermelon cool'
                ],
                'smiles_templates': [
                    'CC(=O)OC',  # Methyl acetate-like
                    'CC(=O)OCC',  # Ethyl acetate-like
                    'CC(=O)OCCC',  # Propyl acetate-like
                    'CC(C)(C)C(=O)OC',  # Branched esters
                    'C=CC(=O)OC'  # Unsaturated esters
                ]
            }
        }
    
    def generate_variation(self, base_smiles: str) -> str:
        """Generate structural variation of base SMILES."""
        try:
            mol = Chem.MolFromSmiles(base_smiles)
            if mol is None:
                return base_smiles
            
            # Simple structural modifications
            modifications = [
                lambda s: s,  # No change
                lambda s: s.replace('C', 'CC', 1) if 'C' in s else s,  # Add methyl
                lambda s: s.replace('CCO', 'CCCO') if 'CCO' in s else s,  # Extend chain
                lambda s: s.replace('C=C', 'CC') if 'C=C' in s else s,  # Saturate double bond
                lambda s: s.replace('c1ccccc1', 'c1ccc(cc1)C') if 'c1ccccc1' in s else s,  # Add substituent
            ]
            
            modified = random.choice(modifications)(base_smiles)
            
            # Validate the modified SMILES
            if Chem.MolFromSmiles(modified) is not None:
                return modified
            else:
                return base_smiles
                
        except Exception:
            return base_smiles
    
    def enhance_description(self, base_description: str, category: str) -> str:
        """Enhance description with additional sensory details."""
        enhancements = {
            'floral': [
                'with silky petals', 'in morning light', 'with honey undertones',
                'dancing in breeze', 'with dewdrop freshness', 'in full bloom'
            ],
            'citrus': [
                'with sparkling brightness', 'bursting with juice', 'with zesty tang',
                'in summer heat', 'with effervescent joy', 'with sunny warmth'
            ],
            'woody': [
                'with earthy depth', 'aged to perfection', 'with resinous warmth',
                'in ancient forest', 'with smoky undertones', 'with noble character'
            ],
            'fresh': [
                'with crystal clarity', 'in pure air', 'with cooling effect',
                'like morning mist', 'with transparent beauty', 'infinitely clean'
            ],
            'spicy': [
                'with warming heat', 'exotic and mysterious', 'with fiery passion',
                'from distant lands', 'with tingling sensation', 'bold and adventurous'
            ],
            'fruity': [
                'with juicy sweetness', 'ripe and luscious', 'with natural sugar',
                'from summer orchard', 'with mouth-watering appeal', 'bursting with flavor'
            ]
        }
        
        enhancement = random.choice(enhancements.get(category, ['with unique character']))
        return f"{base_description} {enhancement}"
    
    def generate_synthetic_dataset(self, 
                                  num_samples: int = 10000,
                                  save_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate synthetic training dataset."""
        logger.info(f"Generating {num_samples} synthetic fragrance samples...")
        
        dataset = []
        
        for i in tqdm(range(num_samples), desc="Generating samples"):
            # Select random category
            category = random.choice(list(self.fragrance_categories.keys()))
            cat_data = self.fragrance_categories[category]
            
            # Select and enhance description
            base_description = random.choice(cat_data['descriptions'])
            description = self.enhance_description(base_description, category)
            
            # Select and vary SMILES
            base_smiles = random.choice(cat_data['smiles_templates'])
            smiles = self.generate_variation(base_smiles)
            
            # Validate molecular weight
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    mw = Descriptors.MolWt(mol)
                    if not (self.config.min_molecular_weight <= mw <= self.config.max_molecular_weight):
                        continue
            except:
                continue
            
            # Create sample
            sample = {
                'id': i,
                'description': description,
                'smiles': smiles,
                'category': category,
                'odor_description': description,
                'molecular_weight': mw if 'mol' in locals() and mol else None
            }
            
            dataset.append(sample)
        
        logger.info(f"Generated {len(dataset)} valid samples")
        
        # Save if requested
        if save_path:
            self.save_dataset(dataset, save_path)
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]], save_path: str):
        """Save dataset to file."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        if save_path.endswith('.json'):
            with open(save_path, 'w') as f:
                json.dump(dataset, f, indent=2)
        elif save_path.endswith('.csv'):
            df = pd.DataFrame(dataset)
            df.to_csv(save_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {save_path}")
        
        logger.info(f"Dataset saved to {save_path}")
    
    def create_train_val_test_split(self, 
                                   dataset: List[Dict[str, Any]], 
                                   save_dir: str) -> Tuple[List, List, List]:
        """Split dataset into train/validation/test sets."""
        random.shuffle(dataset)
        
        total_size = len(dataset)
        train_size = int(total_size * self.config.train_split)
        val_size = int(total_size * self.config.val_split)
        
        train_data = dataset[:train_size]
        val_data = dataset[train_size:train_size + val_size]
        test_data = dataset[train_size + val_size:]
        
        # Save splits
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_dataset(train_data, save_dir / "train.json")
        self.save_dataset(val_data, save_dir / "val.json")
        self.save_dataset(test_data, save_dir / "test.json")
        
        logger.info(f"Dataset split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return train_data, val_data, test_data


def create_dataloaders(train_dataset: FragranceDataset,
                      val_dataset: Optional[FragranceDataset] = None,
                      batch_size: int = 32,
                      num_workers: int = 4) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create train and validation dataloaders."""
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
    
    return train_loader, val_loader


# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic dataset
    generator = SyntheticDatasetGenerator()
    synthetic_data = generator.generate_synthetic_dataset(num_samples=1000)
    
    # Create dataset
    config = DatasetConfig()
    dataset = FragranceDataset(config=config)
    dataset.data = synthetic_data
    
    # Test dataset
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print("Sample keys:", list(sample.keys()))
    print("Text embedding shape:", sample['text_embeddings'].shape)
    print("SMILES tokens shape:", sample['smiles_tokens'].shape)
    print("Fingerprint shape:", sample['molecular_fingerprints'].shape)
    print("Odor properties shape:", sample['odor_properties'].shape)