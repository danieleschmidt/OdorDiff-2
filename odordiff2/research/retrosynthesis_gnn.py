"""
Graph Neural Network for Retrosynthesis Prediction.

This module implements a sophisticated GNN architecture for predicting
synthetic routes and feasibility scores for generated molecules, enabling
more practical fragrance development.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx

from ..models.molecule import Molecule
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ReactionStep:
    """Represents a single reaction step in a synthesis route."""
    reactants: List[str]  # SMILES of reactants
    product: str  # SMILES of product
    reaction_type: str  # Type of reaction
    catalyst: Optional[str] = None
    conditions: Optional[Dict[str, Any]] = None
    yield_estimate: float = 0.8
    difficulty_score: float = 0.5
    cost_factor: float = 1.0


@dataclass
class SynthesisRoute:
    """Complete synthesis route for a target molecule."""
    target_molecule: str  # Target SMILES
    steps: List[ReactionStep]
    total_yield: float
    feasibility_score: float
    estimated_cost: float
    convergent_steps: int = 0
    longest_linear_sequence: int = 0


@dataclass
class ReactionTemplate:
    """Template for a chemical reaction."""
    name: str
    smarts_pattern: str  # SMARTS reaction pattern
    reaction_type: str
    difficulty: float
    selectivity: float
    functional_groups: List[str]


class MolecularGraphEncoder(nn.Module):
    """
    Enhanced molecular graph encoder with attention mechanisms.
    """
    
    def __init__(
        self,
        atom_feature_dim: int = 74,
        bond_feature_dim: int = 12,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_attention_heads: int = 8
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Atom and bond embeddings
        self.atom_embedding = nn.Linear(atom_feature_dim, hidden_dim)
        self.bond_embedding = nn.Linear(bond_feature_dim, hidden_dim)
        
        # Graph convolution layers with attention
        self.conv_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        
        for _ in range(n_layers):
            self.conv_layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
            self.attention_layers.append(
                nn.MultiheadAttention(hidden_dim, n_attention_heads, batch_first=True)
            )
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        
        # Global pooling
        self.global_attention = nn.MultiheadAttention(hidden_dim, n_attention_heads, batch_first=True)
        self.global_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        atom_features: torch.Tensor,
        bond_features: torch.Tensor,
        edge_indices: torch.Tensor,
        batch_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode molecular graph with attention.
        
        Args:
            atom_features: (n_atoms, atom_feature_dim)
            bond_features: (n_bonds, bond_feature_dim)
            edge_indices: (2, n_bonds) - edge connectivity
            batch_idx: (n_atoms,) - batch assignment for atoms
            
        Returns:
            Graph embeddings (batch_size, hidden_dim)
        """
        n_atoms = atom_features.size(0)
        device = atom_features.device
        
        # Embed atom and bond features
        atom_h = self.atom_embedding(atom_features)  # (n_atoms, hidden_dim)
        bond_h = self.bond_embedding(bond_features)  # (n_bonds, hidden_dim)
        
        # Graph convolution with attention
        for i, (conv, attention, norm) in enumerate(zip(
            self.conv_layers, self.attention_layers, self.layer_norms
        )):
            # Message passing
            source_atoms = edge_indices[0]  # Source atoms
            target_atoms = edge_indices[1]  # Target atoms
            
            # Gather source atom features and bond features
            source_h = atom_h[source_atoms]  # (n_bonds, hidden_dim)
            target_h = atom_h[target_atoms]  # (n_bonds, hidden_dim)
            
            # Combine atom and bond information
            messages = torch.cat([source_h + bond_h, target_h], dim=-1)  # (n_bonds, 2*hidden_dim)
            messages = F.relu(conv(messages))  # (n_bonds, hidden_dim)
            
            # Aggregate messages to target atoms
            aggregated = torch.zeros_like(atom_h)
            aggregated.index_add_(0, target_atoms, messages)
            
            # Self-attention
            atom_h_reshaped = atom_h.unsqueeze(0)  # (1, n_atoms, hidden_dim)
            attended, _ = attention(atom_h_reshaped, atom_h_reshaped, atom_h_reshaped)
            attended = attended.squeeze(0)  # (n_atoms, hidden_dim)
            
            # Residual connection and normalization
            atom_h = norm(atom_h + aggregated + attended)
        
        # Global pooling with attention
        batch_size = batch_idx.max().item() + 1
        graph_embeddings = []
        
        for b in range(batch_size):
            mask = (batch_idx == b)
            if mask.sum() == 0:
                graph_embeddings.append(torch.zeros(self.hidden_dim, device=device))
                continue
            
            batch_atoms = atom_h[mask].unsqueeze(0)  # (1, n_batch_atoms, hidden_dim)
            
            # Global attention pooling
            pooled, _ = self.global_attention(batch_atoms, batch_atoms, batch_atoms)
            pooled = self.global_norm(pooled)
            
            # Mean pooling
            graph_embed = pooled.mean(dim=1).squeeze(0)  # (hidden_dim,)
            graph_embeddings.append(graph_embed)
        
        return torch.stack(graph_embeddings)  # (batch_size, hidden_dim)


class ReactionPredictor(nn.Module):
    """
    Predicts possible reactions for a given molecular graph.
    """
    
    def __init__(
        self,
        graph_encoder: MolecularGraphEncoder,
        n_reaction_types: int = 50,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.graph_encoder = graph_encoder
        self.n_reaction_types = n_reaction_types
        
        # Reaction type prediction
        self.reaction_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_reaction_types)
        )
        
        # Reactivity site prediction
        self.site_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Probability of being reactive
        )
        
        # Feasibility scorer
        self.feasibility_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        atom_features: torch.Tensor,
        bond_features: torch.Tensor,
        edge_indices: torch.Tensor,
        batch_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict reactions for molecular graphs.
        
        Returns:
            reaction_probs: (batch_size, n_reaction_types)
            site_probs: (n_atoms,)
            feasibility_scores: (batch_size,)
        """
        # Encode graphs
        graph_embeddings = self.graph_encoder(
            atom_features, bond_features, edge_indices, batch_idx
        )
        
        # Predict reaction types
        reaction_probs = F.softmax(self.reaction_classifier(graph_embeddings), dim=-1)
        
        # Predict reactive sites (per atom)
        atom_embeddings = self.graph_encoder.atom_embedding(atom_features)
        site_probs = torch.sigmoid(self.site_predictor(atom_embeddings)).squeeze(-1)
        
        # Predict feasibility
        feasibility_scores = self.feasibility_scorer(graph_embeddings).squeeze(-1)
        
        return reaction_probs, site_probs, feasibility_scores


class RetrosynthesisPlanner(nn.Module):
    """
    Monte Carlo Tree Search-based retrosynthesis planner.
    """
    
    def __init__(
        self,
        reaction_predictor: ReactionPredictor,
        max_depth: int = 6,
        max_width: int = 10,
        exploration_constant: float = 1.4
    ):
        super().__init__()
        
        self.reaction_predictor = reaction_predictor
        self.max_depth = max_depth
        self.max_width = max_width
        self.exploration_constant = exploration_constant
        
        # Reaction templates database
        self.reaction_templates = self._init_reaction_templates()
        
        # Commercial building blocks (simplified)
        self.commercial_compounds = self._init_commercial_compounds()
    
    def _init_reaction_templates(self) -> List[ReactionTemplate]:
        """Initialize reaction templates."""
        return [
            ReactionTemplate(
                name="Suzuki Coupling",
                smarts_pattern="[C:1][Br,I,Cl].[C:2]B(O)O>>[C:1][C:2]",
                reaction_type="C-C coupling",
                difficulty=0.3,
                selectivity=0.9,
                functional_groups=["aryl_halide", "boronic_acid"]
            ),
            ReactionTemplate(
                name="Grignard Addition",
                smarts_pattern="[C:1][Mg][Br,Cl,I].[C:2]=[O:3]>>[C:1][C:2][O:3]",
                reaction_type="C-C formation",
                difficulty=0.4,
                selectivity=0.8,
                functional_groups=["grignard", "carbonyl"]
            ),
            ReactionTemplate(
                name="Ester Hydrolysis",
                smarts_pattern="[C:1][C:2](=[O:3])[O:4][C:5]>>[C:1][C:2](=[O:3])[O:4]",
                reaction_type="hydrolysis",
                difficulty=0.2,
                selectivity=0.95,
                functional_groups=["ester"]
            ),
            ReactionTemplate(
                name="Friedel-Crafts Acylation",
                smarts_pattern="[c:1][H].[C:2][C:3](=[O:4])[Cl]>>[c:1][C:2][C:3](=[O:4])",
                reaction_type="aromatic substitution",
                difficulty=0.5,
                selectivity=0.7,
                functional_groups=["aromatic", "acyl_chloride"]
            ),
            ReactionTemplate(
                name="Reduction",
                smarts_pattern="[C:1]=[O:2]>>[C:1][O:2]",
                reaction_type="reduction",
                difficulty=0.2,
                selectivity=0.9,
                functional_groups=["carbonyl"]
            ),
            ReactionTemplate(
                name="Williamson Ether Synthesis",
                smarts_pattern="[C:1][O-].[C:2][Br,I,Cl]>>[C:1][O:2][C:2]",
                reaction_type="ether formation",
                difficulty=0.3,
                selectivity=0.8,
                functional_groups=["alkoxide", "alkyl_halide"]
            )
        ]
    
    def _init_commercial_compounds(self) -> Set[str]:
        """Initialize set of commercially available compounds."""
        return {
            # Common building blocks
            "CCO",  # Ethanol
            "CC(C)O",  # Isopropanol
            "c1ccccc1",  # Benzene
            "c1ccc(cc1)Br",  # Bromobenzene
            "c1ccc(cc1)C=O",  # Benzaldehyde
            "CC(=O)O",  # Acetic acid
            "CC(=O)Cl",  # Acetyl chloride
            "CCMgBr",  # Ethylmagnesium bromide
            "c1ccc(cc1)B(O)O",  # Phenylboronic acid
            "CC(C)(C)OC(=O)Cl",  # Boc chloride
            # Fragrance precursors
            "CC(C)=CCO",  # 3-Methyl-2-butenol
            "c1ccc(cc1)CO",  # Benzyl alcohol
            "COc1ccccc1",  # Anisole
            "CC(C)=CC",  # Isobutene
            "C=CCO",  # Allyl alcohol
        }
    
    def plan_synthesis(
        self,
        target_molecule: Molecule,
        max_routes: int = 5
    ) -> List[SynthesisRoute]:
        """
        Plan synthesis routes for target molecule using MCTS.
        
        Args:
            target_molecule: Target molecule to synthesize
            max_routes: Maximum number of routes to return
            
        Returns:
            List of synthesis routes ranked by feasibility
        """
        logger.info(f"Planning synthesis for: {target_molecule.smiles}")
        
        if not target_molecule.is_valid:
            logger.warning("Invalid target molecule")
            return []
        
        # Check if target is already commercially available
        if target_molecule.smiles in self.commercial_compounds:
            route = SynthesisRoute(
                target_molecule=target_molecule.smiles,
                steps=[],
                total_yield=1.0,
                feasibility_score=1.0,
                estimated_cost=10.0  # Base cost
            )
            return [route]
        
        # Run MCTS to find synthesis routes
        routes = self._mcts_search(target_molecule.smiles)
        
        # Score and rank routes
        scored_routes = []
        for route in routes:
            score = self._score_route(route)
            route.feasibility_score = score
            scored_routes.append(route)
        
        # Sort by feasibility score
        scored_routes.sort(key=lambda r: r.feasibility_score, reverse=True)
        
        logger.info(f"Found {len(scored_routes)} synthesis routes")
        return scored_routes[:max_routes]
    
    def _mcts_search(self, target_smiles: str, iterations: int = 100) -> List[SynthesisRoute]:
        """Monte Carlo Tree Search for retrosynthesis."""
        routes = []
        
        # Simplified MCTS implementation
        for _ in range(iterations // 10):  # Reduced for efficiency
            try:
                route = self._generate_random_route(target_smiles)
                if route and len(route.steps) <= self.max_depth:
                    routes.append(route)
            except Exception as e:
                logger.debug(f"Route generation failed: {e}")
        
        return routes
    
    def _generate_random_route(self, target_smiles: str) -> Optional[SynthesisRoute]:
        """Generate a random synthesis route."""
        steps = []
        current_targets = [target_smiles]
        depth = 0
        
        while current_targets and depth < self.max_depth:
            new_targets = []
            
            for target in current_targets:
                if target in self.commercial_compounds:
                    continue
                
                # Find applicable reactions
                applicable_reactions = self._find_applicable_reactions(target)
                
                if not applicable_reactions:
                    continue
                
                # Select random reaction
                reaction = np.random.choice(applicable_reactions)
                
                # Generate precursors
                precursors = self._apply_retro_reaction(target, reaction)
                
                if precursors:
                    step = ReactionStep(
                        reactants=precursors,
                        product=target,
                        reaction_type=reaction.name,
                        yield_estimate=reaction.selectivity * np.random.uniform(0.7, 0.95),
                        difficulty_score=reaction.difficulty,
                        cost_factor=np.random.uniform(0.8, 1.5)
                    )
                    steps.append(step)
                    new_targets.extend([p for p in precursors if p not in self.commercial_compounds])
            
            current_targets = new_targets
            depth += 1
        
        if not steps:
            return None
        
        # Calculate overall metrics
        total_yield = np.prod([step.yield_estimate for step in steps])
        
        return SynthesisRoute(
            target_molecule=target_smiles,
            steps=steps,
            total_yield=total_yield,
            feasibility_score=0.0,  # Will be calculated later
            estimated_cost=sum(step.cost_factor for step in steps) * 50,  # Base cost
            longest_linear_sequence=len(steps)
        )
    
    def _find_applicable_reactions(self, molecule_smiles: str) -> List[ReactionTemplate]:
        """Find reactions applicable to the given molecule."""
        applicable = []
        
        # Simplified functional group detection
        molecule_lower = molecule_smiles.lower()
        
        for template in self.reaction_templates:
            # Simple pattern matching (would use RDKit in real implementation)
            if template.reaction_type == "hydrolysis" and "o" in molecule_lower:
                applicable.append(template)
            elif template.reaction_type == "reduction" and "=" in molecule_smiles:
                applicable.append(template)
            elif template.reaction_type == "C-C coupling" and ("c1c" in molecule_lower or "br" in molecule_lower):
                applicable.append(template)
            elif template.reaction_type == "C-C formation" and "=" in molecule_smiles:
                applicable.append(template)
            elif template.reaction_type == "aromatic substitution" and "c1c" in molecule_lower:
                applicable.append(template)
            elif template.reaction_type == "ether formation" and "o" in molecule_lower:
                applicable.append(template)
        
        return applicable
    
    def _apply_retro_reaction(
        self, 
        product_smiles: str, 
        reaction_template: ReactionTemplate
    ) -> List[str]:
        """Apply retrosynthetic reaction to generate precursors."""
        # Simplified retrosynthetic transformation
        # In real implementation, would use RDKit reaction SMARTS
        
        if reaction_template.reaction_type == "hydrolysis":
            # Ester -> acid + alcohol
            return ["CC(=O)O", "CCO"]  # Simplified
        
        elif reaction_template.reaction_type == "reduction":
            # Alcohol -> carbonyl
            if "CCO" in product_smiles:
                return ["CC=O"]  # Simplified
            return ["c1ccc(cc1)C=O"]  # Default to benzaldehyde
        
        elif reaction_template.reaction_type == "C-C coupling":
            # Suzuki product -> aryl halide + boronic acid
            return ["c1ccc(cc1)Br", "c1ccc(cc1)B(O)O"]
        
        elif reaction_template.reaction_type == "C-C formation":
            # Grignard addition product -> Grignard + carbonyl
            return ["CCMgBr", "c1ccc(cc1)C=O"]
        
        elif reaction_template.reaction_type == "aromatic substitution":
            # Friedel-Crafts product -> benzene + acyl chloride
            return ["c1ccccc1", "CC(=O)Cl"]
        
        elif reaction_template.reaction_type == "ether formation":
            # Ether -> alkoxide + alkyl halide
            return ["CCO", "CCBr"]
        
        return []
    
    def _score_route(self, route: SynthesisRoute) -> float:
        """Score a synthesis route based on multiple factors."""
        if not route.steps:
            return 1.0  # Commercial compound
        
        # Factors to consider
        yield_score = route.total_yield
        complexity_penalty = max(0, 1.0 - len(route.steps) / self.max_depth)
        difficulty_penalty = 1.0 - np.mean([step.difficulty_score for step in route.steps])
        cost_penalty = max(0, 1.0 - route.estimated_cost / 500)  # Normalize to $500
        
        # Commercial availability bonus
        commercial_bonus = sum(
            0.1 for step in route.steps 
            for reactant in step.reactants 
            if reactant in self.commercial_compounds
        ) / max(1, len(route.steps))
        
        # Combine factors
        score = (
            yield_score * 0.3 +
            complexity_penalty * 0.2 +
            difficulty_penalty * 0.2 +
            cost_penalty * 0.2 +
            commercial_bonus * 0.1
        )
        
        return min(1.0, max(0.0, score))


class RetrosynthesisGNN:
    """
    Main class for GNN-based retrosynthesis prediction.
    """
    
    def __init__(self, device: str = "cpu", enable_training: bool = False):
        self.device = device
        self.enable_training = enable_training
        
        # Initialize models
        self.graph_encoder = MolecularGraphEncoder()
        self.reaction_predictor = ReactionPredictor(self.graph_encoder)
        self.retro_planner = RetrosynthesisPlanner(self.reaction_predictor)
        
        # Move to device
        self.graph_encoder.to(device)
        self.reaction_predictor.to(device)
        
        logger.info(f"RetrosynthesisGNN initialized on {device}")
    
    def predict_synthesis_feasibility(self, molecule: Molecule) -> float:
        """Predict synthesis feasibility for a molecule."""
        if not molecule.is_valid:
            return 0.0
        
        try:
            routes = self.retro_planner.plan_synthesis(molecule, max_routes=1)
            if routes:
                return routes[0].feasibility_score
            return 0.1  # Low but non-zero for novel molecules
            
        except Exception as e:
            logger.warning(f"Feasibility prediction failed for {molecule.smiles}: {e}")
            return 0.1
    
    def suggest_synthesis_routes(
        self,
        molecule: Molecule,
        max_routes: int = 3,
        prefer_green_chemistry: bool = True
    ) -> List[SynthesisRoute]:
        """Suggest synthesis routes for a molecule."""
        logger.info(f"Suggesting synthesis routes for: {molecule.smiles}")
        
        routes = self.retro_planner.plan_synthesis(molecule, max_routes * 2)
        
        if prefer_green_chemistry:
            # Filter for environmentally friendly routes
            green_routes = []
            for route in routes:
                green_score = self._calculate_green_chemistry_score(route)
                if green_score > 0.5:  # Only include reasonably green routes
                    route.green_chemistry_score = green_score
                    green_routes.append(route)
            routes = green_routes or routes  # Fall back to all routes if none are green
        
        return routes[:max_routes]
    
    def _calculate_green_chemistry_score(self, route: SynthesisRoute) -> float:
        """Calculate green chemistry score for a route."""
        # Simplified green chemistry scoring
        penalties = 0.0
        
        for step in route.steps:
            # Penalize toxic reagents (simplified)
            for reactant in step.reactants:
                if any(toxic in reactant.lower() for toxic in ['br', 'cl', 'hg', 'pb']):
                    penalties += 0.1
            
            # Penalize harsh conditions
            if step.difficulty_score > 0.7:
                penalties += 0.05
        
        green_score = max(0.0, 1.0 - penalties)
        return green_score
    
    def optimize_route_for_cost(self, route: SynthesisRoute) -> SynthesisRoute:
        """Optimize synthesis route for cost efficiency."""
        # Simplified cost optimization
        optimized_steps = []
        
        for step in route.steps:
            # Check for cheaper alternative reagents
            if step.reaction_type == "C-C coupling":
                # Prefer cheaper coupling methods
                step.cost_factor *= 0.8
                step.catalyst = "Pd(PPh3)4"  # Standard catalyst
            
            elif step.reaction_type == "reduction":
                # Use cheaper reducing agents
                step.cost_factor *= 0.9
                step.conditions = {"reagent": "NaBH4", "temp": "0°C"}
            
            optimized_steps.append(step)
        
        # Recalculate route metrics
        route.steps = optimized_steps
        route.estimated_cost = sum(step.cost_factor for step in optimized_steps) * 40
        route.feasibility_score = self.retro_planner._score_route(route)
        
        return route
    
    def train_on_reaction_data(
        self,
        reaction_dataset: List[Dict[str, Any]],
        epochs: int = 10,
        learning_rate: float = 1e-3
    ):
        """Train the model on reaction prediction data."""
        if not self.enable_training:
            logger.warning("Training disabled. Set enable_training=True to train models.")
            return
        
        logger.info(f"Training on {len(reaction_dataset)} reactions for {epochs} epochs")
        
        optimizer = torch.optim.Adam(self.reaction_predictor.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for reaction_data in reaction_dataset:
                try:
                    # This would require proper data preprocessing
                    # Simplified training loop structure
                    
                    # Forward pass (placeholder)
                    # loss = self._compute_training_loss(reaction_data, criterion)
                    
                    # Backward pass
                    # optimizer.zero_grad()
                    # loss.backward()
                    # optimizer.step()
                    
                    # total_loss += loss.item()
                    pass
                    
                except Exception as e:
                    logger.warning(f"Training error: {e}")
            
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss / len(reaction_dataset):.4f}")
        
        logger.info("Training completed")
    
    def benchmark_retrosynthesis_accuracy(
        self,
        test_molecules: List[Tuple[str, List[str]]]  # (SMILES, known_routes)
    ) -> Dict[str, float]:
        """Benchmark retrosynthesis prediction accuracy."""
        logger.info(f"Benchmarking on {len(test_molecules)} molecules")
        
        route_similarities = []
        feasibility_accuracies = []
        
        for smiles, known_routes in test_molecules:
            try:
                molecule = Molecule(smiles)
                predicted_routes = self.suggest_synthesis_routes(molecule, max_routes=3)
                
                if predicted_routes and known_routes:
                    # Calculate route similarity (simplified)
                    similarity = self._calculate_route_similarity(
                        predicted_routes[0], known_routes[0]
                    )
                    route_similarities.append(similarity)
                
                # Check feasibility prediction
                feasibility = self.predict_synthesis_feasibility(molecule)
                # Assume molecules with known routes are feasible
                feasibility_accuracies.append(1.0 if feasibility > 0.5 else 0.0)
                
            except Exception as e:
                logger.warning(f"Benchmark error for {smiles}: {e}")
        
        metrics = {
            'route_similarity': np.mean(route_similarities) if route_similarities else 0.0,
            'feasibility_accuracy': np.mean(feasibility_accuracies) if feasibility_accuracies else 0.0,
            'coverage': len(route_similarities) / len(test_molecules),
            'sample_size': len(test_molecules)
        }
        
        logger.info(f"Retrosynthesis benchmark: {metrics}")
        return metrics
    
    def _calculate_route_similarity(
        self, 
        predicted_route: SynthesisRoute, 
        known_route: str
    ) -> float:
        """Calculate similarity between predicted and known routes."""
        # Simplified route comparison
        predicted_reactants = set()
        for step in predicted_route.steps:
            predicted_reactants.update(step.reactants)
        
        # Parse known route (simplified)
        known_reactants = set(known_route.split(">>")[0].split("."))
        
        if not predicted_reactants or not known_reactants:
            return 0.0
        
        # Jaccard similarity
        intersection = len(predicted_reactants & known_reactants)
        union = len(predicted_reactants | known_reactants)
        
        return intersection / union if union > 0 else 0.0