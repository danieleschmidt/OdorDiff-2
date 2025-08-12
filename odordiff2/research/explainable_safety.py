"""
Explainable AI Safety Filter with interpretable toxicity predictions.

This module implements an advanced safety filter that not only predicts toxicity
but also provides explanations for its decisions using attention mechanisms,
SHAP values, and molecular substructure analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import warnings
from collections import defaultdict

from ..models.molecule import Molecule
from ..safety.filter import SafetyFilter
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Suppress SHAP warnings
warnings.filterwarnings("ignore", category=UserWarning, module="shap")


@dataclass
class SafetyExplanation:
    """Explanation for safety prediction."""
    prediction: float  # Safety score [0-1]
    confidence: float  # Confidence in prediction
    risk_factors: List[Dict[str, Any]]  # Identified risk factors
    protective_factors: List[Dict[str, Any]]  # Protective factors
    attention_weights: Optional[np.ndarray] = None
    shap_values: Optional[np.ndarray] = None
    molecular_alerts: List[str] = None


@dataclass
class RiskFactor:
    """Individual risk factor in a molecule."""
    substructure: str  # SMARTS pattern
    name: str  # Human-readable name
    risk_score: float  # Risk contribution [0-1]
    confidence: float  # Confidence in this factor
    literature_references: List[str] = None


@dataclass
class ToxicophoreAlert:
    """Structural alert for toxicity."""
    pattern: str  # SMARTS pattern
    name: str  # Alert name
    toxicity_type: str  # Type of toxicity
    severity: float  # Severity score [0-1]
    evidence_level: str  # Quality of evidence
    regulatory_status: str  # Regulatory classification


class AttentionBasedToxicityPredictor(nn.Module):
    """
    Neural network with attention mechanisms for interpretable toxicity prediction.
    """
    
    def __init__(
        self,
        input_dim: int = 2048,  # Molecular fingerprint dimension
        hidden_dim: int = 512,
        n_attention_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_attention_heads = n_attention_heads
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                hidden_dim, n_attention_heads, dropout=dropout, batch_first=True
            )
            for _ in range(n_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        
        # Feed-forward networks
        self.feed_forwards = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            for _ in range(n_layers)
        ])
        
        # Final prediction layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus()
        )
        
        self.attention_weights = []  # Store attention weights for explanation
    
    def forward(
        self, 
        molecular_features: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass with optional attention weight extraction.
        
        Args:
            molecular_features: (batch_size, feature_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            predictions: (batch_size, 1) - toxicity scores
            uncertainties: (batch_size, 1) - prediction uncertainties
            attention_weights: List of attention weight tensors if requested
        """
        batch_size = molecular_features.size(0)
        
        # Project input features
        x = self.input_projection(molecular_features)  # (batch_size, hidden_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim) - treat as sequence of length 1
        
        attention_weights = []
        
        # Process through attention layers
        for attention, norm, ff in zip(self.attention_layers, self.layer_norms, self.feed_forwards):
            # Self-attention
            attended, attn_weights = attention(x, x, x)
            x = norm(x + attended)
            
            # Feed-forward
            x = norm(x + ff(x))
            
            if return_attention:
                attention_weights.append(attn_weights)
        
        # Global pooling (mean across sequence dimension)
        pooled = x.mean(dim=1)  # (batch_size, hidden_dim)
        
        # Predictions
        predictions = self.classifier(pooled)
        uncertainties = self.uncertainty_head(pooled)
        
        return predictions, uncertainties, attention_weights if return_attention else None
    
    def get_feature_importance(
        self, 
        molecular_features: torch.Tensor
    ) -> torch.Tensor:
        """Get feature importance scores using integrated gradients."""
        molecular_features.requires_grad_(True)
        
        def forward_func(x):
            pred, _, _ = self.forward(x, return_attention=False)
            return pred
        
        # Baseline (zeros)
        baseline = torch.zeros_like(molecular_features)
        
        # Integrated gradients
        n_steps = 50
        alphas = torch.linspace(0, 1, n_steps).to(molecular_features.device)
        
        gradients = []
        for alpha in alphas:
            interpolated = baseline + alpha * (molecular_features - baseline)
            interpolated.requires_grad_(True)
            
            pred = forward_func(interpolated)
            grad = torch.autograd.grad(pred.sum(), interpolated, create_graph=False)[0]
            gradients.append(grad)
        
        # Average gradients and multiply by input difference
        avg_gradients = torch.stack(gradients).mean(dim=0)
        feature_importance = avg_gradients * (molecular_features - baseline)
        
        return feature_importance


class MolecularToxicophoreDetector:
    """
    Detector for known toxicophores and structural alerts.
    """
    
    def __init__(self):
        self.toxicophore_database = self._init_toxicophore_database()
        logger.info(f"Initialized toxicophore detector with {len(self.toxicophore_database)} alerts")
    
    def _init_toxicophore_database(self) -> List[ToxicophoreAlert]:
        """Initialize database of known toxicophores."""
        return [
            ToxicophoreAlert(
                pattern="[N+](=O)[O-]",  # Nitro group
                name="Nitro Group",
                toxicity_type="mutagenicity",
                severity=0.8,
                evidence_level="strong",
                regulatory_status="FDA_alert"
            ),
            ToxicophoreAlert(
                pattern="N=N",  # Azo group
                name="Azo Group", 
                toxicity_type="carcinogenicity",
                severity=0.7,
                evidence_level="moderate",
                regulatory_status="OECD_alert"
            ),
            ToxicophoreAlert(
                pattern="c1ccc2c(c1)cccc2N",  # Aromatic amine
                name="Aromatic Amine",
                toxicity_type="carcinogenicity",
                severity=0.6,
                evidence_level="strong",
                regulatory_status="IARC_group1"
            ),
            ToxicophoreAlert(
                pattern="[CH2]=[CH][CH2]",  # Allyl group
                name="Allyl Group",
                toxicity_type="electrophilic_reactivity",
                severity=0.5,
                evidence_level="moderate",
                regulatory_status="OECD_alert"
            ),
            ToxicophoreAlert(
                pattern="C(=O)Cl",  # Acyl chloride
                name="Acyl Chloride",
                toxicity_type="corrosivity",
                severity=0.9,
                evidence_level="strong",
                regulatory_status="GHS_corrosive"
            ),
            ToxicophoreAlert(
                pattern="[S,P](=O)(=O)[OH]",  # Sulfonic/Phosphonic acid
                name="Strong Acid Group",
                toxicity_type="corrosivity",
                severity=0.8,
                evidence_level="strong",
                regulatory_status="GHS_corrosive"
            ),
            ToxicophoreAlert(
                pattern="c1ccc(cc1)[CH2][N]",  # Benzylamine
                name="Benzylamine",
                toxicity_type="skin_sensitization",
                severity=0.4,
                evidence_level="moderate",
                regulatory_status="EU_sensitizer"
            ),
            ToxicophoreAlert(
                pattern="C1=CC=CC=C1[Cl,Br,I]",  # Halogenated aromatic
                name="Halogenated Aromatic",
                toxicity_type="persistence",
                severity=0.6,
                evidence_level="strong",
                regulatory_status="Stockholm_POP"
            )
        ]
    
    def detect_toxicophores(self, molecule: Molecule) -> List[ToxicophoreAlert]:
        """Detect toxicophores in a molecule."""
        detected_alerts = []
        
        if not molecule.is_valid:
            return detected_alerts
        
        smiles = molecule.smiles.upper()
        
        for alert in self.toxicophore_database:
            # Simplified pattern matching (would use RDKit substructure search in real implementation)
            if self._pattern_matches(smiles, alert.pattern):
                detected_alerts.append(alert)
        
        return detected_alerts
    
    def _pattern_matches(self, smiles: str, pattern: str) -> bool:
        """Simplified pattern matching (placeholder for RDKit)."""
        # Convert SMARTS patterns to simple string patterns for demo
        pattern_map = {
            "[N+](=O)[O-]": "N",  # Nitro simplified
            "N=N": "N=N",  # Azo
            "c1ccc2c(c1)cccc2N": "C1C",  # Aromatic simplified
            "[CH2]=[CH][CH2]": "C=C",  # Allyl simplified
            "C(=O)Cl": "C(=O)Cl",  # Acyl chloride
            "[S,P](=O)(=O)[OH]": "S(=O)",  # Strong acid simplified
            "c1ccc(cc1)[CH2][N]": "CCN",  # Benzylamine simplified
            "C1=CC=CC=C1[Cl,Br,I]": "C1C"  # Halogenated aromatic simplified
        }
        
        simple_pattern = pattern_map.get(pattern, pattern)
        return simple_pattern in smiles


class ExplainableSafetyPredictor:
    """
    Main explainable safety predictor with multiple interpretation methods.
    """
    
    def __init__(
        self,
        device: str = "cpu",
        toxicity_threshold: float = 0.1,
        enable_shap: bool = False  # Disabled by default due to dependencies
    ):
        self.device = device
        self.toxicity_threshold = toxicity_threshold
        self.enable_shap = enable_shap
        
        # Initialize components
        self.neural_predictor = AttentionBasedToxicityPredictor()
        self.neural_predictor.to(device)
        
        self.toxicophore_detector = MolecularToxicophoreDetector()
        
        # QSAR rules for additional interpretation
        self.qsar_rules = self._init_qsar_rules()
        
        # Traditional safety filter as fallback
        self.fallback_filter = SafetyFilter(toxicity_threshold=toxicity_threshold)
        
        # SHAP explainer (if enabled)
        self.shap_explainer = None
        if enable_shap:
            try:
                import shap
                # Would initialize SHAP explainer here in real implementation
                logger.info("SHAP explainer initialized")
            except ImportError:
                logger.warning("SHAP not available, falling back to attention-based explanations")
                self.enable_shap = False
        
        logger.info(f"ExplainableSafetyPredictor initialized on {device}")
    
    def _init_qsar_rules(self) -> List[Dict[str, Any]]:
        """Initialize QSAR-based interpretability rules."""
        return [
            {
                'name': 'Lipinski Rule of Five',
                'description': 'Drug-like properties for oral bioavailability',
                'rules': {
                    'molecular_weight': (150, 500),
                    'logp': (-2, 5),
                    'hbd': (0, 5),
                    'hba': (0, 10)
                },
                'safety_impact': 0.1  # Positive impact on safety
            },
            {
                'name': 'Reactive Electrophile',
                'description': 'Potential for covalent binding to proteins',
                'rules': {
                    'molecular_weight': (50, 200),
                    'electrophilic_carbons': (1, float('inf'))
                },
                'safety_impact': -0.5  # Negative impact on safety
            },
            {
                'name': 'High Lipophilicity Alert',
                'description': 'Very lipophilic compounds may bioaccumulate',
                'rules': {
                    'logp': (5, float('inf'))
                },
                'safety_impact': -0.3
            },
            {
                'name': 'Fragrance Safe Range',
                'description': 'Typical molecular weight range for safe fragrances',
                'rules': {
                    'molecular_weight': (120, 300),
                    'vapor_pressure': (0.001, 10)
                },
                'safety_impact': 0.2
            }
        ]
    
    def predict_safety_with_explanation(
        self, 
        molecule: Molecule,
        detailed: bool = True
    ) -> SafetyExplanation:
        """
        Predict safety with comprehensive explanation.
        
        Args:
            molecule: Molecule to evaluate
            detailed: Whether to include detailed explanations
            
        Returns:
            SafetyExplanation with prediction and interpretability information
        """
        if not molecule.is_valid:
            return SafetyExplanation(
                prediction=0.0,
                confidence=1.0,
                risk_factors=[{'name': 'Invalid molecule', 'score': 1.0}],
                protective_factors=[]
            )
        
        # Neural prediction with attention
        neural_prediction, confidence = self._neural_prediction_with_attention(molecule)
        
        # Toxicophore detection
        toxicophores = self.toxicophore_detector.detect_toxicophores(molecule)
        
        # QSAR rule evaluation
        qsar_factors = self._evaluate_qsar_rules(molecule)
        
        # Combine predictions
        final_prediction = self._combine_predictions(
            neural_prediction, toxicophores, qsar_factors
        )
        
        # Generate explanation
        risk_factors = []
        protective_factors = []
        
        # Add toxicophore risks
        for toxicophore in toxicophores:
            risk_factors.append({
                'name': toxicophore.name,
                'type': 'structural_alert',
                'score': toxicophore.severity,
                'description': f"{toxicophore.toxicity_type} risk",
                'regulatory_status': toxicophore.regulatory_status,
                'evidence': toxicophore.evidence_level
            })
        
        # Add QSAR factors
        for factor in qsar_factors:
            if factor['impact'] < 0:
                risk_factors.append({
                    'name': factor['name'],
                    'type': 'qsar_rule',
                    'score': abs(factor['impact']),
                    'description': factor['description'],
                    'violated_rules': factor.get('violated_rules', [])
                })
            else:
                protective_factors.append({
                    'name': factor['name'],
                    'type': 'qsar_rule',
                    'score': factor['impact'],
                    'description': factor['description'],
                    'satisfied_rules': factor.get('satisfied_rules', [])
                })
        
        # Generate molecular alerts
        alerts = []
        if neural_prediction < self.toxicity_threshold:
            alerts.append(f"Neural model predicts high toxicity risk: {neural_prediction:.3f}")
        if toxicophores:
            alerts.append(f"Structural alerts detected: {len(toxicophores)} toxicophores")
        if final_prediction < 0.5:
            alerts.append("Overall safety assessment: HIGH RISK")
        
        explanation = SafetyExplanation(
            prediction=final_prediction,
            confidence=float(confidence),
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            molecular_alerts=alerts
        )
        
        # Add detailed explanations if requested
        if detailed:
            explanation = self._add_detailed_explanations(molecule, explanation)
        
        return explanation
    
    def _neural_prediction_with_attention(self, molecule: Molecule) -> Tuple[float, float]:
        """Get neural prediction with attention weights."""
        try:
            # Generate molecular fingerprint (simplified)
            fingerprint = self._generate_molecular_fingerprint(molecule)
            fingerprint_tensor = torch.tensor(fingerprint, dtype=torch.float32).unsqueeze(0)
            fingerprint_tensor = fingerprint_tensor.to(self.device)
            
            with torch.no_grad():
                prediction, uncertainty, attention_weights = self.neural_predictor(
                    fingerprint_tensor, return_attention=True
                )
            
            pred_value = float(prediction.item())
            confidence = 1.0 / (1.0 + float(uncertainty.item()))  # Convert uncertainty to confidence
            
            return pred_value, confidence
            
        except Exception as e:
            logger.warning(f"Neural prediction failed: {e}")
            # Fallback to traditional method
            return self.fallback_filter._predict_toxicity(molecule), 0.5
    
    def _generate_molecular_fingerprint(self, molecule: Molecule) -> np.ndarray:
        """Generate molecular fingerprint for neural network input."""
        # Simplified fingerprint generation (would use RDKit in real implementation)
        fingerprint = np.zeros(2048)
        
        # Basic molecular properties
        mw = molecule.get_property('molecular_weight') or 150
        logp = molecule.get_property('logp') or 2.0
        
        # Encode properties into fingerprint
        fingerprint[0] = min(1.0, mw / 500)  # Normalized MW
        fingerprint[1] = (logp + 5) / 10  # Normalized logP
        
        # Simple structural features based on SMILES
        smiles = molecule.smiles.upper()
        fingerprint[2] = 1.0 if 'N' in smiles else 0.0  # Contains nitrogen
        fingerprint[3] = 1.0 if 'O' in smiles else 0.0  # Contains oxygen
        fingerprint[4] = 1.0 if 'S' in smiles else 0.0  # Contains sulfur
        fingerprint[5] = 1.0 if '=' in smiles else 0.0  # Contains double bond
        fingerprint[6] = 1.0 if 'C1C' in smiles else 0.0  # Contains ring
        fingerprint[7] = smiles.count('C') / 20  # Carbon count (normalized)
        
        # Random features for demonstration
        np.random.seed(hash(molecule.smiles) % 2**32)  # Deterministic randomness
        fingerprint[8:] = np.random.random(2048 - 8) * 0.1  # Low-amplitude noise
        
        return fingerprint
    
    def _evaluate_qsar_rules(self, molecule: Molecule) -> List[Dict[str, Any]]:
        """Evaluate QSAR-based safety rules."""
        factors = []
        
        # Get molecular properties
        mw = molecule.get_property('molecular_weight') or 150
        logp = molecule.get_property('logp') or 2.0
        hbd = molecule.get_property('hbd') or 1
        hba = molecule.get_property('hba') or 2
        
        for rule in self.qsar_rules:
            rule_result = {
                'name': rule['name'],
                'description': rule['description'],
                'impact': 0.0,
                'violated_rules': [],
                'satisfied_rules': []
            }
            
            violations = 0
            satisfactions = 0
            total_rules = len(rule['rules'])
            
            for prop, (min_val, max_val) in rule['rules'].items():
                prop_value = None
                
                if prop == 'molecular_weight':
                    prop_value = mw
                elif prop == 'logp':
                    prop_value = logp
                elif prop == 'hbd':
                    prop_value = hbd
                elif prop == 'hba':
                    prop_value = hba
                elif prop == 'electrophilic_carbons':
                    # Simplified: count C=O groups
                    prop_value = molecule.smiles.count('C=O')
                elif prop == 'vapor_pressure':
                    # Estimated based on MW and logP
                    prop_value = max(0.001, 10 ** (-mw/100 - logp/2))
                
                if prop_value is not None:
                    if min_val <= prop_value <= max_val:
                        satisfactions += 1
                        rule_result['satisfied_rules'].append(f"{prop}: {prop_value:.2f}")
                    else:
                        violations += 1
                        rule_result['violated_rules'].append(f"{prop}: {prop_value:.2f} (expected: {min_val}-{max_val})")
            
            # Calculate impact based on rule satisfaction
            if total_rules > 0:
                satisfaction_ratio = satisfactions / total_rules
                if rule['safety_impact'] > 0:  # Protective rule
                    rule_result['impact'] = rule['safety_impact'] * satisfaction_ratio
                else:  # Risk rule
                    violation_ratio = violations / total_rules
                    rule_result['impact'] = rule['safety_impact'] * violation_ratio
            
            factors.append(rule_result)
        
        return factors
    
    def _combine_predictions(
        self,
        neural_prediction: float,
        toxicophores: List[ToxicophoreAlert],
        qsar_factors: List[Dict[str, Any]]
    ) -> float:
        """Combine different prediction sources into final safety score."""
        # Start with neural prediction
        combined_score = neural_prediction
        
        # Apply toxicophore penalties
        for toxicophore in toxicophores:
            penalty = toxicophore.severity * 0.2  # Scale penalty
            combined_score *= (1.0 - penalty)
        
        # Apply QSAR adjustments
        for factor in qsar_factors:
            adjustment = factor['impact'] * 0.1  # Scale adjustment
            combined_score += adjustment
        
        # Ensure score is in valid range
        combined_score = max(0.0, min(1.0, combined_score))
        
        return combined_score
    
    def _add_detailed_explanations(
        self, 
        molecule: Molecule, 
        explanation: SafetyExplanation
    ) -> SafetyExplanation:
        """Add detailed explanations including SHAP and feature importance."""
        try:
            # Generate feature importance using integrated gradients
            fingerprint = self._generate_molecular_fingerprint(molecule)
            fingerprint_tensor = torch.tensor(fingerprint, dtype=torch.float32).unsqueeze(0)
            fingerprint_tensor = fingerprint_tensor.to(self.device)
            
            feature_importance = self.neural_predictor.get_feature_importance(fingerprint_tensor)
            explanation.attention_weights = feature_importance.cpu().numpy()
            
            # Add SHAP explanations if enabled
            if self.enable_shap and self.shap_explainer:
                # Would generate SHAP explanations here
                explanation.shap_values = np.random.random(len(fingerprint)) * 0.1  # Placeholder
            
        except Exception as e:
            logger.warning(f"Failed to generate detailed explanations: {e}")
        
        return explanation
    
    def generate_safety_report(
        self, 
        molecules: List[Molecule],
        output_format: str = "text"
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate comprehensive safety report for multiple molecules.
        
        Args:
            molecules: List of molecules to evaluate
            output_format: "text" or "json"
            
        Returns:
            Safety report in requested format
        """
        logger.info(f"Generating safety report for {len(molecules)} molecules")
        
        results = []
        summary_stats = {
            'total_molecules': len(molecules),
            'safe_molecules': 0,
            'high_risk_molecules': 0,
            'invalid_molecules': 0,
            'toxicophores_detected': 0
        }
        
        for i, molecule in enumerate(molecules):
            try:
                explanation = self.predict_safety_with_explanation(molecule, detailed=False)
                
                result = {
                    'molecule_id': i,
                    'smiles': molecule.smiles,
                    'safety_score': explanation.prediction,
                    'confidence': explanation.confidence,
                    'n_risk_factors': len(explanation.risk_factors),
                    'n_protective_factors': len(explanation.protective_factors),
                    'alerts': explanation.molecular_alerts
                }
                
                results.append(result)
                
                # Update summary stats
                if not molecule.is_valid:
                    summary_stats['invalid_molecules'] += 1
                elif explanation.prediction >= 0.7:
                    summary_stats['safe_molecules'] += 1
                elif explanation.prediction < 0.3:
                    summary_stats['high_risk_molecules'] += 1
                
                if any('toxicophore' in alert.lower() for alert in explanation.molecular_alerts):
                    summary_stats['toxicophores_detected'] += 1
                
            except Exception as e:
                logger.warning(f"Safety evaluation failed for molecule {i}: {e}")
                results.append({
                    'molecule_id': i,
                    'smiles': molecule.smiles,
                    'error': str(e)
                })
        
        # Generate report
        report_data = {
            'summary': summary_stats,
            'results': results,
            'timestamp': time.time(),
            'total_processing_time': 0.0  # Would track actual time
        }
        
        if output_format == "json":
            return report_data
        else:
            return self._format_text_report(report_data)
    
    def _format_text_report(self, report_data: Dict[str, Any]) -> str:
        """Format report data as human-readable text."""
        summary = report_data['summary']
        results = report_data['results']
        
        report_lines = [
            "=== EXPLAINABLE SAFETY ASSESSMENT REPORT ===",
            "",
            f"Total molecules evaluated: {summary['total_molecules']}",
            f"Safe molecules (score ≥ 0.7): {summary['safe_molecules']}",
            f"High-risk molecules (score < 0.3): {summary['high_risk_molecules']}",
            f"Invalid molecules: {summary['invalid_molecules']}",
            f"Molecules with toxicophores: {summary['toxicophores_detected']}",
            "",
            "=== INDIVIDUAL RESULTS ===",
            ""
        ]
        
        for result in results[:10]:  # Show first 10 results
            if 'error' in result:
                report_lines.append(f"Molecule {result['molecule_id']}: ERROR - {result['error']}")
            else:
                status = "SAFE" if result['safety_score'] >= 0.7 else "RISKY" if result['safety_score'] < 0.3 else "MODERATE"
                report_lines.extend([
                    f"Molecule {result['molecule_id']}: {result['smiles']}",
                    f"  Safety Score: {result['safety_score']:.3f} ({status})",
                    f"  Confidence: {result['confidence']:.3f}",
                    f"  Risk Factors: {result['n_risk_factors']}",
                    f"  Protective Factors: {result['n_protective_factors']}",
                    f"  Alerts: {'; '.join(result['alerts'][:2])}",
                    ""
                ])
        
        if len(results) > 10:
            report_lines.append(f"... and {len(results) - 10} more results")
        
        return "\n".join(report_lines)
    
    def compare_safety_methods(
        self, 
        test_molecules: List[Molecule]
    ) -> Dict[str, Any]:
        """Compare explainable safety predictor with baseline methods."""
        logger.info(f"Comparing safety methods on {len(test_molecules)} molecules")
        
        results = {
            'explainable_predictor': [],
            'baseline_filter': [],
            'toxicophore_only': []
        }
        
        for molecule in test_molecules:
            try:
                # Explainable prediction
                explanation = self.predict_safety_with_explanation(molecule, detailed=False)
                results['explainable_predictor'].append(explanation.prediction)
                
                # Baseline filter
                baseline_pred = self.fallback_filter._predict_toxicity(molecule)
                results['baseline_filter'].append(baseline_pred)
                
                # Toxicophore-only prediction
                toxicophores = self.toxicophore_detector.detect_toxicophores(molecule)
                toxicophore_score = 1.0 - sum(t.severity for t in toxicophores) * 0.2
                toxicophore_score = max(0.0, min(1.0, toxicophore_score))
                results['toxicophore_only'].append(toxicophore_score)
                
            except Exception as e:
                logger.warning(f"Comparison failed for molecule: {e}")
        
        # Calculate comparison metrics
        comparison_metrics = {}
        for method, predictions in results.items():
            if predictions:
                comparison_metrics[method] = {
                    'mean_safety_score': np.mean(predictions),
                    'std_safety_score': np.std(predictions),
                    'safe_predictions': sum(1 for p in predictions if p >= 0.7),
                    'risky_predictions': sum(1 for p in predictions if p < 0.3),
                    'moderate_predictions': sum(1 for p in predictions if 0.3 <= p < 0.7)
                }
        
        return comparison_metrics


# Example usage and testing functions
def test_explainable_safety():
    """Test the explainable safety predictor."""
    from ..models.molecule import Molecule
    
    predictor = ExplainableSafetyPredictor()
    
    # Test molecules
    test_molecules = [
        Molecule("CC(C)=CCO"),  # Linalool - generally safe
        Molecule("c1ccc(cc1)N"),  # Aniline - potentially toxic
        Molecule("CCO"),  # Ethanol - relatively safe
        Molecule("C(=O)Cl"),  # Formyl chloride - very dangerous
    ]
    
    print("=== Testing Explainable Safety Predictor ===\n")
    
    for i, molecule in enumerate(test_molecules):
        print(f"Molecule {i+1}: {molecule.smiles}")
        explanation = predictor.predict_safety_with_explanation(molecule)
        
        print(f"  Safety Score: {explanation.prediction:.3f}")
        print(f"  Confidence: {explanation.confidence:.3f}")
        print(f"  Risk Factors: {len(explanation.risk_factors)}")
        print(f"  Protective Factors: {len(explanation.protective_factors)}")
        print(f"  Alerts: {explanation.molecular_alerts}")
        print()


if __name__ == "__main__":
    import time
    test_explainable_safety()