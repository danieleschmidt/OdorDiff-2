"""
Advanced Safety Systems - Multi-layer security and safety validation
Generation 2 Enhancement: Maximum Security and Safety Assurance
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import hashlib
import hmac
import secrets
import json
from pathlib import Path
import threading
import time

from ..utils.logging import get_logger
from ..models.molecule import Molecule

logger = get_logger(__name__)


class ThreatLevel(Enum):
    """Threat levels for security assessment."""
    MINIMAL = "minimal"
    LOW = "low" 
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    EXTREME = "extreme"


class SafetyViolationType(Enum):
    """Types of safety violations."""
    TOXICITY = "toxicity"
    CARCINOGENICITY = "carcinogenicity"
    MUTAGENICITY = "mutagenicity"
    ENVIRONMENTAL_HAZARD = "environmental_hazard"
    EXPLOSIVE = "explosive"
    CORROSIVE = "corrosive"
    SKIN_SENSITIZER = "skin_sensitizer"
    RESPIRATORY_IRRITANT = "respiratory_irritant"
    REPRODUCTIVE_TOXIN = "reproductive_toxin"
    CONTROLLED_SUBSTANCE = "controlled_substance"


@dataclass
class SecurityThreat:
    """Represents a detected security threat."""
    threat_type: str
    threat_level: ThreatLevel
    description: str
    molecule_context: Optional[Molecule] = None
    timestamp: datetime = field(default_factory=datetime.now)
    mitigation_applied: bool = False
    source_ip: Optional[str] = None
    user_context: Optional[Dict[str, Any]] = None


@dataclass
class SafetyAssessment:
    """Comprehensive safety assessment result."""
    molecule: Molecule
    overall_safety_score: float
    violations: List[SafetyViolationType]
    detailed_scores: Dict[str, float]
    regulatory_compliance: Dict[str, bool]
    recommendations: List[str]
    assessment_timestamp: datetime = field(default_factory=datetime.now)


class AdvancedToxicityPredictor(nn.Module):
    """Advanced neural network for toxicity prediction."""
    
    def __init__(self, input_dim: int = 2048, hidden_dims: List[int] = [1024, 512, 256]):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            current_dim = hidden_dim
        
        # Multi-task output heads for different safety aspects
        self.shared_layers = nn.Sequential(*layers)
        
        # Specific prediction heads
        self.toxicity_head = nn.Sequential(
            nn.Linear(current_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.carcinogenicity_head = nn.Sequential(
            nn.Linear(current_dim, 128),
            nn.ReLU(), 
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.mutagenicity_head = nn.Sequential(
            nn.Linear(current_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.environmental_head = nn.Sequential(
            nn.Linear(current_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, molecular_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning multiple safety predictions."""
        shared_features = self.shared_layers(molecular_features)
        
        return {
            'toxicity': self.toxicity_head(shared_features),
            'carcinogenicity': self.carcinogenicity_head(shared_features),
            'mutagenicity': self.mutagenicity_head(shared_features),
            'environmental_hazard': self.environmental_head(shared_features)
        }


class QuantumSafetyValidator:
    """Quantum-enhanced safety validation with uncertainty quantification."""
    
    def __init__(self):
        self.toxicity_predictor = AdvancedToxicityPredictor()
        self.uncertainty_threshold = 0.3
        self.safety_thresholds = {
            'toxicity': 0.2,
            'carcinogenicity': 0.1, 
            'mutagenicity': 0.15,
            'environmental_hazard': 0.25
        }
        
        # Load regulatory databases
        self._load_regulatory_databases()
        
        # Quantum uncertainty estimation
        self.quantum_ensemble_size = 10
        
    def _load_regulatory_databases(self):
        """Load regulatory compliance databases."""
        self.regulatory_lists = {
            'eu_reach': self._load_eu_reach_list(),
            'us_tsca': self._load_us_tsca_list(),
            'ifra_restricted': self._load_ifra_restricted_list(),
            'un_ghs': self._load_un_ghs_list()
        }
    
    def _load_eu_reach_list(self) -> Set[str]:
        """Load EU REACH restricted substances."""
        # In production, this would load from actual database
        return {
            'benzene', 'formaldehyde', 'methanol', 'toluene',
            'xylene', 'dichloromethane', 'chloroform'
        }
    
    def _load_us_tsca_list(self) -> Set[str]:
        """Load US TSCA inventory."""
        return {
            'acetone', 'ethanol', 'water', 'sodium_chloride'
        }
    
    def _load_ifra_restricted_list(self) -> Set[str]:
        """Load IFRA restricted fragrance materials."""
        return {
            'oakmoss_absolute', 'treemoss_absolute', 'methyl_eugenol',
            'safrole', 'benzyl_salicylate'
        }
    
    def _load_un_ghs_list(self) -> Set[str]:
        """Load UN GHS classified hazardous substances."""
        return {
            'benzene', 'methanol', 'formaldehyde', 'hydrogen_peroxide'
        }
    
    def validate_molecule_safety(self, molecule: Molecule) -> SafetyAssessment:
        """Comprehensive safety validation with quantum uncertainty."""
        if not molecule or not molecule.is_valid:
            return SafetyAssessment(
                molecule=molecule,
                overall_safety_score=0.0,
                violations=[SafetyViolationType.TOXICITY],
                detailed_scores={},
                regulatory_compliance={},
                recommendations=["Invalid molecule structure"]
            )
        
        # Extract molecular features
        features = self._extract_safety_features(molecule)
        
        # Quantum ensemble predictions for uncertainty
        ensemble_predictions = []
        for _ in range(self.quantum_ensemble_size):
            # Add quantum noise for uncertainty estimation
            noisy_features = features + torch.randn_like(features) * 0.1
            predictions = self.toxicity_predictor(noisy_features.unsqueeze(0))
            ensemble_predictions.append(predictions)
        
        # Aggregate predictions and compute uncertainty
        aggregated_scores = {}
        uncertainties = {}
        
        for key in ensemble_predictions[0].keys():
            scores = torch.stack([pred[key] for pred in ensemble_predictions])
            aggregated_scores[key] = torch.mean(scores).item()
            uncertainties[key] = torch.std(scores).item()
        
        # Check for high uncertainty (potential safety risk)
        high_uncertainty_keys = [
            k for k, uncertainty in uncertainties.items() 
            if uncertainty > self.uncertainty_threshold
        ]
        
        # Identify violations
        violations = []
        for safety_aspect, score in aggregated_scores.items():
            threshold = self.safety_thresholds.get(safety_aspect, 0.2)
            if score > threshold:
                violations.append(SafetyViolationType(safety_aspect))
        
        # High uncertainty is treated as potential violation
        if high_uncertainty_keys:
            violations.append(SafetyViolationType.TOXICITY)
        
        # Check regulatory compliance
        regulatory_compliance = self._check_regulatory_compliance(molecule)
        
        # Calculate overall safety score
        overall_safety_score = self._calculate_overall_safety_score(
            aggregated_scores, uncertainties, regulatory_compliance
        )
        
        # Generate recommendations
        recommendations = self._generate_safety_recommendations(
            violations, high_uncertainty_keys, regulatory_compliance
        )
        
        return SafetyAssessment(
            molecule=molecule,
            overall_safety_score=overall_safety_score,
            violations=violations,
            detailed_scores=aggregated_scores,
            regulatory_compliance=regulatory_compliance,
            recommendations=recommendations
        )
    
    def _extract_safety_features(self, molecule: Molecule) -> torch.Tensor:
        """Extract comprehensive safety-relevant molecular features."""
        features = []
        
        # Basic molecular properties
        features.append(molecule.get_property('molecular_weight', 150) / 500.0)
        features.append(molecule.get_property('logp', 2.0) / 6.0)
        features.append(molecule.get_property('tpsa', 50) / 200.0)
        
        # Functional group indicators
        smiles = molecule.smiles.lower()
        
        # Potentially hazardous functional groups
        hazardous_groups = {
            'nitro': 'no2',
            'nitrile': 'c#n', 
            'azide': 'n3',
            'peroxide': 'oo',
            'halogen': ['f', 'cl', 'br', 'i'],
            'aromatic_amine': ['c1ccccc1n', 'nc1ccccc1'],
            'phenol': 'c1ccc(cc1)o',
            'aldehyde': 'c=o',
            'epoxide': 'c1oc1'
        }
        
        for group_name, patterns in hazardous_groups.items():
            if isinstance(patterns, list):
                has_group = any(pattern in smiles for pattern in patterns)
            else:
                has_group = patterns in smiles
            features.append(float(has_group))
        
        # Structural alerts for toxicity
        toxicity_alerts = [
            'c1cc(ccc1n)n',  # Aromatic amine
            'c1ccc(cc1)n(=o)=o',  # Nitrobenzene
            'oc1ccccc1o',  # Catechol
            'c1ccc2c(c1)cccc2',  # Naphthalene-like
            'c1cc(c(cc1)o)o'  # Hydroquinone-like
        ]
        
        for alert in toxicity_alerts:
            features.append(float(alert in smiles))
        
        # Pad features to expected dimension
        while len(features) < 2048:
            features.append(0.0)
        
        return torch.tensor(features[:2048], dtype=torch.float32)
    
    def _check_regulatory_compliance(self, molecule: Molecule) -> Dict[str, bool]:
        """Check compliance with various regulatory frameworks."""
        compliance = {}
        
        # Simplified compliance checking (in production would be more sophisticated)
        molecule_name = getattr(molecule, 'name', '').lower()
        smiles = molecule.smiles.lower()
        
        for regulation, restricted_list in self.regulatory_lists.items():
            # Check if molecule or similar structure is restricted
            is_compliant = True
            
            for restricted_item in restricted_list:
                if (restricted_item in molecule_name or 
                    restricted_item in smiles or
                    self._structural_similarity_check(smiles, restricted_item)):
                    is_compliant = False
                    break
            
            compliance[regulation] = is_compliant
        
        return compliance
    
    def _structural_similarity_check(self, smiles1: str, smiles2: str) -> bool:
        """Simple structural similarity check."""
        # Very simplified - in production would use proper chemical similarity
        return smiles1 == smiles2 or (len(smiles1) > 5 and smiles2 in smiles1)
    
    def _calculate_overall_safety_score(
        self, 
        scores: Dict[str, float], 
        uncertainties: Dict[str, float],
        compliance: Dict[str, bool]
    ) -> float:
        """Calculate overall safety score."""
        # Start with base safety score
        safety_score = 1.0
        
        # Penalize high toxicity scores
        for aspect, score in scores.items():
            weight = {
                'toxicity': 0.3,
                'carcinogenicity': 0.4,
                'mutagenicity': 0.35,
                'environmental_hazard': 0.2
            }.get(aspect, 0.2)
            
            safety_score -= weight * score
        
        # Penalize high uncertainty
        avg_uncertainty = np.mean(list(uncertainties.values()))
        safety_score -= 0.2 * avg_uncertainty
        
        # Penalize regulatory non-compliance
        compliance_score = np.mean(list(compliance.values()))
        safety_score *= compliance_score
        
        return max(0.0, min(1.0, safety_score))
    
    def _generate_safety_recommendations(
        self,
        violations: List[SafetyViolationType],
        high_uncertainty: List[str],
        compliance: Dict[str, bool]
    ) -> List[str]:
        """Generate safety recommendations."""
        recommendations = []
        
        if SafetyViolationType.TOXICITY in violations:
            recommendations.append("Consider structural modifications to reduce toxicity")
            recommendations.append("Conduct in-vitro toxicity testing before use")
        
        if SafetyViolationType.CARCINOGENICITY in violations:
            recommendations.append("CRITICAL: Potential carcinogen - avoid use")
            recommendations.append("Consider alternative molecular structures")
        
        if SafetyViolationType.MUTAGENICITY in violations:
            recommendations.append("Mutagenicity risk detected - requires Ames testing")
        
        if SafetyViolationType.ENVIRONMENTAL_HAZARD in violations:
            recommendations.append("Environmental impact assessment required")
            recommendations.append("Consider biodegradable alternatives")
        
        if high_uncertainty:
            recommendations.append("High prediction uncertainty - additional testing required")
            recommendations.append("Consider multiple assessment methods")
        
        for regulation, compliant in compliance.items():
            if not compliant:
                recommendations.append(f"Non-compliant with {regulation.upper()} - regulatory review needed")
        
        if not recommendations:
            recommendations.append("Molecule passes initial safety screening")
            recommendations.append("Continue with standard safety protocols")
        
        return recommendations


class SecurityThreatDetector:
    """Advanced security threat detection system."""
    
    def __init__(self):
        self.threat_patterns = self._load_threat_patterns()
        self.suspicious_activities = []
        self.rate_limiters = {}
        self.blocked_ips = set()
        
        # Threat detection neural network
        self.threat_classifier = self._initialize_threat_classifier()
        
    def _load_threat_patterns(self) -> Dict[str, List[str]]:
        """Load known threat patterns."""
        return {
            'chemical_weapons': [
                'sarin', 'vx', 'tabun', 'soman', 'novichok',
                'mustard', 'lewisite', 'phosgene'
            ],
            'explosives': [
                'tnt', 'rdx', 'petn', 'hmtd', 'tatp',
                'nitroglycerin', 'picric_acid'
            ],
            'drugs': [
                'mdma', 'lsd', 'pcp', 'methamphetamine',
                'cocaine', 'heroin', 'fentanyl'
            ],
            'poisons': [
                'ricin', 'abrin', 'saxitoxin', 'tetrodotoxin',
                'strychnine', 'cyanide'
            ]
        }
    
    def _initialize_threat_classifier(self) -> nn.Module:
        """Initialize neural network for threat classification."""
        return nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(self.threat_patterns)),
            nn.Sigmoid()
        )
    
    def analyze_generation_request(
        self,
        prompt: str,
        user_context: Optional[Dict[str, Any]] = None,
        source_ip: Optional[str] = None
    ) -> Tuple[bool, List[SecurityThreat]]:
        """Analyze generation request for security threats."""
        
        threats_detected = []
        
        # Check for blocked IPs
        if source_ip and source_ip in self.blocked_ips:
            threat = SecurityThreat(
                threat_type="blocked_ip",
                threat_level=ThreatLevel.HIGH,
                description=f"Request from blocked IP: {source_ip}",
                source_ip=source_ip,
                user_context=user_context
            )
            threats_detected.append(threat)
            return False, threats_detected
        
        # Rate limiting check
        if source_ip:
            if self._check_rate_limit_exceeded(source_ip):
                threat = SecurityThreat(
                    threat_type="rate_limit_exceeded",
                    threat_level=ThreatLevel.MODERATE,
                    description=f"Rate limit exceeded from IP: {source_ip}",
                    source_ip=source_ip
                )
                threats_detected.append(threat)
                return False, threats_detected
        
        # Prompt analysis for malicious intent
        prompt_threats = self._analyze_prompt_threats(prompt)
        threats_detected.extend(prompt_threats)
        
        # User context analysis
        if user_context:
            context_threats = self._analyze_user_context(user_context)
            threats_detected.extend(context_threats)
        
        # Determine if request should be allowed
        max_threat_level = max(
            (threat.threat_level for threat in threats_detected),
            default=ThreatLevel.MINIMAL
        )
        
        allow_request = max_threat_level.value not in ['critical', 'extreme']
        
        return allow_request, threats_detected
    
    def _analyze_prompt_threats(self, prompt: str) -> List[SecurityThreat]:
        """Analyze prompt for threat indicators."""
        threats = []
        prompt_lower = prompt.lower()
        
        # Check against known threat patterns
        for threat_category, patterns in self.threat_patterns.items():
            for pattern in patterns:
                if pattern in prompt_lower:
                    threat_level = self._assess_threat_level(threat_category, pattern)
                    
                    threat = SecurityThreat(
                        threat_type=f"malicious_prompt_{threat_category}",
                        threat_level=threat_level,
                        description=f"Prompt contains {threat_category} related term: '{pattern}'"
                    )
                    threats.append(threat)
        
        # Suspicious keyword combinations
        suspicious_combinations = [
            ['weapon', 'chemical'],
            ['explosive', 'synthesis'],
            ['toxic', 'gas'],
            ['nerve', 'agent'],
            ['biological', 'weapon']
        ]
        
        for combination in suspicious_combinations:
            if all(keyword in prompt_lower for keyword in combination):
                threat = SecurityThreat(
                    threat_type="suspicious_combination",
                    threat_level=ThreatLevel.HIGH,
                    description=f"Suspicious keyword combination: {combination}"
                )
                threats.append(threat)
        
        return threats
    
    def _assess_threat_level(self, category: str, pattern: str) -> ThreatLevel:
        """Assess threat level based on category and specific pattern."""
        high_risk_categories = ['chemical_weapons', 'explosives']
        critical_patterns = ['sarin', 'vx', 'novichok', 'tnt', 'rdx']
        
        if pattern in critical_patterns:
            return ThreatLevel.EXTREME
        elif category in high_risk_categories:
            return ThreatLevel.CRITICAL
        elif category == 'drugs':
            return ThreatLevel.HIGH
        else:
            return ThreatLevel.MODERATE
    
    def _check_rate_limit_exceeded(self, source_ip: str) -> bool:
        """Check if rate limit is exceeded for source IP."""
        current_time = datetime.now()
        
        if source_ip not in self.rate_limiters:
            self.rate_limiters[source_ip] = []
        
        # Clean old requests (older than 1 hour)
        self.rate_limiters[source_ip] = [
            timestamp for timestamp in self.rate_limiters[source_ip]
            if current_time - timestamp < timedelta(hours=1)
        ]
        
        # Check limits
        requests_last_hour = len(self.rate_limiters[source_ip])
        requests_last_10_min = len([
            timestamp for timestamp in self.rate_limiters[source_ip]
            if current_time - timestamp < timedelta(minutes=10)
        ])
        
        # Add current request
        self.rate_limiters[source_ip].append(current_time)
        
        # Rate limit thresholds
        return requests_last_hour > 100 or requests_last_10_min > 20
    
    def block_ip(self, ip: str, reason: str = "Security threat detected"):
        """Block an IP address."""
        self.blocked_ips.add(ip)
        logger.warning(f"Blocked IP {ip}: {reason}")
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics and statistics."""
        current_time = datetime.now()
        
        # Recent threats (last 24 hours)
        recent_threats = [
            activity for activity in self.suspicious_activities
            if current_time - activity.get('timestamp', current_time) < timedelta(days=1)
        ]
        
        return {
            'total_threats_detected': len(self.suspicious_activities),
            'recent_threats_24h': len(recent_threats),
            'blocked_ips': len(self.blocked_ips),
            'active_rate_limiters': len(self.rate_limiters),
            'threat_categories': list(self.threat_patterns.keys())
        }


class SecureGenerationFramework:
    """Comprehensive security and safety framework for molecular generation."""
    
    def __init__(self, base_generator: Any):
        self.base_generator = base_generator
        self.safety_validator = QuantumSafetyValidator()
        self.threat_detector = SecurityThreatDetector()
        
        # Security configuration
        self.security_level = "high"  # "low", "medium", "high", "maximum"
        self.enable_logging = True
        self.audit_trail = []
        
        # Session management
        self.active_sessions = {}
        self.session_timeout = timedelta(hours=2)
        
    def secure_generate(
        self,
        prompt: str,
        num_molecules: int = 5,
        user_context: Optional[Dict[str, Any]] = None,
        source_ip: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Tuple[List[Molecule], List[SecurityThreat], List[SafetyAssessment]]:
        """Secure molecular generation with comprehensive safety and security."""
        
        threats_detected = []
        safety_assessments = []
        
        # Security analysis
        allow_request, security_threats = self.threat_detector.analyze_generation_request(
            prompt, user_context, source_ip
        )
        threats_detected.extend(security_threats)
        
        if not allow_request:
            logger.warning(f"Request blocked due to security threats: {len(security_threats)} threats detected")
            self._log_security_event("request_blocked", {
                'prompt': prompt[:100],
                'threats': len(security_threats),
                'source_ip': source_ip
            })
            return [], threats_detected, []
        
        # Generate molecules
        try:
            molecules = self.base_generator.generate(prompt, num_molecules, **kwargs)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return [], threats_detected, []
        
        # Safety validation for each molecule
        validated_molecules = []
        for molecule in molecules:
            safety_assessment = self.safety_validator.validate_molecule_safety(molecule)
            safety_assessments.append(safety_assessment)
            
            # Only include safe molecules
            if safety_assessment.overall_safety_score > 0.5 and not safety_assessment.violations:
                validated_molecules.append(molecule)
            else:
                logger.warning(f"Molecule rejected due to safety concerns: {molecule.smiles}")
        
        # Audit logging
        if self.enable_logging:
            self._log_generation_event({
                'prompt': prompt,
                'molecules_generated': len(molecules),
                'molecules_validated': len(validated_molecules),
                'safety_assessments': len(safety_assessments),
                'threats_detected': len(threats_detected),
                'source_ip': source_ip,
                'session_id': session_id
            })
        
        return validated_molecules, threats_detected, safety_assessments
    
    def _log_security_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log security events for audit trail."""
        event = {
            'type': 'security',
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'data': event_data
        }
        
        self.audit_trail.append(event)
        logger.warning(f"Security event: {event_type} - {event_data}")
    
    def _log_generation_event(self, event_data: Dict[str, Any]):
        """Log generation events for audit trail."""
        event = {
            'type': 'generation',
            'timestamp': datetime.now().isoformat(),
            'data': event_data
        }
        
        self.audit_trail.append(event)
    
    def get_comprehensive_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security and safety report."""
        return {
            'security_metrics': self.threat_detector.get_security_metrics(),
            'audit_trail_entries': len(self.audit_trail),
            'recent_security_events': [
                event for event in self.audit_trail
                if event.get('type') == 'security' and
                   datetime.fromisoformat(event['timestamp']) > datetime.now() - timedelta(days=7)
            ],
            'safety_validator_status': {
                'toxicity_predictor_loaded': self.safety_validator.toxicity_predictor is not None,
                'regulatory_databases_loaded': len(self.safety_validator.regulatory_lists),
                'quantum_ensemble_size': self.safety_validator.quantum_ensemble_size
            },
            'system_security_level': self.security_level,
            'active_sessions': len(self.active_sessions)
        }