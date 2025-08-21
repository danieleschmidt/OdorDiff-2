"""Real-time adaptive learning modules."""

from .adaptive_learning import (
    AdaptiveLearningSystem,
    UserFeedback,
    AdaptationResult,
    ContinualLearningCore,
    FewShotPersonalizer,
    FederatedLearningCoordinator
)

__all__ = [
    "AdaptiveLearningSystem",
    "UserFeedback",
    "AdaptationResult", 
    "ContinualLearningCore",
    "FewShotPersonalizer",
    "FederatedLearningCoordinator"
]