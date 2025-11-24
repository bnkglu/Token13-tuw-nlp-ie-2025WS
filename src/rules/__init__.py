"""Rule-based relation extraction system.

This package provides a scalable, ML-driven approach to rule-based
relation extraction using shortest dependency path (SDP) pattern learning.

The pipeline:
1. Extract SDP features from training data
2. Train ML classifier to find discriminative patterns
3. Convert high-weight patterns to spaCy DependencyMatcher rules
4. Apply rules with configurable constraints
"""

__version__ = "1.0.0"

from .sdp_extractor import SDPExtractor, SDPSignature
from .feature_extractor import SDPFeatureExtractor, FeatureMatrix
from .ml_pattern_learner import MLPatternLearner, ImportantFeature, MLPatternResult
from .ml_rule_pipeline import MLRulePipeline, MLRulePipelineResult
from .pattern_generator import PatternGenerator, GeneratedPattern
from .constraints import ConstraintEngine, MatchResult
from .rule_matcher import RuleMatcher

__all__ = [
    # Core SDP extraction
    "SDPExtractor",
    "SDPSignature",
    # Feature extraction
    "SDPFeatureExtractor",
    "FeatureMatrix",
    # ML-based learning
    "MLPatternLearner",
    "ImportantFeature",
    "MLPatternResult",
    # Pipeline
    "MLRulePipeline",
    "MLRulePipelineResult",
    # Pattern generation & matching
    "PatternGenerator",
    "GeneratedPattern",
    "ConstraintEngine",
    "MatchResult",
    "RuleMatcher",
]
