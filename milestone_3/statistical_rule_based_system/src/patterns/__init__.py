"""
Pattern extraction module.

Provides modular pattern extractors for relation extraction:
- Lexical patterns (LEMMA, BIGRAM, PREP, BEFORE_E1, AFTER_E2, ENTITY_POS)
- Dependency patterns (DEP_VERB, DEP_LABELS)
- Semantic patterns (SYNSET, LEXNAME, HYPERNYM, FRAME)
- Preposition patterns (PREP_STRUCT, PREP_GOV_LEX, PREP_ROLES)

Usage:
    from src.patterns import PatternExtractor, Rule

    extractor = PatternExtractor(min_precision=0.6, min_support=2)
    patterns = extractor.extract_patterns(processed_data)
    rules = extractor.filter_and_rank(patterns)
"""

from .base import PatternExtractorBase, Rule, split_directed_label
from .dependency import DependencyPatternExtractor
from .extractor import PatternExtractor
from .lexical import LexicalPatternExtractor
from .preposition import PrepositionPatternExtractor
from .semantic_patterns import SemanticPatternExtractor

__all__ = [
    # Main extractor
    'PatternExtractor',
    # Base classes
    'PatternExtractorBase',
    'Rule',
    'split_directed_label',
    # Sub-extractors
    'LexicalPatternExtractor',
    'DependencyPatternExtractor',
    'SemanticPatternExtractor',
    'PrepositionPatternExtractor',
]
