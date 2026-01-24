"""
Base classes and types for pattern extraction.

This module defines the core Rule dataclass and abstract interfaces
that all pattern extractors implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class Rule:
    """A single extraction rule with metadata."""

    name: str
    relation: str  # Full directed label (e.g., 'Cause-Effect(e1,e2)')
    base_relation: str  # Undirected (e.g., 'Cause-Effect')
    direction: Optional[str]  # 'e1,e2' or 'e2,e1' or None for Other
    pattern_type: str
    pattern_data: tuple
    precision: float
    support: int
    priority: float
    explanation: str

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        # Infer matcher_type for legacy compatibility
        lexical_types = {'LEMMA', 'BIGRAM', 'PREP', 'BEFORE_E1', 'AFTER_E2', 'ENTITY_POS'}
        dep_types = {'DEP_VERB', 'DEP_LABELS'}
        
        if self.pattern_type in lexical_types:
            matcher_type = 'lexical'
        elif self.pattern_type in dep_types:
            matcher_type = 'dependency'
        else:
            matcher_type = 'semantic'

        return {
            "name": self.name,
            "relation": self.relation,
            "base_relation": self.base_relation,
            "direction": self.direction,
            "matcher_type": matcher_type,
            "pattern_type": self.pattern_type,
            "pattern_data": self.pattern_data,
            "precision": self.precision,
            "support": self.support,
            "priority": self.priority,
            "explanation": self.explanation,
        }


def split_directed_label(rel_directed: str) -> tuple[str, Optional[str]]:
    """
    Split directed label into base relation and direction.

    Args:
        rel_directed: e.g., 'Cause-Effect(e1,e2)' or 'Other'

    Returns:
        (base_relation, direction) tuple
    """
    if '(' in rel_directed and rel_directed.endswith(')'):
        base, dir_part = rel_directed.split('(', 1)
        direction = dir_part[:-1]
        return base.strip(), direction.strip()
    return rel_directed, None


class PatternExtractorBase(ABC):
    """Abstract base class for pattern extractors."""

    @property
    @abstractmethod
    def pattern_types(self) -> list[str]:
        """Return list of pattern types this extractor handles."""
        pass

    @abstractmethod
    def extract(
        self,
        doc,
        e1_span,
        e2_span,
        relation: str,
        patterns: dict,
    ) -> None:
        """
        Extract patterns from a single sample and update patterns dict.

        Args:
            doc: spaCy Doc
            e1_span: First entity span
            e2_span: Second entity span
            relation: Directed relation label
            patterns: Dict to update with pattern counts
        """
        pass
