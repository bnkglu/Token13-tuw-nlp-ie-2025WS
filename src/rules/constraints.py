"""Constraint engine for filtering pattern matches.

This module provides configurable constraints to filter out false positive
matches from rule-based extraction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Type

from spacy.tokens import Doc, Span, Token


@dataclass
class MatchResult:
    """Result of a pattern match.

    Attributes
    ----------
    pattern_id : str
        ID of the matched pattern
    relation_type : str
        Predicted relation type
    e1_span : Tuple[int, int]
        Token indices for entity 1 (start, end)
    e2_span : Tuple[int, int]
        Token indices for entity 2 (start, end)
    confidence : float
        Pattern confidence score
    matched_tokens : List[int]
        All token indices in the match
    """

    pattern_id: str
    relation_type: str
    e1_span: Tuple[int, int]
    e2_span: Tuple[int, int]
    confidence: float
    matched_tokens: List[int] = field(default_factory=list)

    @property
    def e1_token(self) -> int:
        """Return primary token index for e1."""
        return self.e1_span[0]

    @property
    def e2_token(self) -> int:
        """Return primary token index for e2."""
        return self.e2_span[0]


class Constraint(ABC):
    """Base class for match constraints."""

    @abstractmethod
    def validate(self, match: MatchResult, doc: Doc) -> bool:
        """Validate if match passes this constraint.

        Parameters
        ----------
        match : MatchResult
            The match to validate
        doc : Doc
            spaCy Doc containing the match

        Returns
        -------
        bool
            True if match passes, False otherwise
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize constraint to dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Constraint":
        """Create constraint from dictionary."""
        pass


class EntityTypeConstraint(Constraint):
    """Constraint based on spaCy NER entity types.

    Parameters
    ----------
    entity : str
        Which entity to check ("e1" or "e2")
    allowed_types : Optional[Set[str]]
        Allowed NER types (None = any)
    blocked_types : Optional[Set[str]]
        Blocked NER types
    """

    def __init__(
        self,
        entity: str = "e2",
        allowed_types: Optional[Set[str]] = None,
        blocked_types: Optional[Set[str]] = None,
    ):
        self.entity = entity
        self.allowed_types = allowed_types
        self.blocked_types = blocked_types or set()

    def validate(self, match: MatchResult, doc: Doc) -> bool:
        token_idx = match.e1_token if self.entity == "e1" else match.e2_token

        if token_idx >= len(doc):
            return False

        token = doc[token_idx]

        # Check NER type
        ent_type = token.ent_type_ if token.ent_type_ else None

        if self.allowed_types and ent_type not in self.allowed_types:
            return False

        if ent_type in self.blocked_types:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "entity_type",
            "entity": self.entity,
            "allowed_types": list(self.allowed_types) if self.allowed_types else None,
            "blocked_types": list(self.blocked_types),
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "EntityTypeConstraint":
        return cls(
            entity=config.get("entity", "e2"),
            allowed_types=set(config["allowed_types"]) if config.get("allowed_types") else None,
            blocked_types=set(config.get("blocked_types", [])),
        )


class LemmaConstraint(Constraint):
    """Constraint based on token lemmas.

    Parameters
    ----------
    entity : str
        Which entity to check ("e1" or "e2")
    allowed_lemmas : Optional[Set[str]]
        Allowed lemmas (None = any)
    blocked_lemmas : Optional[Set[str]]
        Blocked lemmas
    """

    def __init__(
        self,
        entity: str = "e2",
        allowed_lemmas: Optional[Set[str]] = None,
        blocked_lemmas: Optional[Set[str]] = None,
    ):
        self.entity = entity
        self.allowed_lemmas = allowed_lemmas
        self.blocked_lemmas = blocked_lemmas or set()

    def validate(self, match: MatchResult, doc: Doc) -> bool:
        token_idx = match.e1_token if self.entity == "e1" else match.e2_token

        if token_idx >= len(doc):
            return False

        lemma = doc[token_idx].lemma_.lower()

        if self.allowed_lemmas and lemma not in self.allowed_lemmas:
            return False

        if lemma in self.blocked_lemmas:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "lemma",
            "entity": self.entity,
            "allowed_lemmas": list(self.allowed_lemmas) if self.allowed_lemmas else None,
            "blocked_lemmas": list(self.blocked_lemmas),
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "LemmaConstraint":
        return cls(
            entity=config.get("entity", "e2"),
            allowed_lemmas=set(config["allowed_lemmas"]) if config.get("allowed_lemmas") else None,
            blocked_lemmas=set(config.get("blocked_lemmas", [])),
        )


class POSConstraint(Constraint):
    """Constraint based on POS tags.

    Parameters
    ----------
    entity : str
        Which entity to check ("e1" or "e2")
    allowed_pos : Optional[Set[str]]
        Allowed POS tags (None = any)
    blocked_pos : Optional[Set[str]]
        Blocked POS tags
    """

    def __init__(
        self,
        entity: str = "e2",
        allowed_pos: Optional[Set[str]] = None,
        blocked_pos: Optional[Set[str]] = None,
    ):
        self.entity = entity
        self.allowed_pos = allowed_pos
        self.blocked_pos = blocked_pos or set()

    def validate(self, match: MatchResult, doc: Doc) -> bool:
        token_idx = match.e1_token if self.entity == "e1" else match.e2_token

        if token_idx >= len(doc):
            return False

        pos = doc[token_idx].pos_

        if self.allowed_pos and pos not in self.allowed_pos:
            return False

        if pos in self.blocked_pos:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "pos",
            "entity": self.entity,
            "allowed_pos": list(self.allowed_pos) if self.allowed_pos else None,
            "blocked_pos": list(self.blocked_pos),
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "POSConstraint":
        return cls(
            entity=config.get("entity", "e2"),
            allowed_pos=set(config["allowed_pos"]) if config.get("allowed_pos") else None,
            blocked_pos=set(config.get("blocked_pos", [])),
        )


class DistanceConstraint(Constraint):
    """Constraint based on token distance between entities.

    Parameters
    ----------
    min_distance : int
        Minimum token distance
    max_distance : int
        Maximum token distance
    """

    def __init__(self, min_distance: int = 0, max_distance: int = 20):
        self.min_distance = min_distance
        self.max_distance = max_distance

    def validate(self, match: MatchResult, doc: Doc) -> bool:
        distance = abs(match.e2_token - match.e1_token)
        return self.min_distance <= distance <= self.max_distance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "distance",
            "min_distance": self.min_distance,
            "max_distance": self.max_distance,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "DistanceConstraint":
        return cls(
            min_distance=config.get("min_distance", 0),
            max_distance=config.get("max_distance", 20),
        )


class DependencyConstraint(Constraint):
    """Constraint based on dependency relations.

    Parameters
    ----------
    entity : str
        Which entity to check ("e1" or "e2")
    allowed_deps : Optional[Set[str]]
        Allowed dependency relations
    blocked_deps : Optional[Set[str]]
        Blocked dependency relations
    """

    def __init__(
        self,
        entity: str = "e2",
        allowed_deps: Optional[Set[str]] = None,
        blocked_deps: Optional[Set[str]] = None,
    ):
        self.entity = entity
        self.allowed_deps = allowed_deps
        self.blocked_deps = blocked_deps or set()

    def validate(self, match: MatchResult, doc: Doc) -> bool:
        token_idx = match.e1_token if self.entity == "e1" else match.e2_token

        if token_idx >= len(doc):
            return False

        dep = doc[token_idx].dep_

        if self.allowed_deps and dep not in self.allowed_deps:
            return False

        if dep in self.blocked_deps:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "dependency",
            "entity": self.entity,
            "allowed_deps": list(self.allowed_deps) if self.allowed_deps else None,
            "blocked_deps": list(self.blocked_deps),
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "DependencyConstraint":
        return cls(
            entity=config.get("entity", "e2"),
            allowed_deps=set(config["allowed_deps"]) if config.get("allowed_deps") else None,
            blocked_deps=set(config.get("blocked_deps", [])),
        )


# Registry of constraint types
CONSTRAINT_REGISTRY: Dict[str, Type[Constraint]] = {
    "entity_type": EntityTypeConstraint,
    "lemma": LemmaConstraint,
    "pos": POSConstraint,
    "distance": DistanceConstraint,
    "dependency": DependencyConstraint,
}


class ConstraintEngine:
    """Engine for managing and applying constraints.

    Parameters
    ----------
    default_constraints : Optional[List[Constraint]]
        Default constraints applied to all patterns
    """

    def __init__(self, default_constraints: Optional[List[Constraint]] = None):
        self.default_constraints = default_constraints or []
        self._pattern_constraints: Dict[str, List[Constraint]] = {}

    def add_constraint(
        self,
        pattern_id: str,
        constraint: Constraint,
    ) -> None:
        """Add a constraint for a specific pattern.

        Parameters
        ----------
        pattern_id : str
            Pattern ID to add constraint to
        constraint : Constraint
            The constraint to add
        """
        if pattern_id not in self._pattern_constraints:
            self._pattern_constraints[pattern_id] = []
        self._pattern_constraints[pattern_id].append(constraint)

    def add_default_constraint(self, constraint: Constraint) -> None:
        """Add a default constraint applied to all patterns.

        Parameters
        ----------
        constraint : Constraint
            The constraint to add
        """
        self.default_constraints.append(constraint)

    def validate_match(self, match: MatchResult, doc: Doc) -> bool:
        """Validate a match against all applicable constraints.

        Parameters
        ----------
        match : MatchResult
            The match to validate
        doc : Doc
            spaCy Doc containing the match

        Returns
        -------
        bool
            True if match passes all constraints
        """
        # Apply default constraints
        for constraint in self.default_constraints:
            if not constraint.validate(match, doc):
                return False

        # Apply pattern-specific constraints
        pattern_constraints = self._pattern_constraints.get(match.pattern_id, [])
        for constraint in pattern_constraints:
            if not constraint.validate(match, doc):
                return False

        return True

    def filter_matches(
        self, matches: List[MatchResult], doc: Doc
    ) -> List[MatchResult]:
        """Filter matches by applying all constraints.

        Parameters
        ----------
        matches : List[MatchResult]
            Matches to filter
        doc : Doc
            spaCy Doc containing the matches

        Returns
        -------
        List[MatchResult]
            Matches that pass all constraints
        """
        return [m for m in matches if self.validate_match(m, doc)]

    def load_from_config(self, config: Dict[str, Any]) -> None:
        """Load constraints from configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration with pattern constraints
        """
        for pattern_id, constraints_config in config.items():
            for c_config in constraints_config:
                constraint_type = c_config.get("type")
                if constraint_type in CONSTRAINT_REGISTRY:
                    constraint_cls = CONSTRAINT_REGISTRY[constraint_type]
                    constraint = constraint_cls.from_dict(c_config)
                    self.add_constraint(pattern_id, constraint)

    def to_config(self) -> Dict[str, List[Dict[str, Any]]]:
        """Export constraints to configuration dictionary.

        Returns
        -------
        Dict[str, List[Dict[str, Any]]]
            Configuration dictionary
        """
        config = {}
        for pattern_id, constraints in self._pattern_constraints.items():
            config[pattern_id] = [c.to_dict() for c in constraints]
        return config


def create_common_constraints() -> List[Constraint]:
    """Create commonly useful default constraints.

    Returns
    -------
    List[Constraint]
        List of common constraints
    """
    return [
        # Block temporal entities for non-temporal relations
        EntityTypeConstraint(
            entity="e2",
            blocked_types={"DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT"},
        ),
        # Require reasonable distance
        DistanceConstraint(min_distance=1, max_distance=15),
    ]
