"""Pattern generator for spaCy DependencyMatcher.

This module converts mined SDP signatures into spaCy DependencyMatcher
patterns and exports them to YAML configuration files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .sdp_extractor import SDPSignature


@dataclass
class GeneratedPattern:
    """A generated DependencyMatcher pattern.

    Attributes
    ----------
    pattern_id : str
        Unique identifier for the pattern
    relation_type : str
        Target relation type
    matcher_pattern : List[Dict[str, Any]]
        spaCy DependencyMatcher pattern specification
    signature : SDPSignature
        Original SDP signature
    confidence : float
        Pattern precision/confidence score
    constraints : List[Dict[str, Any]]
        Additional constraints for filtering
    """

    pattern_id: str
    relation_type: str
    matcher_pattern: List[Dict[str, Any]]
    signature: SDPSignature
    confidence: float
    constraints: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML export."""
        return {
            "id": self.pattern_id,
            "relation": self.relation_type,
            "signature": self.signature.to_string(),
            "confidence": round(self.confidence, 4),
            "matcher_pattern": self.matcher_pattern,
            "constraints": self.constraints,
        }


class PatternGenerator:
    """Generate spaCy DependencyMatcher patterns from SDP signatures.

    Parameters
    ----------
    include_lemma_constraints : bool
        Whether to include lemma constraints for trigger words
    """

    # Mapping from SDP dependency labels to spaCy REL_OP
    # ">" means "A is the head of B" (A > B)
    # "<" means "A is a child of B" (A < B)
    DEP_TO_REL_OP = {
        "up": "<",  # child -> head direction
        "down": ">",  # head -> child direction
    }

    def __init__(self, include_lemma_constraints: bool = True):
        self.include_lemma_constraints = include_lemma_constraints
        self._pattern_counter: Dict[str, int] = {}

    def _generate_pattern_id(self, relation_type: str) -> str:
        """Generate a unique pattern ID."""
        # Create abbreviation from relation type
        abbrev = "".join(word[0].upper() for word in relation_type.split("-"))

        if relation_type not in self._pattern_counter:
            self._pattern_counter[relation_type] = 0

        self._pattern_counter[relation_type] += 1
        return f"{abbrev}_{self._pattern_counter[relation_type]:03d}"

    def signature_to_matcher(
        self,
        signature: SDPSignature,
        relation_type: str,
        confidence: float = 0.0,
    ) -> GeneratedPattern:
        """Convert an SDP signature to a DependencyMatcher pattern.

        Parameters
        ----------
        signature : SDPSignature
            The SDP signature to convert
        relation_type : str
            Target relation type
        confidence : float
            Pattern confidence score

        Returns
        -------
        GeneratedPattern
            Generated pattern with matcher specification
        """
        pattern_id = self._generate_pattern_id(relation_type)
        matcher_pattern = self._build_matcher_pattern(signature)

        return GeneratedPattern(
            pattern_id=pattern_id,
            relation_type=relation_type,
            matcher_pattern=matcher_pattern,
            signature=signature,
            confidence=confidence,
        )

    def _build_matcher_pattern(
        self, signature: SDPSignature
    ) -> List[Dict[str, Any]]:
        """Build spaCy DependencyMatcher pattern from signature.

        The pattern follows the structure:
        1. First token is the anchor (RIGHT_ID, RIGHT_ATTRS)
        2. Subsequent tokens reference previous ones (LEFT_ID, REL_OP, RIGHT_ID, RIGHT_ATTRS)
        """
        if not signature.pos_pattern:
            return []

        pattern = []
        pos_list = list(signature.pos_pattern)
        trigger_list = list(signature.trigger_words)
        dep_list = list(signature.dep_pattern)

        # First token (anchor)
        first_attrs = self._build_token_attrs(
            pos_list[0],
            trigger_list[0] if trigger_list else None,
        )
        pattern.append({
            "RIGHT_ID": "token_0",
            "RIGHT_ATTRS": first_attrs,
        })

        # Subsequent tokens
        for i in range(1, len(pos_list)):
            token_attrs = self._build_token_attrs(
                pos_list[i],
                trigger_list[i] if i < len(trigger_list) else None,
            )

            # Get dependency relation between token i-1 and token i
            dep_rel = dep_list[i - 1] if i - 1 < len(dep_list) else None

            # Determine relationship operator
            # In SDP, we traverse the tree, so we use ">" (head-child)
            # or "<" (child-head) based on the path direction
            rel_op = ">"  # Default: previous token is head of current

            token_spec = {
                "LEFT_ID": f"token_{i - 1}",
                "REL_OP": rel_op,
                "RIGHT_ID": f"token_{i}",
                "RIGHT_ATTRS": token_attrs,
            }

            # Add dependency constraint if available
            if dep_rel:
                token_spec["RIGHT_ATTRS"]["DEP"] = dep_rel

            pattern.append(token_spec)

        return pattern

    def _build_token_attrs(
        self, pos: str, trigger_word: Optional[str]
    ) -> Dict[str, Any]:
        """Build token attributes dictionary."""
        attrs: Dict[str, Any] = {"POS": pos}

        # Add lemma constraint for trigger words
        if self.include_lemma_constraints and trigger_word:
            attrs["LEMMA"] = trigger_word

        return attrs

    def export_to_yaml(
        self,
        patterns: Dict[str, List[GeneratedPattern]],
        output_path: str,
    ) -> None:
        """Export patterns to YAML configuration file.

        Parameters
        ----------
        patterns : Dict[str, List[GeneratedPattern]]
            Patterns organized by relation type
        output_path : str
            Output YAML file path
        """
        config = {"version": "1.0", "relations": {}}

        for rel_type, gen_patterns in patterns.items():
            config["relations"][rel_type] = {
                "patterns": [p.to_dict() for p in gen_patterns]
            }

        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"Exported {sum(len(p) for p in patterns.values())} patterns to: {output_path}")

    def load_from_yaml(self, yaml_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Load patterns from YAML configuration.

        Parameters
        ----------
        yaml_path : str
            Path to YAML config file

        Returns
        -------
        Dict[str, List[Dict[str, Any]]]
            Patterns organized by relation type
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return config.get("relations", {})


class PatternOptimizer:
    """Optimize and merge similar patterns.

    This class provides utilities to:
    - Merge similar patterns with slight variations
    - Remove redundant patterns
    - Generalize patterns for better coverage
    """

    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold

    def merge_similar(
        self, patterns: List[GeneratedPattern]
    ) -> List[GeneratedPattern]:
        """Merge similar patterns into more general ones.

        Parameters
        ----------
        patterns : List[GeneratedPattern]
            Patterns to merge

        Returns
        -------
        List[GeneratedPattern]
            Merged patterns
        """
        # Group patterns by structure (POS + DEP pattern)
        groups: Dict[Tuple, List[GeneratedPattern]] = {}

        for p in patterns:
            key = (p.signature.pos_pattern, p.signature.dep_pattern)
            if key not in groups:
                groups[key] = []
            groups[key].append(p)

        # Merge each group
        merged = []
        for group in groups.values():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # Keep the pattern with highest confidence
                best = max(group, key=lambda x: x.confidence)
                # Remove lemma constraints to make it more general
                generalized = self._generalize_pattern(best)
                merged.append(generalized)

        return merged

    def _generalize_pattern(
        self, pattern: GeneratedPattern
    ) -> GeneratedPattern:
        """Remove specific lemma constraints to generalize pattern."""
        new_matcher = []

        for spec in pattern.matcher_pattern:
            new_spec = spec.copy()
            if "RIGHT_ATTRS" in new_spec:
                attrs = new_spec["RIGHT_ATTRS"].copy()
                # Keep LEMMA only for function words (prepositions, etc.)
                if "LEMMA" in attrs and attrs.get("POS") not in ("ADP", "SCONJ"):
                    del attrs["LEMMA"]
                new_spec["RIGHT_ATTRS"] = attrs
            new_matcher.append(new_spec)

        return GeneratedPattern(
            pattern_id=pattern.pattern_id + "_gen",
            relation_type=pattern.relation_type,
            matcher_pattern=new_matcher,
            signature=pattern.signature,
            confidence=pattern.confidence,
            constraints=pattern.constraints,
        )

    def remove_redundant(
        self, patterns: List[GeneratedPattern]
    ) -> List[GeneratedPattern]:
        """Remove redundant patterns that are subsets of others.

        Parameters
        ----------
        patterns : List[GeneratedPattern]
            Patterns to filter

        Returns
        -------
        List[GeneratedPattern]
            Non-redundant patterns
        """
        # Sort by specificity (more constraints = more specific)
        sorted_patterns = sorted(
            patterns,
            key=lambda p: len(p.matcher_pattern),
            reverse=True,
        )

        non_redundant = []
        seen_structures = set()

        for p in sorted_patterns:
            # Create structure signature
            struct = (p.signature.pos_pattern, p.signature.dep_pattern)

            if struct not in seen_structures:
                non_redundant.append(p)
                seen_structures.add(struct)

        return non_redundant
