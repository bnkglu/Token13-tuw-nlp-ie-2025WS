"""Rule matcher for applying learned patterns to new text.

This module provides the deployment interface for rule-based relation
extraction using mined SDP patterns.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import spacy
import yaml
from spacy.matcher import DependencyMatcher
from spacy.tokens import Doc

from .constraints import ConstraintEngine, MatchResult, create_common_constraints


@dataclass
class ExtractionResult:
    """Result of relation extraction from a document.

    Attributes
    ----------
    text : str
        Original text
    matches : List[MatchResult]
        All matches found
    predictions : Dict[Tuple[int, int], str]
        Predicted relations for entity pairs (e1_idx, e2_idx) -> relation
    """

    text: str
    matches: List[MatchResult] = field(default_factory=list)
    predictions: Dict[Tuple[int, int], str] = field(default_factory=dict)


class RuleMatcher:
    """Apply rule-based patterns to extract relations.

    Parameters
    ----------
    model_name : str
        spaCy model to use for parsing
    use_default_constraints : bool
        Whether to use common default constraints
    """

    def __init__(
        self,
        model_name: str = "en_core_web_lg",
        use_default_constraints: bool = True,
    ):
        self.nlp = spacy.load(model_name)
        self.matcher = DependencyMatcher(self.nlp.vocab)
        self.constraint_engine = ConstraintEngine()

        if use_default_constraints:
            for constraint in create_common_constraints():
                self.constraint_engine.add_default_constraint(constraint)

        # Pattern metadata
        self._pattern_info: Dict[str, Dict[str, Any]] = {}
        self._relation_patterns: Dict[str, List[str]] = defaultdict(list)

    def load_patterns(self, yaml_path: str) -> int:
        """Load patterns from YAML configuration.

        Parameters
        ----------
        yaml_path : str
            Path to YAML config file

        Returns
        -------
        int
            Number of patterns loaded
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        count = 0
        relations = config.get("relations", {})

        for rel_type, rel_config in relations.items():
            patterns = rel_config.get("patterns", [])

            for pattern_config in patterns:
                pattern_id = pattern_config["id"]
                matcher_pattern = pattern_config.get("matcher_pattern", [])
                confidence = pattern_config.get("confidence", 0.0)
                constraints = pattern_config.get("constraints", [])

                if not matcher_pattern:
                    continue

                # Add to spaCy matcher
                try:
                    self.matcher.add(pattern_id, [matcher_pattern])
                except Exception as e:
                    print(f"Warning: Failed to add pattern {pattern_id}: {e}")
                    continue

                # Store metadata
                self._pattern_info[pattern_id] = {
                    "relation": rel_type,
                    "confidence": confidence,
                    "signature": pattern_config.get("signature", ""),
                }

                # Track patterns by relation
                self._relation_patterns[rel_type].append(pattern_id)

                # Load constraints
                if constraints:
                    self.constraint_engine.load_from_config({pattern_id: constraints})

                count += 1

        print(f"Loaded {count} patterns for {len(relations)} relation types")
        return count

    def add_pattern(
        self,
        pattern_id: str,
        relation_type: str,
        matcher_pattern: List[Dict[str, Any]],
        confidence: float = 0.0,
    ) -> None:
        """Add a single pattern programmatically.

        Parameters
        ----------
        pattern_id : str
            Unique pattern identifier
        relation_type : str
            Target relation type
        matcher_pattern : List[Dict[str, Any]]
            spaCy DependencyMatcher pattern
        confidence : float
            Pattern confidence score
        """
        self.matcher.add(pattern_id, [matcher_pattern])

        self._pattern_info[pattern_id] = {
            "relation": relation_type,
            "confidence": confidence,
        }
        self._relation_patterns[relation_type].append(pattern_id)

    def match(self, doc: Doc) -> List[MatchResult]:
        """Find all pattern matches in a document.

        Parameters
        ----------
        doc : Doc
            spaCy processed document

        Returns
        -------
        List[MatchResult]
            All matches found (before constraint filtering)
        """
        matches = []
        raw_matches = self.matcher(doc)

        for match_id, token_ids in raw_matches:
            pattern_id = self.nlp.vocab.strings[match_id]
            info = self._pattern_info.get(pattern_id, {})

            # Identify entities (first and last NOUN in match)
            nouns = [i for i in token_ids if doc[i].pos_ == "NOUN"]

            if len(nouns) < 2:
                # Fall back to first and last tokens
                e1_idx = token_ids[0]
                e2_idx = token_ids[-1]
            else:
                e1_idx = nouns[0]
                e2_idx = nouns[-1]

            match_result = MatchResult(
                pattern_id=pattern_id,
                relation_type=info.get("relation", "Unknown"),
                e1_span=(e1_idx, e1_idx + 1),
                e2_span=(e2_idx, e2_idx + 1),
                confidence=info.get("confidence", 0.0),
                matched_tokens=list(token_ids),
            )
            matches.append(match_result)

        return matches

    def extract(
        self,
        text: str,
        apply_constraints: bool = True,
    ) -> ExtractionResult:
        """Extract relations from text.

        Parameters
        ----------
        text : str
            Input text
        apply_constraints : bool
            Whether to apply constraint filtering

        Returns
        -------
        ExtractionResult
            Extraction results with matches and predictions
        """
        doc = self.nlp(text)
        matches = self.match(doc)

        if apply_constraints:
            matches = self.constraint_engine.filter_matches(matches, doc)

        # Resolve predictions (use highest confidence match for each entity pair)
        predictions = self._resolve_predictions(matches)

        return ExtractionResult(
            text=text,
            matches=matches,
            predictions=predictions,
        )

    def _resolve_predictions(
        self, matches: List[MatchResult]
    ) -> Dict[Tuple[int, int], str]:
        """Resolve multiple matches to single predictions.

        Uses highest confidence match for each entity pair.

        Parameters
        ----------
        matches : List[MatchResult]
            All matches

        Returns
        -------
        Dict[Tuple[int, int], str]
            (e1_idx, e2_idx) -> relation type
        """
        # Group by entity pair
        pair_matches: Dict[Tuple[int, int], List[MatchResult]] = defaultdict(list)

        for m in matches:
            pair = (m.e1_token, m.e2_token)
            pair_matches[pair].append(m)

        # Select best match for each pair
        predictions = {}
        for pair, pair_ms in pair_matches.items():
            best = max(pair_ms, key=lambda x: x.confidence)
            predictions[pair] = best.relation_type

        return predictions

    def extract_for_semeval(
        self,
        doc: Doc,
        e1_token_idx: int,
        e2_token_idx: int,
        apply_constraints: bool = True,
    ) -> Optional[str]:
        """Extract relation for a specific entity pair (SemEval format).

        Parameters
        ----------
        doc : Doc
            spaCy processed document
        e1_token_idx : int
            Token index of entity 1
        e2_token_idx : int
            Token index of entity 2
        apply_constraints : bool
            Whether to apply constraint filtering

        Returns
        -------
        Optional[str]
            Predicted relation with direction, or None
        """
        matches = self.match(doc)

        if apply_constraints:
            matches = self.constraint_engine.filter_matches(matches, doc)

        # Find matches that include both entities
        relevant_matches = []
        for m in matches:
            matched_set = set(m.matched_tokens)
            if e1_token_idx in matched_set or e2_token_idx in matched_set:
                # Check if entities are at the expected positions
                nouns_in_match = [
                    i for i in m.matched_tokens if doc[i].pos_ == "NOUN"
                ]
                if e1_token_idx in nouns_in_match and e2_token_idx in nouns_in_match:
                    relevant_matches.append(m)

        if not relevant_matches:
            return None

        # Return best match
        best = max(relevant_matches, key=lambda x: x.confidence)

        # Determine direction based on entity positions in match
        if best.e1_token <= best.e2_token:
            direction = "(e1,e2)"
        else:
            direction = "(e2,e1)"

        return f"{best.relation_type}{direction}"

    def batch_extract(
        self,
        texts: List[str],
        apply_constraints: bool = True,
    ) -> List[ExtractionResult]:
        """Extract relations from multiple texts efficiently.

        Parameters
        ----------
        texts : List[str]
            Input texts
        apply_constraints : bool
            Whether to apply constraint filtering

        Returns
        -------
        List[ExtractionResult]
            Extraction results for each text
        """
        results = []

        for doc in self.nlp.pipe(texts):
            matches = self.match(doc)

            if apply_constraints:
                matches = self.constraint_engine.filter_matches(matches, doc)

            predictions = self._resolve_predictions(matches)

            results.append(ExtractionResult(
                text=doc.text,
                matches=matches,
                predictions=predictions,
            ))

        return results

    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded patterns.

        Returns
        -------
        Dict[str, Any]
            Pattern statistics
        """
        return {
            "total_patterns": len(self._pattern_info),
            "patterns_by_relation": {
                rel: len(patterns)
                for rel, patterns in self._relation_patterns.items()
            },
            "relations": list(self._relation_patterns.keys()),
        }

    def explain_match(self, match: MatchResult, doc: Doc) -> str:
        """Generate human-readable explanation of a match.

        Parameters
        ----------
        match : MatchResult
            The match to explain
        doc : Doc
            spaCy Doc containing the match

        Returns
        -------
        str
            Human-readable explanation
        """
        info = self._pattern_info.get(match.pattern_id, {})

        tokens = [doc[i].text for i in match.matched_tokens]
        e1_text = doc[match.e1_token].text
        e2_text = doc[match.e2_token].text

        explanation = [
            f"Pattern: {match.pattern_id}",
            f"Relation: {match.relation_type}",
            f"Confidence: {match.confidence:.2%}",
            f"Entity 1: {e1_text}",
            f"Entity 2: {e2_text}",
            f"Matched path: {' -> '.join(tokens)}",
            f"Signature: {info.get('signature', 'N/A')}",
        ]

        return "\n".join(explanation)


