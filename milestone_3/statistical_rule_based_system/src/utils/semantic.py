"""Semantic feature extraction using WordNet and FrameNet.

This module provides:
- WordNet synset and lexname lookup
- Hypernym extraction
- FrameNet frame lookup
- Frame-to-relation mapping
- Prepositional structure analysis
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import nltk


def setup_nltk() -> None:
    """Download required NLTK corpora. Run once before using this module."""
    print("Downloading NLTK resources...")
    nltk.download("wordnet", quiet=True)
    nltk.download("framenet_v17", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    from nltk.corpus import framenet as fn
    from nltk.corpus import wordnet as wn

    try:
        list(wn.all_synsets())[:1]  # Just check access
        list(fn.frames())[:1]
        print("✓ NLTK resources loaded.")
    except Exception as e:
        print(f"Warning: NLTK load failed: {e}")


def _get_wn():
    """Lazy accessor for WordNet."""
    try:
        from nltk.corpus import wordnet as wn
        return wn
    except LookupError:
        return None


def _get_fn():
    """Lazy accessor for FrameNet."""
    try:
        from nltk.corpus import framenet as fn
        return fn
    except LookupError:
        return None


@lru_cache(maxsize=10000)
def get_synset_name(lemma: str, pos: str = "v") -> Optional[str]:
    """
    Get the first synset name for a lemma.

    Args:
        lemma: Word lemma.
        pos: Part of speech ('v' for verb, 'n' for noun).

    Returns:
        Synset name or None if not found.
    """
    wn = _get_wn()
    if wn is None:
        return None
    wn_pos = wn.VERB if pos == "v" else wn.NOUN
    synsets = wn.synsets(lemma.lower(), pos=wn_pos)
    return synsets[0].name() if synsets else None


@lru_cache(maxsize=10000)
def get_lexname(lemma: str, pos: str = "v") -> Optional[str]:
    """
    Get the lexicographer filename (lexname) for a lemma.

    Args:
        lemma: Word lemma.
        pos: Part of speech ('v' for verb, 'n' for noun).

    Returns:
        Lexname or None if not found.
    """
    wn = _get_wn()
    if wn is None:
        return None
    wn_pos = wn.VERB if pos == "v" else wn.NOUN
    synsets = wn.synsets(lemma.lower(), pos=wn_pos)
    return synsets[0].lexname() if synsets else None


@lru_cache(maxsize=10000)
def get_shallow_hypernym(lemma: str, pos: str = "n") -> Optional[str]:
    """
    Get the first hypernym for a lemma.

    Args:
        lemma: Word lemma.
        pos: Part of speech ('n' for noun, 'v' for verb).

    Returns:
        Hypernym synset name or None if not found.
    """
    wn = _get_wn()
    if wn is None:
        return None
    wn_pos = wn.NOUN if pos == "n" else wn.VERB
    synsets = wn.synsets(lemma.lower(), pos=wn_pos)
    if not synsets:
        return None
    hypernyms = synsets[0].hypernyms()
    return hypernyms[0].name() if hypernyms else None


@lru_cache(maxsize=5000)
def get_frames_for_verb(lemma: str, max_frames: int = 3) -> tuple[str, ...]:
    """
    Get FrameNet frames for a verb lemma.

    Args:
        lemma: Verb lemma.
        max_frames: Maximum number of frames to return.

    Returns:
        Tuple of frame names.
    """
    fn = _get_fn()
    if fn is None:
        return ()
    try:
        import re
        pattern = r"(?i)^" + re.escape(lemma) + r"\."
        lus = fn.lus(pattern)
        frames = []
        for lu in lus:
            frame_name = lu.frame.name
            if frame_name not in frames:
                frames.append(frame_name)
                if len(frames) >= max_frames:
                    break
        return tuple(frames)
    except Exception:
        return ()


class FrameMapper:
    """Maps semantic frames to relation types based on training data."""

    def __init__(self):
        self.frame_relation_counts = {}
        self.frame_to_relation = {}
        self._fitted = False

    def learn_from_data(
        self,
        verb_lemmas: list[str],
        relations: list[str],
        min_support: int = 3,
        min_precision: float = 0.6,
    ) -> None:
        """
        Learn frame-to-relation mappings from training data.

        Args:
            verb_lemmas: List of verb lemmas.
            relations: Corresponding relation labels.
            min_support: Minimum frame occurrence count.
            min_precision: Minimum precision for mapping.
        """
        from collections import defaultdict

        self.frame_relation_counts = defaultdict(lambda: defaultdict(int))

        for lemma, relation in zip(verb_lemmas, relations):
            frames = get_frames_for_verb(lemma)
            for frame in frames:
                self.frame_relation_counts[frame][relation] += 1

        self.frame_to_relation = {}
        for frame, rel_counts in self.frame_relation_counts.items():
            total = sum(rel_counts.values())
            if total < min_support:
                continue
            best_rel = max(rel_counts, key=rel_counts.get)
            precision = rel_counts[best_rel] / total
            if precision >= min_precision:
                self.frame_to_relation[frame] = best_rel

        self._fitted = True
        print(f"Learned {len(self.frame_to_relation)} frame-relation mappings")

    def predict_relation(self, verb_lemma: str) -> Optional[str]:
        """
        Predict relation based on verb's semantic frame.

        Args:
            verb_lemma: Verb lemma.

        Returns:
            Predicted relation or None.
        """
        if not self._fitted:
            return None
        frames = get_frames_for_verb(verb_lemma)
        for frame in frames:
            if frame in self.frame_to_relation:
                return self.frame_to_relation[frame]
        return None


@dataclass
class PrepStructureFeatures:
    """Features extracted from prepositional structure analysis."""

    prep: str
    gov_lemma: str
    gov_lexname: Optional[str]
    e1_role: str
    e2_role: str
    prep_obj_lemma: str
    prep_obj_lexname: Optional[str]

    def to_pattern_key(self, pattern_type: str = "PREP_STRUCT") -> tuple:
        """
        Convert features to a pattern key tuple.

        Args:
            pattern_type: Type of pattern key to generate.

        Returns:
            Tuple suitable for use as a dictionary key.
        """
        if pattern_type == "PREP_STRUCT":
            return (
                "PREP_STRUCT",
                self.prep,
                self.gov_lemma,
                self.e1_role,
                self.e2_role,
            )
        elif pattern_type == "PREP_GOV_LEX":
            return ("PREP_GOV_LEX", self.prep, self.gov_lexname)
        elif pattern_type == "PREP_ROLES":
            return ("PREP_ROLES", self.prep, self.e1_role, self.e2_role)
        return (
            "PREP_STRUCT",
            self.prep,
            self.gov_lemma,
            self.e1_role,
            self.e2_role,
        )


def _get_role_to_ancestor(token, ancestor, max_depth: int = 5) -> str:
    """Get the dependency role of token relative to an ancestor."""
    current = token
    for _ in range(max_depth):
        if current == ancestor:
            return token.dep_
        if current.head == current:
            break
        if current.head == ancestor:
            return current.dep_
        current = current.head
    return "none"


def extract_prep_structure_features(
    doc,
    e1_span,
    e2_span,
) -> Optional[PrepStructureFeatures]:
    """
    Extract prepositional structure features between entities.

    Args:
        doc: spaCy Doc.
        e1_span: Entity 1 span.
        e2_span: Entity 2 span.

    Returns:
        PrepStructureFeatures or None if no preposition found.
    """
    e1_root = e1_span.root
    e2_root = e2_span.root

    start = min(e1_span.end, e2_span.end)
    end = max(e1_span.start, e2_span.start)

    for token in doc[start:end]:
        if token.pos_ == "ADP":
            prep = token.lemma_.lower()
            gov = token.head
            gov_lemma = gov.lemma_.lower()
            gov_pos = "v" if gov.pos_ == "VERB" else "n"
            gov_lexname = get_lexname(gov_lemma, gov_pos)

            e1_role = _get_role_to_ancestor(e1_root, gov)
            e2_role = _get_role_to_ancestor(e2_root, gov)

            prep_obj_lemma = ""
            prep_obj_lexname = None

            for child in token.children:
                if child.dep_ == "pobj":
                    prep_obj_lemma = child.lemma_.lower()
                    prep_obj_lexname = get_lexname(prep_obj_lemma, "n")
                    break

            return PrepStructureFeatures(
                prep=prep,
                gov_lemma=gov_lemma,
                gov_lexname=gov_lexname,
                e1_role=e1_role,
                e2_role=e2_role,
                prep_obj_lemma=prep_obj_lemma,
                prep_obj_lexname=prep_obj_lexname,
            )

    return None

# =============================================================================
# Rule Priority Tiers
# =============================================================================


class RulePriority:
    """
    Rule priority tier constants.

    Higher tier = higher priority = checked first.
    """

    TIER_1_COMBINED = 100 # Combined Syntax + Semantics (Highest Priority)
    TIER_2_PREP_STRUCTURE = 80  # Preposition with specific structure
    TIER_3_LEXNAME_HYPERNYM = 60  # Lexname + hypernym pattern
    TIER_4_BIGRAM_DEP = 40  # Bigram + dependency pattern
    TIER_5_LEXICAL = 20  # Fallback (Simple lexical, etc.)
    TIER_6_FALLBACK = 0  # Default/Other


def compute_rule_priority(
    pattern_type: str,
    precision: float,
    support: int,
    has_synset: bool = False,
    has_frame: bool = False,
    rule_key: tuple = None,
) -> float:
    """
    Compute priority score for a rule with deterministic tie-breaking.

    Priority = base_tier * 100000 + precision_bonus + tie_break

    Args:
        pattern_type: Type of pattern (SYNSET, PREP_STRUCT, LEXNAME, etc.)
        precision: Rule precision (0-1)
        support: Rule support count
        has_synset: Whether rule uses synset features
        has_frame: Whether rule uses frame features
        rule_key: Tuple representing rule pattern for stable tie-breaking.

    Returns:
        Priority score (higher = checked first)
    """
    import hashlib

    # Base tier calculation (Always applied)
    if pattern_type in ('PREP_STRUCT_LEXNAME', 'PREP_STRUCT_HYPERNYM'):
        base = RulePriority.TIER_1_COMBINED
    elif pattern_type == 'PREP_STRUCT':
        base = RulePriority.TIER_2_PREP_STRUCTURE
    elif pattern_type in ('LEXNAME', 'HYPERNYM'):
        base = RulePriority.TIER_3_LEXNAME_HYPERNYM
    elif pattern_type in ('BIGRAM', 'DEP_VERB'):
        base = RulePriority.TIER_4_BIGRAM_DEP
    else:
        # Fallback (including FRAME, SYNSET, LEMMA, etc.)
        base = RulePriority.TIER_5_LEXICAL

    # Precision bonus (max ~10000)
    precision_bonus = precision * 10000
    
    # Stable tie-breaking: use deterministic hash of rule_key
    # This ensures identical (tier, precision) still have deterministic order
    tie_break = 0
    if rule_key:
        # Convert rule_key to stable hash (0.0 to 0.9999...)
        key_str = str(rule_key)
        hash_digest = hashlib.md5(key_str.encode()).hexdigest()
        # Take first 8 hex chars and normalize to [0, 1)
        tie_break = int(hash_digest[:8], 16) / (16**8)

    return base * 100000 + precision_bonus + tie_break
