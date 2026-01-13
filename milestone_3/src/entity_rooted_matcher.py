"""
Entity-rooted pattern matching for Milestone 3.

Instead of using DependencyMatcher freely (which finds patterns anywhere in the doc),
this module checks if pattern structures exist specifically at entity root positions.
This guarantees 100% anchoring alignment.

Key advantage: No anchoring failures since we start from entity positions.

Enhanced features:
- WordNet hypernym matching for generalized pattern matching
- FrameNet semantic role scoring for validation
- ML resolver integration for Other class refinement
"""

import json
from pathlib import Path
from tqdm.auto import tqdm
from typing import Tuple, Optional

# Try to import WordNet and FrameNet modules
try:
    from wordnet_augmentor import (
        hypernym_matches as wn_hypernym_matches,
        relation_specific_match,
        WORDNET_AVAILABLE,
        # Entity type disambiguation
        get_entity_type,
        disambiguate_by_entity_type,
        ENTITY_TYPE_RELATION_RULES,
        ENTITY_TYPE_RELATION_RULES_MEDIUM,
    )
    ENTITY_TYPE_DISAMBIGUATION_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False
    ENTITY_TYPE_DISAMBIGUATION_AVAILABLE = False
    wn_hypernym_matches = None
    relation_specific_match = None
    get_entity_type = None
    disambiguate_by_entity_type = None
    ENTITY_TYPE_RELATION_RULES = {}
    ENTITY_TYPE_RELATION_RULES_MEDIUM = {}

try:
    from framenet_scorer import score_frame_compatibility
except ImportError:
    score_frame_compatibility = None


# Load concept clusters for concept matching
def load_concept_clusters():
    """Load concept clusters from data directory."""
    data_dir = Path(__file__).parent.parent / "data"
    concept_path = data_dir / "concept_clusters.json"

    if concept_path.exists():
        with open(concept_path) as f:
            data = json.load(f)
        clusters = data.get('expanded_clusters', {})

        # Build lemma -> concept reverse mapping
        lemma_to_concept = {}
        for concept_name, cluster in clusters.items():
            words = cluster.get('seeds', []) + cluster.get('expanded', [])
            for word in words:
                lemma_to_concept[word.lower()] = concept_name

        return clusters, lemma_to_concept
    return {}, {}


# Global concept lookup (loaded once)
CONCEPT_CLUSTERS, LEMMA_TO_CONCEPT = load_concept_clusters()


def lemma_matches(token_lemma, pattern_lemma):
    """
    Check if token lemma matches pattern lemma (handles concepts).

    Args:
        token_lemma: Actual token lemma from doc
        pattern_lemma: Pattern's expected lemma (may be concept like "CAUSE_VERB")

    Returns:
        bool: True if matches
    """
    token_lemma = token_lemma.lower()
    pattern_lemma_lower = pattern_lemma.lower()

    # Check literal match first
    if token_lemma == pattern_lemma_lower:
        return True

    # Check if pattern_lemma is a concept (uppercase with underscore)
    if pattern_lemma.isupper() and '_' in pattern_lemma:
        # Get concept cluster words
        cluster = CONCEPT_CLUSTERS.get(pattern_lemma, {})
        words = cluster.get('seeds', []) + cluster.get('expanded', [])
        if token_lemma in [w.lower() for w in words]:
            return True

    return False


def lemma_matches_enhanced(
    token_lemma: str,
    pattern_lemma: str,
    relation: Optional[str] = None,
    pos: str = 'v'
) -> Tuple[bool, float]:
    """
    Enhanced lemma matching with WordNet hypernyms and confidence scoring.

    Conservative matching - only use curated relation-specific word lists,
    not general hypernym matching (which is too permissive).

    Matching priority (highest to lowest confidence):
    1. Direct match (confidence=1.0)
    2. Concept cluster match (confidence=0.9)
    3. Relation-specific word list match (confidence=0.85)

    Args:
        token_lemma: Actual token lemma from doc
        pattern_lemma: Pattern's expected lemma
        relation: Optional relation type for relation-specific matching
        pos: Part of speech for WordNet lookup ('v' for verb, 'n' for noun)

    Returns:
        Tuple of (matched, confidence)
    """
    token_lemma = token_lemma.lower()
    pattern_lemma_lower = pattern_lemma.lower()

    # 1. Direct match - highest confidence
    if token_lemma == pattern_lemma_lower:
        return True, 1.0

    # 2. Concept cluster match
    if pattern_lemma.isupper() and '_' in pattern_lemma:
        cluster = CONCEPT_CLUSTERS.get(pattern_lemma, {})
        words = cluster.get('seeds', []) + cluster.get('expanded', [])
        if token_lemma in [w.lower() for w in words]:
            return True, 0.9

    # 3. Relation-specific word list match (conservative - only curated lists)
    # Only match if BOTH token and pattern are in the same relation's word group
    if relation and relation_specific_match is not None:
        # Check if token is in any group for this relation
        token_matched, token_conf, token_group = relation_specific_match(token_lemma, relation, pos)
        # Check if pattern is in any group for this relation
        pattern_matched, pattern_conf, pattern_group = relation_specific_match(pattern_lemma_lower, relation, pos)

        # Only consider a match if both are in the SAME group
        if token_matched and pattern_matched and token_group == pattern_group:
            return True, min(token_conf, pattern_conf)

    # DISABLED: General WordNet hypernym match (too permissive, hurts accuracy)
    # if wn_hypernym_matches is not None and WORDNET_AVAILABLE:
    #     if wn_hypernym_matches(token_lemma, pattern_lemma_lower, pos):
    #         return True, 0.6

    return False, 0.0


# Flag to enable/disable enhanced matching globally
# DISABLED - Enhanced matching was causing accuracy drop (too permissive)
ENHANCED_MATCHING_ENABLED = False

# Flag to enable/disable entity type disambiguation
# ENABLED - Uses WordNet supersenses to disambiguate relations based on entity types
ENTITY_TYPE_DISAMBIGUATION_ENABLED = True

# Minimum FrameNet score for semantic validation (0.0-1.0)
# Set to 0.5 (neutral base score) so unknown verbs pass but mismatches fail
# Higher values (0.55+) are too strict as they reject patterns with unknown verbs
MIN_FRAME_SCORE = 0.5


def get_matching_config():
    """
    Return current matching configuration for notebook display.
    
    This helper function exposes the configuration state for verification
    and debugging in notebooks.
    
    Returns:
        dict: Configuration values including:
            - enhanced_matching_enabled: Whether WordNet enhanced matching is on
            - entity_type_disambiguation_enabled: Whether entity type rules are used
            - min_frame_score: Threshold for FrameNet semantic validation
            - wordnet_available: Whether WordNet is available for use
            - entity_type_rules_count: Number of entity type rules available
    """
    return {
        'enhanced_matching_enabled': ENHANCED_MATCHING_ENABLED,
        'entity_type_disambiguation_enabled': ENTITY_TYPE_DISAMBIGUATION_ENABLED,
        'min_frame_score': MIN_FRAME_SCORE,
        'wordnet_available': WORDNET_AVAILABLE,
        'entity_type_disambiguation_available': ENTITY_TYPE_DISAMBIGUATION_AVAILABLE,
        'entity_type_rules_count': len(ENTITY_TYPE_RELATION_RULES) + len(ENTITY_TYPE_RELATION_RULES_MEDIUM),
    }


def semantic_validation(
    relation: str,
    anchor_lemma: str,
    anchor_pos: str,
    e1_dep: str,
    e2_dep: str
) -> Tuple[bool, float]:
    """
    Combined WordNet + FrameNet semantic validation.

    Uses both knowledge sources to validate that a pattern match
    makes semantic sense for the predicted relation.

    Returns:
        Tuple of (is_valid, combined_confidence)
    """
    # Default: valid with neutral confidence if no semantic modules available
    if score_frame_compatibility is None and relation_specific_match is None:
        return True, 0.5

    frame_score = 0.5  # Default neutral score
    wn_match = False
    wn_conf = 0.0

    # Get FrameNet score
    if score_frame_compatibility is not None:
        frame_score = score_frame_compatibility(relation, anchor_lemma, e1_dep, e2_dep)

    # Get WordNet relation-specific match
    if relation_specific_match is not None and WORDNET_AVAILABLE:
        pos = 'v' if anchor_pos == 'VERB' else 'n'
        wn_match, wn_conf, _ = relation_specific_match(anchor_lemma, relation, pos)

    # Combined scoring logic (stricter thresholds for better precision):
    # - Both agree strongly: high confidence
    # - FrameNet alone confident: accept with FrameNet score
    # - WordNet + moderate FrameNet: moderate confidence
    # - Low FrameNet only: lower confidence
    # - Below MIN_FRAME_SCORE: reject
    if wn_match and frame_score >= 0.65:
        # Both sources agree strongly - highest confidence
        return True, (wn_conf + frame_score) / 2
    elif frame_score >= 0.65:
        # FrameNet alone is highly confident
        return True, frame_score
    elif wn_match and frame_score >= MIN_FRAME_SCORE:
        # WordNet agrees + moderate FrameNet support
        return True, 0.70
    elif frame_score >= MIN_FRAME_SCORE:
        # Only moderate FrameNet support - lower confidence
        return True, frame_score * 0.75
    else:
        # Below MIN_FRAME_SCORE threshold - reject match
        return False, frame_score


def check_direct_pattern(e1_root, e2_root, pattern):
    """
    Check if DIRECT pattern exists at entity roots.

    Pattern: e1 > e2 or e2 > e1 with specific dependency.

    Args:
        e1_root: Entity 1 root token
        e2_root: Entity 2 root token
        pattern: Pattern dict with pattern_key

    Returns:
        bool: True if pattern matches
    """
    pattern_key = pattern['pattern_key']
    if pattern_key[0] != 'DIRECT':
        return False

    _, dep_label, direction = pattern_key

    if direction == 'e1->e2':
        # e1 is parent of e2
        return e2_root.head == e1_root and e2_root.dep_ == dep_label
    elif direction == 'e2->e1':
        # e2 is parent of e1
        return e1_root.head == e2_root and e1_root.dep_ == dep_label

    return False


def check_direct_2hop_pattern(e1_root, e2_root, pattern):
    """
    Check if DIRECT_2HOP pattern exists at entity roots.

    Pattern: e1 > mid > e2 or e2 > mid > e1 (grandparent relationship)

    Args:
        e1_root: Entity 1 root token
        e2_root: Entity 2 root token
        pattern: Pattern dict with pattern_key

    Returns:
        bool: True if pattern matches
    """
    pattern_key = pattern['pattern_key']
    if pattern_key[0] != 'DIRECT_2HOP':
        return False

    # pattern_key: ("DIRECT_2HOP", mid_dep, child_dep, direction)
    _, mid_dep, child_dep, direction = pattern_key

    if direction == 'e1->e2':
        # e1 > mid > e2: e2's grandparent is e1
        mid = e2_root.head
        if mid == e2_root:  # e2 is root
            return False
        if mid.head != e1_root:
            return False
        if mid.dep_ != mid_dep:
            return False
        if e2_root.dep_ != child_dep:
            return False
        return True
    elif direction == 'e2->e1':
        # e2 > mid > e1: e1's grandparent is e2
        mid = e1_root.head
        if mid == e1_root:  # e1 is root
            return False
        if mid.head != e2_root:
            return False
        if mid.dep_ != mid_dep:
            return False
        if e1_root.dep_ != child_dep:
            return False
        return True

    return False


def check_direct_sibling_pattern(e1_root, e2_root, pattern):
    """
    Check if DIRECT_SIBLING pattern exists at entity roots.

    Pattern: head > e1, head > e2 (shared parent)

    Args:
        e1_root: Entity 1 root token
        e2_root: Entity 2 root token
        pattern: Pattern dict with pattern_key

    Returns:
        bool: True if pattern matches
    """
    pattern_key = pattern['pattern_key']
    if pattern_key[0] != 'DIRECT_SIBLING':
        return False

    # pattern_key: ("DIRECT_SIBLING", e1_dep, e2_dep)
    _, e1_dep, e2_dep = pattern_key

    # Check if they share the same head
    if e1_root.head != e2_root.head:
        return False

    # Check if head is not the token itself (not root of sentence)
    if e1_root.head == e1_root:
        return False

    # Check dependency labels
    if e1_root.dep_ != e1_dep:
        return False
    if e2_root.dep_ != e2_dep:
        return False

    return True


def get_ancestors(token, max_depth=10):
    """Get all ancestors of a token up to max_depth."""
    ancestors = []
    current = token
    depth = 0
    while current.head != current and depth < max_depth:
        current = current.head
        ancestors.append(current)
        depth += 1
    return ancestors


def check_triangle_pattern(e1_root, e2_root, pattern, relation=None):
    """
    Check if TRIANGLE pattern exists at entity roots.

    Pattern: anchor > e1, anchor > e2 (or with >> for indirect)
    where anchor has specific lemma and POS.

    Args:
        e1_root: Entity 1 root token
        e2_root: Entity 2 root token
        pattern: Pattern dict with pattern_key
        relation: Optional relation for enhanced WordNet matching

    Returns:
        bool: True if pattern matches
    """
    pattern_key = pattern['pattern_key']
    if pattern_key[0] != 'TRIANGLE':
        return False

    # Handle both old (5-element) and new (7-element) pattern keys
    if len(pattern_key) == 5:
        _, anchor_lemma, anchor_pos, e1_dep, e2_dep = pattern_key
        e1_rel_op, e2_rel_op = '>', '>'
    elif len(pattern_key) == 7:
        _, anchor_lemma, anchor_pos, e1_rel_op, e1_dep, e2_rel_op, e2_dep = pattern_key
    else:
        return False

    e1_ancestors = get_ancestors(e1_root)
    e2_ancestors = get_ancestors(e2_root)

    # Find lowest common ancestor (LCA)
    lca = None
    for anc in e1_ancestors:
        if anc in e2_ancestors:
            lca = anc
            break

    if not lca:
        return False

    # Check if LCA matches anchor constraints
    if lca.pos_ != anchor_pos:
        return False

    # Check lemma with enhanced WordNet matching
    if ENHANCED_MATCHING_ENABLED and WORDNET_AVAILABLE:
        pos = 'v' if anchor_pos == 'VERB' else 'n'
        matched, conf = lemma_matches_enhanced(lca.lemma_, anchor_lemma, relation, pos)
        if not matched:
            return False
    else:
        if not lemma_matches(lca.lemma_, anchor_lemma):
            return False

    # Semantic validation using FrameNet + WordNet (if available)
    # Get entity dependency labels for frame scoring
    if relation:
        is_valid, semantic_conf = semantic_validation(
            relation, lca.lemma_, anchor_pos, e1_root.dep_, e2_root.dep_
        )
        if not is_valid:
            return False

    # Check e1 relationship to LCA
    if e1_rel_op == '>':
        # e1 must be direct child of LCA
        if e1_root.head != lca:
            return False
    elif e1_rel_op == '>>':
        # e1 must be descendant of LCA
        if lca not in e1_ancestors:
            return False

    # Check e2 relationship to LCA
    if e2_rel_op == '>':
        # e2 must be direct child of LCA
        if e2_root.head != lca:
            return False
    elif e2_rel_op == '>>':
        # e2 must be descendant of LCA
        if lca not in e2_ancestors:
            return False

    # Check dependency labels
    if e1_root.dep_ != e1_dep:
        return False
    if e2_root.dep_ != e2_dep:
        return False

    return True


def check_bridge_pattern(e1_root, e2_root, pattern, relation=None, e1_text=None, e2_text=None):
    """
    Check if BRIDGE pattern exists at entity roots.

    Pattern: e1 > prep > e2

    Args:
        e1_root: Entity 1 root token
        e2_root: Entity 2 root token
        pattern: Pattern dict with pattern_key
        relation: Optional relation for enhanced WordNet matching
        e1_text: Entity 1 text (for entity type gating)
        e2_text: Entity 2 text (for entity type gating)

    Returns:
        bool: True if pattern matches
    """
    pattern_key = pattern['pattern_key']
    if pattern_key[0] != 'BRIDGE':
        return False

    # Handle pattern key format: ("BRIDGE", prep_lemma, e2_rel_op, e2_dep)
    if len(pattern_key) != 4:
        return False

    _, prep_lemma, third, fourth = pattern_key
    # Check if third is a REL_OP or dependency
    if third in {'>', '>>'}:
        e2_rel_op, e2_dep = third, fourth
    else:
        # Old format
        e2_rel_op, e2_dep = '>', fourth

    # Acceptable dependency labels for e2 attached to preposition
    VALID_PREP_DEPS = {'pobj', 'pcomp', 'nmod', 'obl', 'dobj', 'attr', 'oprd', 'conj'}

    # Entity type gating for PART_PREP patterns
    # Disambiguate Member-Collection vs Component-Whole based on e2 entity type
    entity_type_gate_passed = True
    if prep_lemma == "PART_PREP" and ENTITY_TYPE_DISAMBIGUATION_AVAILABLE and e2_text:
        e2_type = get_entity_type(e2_text)
        # Filter based on relation and entity type compatibility
        if relation == "Member-Collection":
            # Member-Collection prefers e2 to be group/organization
            if e2_type not in {"group", "organization", "person", None}:
                entity_type_gate_passed = False
        elif relation == "Component-Whole":
            # Component-Whole prefers e2 to be artifact/object/body
            if e2_type not in {"artifact", "object", "body", "substance", "location", None}:
                entity_type_gate_passed = False
    
    if not entity_type_gate_passed:
        return False

    # Find prepositions that are children of e1
    for child in e1_root.children:
        if child.pos_ != 'ADP':
            continue

        # Check if prep lemma matches with enhanced WordNet matching
        if ENHANCED_MATCHING_ENABLED and WORDNET_AVAILABLE:
            matched, conf = lemma_matches_enhanced(child.lemma_, prep_lemma, relation, 'n')
            if not matched:
                continue
        else:
            if not lemma_matches(child.lemma_, prep_lemma):
                continue

        # Semantic validation using FrameNet + WordNet (if available)
        if relation:
            is_valid, semantic_conf = semantic_validation(
                relation, child.lemma_, 'ADP', e1_root.dep_, e2_root.dep_
            )
            if not is_valid:
                continue  # Try next preposition

        # Check if e2 is related to this prep
        if e2_rel_op == '>':
            # e2 should be direct child of prep
            if e2_root.head == child:
                # Check dependency
                if e2_dep in VALID_PREP_DEPS or e2_root.dep_ == e2_dep or e2_root.dep_ in VALID_PREP_DEPS:
                    return True
        elif e2_rel_op == '>>':
            # e2 should be descendant of prep
            e2_ancs = get_ancestors(e2_root, max_depth=5)
            if child in e2_ancs:
                return True

    return False


def check_linear_pattern(e1_root, e2_root, pattern, doc, relation=None):
    """
    Check if LINEAR pattern exists at entity roots.

    Pattern: e1 .* token1 .* token2 .* e2 (precedence-based)

    Args:
        e1_root: Entity 1 root token
        e2_root: Entity 2 root token
        pattern: Pattern dict with pattern_key
        doc: spaCy Doc
        relation: Optional relation for enhanced WordNet matching

    Returns:
        bool: True if pattern matches
    """
    pattern_key = pattern['pattern_key']
    if pattern_key[0] != 'LINEAR':
        return False

    _, tokens_tuple = pattern_key

    # Get positions
    e1_pos = e1_root.i
    e2_pos = e2_root.i

    # Determine order - e1 should come before e2 for LINEAR patterns
    if e1_pos >= e2_pos:
        return False  # Linear patterns expect e1 before e2

    start, end = e1_pos, e2_pos

    # Check if required tokens appear in order between entities
    last_found_pos = start

    for required_lemma, required_pos in tokens_tuple:
        found = False
        for token in doc[last_found_pos + 1:end]:
            # Match POS first
            if token.pos_ != required_pos:
                continue

            # Match lemma with enhanced WordNet matching
            if ENHANCED_MATCHING_ENABLED and WORDNET_AVAILABLE:
                pos_wn = 'v' if required_pos == 'VERB' else 'n'
                matched, conf = lemma_matches_enhanced(token.lemma_, required_lemma, relation, pos_wn)
                if matched:
                    last_found_pos = token.i
                    found = True
                    break
            else:
                if lemma_matches(token.lemma_, required_lemma):
                    last_found_pos = token.i
                    found = True
                    break

        if not found:
            return False

    return True


def check_fallback_pattern(e1_root, e2_root, pattern):
    """
    Check if FALLBACK pattern matches.

    FALLBACK patterns are very general and match based on POS tags
    and approximate path length.

    Args:
        e1_root: Entity 1 root token
        e2_root: Entity 2 root token
        pattern: Pattern dict with pattern_key

    Returns:
        bool: True if pattern matches
    """
    pattern_key = pattern['pattern_key']
    if pattern_key[0] != 'FALLBACK':
        return False

    # pattern_key: ("FALLBACK", e1_pos, e2_pos, path_length)
    _, expected_e1_pos, expected_e2_pos, expected_path_len = pattern_key

    # Check POS tags
    if e1_root.pos_ != expected_e1_pos:
        return False
    if e2_root.pos_ != expected_e2_pos:
        return False

    # Check approximate path length (with tolerance)
    actual_path_len = min(abs(e1_root.i - e2_root.i), 10)
    if abs(actual_path_len - expected_path_len) > 2:
        return False

    return True


def apply_patterns_entity_rooted(samples, patterns, nlp):
    """
    Apply patterns using entity-rooted matching instead of DependencyMatcher.

    For each sample:
      1. Get entity root positions
      2. For each pattern (sorted by priority):
         a. Check if pattern structure exists at entity roots
         b. Return first match
      3. If no pattern matched and entity type disambiguation is enabled,
         use WordNet entity types to predict relation
      4. Default to "Other" if no match and no entity type rule

    Args:
        samples: List of processed samples
        patterns: List of pattern dicts (sorted by priority)
        nlp: spaCy model

    Returns:
        predictions: List of predicted relations
        directions: List of predicted directions
        explanations: List of explanations
        stats: Dict with matching statistics
    """
    predictions = []
    directions = []
    explanations = []

    # Statistics
    stats = {
        'total_samples': len(samples),
        'matched': 0,
        'entity_type_fallback': 0,  # NEW: predictions from entity type rules
        'default_other': 0,
        'pattern_usage': {},
        'matches_by_type': {},
        'failed_anchoring': 0,  # Always 0 for entity-rooted (no anchoring failures)
        'match_attempts': 0,
    }

    print(f"\nApplying {len(patterns)} patterns (entity-rooted) to {len(samples)} samples...")
    if ENTITY_TYPE_DISAMBIGUATION_ENABLED and ENTITY_TYPE_DISAMBIGUATION_AVAILABLE:
        print(f"  Entity type disambiguation: ENABLED ({len(ENTITY_TYPE_RELATION_RULES)} high-conf rules)")

    for sample in tqdm(samples, desc="Classifying"):
        doc = sample['doc']
        e1_root = sample['e1_span'].root
        e2_root = sample['e2_span'].root
        e1_text = sample['e1_span'].text
        e2_text = sample['e2_span'].text

        matched_pattern = None

        # Try patterns in priority order (already sorted)
        for pattern in patterns:
            pattern_type = pattern['pattern_type']
            stats['match_attempts'] += 1

            # Check if pattern matches at entity roots
            matched = False

            # Get pattern's relation for enhanced WordNet matching
            pattern_relation = pattern.get('relation', None)

            if pattern_type == 'DIRECT':
                matched = check_direct_pattern(e1_root, e2_root, pattern)
            elif pattern_type == 'DIRECT_2HOP':
                matched = check_direct_2hop_pattern(e1_root, e2_root, pattern)
            elif pattern_type == 'DIRECT_SIBLING':
                matched = check_direct_sibling_pattern(e1_root, e2_root, pattern)
            elif pattern_type == 'TRIANGLE':
                matched = check_triangle_pattern(e1_root, e2_root, pattern, pattern_relation)
            elif pattern_type == 'BRIDGE':
                matched = check_bridge_pattern(e1_root, e2_root, pattern, pattern_relation, e1_text, e2_text)
            elif pattern_type == 'LINEAR':
                matched = check_linear_pattern(e1_root, e2_root, pattern, doc, pattern_relation)
            elif pattern_type == 'FALLBACK':
                matched = check_fallback_pattern(e1_root, e2_root, pattern)

            if matched:
                # Confidence-based pattern skipping: skip low-quality patterns
                pattern_precision = pattern.get('precision', 1.0)
                pattern_support = pattern.get('support', 999)
                
                # Skip if low confidence (precision < 0.55 AND support < 3)
                # This allows fallback to higher-quality patterns in the pipeline
                if pattern_precision < 0.55 and pattern_support < 3:
                    # Continue searching for better patterns
                    continue
                
                matched_pattern = pattern
                stats['matches_by_type'][pattern_type] = stats['matches_by_type'].get(pattern_type, 0) + 1
                break

        # Record prediction
        if matched_pattern:
            predictions.append(matched_pattern['relation'])
            directions.append(matched_pattern['direction'])
            explanations.append(
                f"Pattern {matched_pattern['pattern_id']} "
                f"(type={matched_pattern['pattern_type']}, "
                f"precision={matched_pattern['precision']:.2f}, "
                f"support={matched_pattern['support']}): "
                f"{matched_pattern['explanation']}"
            )
            stats['matched'] += 1

            # Track pattern usage
            pattern_id = matched_pattern['pattern_id']
            stats['pattern_usage'][pattern_id] = stats['pattern_usage'].get(pattern_id, 0) + 1
        
        # NEW: Entity type fallback when no pattern matched
        elif ENTITY_TYPE_DISAMBIGUATION_ENABLED and ENTITY_TYPE_DISAMBIGUATION_AVAILABLE:
            # Get entity types
            e1_type = get_entity_type(e1_text)
            e2_type = get_entity_type(e2_text)
            type_pair = (e1_type, e2_type)
            
            # Check high-confidence rules first
            if type_pair in ENTITY_TYPE_RELATION_RULES:
                predicted_relation = ENTITY_TYPE_RELATION_RULES[type_pair]
                predictions.append(predicted_relation)
                # Extract direction from relation string like "Member-Collection(e2,e1)"
                if '(' in predicted_relation:
                    dir_str = predicted_relation.split('(')[1].rstrip(')')
                    directions.append(dir_str)
                else:
                    directions.append(None)
                explanations.append(
                    f"Entity type rule: ({e1_type}, {e2_type}) -> {predicted_relation} "
                    f"(high confidence, no pattern matched)"
                )
                stats['entity_type_fallback'] += 1
                stats['matches_by_type']['ENTITY_TYPE'] = stats['matches_by_type'].get('ENTITY_TYPE', 0) + 1
            
            # Check medium-confidence rules
            elif type_pair in ENTITY_TYPE_RELATION_RULES_MEDIUM:
                predicted_relation = ENTITY_TYPE_RELATION_RULES_MEDIUM[type_pair]
                predictions.append(predicted_relation)
                if '(' in predicted_relation:
                    dir_str = predicted_relation.split('(')[1].rstrip(')')
                    directions.append(dir_str)
                else:
                    directions.append(None)
                explanations.append(
                    f"Entity type rule: ({e1_type}, {e2_type}) -> {predicted_relation} "
                    f"(medium confidence, no pattern matched)"
                )
                stats['entity_type_fallback'] += 1
                stats['matches_by_type']['ENTITY_TYPE_MEDIUM'] = stats['matches_by_type'].get('ENTITY_TYPE_MEDIUM', 0) + 1
            
            else:
                # No entity type rule - default to Other
                predictions.append('Other')
                directions.append(None)
                explanations.append(
                    f'No pattern matched; entity types ({e1_type}, {e2_type}) have no rule; '
                    f'defaulting to Other.'
                )
                stats['default_other'] += 1
        
        else:
            predictions.append('Other')
            directions.append(None)
            explanations.append('No pattern matched at entity roots; defaulting to Other.')
            stats['default_other'] += 1

    # Calculate statistics
    stats['match_rate'] = stats['matched'] / stats['total_samples'] if stats['total_samples'] > 0 else 0
    stats['entity_type_rate'] = stats['entity_type_fallback'] / stats['total_samples'] if stats['total_samples'] > 0 else 0
    stats['default_rate'] = stats['default_other'] / stats['total_samples'] if stats['total_samples'] > 0 else 0
    stats['unique_patterns_used'] = len(stats['pattern_usage'])

    print(f"\nClassification complete!")
    print(f"  Pattern matched: {stats['matched']} ({stats['match_rate']:.1%})")
    print(f"  Entity type fallback: {stats['entity_type_fallback']} ({stats['entity_type_rate']:.1%})")
    print(f"  Default to Other: {stats['default_other']} ({stats['default_rate']:.1%})")
    print(f"  Unique patterns used: {stats['unique_patterns_used']}")
    print(f"  Matches by type: {stats['matches_by_type']}")

    return predictions, directions, explanations, stats


def apply_patterns_with_lexical_fallback(samples, patterns, nlp, lexical_matcher=None):
    """
    Apply patterns with lexical fallback for unmatched samples.

    Three-stage classification:
      1. Entity-rooted dependency patterns (highest priority)
      2. Lexical patterns from between-entity span (M2-style fallback)
      3. Entity type disambiguation (WordNet-based)
      4. Default to "Other"

    Args:
        samples: List of processed samples
        patterns: List of pattern dicts (sorted by priority)
        nlp: spaCy model
        lexical_matcher: Optional LexicalPatternMatcher instance

    Returns:
        predictions: List of predicted relations
        directions: List of predicted directions
        explanations: List of explanations
        stats: Dict with matching statistics
    """
    predictions = []
    directions = []
    explanations = []

    # Statistics
    stats = {
        'total_samples': len(samples),
        'dep_pattern_matched': 0,
        'lexical_fallback': 0,
        'entity_type_fallback': 0,
        'default_other': 0,
        'pattern_usage': {},
        'matches_by_type': {},
    }

    use_lexical = lexical_matcher is not None and lexical_matcher.is_compiled
    use_entity_type = ENTITY_TYPE_DISAMBIGUATION_ENABLED and ENTITY_TYPE_DISAMBIGUATION_AVAILABLE

    print(f"\nApplying {len(patterns)} patterns to {len(samples)} samples...")
    print(f"  Lexical fallback: {'ENABLED' if use_lexical else 'DISABLED'}")
    print(f"  Entity type disambiguation: {'ENABLED' if use_entity_type else 'DISABLED'}")

    for sample in tqdm(samples, desc="Classifying"):
        doc = sample['doc']
        e1_root = sample['e1_span'].root
        e2_root = sample['e2_span'].root
        e1_text = sample['e1_span'].text
        e2_text = sample['e2_span'].text

        matched_pattern = None
        prediction = None
        direction = None
        explanation = None

        # Stage 1: Try dependency patterns
        for pattern in patterns:
            pattern_type = pattern['pattern_type']
            pattern_relation = pattern.get('relation', None)
            matched = False

            if pattern_type == 'DIRECT':
                matched = check_direct_pattern(e1_root, e2_root, pattern)
            elif pattern_type == 'DIRECT_2HOP':
                matched = check_direct_2hop_pattern(e1_root, e2_root, pattern)
            elif pattern_type == 'DIRECT_SIBLING':
                matched = check_direct_sibling_pattern(e1_root, e2_root, pattern)
            elif pattern_type == 'TRIANGLE':
                matched = check_triangle_pattern(e1_root, e2_root, pattern, pattern_relation)
            elif pattern_type == 'BRIDGE':
                matched = check_bridge_pattern(e1_root, e2_root, pattern, pattern_relation, e1_text, e2_text)
            elif pattern_type == 'LINEAR':
                matched = check_linear_pattern(e1_root, e2_root, pattern, doc, pattern_relation)
            elif pattern_type == 'FALLBACK':
                matched = check_fallback_pattern(e1_root, e2_root, pattern)

            if matched:
                # Confidence-based pattern skipping: skip low-quality patterns
                pattern_precision = pattern.get('precision', 1.0)
                pattern_support = pattern.get('support', 999)
                
                # Skip if low confidence (precision < 0.55 AND support < 3)
                if pattern_precision < 0.55 and pattern_support < 3:
                    continue
                
                matched_pattern = pattern
                break

        if matched_pattern:
            prediction = matched_pattern['relation']
            direction = matched_pattern['direction']
            explanation = (
                f"Pattern {matched_pattern['pattern_id']} "
                f"(type={matched_pattern['pattern_type']}, "
                f"precision={matched_pattern['precision']:.2f})"
            )
            stats['dep_pattern_matched'] += 1
            stats['matches_by_type'][matched_pattern['pattern_type']] = \
                stats['matches_by_type'].get(matched_pattern['pattern_type'], 0) + 1

            pattern_id = matched_pattern['pattern_id']
            stats['pattern_usage'][pattern_id] = stats['pattern_usage'].get(pattern_id, 0) + 1

        # Stage 2: Try lexical fallback
        elif use_lexical:
            lex_pred, lex_dir, lex_expl = lexical_matcher.match(sample)
            if lex_pred:
                prediction = lex_pred
                direction = lex_dir
                explanation = lex_expl
                stats['lexical_fallback'] += 1
                stats['matches_by_type']['LEXICAL'] = stats['matches_by_type'].get('LEXICAL', 0) + 1

        # Stage 3: Try entity type disambiguation
        if prediction is None and use_entity_type:
            e1_type = get_entity_type(e1_text)
            e2_type = get_entity_type(e2_text)
            type_pair = (e1_type, e2_type)

            if type_pair in ENTITY_TYPE_RELATION_RULES:
                prediction = ENTITY_TYPE_RELATION_RULES[type_pair]
                if '(' in prediction:
                    direction = prediction.split('(')[1].rstrip(')')
                explanation = f"Entity type rule: ({e1_type}, {e2_type}) -> {prediction}"
                stats['entity_type_fallback'] += 1
                stats['matches_by_type']['ENTITY_TYPE'] = stats['matches_by_type'].get('ENTITY_TYPE', 0) + 1

            elif type_pair in ENTITY_TYPE_RELATION_RULES_MEDIUM:
                prediction = ENTITY_TYPE_RELATION_RULES_MEDIUM[type_pair]
                if '(' in prediction:
                    direction = prediction.split('(')[1].rstrip(')')
                explanation = f"Entity type rule (medium): ({e1_type}, {e2_type}) -> {prediction}"
                stats['entity_type_fallback'] += 1
                stats['matches_by_type']['ENTITY_TYPE_MEDIUM'] = stats['matches_by_type'].get('ENTITY_TYPE_MEDIUM', 0) + 1

        # Stage 4: Default to Other
        if prediction is None:
            prediction = 'Other'
            direction = None
            explanation = 'No pattern matched; defaulting to Other.'
            stats['default_other'] += 1

        predictions.append(prediction)
        directions.append(direction)
        explanations.append(explanation)

    # Calculate rates
    stats['dep_pattern_rate'] = stats['dep_pattern_matched'] / stats['total_samples'] if stats['total_samples'] > 0 else 0
    stats['lexical_rate'] = stats['lexical_fallback'] / stats['total_samples'] if stats['total_samples'] > 0 else 0
    stats['entity_type_rate'] = stats['entity_type_fallback'] / stats['total_samples'] if stats['total_samples'] > 0 else 0
    stats['default_rate'] = stats['default_other'] / stats['total_samples'] if stats['total_samples'] > 0 else 0

    print(f"\nClassification complete!")
    print(f"  Dependency patterns: {stats['dep_pattern_matched']} ({stats['dep_pattern_rate']:.1%})")
    print(f"  Lexical fallback: {stats['lexical_fallback']} ({stats['lexical_rate']:.1%})")
    print(f"  Entity type fallback: {stats['entity_type_fallback']} ({stats['entity_type_rate']:.1%})")
    print(f"  Default to Other: {stats['default_other']} ({stats['default_rate']:.1%})")
    print(f"  Matches by type: {stats['matches_by_type']}")

    return predictions, directions, explanations, stats


def apply_patterns_with_ml_resolver(samples, patterns, nlp, resolver=None, high_conf_threshold=0.98):
    """
    Apply patterns with optional ML resolver for Other class refinement.

    Two-stage classification:
      1. Entity-rooted patterns predict relation
      2. ML resolver refines predictions involving Other class (conservatively)

    Uses high confidence threshold to avoid over-correction.

    Args:
        samples: List of processed samples
        patterns: List of pattern dicts (sorted by priority)
        nlp: spaCy model
        resolver: Optional OtherResolver instance (if None, no ML refinement)
        high_conf_threshold: Only override if ML is this confident (default 0.98)

    Returns:
        predictions: List of predicted relations
        directions: List of predicted directions
        explanations: List of explanations
        stats: Dict with matching statistics (includes resolver stats)
    """
    # First, apply entity-rooted patterns
    base_preds, dirs, expls, stats = apply_patterns_entity_rooted(samples, patterns, nlp)

    if resolver is None or not hasattr(resolver, 'is_trained') or not resolver.is_trained:
        return base_preds, dirs, expls, stats

    # Apply ML resolver CONSERVATIVELY - only override with high confidence
    refined_preds = []
    resolver_stats = {
        'overridden_to_other': 0,
        'kept_as_other': 0,
        'unchanged': 0,
    }

    for sample, pred, direction, expl in zip(samples, base_preds, dirs, expls):
        # Get probability from ML resolver
        other_prob = resolver.predict_proba_other(sample)

        # Only override non-Other to Other if VERY confident
        if pred != 'Other' and other_prob > high_conf_threshold:
            refined_preds.append('Other')
            resolver_stats['overridden_to_other'] += 1
        else:
            # Keep original prediction
            refined_preds.append(pred)
            if pred == 'Other':
                resolver_stats['kept_as_other'] += 1
            else:
                resolver_stats['unchanged'] += 1

    # Update stats
    stats['resolver_stats'] = resolver_stats
    stats['resolver_overrides'] = resolver_stats['overridden_to_other']

    print(f"\nML Resolver applied (threshold={high_conf_threshold}):")
    print(f"  Overridden to Other: {resolver_stats['overridden_to_other']}")
    print(f"  Kept as Other: {resolver_stats['kept_as_other']}")
    print(f"  Unchanged: {resolver_stats['unchanged']}")

    return refined_preds, dirs, expls, stats
