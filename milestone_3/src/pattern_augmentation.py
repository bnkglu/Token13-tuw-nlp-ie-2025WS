"""
Pattern augmentation module for Milestone 3.

Generates passive voice variants from high-precision active voice patterns.
Implements tiered filtering based on pattern complexity.
"""

import copy
from collections import defaultdict


def count_pattern_nodes(pattern_key):
    """
    Count number of nodes in a pattern.

    Args:
        pattern_key: Pattern tuple identifier

    Returns:
        int: Number of nodes
    """
    pattern_type = pattern_key[0]

    if pattern_type == "DIRECT":
        return 2  # e1, e2
    elif pattern_type == "DIRECT_2HOP":
        return 3  # e1, mid, e2
    elif pattern_type == "DIRECT_SIBLING":
        return 3  # head, e1, e2
    elif pattern_type == "TRIANGLE":
        return 3  # anchor, e1, e2
    elif pattern_type == "BRIDGE":
        return 3  # e1, prep, e2
    elif pattern_type == "LINEAR":
        # Count: e1 + tokens + e2
        tokens_tuple = pattern_key[1]
        return 2 + len(tokens_tuple)
    elif pattern_type == "FALLBACK":
        # Not compiled into DependencyMatcher; treat as minimal size.
        return 2
    else:
        return 1


# Relation-specific precision floors (original values restored)
# Lower thresholds = more patterns = better coverage but potentially more errors
RELATION_MIN_PRECISION = {
    "Other": 0.80,              # Strict for Other patterns
    "Cause-Effect": 0.45,       # Original value
    "Component-Whole": 0.55,    # Original value
    "Entity-Origin": 0.55,      # Original value
    "Member-Collection": 0.55,  # Original value
    "Message-Topic": 0.50,      # Original value
    "Instrument-Agency": 0.45,  # Original value
    "Content-Container": 0.50,  # Original value
    "Entity-Destination": 0.45, # Original value
    "Product-Producer": 0.50,   # Original value
}

# Blacklisted anchors - too generic to be discriminative
# These lemmas appear in many different relations and cause false positives
BLACKLISTED_ANCHORS = {
    "be",      # Extremely common, matches almost any sentence
    "have",    # Very common auxiliary/possessive
    "do",      # Common auxiliary
    "get",     # Common light verb
    # "make",  # Keep - useful for Product-Producer
    # "take",  # Keep - might be useful
    # "give",  # Keep - might be useful
    # "go",    # Keep - useful for Entity-Destination
    # "come",  # Keep - useful for Entity-Origin
    # "say",   # Keep - useful for Message-Topic
    "know",    # Mental state, not relational
    "see",     # Perception, not relational
    "think",   # Mental state, not relational
    "want",    # Mental state, not relational
    # "use",   # Keep - crucial for Instrument-Agency
}


def pattern_uses_generic_concept(pattern_key):
    """Check if pattern uses generic/ambiguous concepts that need higher support."""
    GENERIC_CONCEPTS = {'PART_PREP', 'OF_PREP', 'CONTAINER_PREP'}

    pattern_type = pattern_key[0]
    if pattern_type == 'BRIDGE':
        # BRIDGE pattern_key[1] is prep lemma
        if len(pattern_key) > 1 and isinstance(pattern_key[1], str):
            return pattern_key[1].upper() in GENERIC_CONCEPTS
    elif pattern_type == 'LINEAR':
        # LINEAR pattern_key[1] is list of (lemma, pos) tuples
        if len(pattern_key) > 1:
            tokens = pattern_key[1]
            return any(token[0].upper() in GENERIC_CONCEPTS for token in tokens)
    return False


def filter_patterns_framenet_aware(patterns):
    """
    Filter TRIANGLE patterns to validate against FrameNet verb-frame mappings.

    Patterns with anchor verbs that have known FrameNet frames are prioritized.
    Patterns with anchor verbs that don't match their claimed relation's frames
    are deprioritized (lower confidence).

    Args:
        patterns: List of pattern dicts

    Returns:
        Filtered list of patterns with adjusted confidence
    """
    try:
        from framenet_scorer import VERB_FRAMES, get_relation_frames
    except ImportError:
        # FrameNet not available, return patterns unchanged
        return patterns

    filtered = []
    for pattern in patterns:
        # Only check TRIANGLE patterns (they have anchor verbs)
        if pattern.get('pattern_type') == 'TRIANGLE':
            pattern_key = pattern.get('pattern_key', [])
            if len(pattern_key) >= 2:
                anchor_lemma = pattern_key[1]
                relation = pattern.get('relation', '').split('(')[0]

                # Check if anchor is in VERB_FRAMES
                if anchor_lemma in VERB_FRAMES:
                    verb_frames = VERB_FRAMES[anchor_lemma]
                    relation_frames = get_relation_frames(relation)

                    # Check frame overlap
                    if any(vf in relation_frames for vf in verb_frames):
                        # Good match - verb evokes frames for this relation
                        pattern['framenet_validated'] = True
                        filtered.append(pattern)
                        continue
                    else:
                        # Verb known but doesn't match relation's frames
                        # Reduce precision as penalty
                        pattern['precision'] = pattern.get('precision', 0.5) * 0.85
                        pattern['framenet_validated'] = False
                        filtered.append(pattern)
                        continue

        # Keep non-TRIANGLE patterns or patterns with unknown anchors
        filtered.append(pattern)

    return filtered


def filter_patterns_tiered(pattern_counts, concept_clusters, min_global_support=1):
    """
    Apply tiered thresholds based on pattern complexity.

    Balanced thresholds for coverage AND accuracy:
        - High support (>=3): Precision >= 0.45 (more permissive)
        - Medium support (2): Precision >= 0.55
        - Low support (1): Precision >= 0.65 (stricter)
        - Other: Precision >= 0.80
        - Relation-specific floors apply on top
        - Patterns using generic concepts need support >= 3

    Args:
        pattern_counts: Dict[pattern_key][relation] -> count
        concept_clusters: Dict of concept definitions (for expansion)
        min_global_support: Minimum support for any pattern (default: 1)

    Returns:
        filtered_patterns: List of pattern dicts
    """
    filtered_patterns = []

    print(f"Filtering {len(pattern_counts)} candidate patterns...")
    print(f"  Global minimum support: {min_global_support}")

    for pattern_key, relation_counts in pattern_counts.items():
        total_count = sum(relation_counts.values())

        # Find dominant relation
        best_relation = max(relation_counts, key=relation_counts.get)
        best_count = relation_counts[best_relation]
        precision = best_count / total_count

        pattern_type = pattern_key[0]
        # Calculate pattern length
        pattern_length = count_pattern_nodes(pattern_key)

        # Global minimum support filter
        if best_count < min_global_support:
            continue

        # Check for blacklisted anchors (too generic to be discriminative)
        if pattern_type == "TRIANGLE":
            # TRIANGLE: pattern_key[1] is anchor lemma
            anchor_lemma = pattern_key[1] if len(pattern_key) > 1 else None
            if anchor_lemma and anchor_lemma.lower() in BLACKLISTED_ANCHORS:
                continue
        elif pattern_type == "BRIDGE":
            # BRIDGE: pattern_key[1] is prep lemma - less strict for preps
            pass  # Don't blacklist prepositions
        elif pattern_type == "LINEAR":
            # LINEAR: check tokens in the pattern
            if len(pattern_key) > 1:
                tokens_tuple = pattern_key[1]
                # Skip if ANY token in linear pattern is blacklisted (too generic)
                if any(token[0].lower() in BLACKLISTED_ANCHORS for token in tokens_tuple):
                    continue

        # Patterns using generic concepts need higher minimum support
        if pattern_uses_generic_concept(pattern_key):
            if best_count < 3:
                continue  # Generic concept patterns need at least 3 support

        # Get base relation for relation-specific thresholds
        base_relation = best_relation.split('(')[0] if '(' in best_relation else best_relation

        # Check relation-specific precision floor
        rel_min_precision = RELATION_MIN_PRECISION.get(base_relation, 0.0)

        # Apply tiered thresholds (Balanced for Coverage + Accuracy)
        if best_relation == "Other":
            # Stricter for Other patterns to reduce false positives
            if precision < 0.80 or best_count < 2:
                continue
        elif pattern_type == "LINEAR":
            # Linear patterns: support-based threshold
            if best_count >= 3:
                min_precision = max(0.45, rel_min_precision)
            else:
                min_precision = max(0.55, rel_min_precision)
            if precision < min_precision:
                continue
        elif pattern_type in {"TRIANGLE", "BRIDGE"}:
            # TRIANGLE/BRIDGE: support-based tiering
            # Special handling for ambiguous PART_PREP patterns
            is_part_prep_pattern = False
            if pattern_type == "BRIDGE" and len(pattern_key) > 1:
                prep_lemma = pattern_key[1]
                if prep_lemma == "PART_PREP":
                    is_part_prep_pattern = True
            
            if is_part_prep_pattern:
                # Stricter threshold for "of" patterns (PART_PREP) to reduce overfitting
                if best_count < 3:
                    continue  # Require at least 3 support for PART_PREP patterns
                if best_count >= 5:
                    min_precision = max(0.45, rel_min_precision)
                else:  # best_count is 3 or 4
                    min_precision = max(0.55, rel_min_precision)
            else:
                # Standard tiering for other TRIANGLE/BRIDGE patterns
                if best_count >= 5:
                    min_precision = max(0.40, rel_min_precision)
                elif best_count >= 2:
                    min_precision = max(0.50, rel_min_precision)
                else:
                    min_precision = max(0.60, rel_min_precision)
            
            if precision < min_precision:
                continue
        elif pattern_type == "DIRECT":
            # DIRECT patterns: moderate threshold
            min_precision = max(0.50, rel_min_precision)
            if precision < min_precision:
                continue
        elif pattern_type in {"DIRECT_2HOP", "DIRECT_SIBLING"}:
            # Multi-hop patterns: moderate threshold
            min_precision = max(0.45, rel_min_precision)
            if precision < min_precision:
                continue
        elif pattern_type == "FALLBACK":
            # FALLBACK: stricter (catch-all is risky)
            if precision < 0.70 or best_count < 3:
                continue
        elif pattern_length > 3:
            # Complex patterns: moderate threshold
            min_precision = max(0.45, rel_min_precision)
            if precision < min_precision:
                continue
        else:
            # Default: moderate threshold
            min_precision = max(0.50, rel_min_precision)
            if precision < min_precision:
                continue

        # Create pattern dict
        pattern_dict = create_pattern_dict(
            pattern_key=pattern_key,
            relation=best_relation,
            precision=precision,
            support=best_count,
            total_count=total_count,
            pattern_length=pattern_length,
            concept_clusters=concept_clusters
        )

        filtered_patterns.append(pattern_dict)

    print(f"Filtered to {len(filtered_patterns)} patterns")
    print(f"  Complex (len > 3): {sum(1 for p in filtered_patterns if p['length'] > 3)}")
    print(f"  Simple (len <= 3): {sum(1 for p in filtered_patterns if p['length'] <= 3)}")
    print(f"  'Other' patterns: {sum(1 for p in filtered_patterns if p['relation'] == 'Other')}")

    # EMERGENCY FIX: Add precision distribution analysis
    print(f"\nPattern precision distribution:")
    prec_bins = [0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
    for i in range(len(prec_bins)-1):
        count = sum(1 for p in filtered_patterns
                    if prec_bins[i] <= p['precision'] < prec_bins[i+1])
        pct = count / len(filtered_patterns) * 100 if filtered_patterns else 0
        print(f"  {prec_bins[i]:.2f}-{prec_bins[i+1]:.2f}: {count:4d} ({pct:5.1f}%)")

    # Pattern type distribution
    print(f"\nPattern type distribution:")
    type_counts = {}
    for p in filtered_patterns:
        ptype = p.get('pattern_type', 'UNKNOWN')
        type_counts[ptype] = type_counts.get(ptype, 0) + 1
    for ptype in sorted(type_counts.keys()):
        count = type_counts[ptype]
        pct = count / len(filtered_patterns) * 100 if filtered_patterns else 0
        print(f"  {ptype:<20}: {count:4d} ({pct:5.1f}%)")

    return filtered_patterns


def create_pattern_dict(pattern_key, relation, precision, support, total_count, pattern_length, concept_clusters):
    """
    Create pattern dictionary with metadata.

    Args:
        pattern_key: Tuple pattern identifier
        relation: Predicted relation
        precision: Pattern precision
        support: Pattern support count
        total_count: Total occurrences
        pattern_length: Number of nodes
        concept_clusters: Concept definitions

    Returns:
        pattern_dict: Complete pattern specification
    """
    pattern_type = pattern_key[0]

    # Parse relation and direction
    if '(' in relation and relation != "Other":
        base_relation = relation.split('(')[0]
        direction = relation.split('(')[1].rstrip(')')
    else:
        base_relation = relation
        direction = None

    # Build DependencyMatcher pattern
    dep_pattern = build_dep_matcher_pattern(pattern_key, concept_clusters)

    pattern_dict = {
        'pattern_id': f"{pattern_type}_{hash(pattern_key) % 100000}",
        'pattern_type': pattern_type,
        'pattern_key': pattern_key,
        'relation': relation,
        'base_relation': base_relation,
        'direction': direction,
        'precision': precision,
        'support': support,
        'total_count': total_count,
        'length': pattern_length,
        'dep_matcher_pattern': dep_pattern,
        'is_augmented': False,
        'parent_pattern_id': None,
        'explanation': f"{pattern_type} pattern (precision={precision:.2f}, support={support})"
    }

    return pattern_dict


def build_dep_matcher_pattern(pattern_key, concept_clusters):
    """
    Build DependencyMatcher pattern from pattern_key.

    Expands concepts to {"IN": [...]} format.

    Args:
        pattern_key: Pattern tuple
        concept_clusters: Concept definitions

    Returns:
        dep_pattern: List of dicts for DependencyMatcher
    """
    pattern_type = pattern_key[0]
    # Entity roots are overwhelmingly nominal. Restrict to NOUN/PROPN for accuracy.
    # Removing ADJ, VERB, NUM, PRON reduces false positive matches.
    ENTITY_POS = {"IN": ["NOUN", "PROPN"]}

    if pattern_type == "DIRECT":
        _, dep_label, direction = pattern_key
        dep_pattern = [
            {
                "RIGHT_ID": "e2" if direction == "e2->e1" else "e1",
                "RIGHT_ATTRS": {"POS": ENTITY_POS},
            },
            {
                "LEFT_ID": "e2" if direction == "e2->e1" else "e1",
                "REL_OP": ">",
                "RIGHT_ID": "e1" if direction == "e2->e1" else "e2",
                "RIGHT_ATTRS": {"DEP": dep_label, "POS": ENTITY_POS},
            }
        ]

    elif pattern_type == "DIRECT_2HOP":
        # pattern_key: ("DIRECT_2HOP", mid_dep, child_dep, direction)
        _, mid_dep, child_dep, direction = pattern_key

        head_id = "e1" if direction == "e1->e2" else "e2"
        tail_id = "e2" if direction == "e1->e2" else "e1"

        dep_pattern = [
            {"RIGHT_ID": head_id, "RIGHT_ATTRS": {"POS": ENTITY_POS}},
            {
                "LEFT_ID": head_id,
                "REL_OP": ">",
                "RIGHT_ID": "mid",
                "RIGHT_ATTRS": {"DEP": mid_dep},
            },
            {
                "LEFT_ID": "mid",
                "REL_OP": ">",
                "RIGHT_ID": tail_id,
                "RIGHT_ATTRS": {"DEP": child_dep, "POS": ENTITY_POS},
            },
        ]

    elif pattern_type == "DIRECT_SIBLING":
        # pattern_key: ("DIRECT_SIBLING", e1_dep, e2_dep)
        _, e1_dep, e2_dep = pattern_key
        dep_pattern = [
            {"RIGHT_ID": "head", "RIGHT_ATTRS": {}},
            {"LEFT_ID": "head", "REL_OP": ">", "RIGHT_ID": "e1", "RIGHT_ATTRS": {"DEP": e1_dep, "POS": ENTITY_POS}},
            {"LEFT_ID": "head", "REL_OP": ">", "RIGHT_ID": "e2", "RIGHT_ATTRS": {"DEP": e2_dep, "POS": ENTITY_POS}},
        ]

    elif pattern_type == "TRIANGLE":
        # NOTE: The TRIANGLE pattern_key shape changed during Milestone 3.
        #
        # Old (early): ("TRIANGLE", anchor_concept, anchor_pos, e1_dep, e2_dep)
        # New (current pattern_miner.py):
        #   ("TRIANGLE", anchor_concept, anchor_pos, e1_rel_op, e1_dep, e2_rel_op, e2_dep)
        #
        # Support both for backwards compatibility.
        if len(pattern_key) == 5:
            _, anchor_lemma, anchor_pos, e1_dep, e2_dep = pattern_key
            e1_rel_op, e2_rel_op = ">", ">"
        elif len(pattern_key) == 7:
            _, anchor_lemma, anchor_pos, e1_rel_op, e1_dep, e2_rel_op, e2_dep = pattern_key
        else:
            raise ValueError(f"Unexpected TRIANGLE pattern_key shape (len={len(pattern_key)}): {pattern_key!r}")

        # Use literal lemma only (no concept expansion)
        anchor_attrs = {"LEMMA": anchor_lemma, "POS": anchor_pos}

        # ENTITY-ROOTED: e1 is the root node to fix anchoring issues.
        # Convert ">" (anchor > e1) to "<" (e1 < anchor) for the e1-to-anchor relationship.
        # The "<" operator means "e1's parent is anchor".
        # The "<<" operator means "e1's ancestor is anchor".
        e1_to_anchor_op = "<" if e1_rel_op == ">" else "<<"

        dep_pattern = [
            # e1 is the ROOT - DependencyMatcher starts matching from e1's position
            {"RIGHT_ID": "e1", "RIGHT_ATTRS": {"DEP": e1_dep, "POS": ENTITY_POS}},
            {
                # anchor is e1's parent (or ancestor)
                "LEFT_ID": "e1",
                "REL_OP": e1_to_anchor_op,
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": anchor_attrs,
            },
            {
                # e2 is anchor's child (or descendant)
                "LEFT_ID": "anchor",
                "REL_OP": e2_rel_op,
                "RIGHT_ID": "e2",
                "RIGHT_ATTRS": {"DEP": e2_dep, "POS": ENTITY_POS},
            },
        ]

    elif pattern_type == "BRIDGE":
        # NOTE: BRIDGE pattern_key shape also changed.
        #
        # Old (early): ("BRIDGE", prep_concept, e1_dep, e2_dep)
        # New (current pattern_miner.py): ("BRIDGE", prep_concept, e2_rel_op, e2_dep)
        #
        # Support both; when we have rel_op, use it. Otherwise default to direct child.
        if len(pattern_key) != 4:
            raise ValueError(f"Unexpected BRIDGE pattern_key shape (len={len(pattern_key)}): {pattern_key!r}")

        _, prep_concept, third, fourth = pattern_key
        if third in {">", ">>"}:
            e2_rel_op, e2_dep = third, fourth
        else:
            # Back-compat: treat (e1_dep, e2_dep) and default the attachment operator.
            e2_rel_op, e2_dep = ">", fourth

        # Check if prep is a concept or literal
        prep_words = get_concept_words(prep_concept, concept_clusters)

        if prep_words:
            prep_attrs = {"LEMMA": {"IN": prep_words}, "POS": "ADP"}
        else:
            prep_attrs = {"LEMMA": prep_concept, "POS": "ADP"}

        # In practice, the entity attached to a preposition isn't always `pobj` (can be
        # `pcomp`, `nmod`, etc.), and spaCy/UD labels can vary. Keep this permissive.
        E2_PREP_DEPS = {"IN": ["pobj", "pcomp", "nmod", "obl", "dobj", "attr", "oprd", "conj"]}

        dep_pattern = [
            {"RIGHT_ID": "e1", "RIGHT_ATTRS": {"POS": ENTITY_POS}},
            {"LEFT_ID": "e1", "REL_OP": ">", "RIGHT_ID": "prep", "RIGHT_ATTRS": prep_attrs},
            {
                "LEFT_ID": "prep",
                "REL_OP": e2_rel_op,
                "RIGHT_ID": "e2",
                "RIGHT_ATTRS": {"DEP": E2_PREP_DEPS, "POS": ENTITY_POS},
            },
        ]

    elif pattern_type == "LINEAR":
        _, tokens_tuple = pattern_key

        dep_pattern = [{"RIGHT_ID": "e1", "RIGHT_ATTRS": {"POS": ENTITY_POS}}]

        for i, (lemma, pos) in enumerate(tokens_tuple):
            left_id = f"token_{i-1}" if i > 0 else "e1"

            # Check if lemma is a concept
            lemma_words = get_concept_words(lemma, concept_clusters)
            if lemma_words:
                lemma_attr = {"LEMMA": {"IN": lemma_words}, "POS": pos}
            else:
                lemma_attr = {"LEMMA": lemma, "POS": pos}

            dep_pattern.append({
                "LEFT_ID": left_id,
                "REL_OP": ".*",
                "RIGHT_ID": f"token_{i}",
                "RIGHT_ATTRS": lemma_attr
            })

        # Add e2 at the end
        dep_pattern.append({
            "LEFT_ID": f"token_{len(tokens_tuple)-1}",
            "REL_OP": ".*",
            "RIGHT_ID": "e2",
            "RIGHT_ATTRS": {"POS": ENTITY_POS},
        })

    else:
        dep_pattern = []

    return dep_pattern


def get_concept_words(concept_or_lemma, concept_clusters):
    """
    Concept expansion disabled - always return empty list.

    Concept expansion was found to cause anchoring issues with DependencyMatcher
    by allowing patterns to match multiple words, leading to ambiguous entity bindings.

    Args:
        concept_or_lemma: Concept name or literal lemma (ignored)
        concept_clusters: Concept definitions (ignored)

    Returns:
        Empty list (forces literal lemma matching)
    """
    return []


def is_active_voice_pattern(pattern):
    """
    Check if pattern is active voice (nsubj + dobj).

    Args:
        pattern: Pattern dict

    Returns:
        bool: True if active voice
    """
    if pattern['pattern_type'] != 'TRIANGLE':
        return False

    pattern_key = pattern['pattern_key']
    if len(pattern_key) < 5:
        return False

    # TRIANGLE pattern_key shapes:
    # - Old: ("TRIANGLE", anchor, pos, e1_dep, e2_dep)
    # - New: ("TRIANGLE", anchor, pos, e1_rel_op, e1_dep, e2_rel_op, e2_dep)
    if len(pattern_key) == 5:
        _, _anchor, _pos, e1_dep, e2_dep = pattern_key
    elif len(pattern_key) == 7:
        _, _anchor, _pos, _e1_rel_op, e1_dep, _e2_rel_op, e2_dep = pattern_key
    else:
        return False

    return e1_dep == 'nsubj' and e2_dep == 'dobj'


def generate_passive_variants(patterns, min_precision_for_flip=0.75):
    """
    Generate passive voice variants from high-precision active patterns.

    Args:
        patterns: List of pattern dicts
        min_precision_for_flip: Minimum precision to generate passive

    Returns:
        augmented_patterns: Original + passive variants
    """
    augmented = list(patterns)
    passive_count = 0

    print(f"\\nGenerating passive voice variants...")

    for pattern in patterns:
        # Only flip Triangle patterns
        if pattern['pattern_type'] != 'TRIANGLE':
            continue

        # Only flip high-precision patterns
        if pattern['precision'] < min_precision_for_flip:
            continue

        # Check if active voice
        if not is_active_voice_pattern(pattern):
            continue

        # Create passive variant
        passive = create_passive_variant(pattern)
        if passive:
            augmented.append(passive)
            passive_count += 1

    print(f"Generated {passive_count} passive variants")
    print(f"Total patterns: {len(augmented)}")

    return augmented


def create_passive_variant(active_pattern):
    """
    Create passive voice variant by swapping dependencies and direction.

    Active:  verb > nsubj(e1), verb > dobj(e2)
    Passive: verb > nsubjpass(e1), verb > agent/pobj(e2)

    Args:
        active_pattern: Active voice pattern dict

    Returns:
        passive_pattern: Passive variant, or None
    """
    passive = copy.deepcopy(active_pattern)

    # Update IDs
    passive['pattern_id'] = active_pattern['pattern_id'] + '_PASSIVE'
    passive['is_augmented'] = True
    passive['parent_pattern_id'] = active_pattern['pattern_id']

    # Swap dependencies in pattern key
    pk = active_pattern['pattern_key']
    if len(pk) == 5:
        _, anchor, pos, _e1_dep, _e2_dep = pk
        passive['pattern_key'] = ('TRIANGLE', anchor, pos, 'nsubjpass', 'agent')
    elif len(pk) == 7:
        _, anchor, pos, e1_rel_op, _e1_dep, e2_rel_op, _e2_dep = pk
        # Keep the structural operators, just flip the dependency labels to passive voice.
        passive['pattern_key'] = ('TRIANGLE', anchor, pos, e1_rel_op, 'nsubjpass', e2_rel_op, 'agent')
    else:
        return None

    # Update DependencyMatcher pattern
    dep_pattern = passive['dep_matcher_pattern']
    for node in dep_pattern:
        if node.get('RIGHT_ID') == 'e1':
            node['RIGHT_ATTRS']['DEP'] = 'nsubjpass'
        elif node.get('RIGHT_ID') == 'e2':
            # Passive can use 'agent' or 'pobj'
            node['RIGHT_ATTRS']['DEP'] = {'IN': ['agent', 'pobj']}

    # Flip direction
    old_direction = passive['direction']
    if old_direction == 'e1,e2':
        new_direction = 'e2,e1'
    elif old_direction == 'e2,e1':
        new_direction = 'e1,e2'
    else:
        new_direction = old_direction

    passive['direction'] = new_direction

    # Update relation label
    base_rel = passive['base_relation']
    if base_rel != "Other":
        passive['relation'] = f"{base_rel}({new_direction})"

    # Reduce precision (heuristic)
    passive['precision'] = active_pattern['precision'] * 0.9
    passive['support'] = 0  # No direct support
    passive['explanation'] = f"Passive variant of {active_pattern['pattern_id']}"

    return passive


def sort_patterns(patterns):
    """
    Sort patterns by (length desc, precision desc).

    Args:
        patterns: List of pattern dicts

    Returns:
        sorted_patterns: Sorted list
    """
    type_rank_map = {
        "DIRECT": 0,
        "DIRECT_2HOP": 1,
        "DIRECT_SIBLING": 2,
        "TRIANGLE": 1,
        "BRIDGE": 2,
        "LINEAR": 3,
        "FALLBACK": 9,
    }
    sorted_patterns = sorted(
        patterns,
        key=lambda p: (
            type_rank_map.get(p.get("pattern_type"), 9),
            -p["length"],
            -p["precision"],
            -p.get("support", 0),
        ),
    )

    print(f"\\nSorted {len(sorted_patterns)} patterns by (length desc, precision desc)")

    return sorted_patterns
