"""
Pattern mining module for Milestone 3.

Extracts all 4 pattern types in priority order:
1. Type D (Direct) - Extract and CONTINUE
2. Type A (Triangle) - Extract and STOP
3. Type B (Bridge) - Extract and STOP
4. Type C (Linear) - Fallback if no A/B

All patterns are converted to DependencyMatcher format.
"""

from collections import defaultdict
from tqdm.auto import tqdm


def find_lca(token1, token2):
    """
    Find Lowest Common Ancestor of two tokens.

    Args:
        token1: First spaCy Token
        token2: Second spaCy Token

    Returns:
        lca: Token that is the LCA, or None if no common ancestor
    """
    # Collect ancestors of token1
    ancestors_1 = set()
    current = token1
    while current is not None:
        ancestors_1.add(current)
        if current.head == current:  # Root
            break
        current = current.head

    # Walk up from token2 until we hit an ancestor of token1
    current = token2
    while current is not None:
        if current in ancestors_1:
            return current
        if current.head == current:  # Root
            break
        current = current.head

    return None


def is_triangle_structure(e1_root, e2_root, lca):
    """
    Check if LCA forms a valid triangle structure.

    Valid triangle: LCA is the head of both e1 and e2 (directly or indirectly).

    Args:
        e1_root: Entity 1 root token
        e2_root: Entity 2 root token
        lca: Candidate LCA token

    Returns:
        bool: True if valid triangle
    """
    if lca is None:
        return False

    # LCA should be an ancestor of both (but not the tokens themselves)
    if lca == e1_root or lca == e2_root:
        return False

    # Triangle is intended to be event-driven: anchor should be verb-like.
    if lca.pos_ not in {"VERB", "AUX"}:
        return False

    def _dist_to_ancestor(token, ancestor, max_steps=10):
        steps = 0
        cur = token
        while cur is not None:
            if cur == ancestor:
                return steps
            if cur.head == cur or steps >= max_steps:
                return None
            cur = cur.head
            steps += 1
        return None

    # Allow indirect attachment, but keep it local to avoid overly generic ROOT anchors.
    d1 = _dist_to_ancestor(e1_root, lca, max_steps=10)
    d2 = _dist_to_ancestor(e2_root, lca, max_steps=10)
    if d1 is None or d2 is None:
        return False
    if d1 == 0 or d2 == 0:
        return False
    return d1 <= 3 and d2 <= 3


def find_bridge_node(e1_root, e2_root):
    """
    Find bridge node (preposition) between two entities.

    Bridge pattern: e1 → prep → e2 (prep is child of e1, e2 is child of prep)

    Args:
        e1_root: Entity 1 root token
        e2_root: Entity 2 root token

    Returns:
        bridge: Token acting as bridge, or None
    """
    # Check if e2 is connected to e1 via a preposition
    # Pattern: e1 > prep > e2
    for child in e1_root.children:
        if child.pos_ == "ADP":
            for prep_child in child.children:
                if prep_child == e2_root or e2_root in list(prep_child.subtree):
                    return child

    # Check reverse: e2 > prep > e1
    for child in e2_root.children:
        if child.pos_ == "ADP":
            for prep_child in child.children:
                if prep_child == e1_root or e1_root in list(prep_child.subtree):
                    return child

    return None


def abstract_lemma(lemma, lemma_to_concept):
    """
    Convert lemma to concept if it exists in clusters.

    Args:
        lemma: Word lemma (string)
        lemma_to_concept: Reverse mapping dict

    Returns:
        concept or original lemma
    """
    return lemma_to_concept.get(lemma.lower(), lemma.lower())


def create_type_d_pattern(head, dependent, dep_label, direction):
    """
    Create Type D (Direct) pattern.

    Pattern: head > dependent

    Args:
        head: Head token
        dependent: Dependent token
        dep_label: Dependency label
        direction: "e1->e2" or "e2->e1"

    Returns:
        pattern_key: Hashable pattern identifier
        dep_pattern: DependencyMatcher pattern
    """
    pattern_key = ("DIRECT", dep_label, direction)

    dep_pattern = [
        {"RIGHT_ID": "e2" if direction == "e2->e1" else "e1", "RIGHT_ATTRS": {}},
        {
            "LEFT_ID": "e2" if direction == "e2->e1" else "e1",
            "REL_OP": ">",
            "RIGHT_ID": "e1" if direction == "e2->e1" else "e2",
            "RIGHT_ATTRS": {"DEP": dep_label}
        }
    ]

    return pattern_key, dep_pattern


def create_type_a_pattern(e1_root, e2_root, lca, lemma_to_concept):
    """
    Create Type A (Triangle) pattern.

    Pattern: anchor > e1, anchor > e2

    Args:
        e1_root: Entity 1 root token
        e2_root: Entity 2 root token
        lca: LCA token (anchor)
        lemma_to_concept: Concept mapping

    Returns:
        pattern_key: Hashable pattern identifier
        dep_pattern: DependencyMatcher pattern
    """
    # Abstract LCA lemma to concept
    lca_lemma = lca.lemma_.lower()
    lca_concept = abstract_lemma(lca_lemma, lemma_to_concept)

    # If the entity roots attach indirectly, use descendant operator.
    e1_rel_op = ">" if e1_root.head == lca else ">>"
    e2_rel_op = ">" if e2_root.head == lca else ">>"

    # Keep each entity root's own DEP label as an additional constraint.
    # Note: this is the token's dependency label to its head (not necessarily to the anchor),
    # but it still helps preserve directionality (e.g. nsubj vs dobj) even with '>>'.
    e1_dep = e1_root.dep_
    e2_dep = e2_root.dep_

    pattern_key = ("TRIANGLE", lca_concept, lca.pos_, e1_rel_op, e1_dep, e2_rel_op, e2_dep)

    # Check if concept (use {"IN": [...]}) or literal lemma
    if lca_concept != lca_lemma:  # It's a concept
        # Need to get all words in this concept for the pattern
        # For now, use the concept as a placeholder - will expand later
        anchor_attr = {"LEMMA": lca_concept, "POS": lca.pos_}
    else:
        anchor_attr = {"LEMMA": lca_lemma, "POS": lca.pos_}

    e1_attrs = {"DEP": e1_dep}
    e2_attrs = {"DEP": e2_dep}

    dep_pattern = [
        {"RIGHT_ID": "anchor", "RIGHT_ATTRS": anchor_attr},
        {"LEFT_ID": "anchor", "REL_OP": e1_rel_op, "RIGHT_ID": "e1", "RIGHT_ATTRS": e1_attrs},
        {"LEFT_ID": "anchor", "REL_OP": e2_rel_op, "RIGHT_ID": "e2", "RIGHT_ATTRS": e2_attrs},
    ]

    return pattern_key, dep_pattern


def create_type_b_pattern(e1_root, e2_root, bridge, lemma_to_concept):
    """
    Create Type B (Bridge) pattern.

    Pattern: e1 > prep > e2

    Args:
        e1_root: Entity 1 root token
        e2_root: Entity 2 root token
        bridge: Bridge token (preposition)
        lemma_to_concept: Concept mapping

    Returns:
        pattern_key: Hashable pattern identifier
        dep_pattern: DependencyMatcher pattern
    """
    # Abstract bridge lemma to concept
    bridge_lemma = bridge.lemma_.lower()
    bridge_concept = abstract_lemma(bridge_lemma, lemma_to_concept)

    # e2 may be directly attached to the preposition, or inside a subtree (e.g. compounds).
    e2_rel_op = ">" if e2_root.head == bridge else ">>"
    e2_dep = e2_root.dep_
    pattern_key = ("BRIDGE", bridge_concept, e2_rel_op, e2_dep)

    # Check if concept or literal
    if bridge_concept != bridge_lemma:
        prep_attr = {"LEMMA": bridge_concept, "POS": "ADP"}
    else:
        prep_attr = {"LEMMA": bridge_lemma, "POS": "ADP"}

    e2_attrs = {"DEP": e2_dep}
    dep_pattern = [
        {"RIGHT_ID": "e1", "RIGHT_ATTRS": {}},
        {"LEFT_ID": "e1", "REL_OP": ">", "RIGHT_ID": "prep", "RIGHT_ATTRS": prep_attr},
        {"LEFT_ID": "prep", "REL_OP": e2_rel_op, "RIGHT_ID": "e2", "RIGHT_ATTRS": e2_attrs},
    ]

    return pattern_key, dep_pattern


def create_type_c_pattern(e1_root, e2_root, between_tokens, lemma_to_concept):
    """
    Create Type C (Linear) pattern.

    Pattern: e1 .* token .* e2 (precedence-based)

    Args:
        e1_root: Entity 1 root token
        e2_root: Entity 2 root token
        between_tokens: Span of tokens between entities
        lemma_to_concept: Concept mapping

    Returns:
        pattern_key: Hashable pattern identifier
        dep_pattern: DependencyMatcher pattern
    """
    # Extract between words
    between_lemmas = []
    for token in between_tokens:
        if not token.is_punct and len(token.lemma_) > 1:
            lemma = token.lemma_.lower()
            concept = abstract_lemma(lemma, lemma_to_concept)
            between_lemmas.append((concept, token.pos_))

    if not between_lemmas:
        return None, None

    # Limit to first 3 tokens for pattern
    between_lemmas = between_lemmas[:3]

    pattern_key = ("LINEAR", tuple(between_lemmas))

    # Build DependencyMatcher pattern with precedence operators
    dep_pattern = [{"RIGHT_ID": "e1", "RIGHT_ATTRS": {}}]

    for i, (lemma, pos) in enumerate(between_lemmas):
        left_id = f"token_{i-1}" if i > 0 else "e1"
        dep_pattern.append({
            "LEFT_ID": left_id,
            "REL_OP": ".*",  # Precedence (follows)
            "RIGHT_ID": f"token_{i}",
            "RIGHT_ATTRS": {"LEMMA": lemma, "POS": pos}
        })

    # Add e2 at the end
    dep_pattern.append({
        "LEFT_ID": f"token_{len(between_lemmas)-1}",
        "REL_OP": ".*",
        "RIGHT_ID": "e2",
        "RIGHT_ATTRS": {}
    })

    return pattern_key, dep_pattern


def extract_all_patterns(processed_samples, lemma_to_concept):
    """
    Extract all 4 pattern types following priority order.

    Priority:
      1. Type D (Direct) - extract and CONTINUE
      2. Type A (Triangle) - extract and STOP
      3. Type B (Bridge) - extract and STOP
      4. Type C (Linear) - fallback if no A/B

    Args:
        processed_samples: List of preprocessed samples
        lemma_to_concept: Dict mapping lemmas to concepts

    Returns:
        pattern_counts: Dict[pattern_key][relation] -> count
    """
    pattern_counts = defaultdict(lambda: defaultdict(int))

    print(f"Mining patterns from {len(processed_samples)} samples...")

    # Debug counters for pattern types
    direct_e1_to_e2 = 0
    direct_e2_to_e1 = 0
    two_hop_count = 0
    sibling_count = 0
    fallback_count = 0

    for sample in tqdm(processed_samples, desc="Mining patterns"):
        e1_root = sample['e1_span'].root
        e2_root = sample['e2_span'].root
        relation = sample['relation_directed']
        doc = sample['doc']
        e1_span = sample['e1_span']
        e2_span = sample['e2_span']

        found_main_pattern = False  # Track if A/B found
        found_any_pattern = False   # Track if ANY pattern found (for fallback)

        # === TYPE D: Direct connection (check first, don't stop) ===
        if e2_root.head == e1_root:
            direct_e2_to_e1 += 1
            pattern_key, dep_pattern = create_type_d_pattern(
                e1_root, e2_root, e2_root.dep_, direction="e2->e1"
            )
            if pattern_key:
                pattern_counts[pattern_key][relation] += 1
                found_any_pattern = True
        elif e1_root.head == e2_root:
            direct_e1_to_e2 += 1
            pattern_key, dep_pattern = create_type_d_pattern(
                e2_root, e1_root, e1_root.dep_, direction="e1->e2"
            )
            if pattern_key:
                pattern_counts[pattern_key][relation] += 1
                found_any_pattern = True
        # === TYPE D_2HOP: 2-hop (grandparent) relationships ===
        elif e2_root.head != e2_root and e2_root.head.head == e1_root:
            # e1 > intermediate > e2
            two_hop_count += 1
            pattern_key = ("DIRECT_2HOP", e2_root.head.dep_, e2_root.dep_, "e1->e2")
            pattern_counts[pattern_key][relation] += 1
            found_any_pattern = True
        elif e1_root.head != e1_root and e1_root.head.head == e2_root:
            # e2 > intermediate > e1
            two_hop_count += 1
            pattern_key = ("DIRECT_2HOP", e1_root.head.dep_, e1_root.dep_, "e2->e1")
            pattern_counts[pattern_key][relation] += 1
            found_any_pattern = True
        # === TYPE D_SIBLING: Shared head (conjunctions, compounds) ===
        elif (e1_root.head == e2_root.head and
              e1_root.head != e1_root and
              e1_root.dep_ in {'conj', 'appos', 'nmod', 'compound'}):
            sibling_count += 1
            pattern_key = ("DIRECT_SIBLING", e1_root.dep_, e2_root.dep_)
            pattern_counts[pattern_key][relation] += 1
            found_any_pattern = True

        # === TYPE A: Triangle (LCA) ===
        lca = find_lca(e1_root, e2_root)
        if lca and is_triangle_structure(e1_root, e2_root, lca):
            pattern_key, dep_pattern = create_type_a_pattern(
                e1_root, e2_root, lca, lemma_to_concept
            )
            if pattern_key:
                pattern_counts[pattern_key][relation] += 1
                found_main_pattern = True
                found_any_pattern = True
                continue  # Stop here

        # === TYPE B: Bridge (Prepositional chain) ===
        bridge = find_bridge_node(e1_root, e2_root)
        if bridge and bridge.pos_ == "ADP":
            pattern_key, dep_pattern = create_type_b_pattern(
                e1_root, e2_root, bridge, lemma_to_concept
            )
            if pattern_key:
                pattern_counts[pattern_key][relation] += 1
                found_main_pattern = True
                found_any_pattern = True
                continue  # Stop here

        # === TYPE C: Linear (Fallback if no A/B) ===
        if not found_main_pattern:
            between_span = sample.get('between_words')
            if between_span is None:
                # Calculate between span
                if e1_span.start < e2_span.start:
                    between_tokens = doc[e1_span.end:e2_span.start]
                else:
                    between_tokens = doc[e2_span.end:e1_span.start]
            else:
                # Use pre-computed between words
                between_tokens = [doc[w['id']] for w in between_span if 'id' in w]
                if not between_tokens:
                    # Fallback: compute from spans
                    if e1_span.start < e2_span.start:
                        between_tokens = doc[e1_span.end:e2_span.start]
                    else:
                        between_tokens = doc[e2_span.end:e1_span.start]

            if len(between_tokens) > 0 and len(between_tokens) <= 5:
                pattern_key, dep_pattern = create_type_c_pattern(
                    e1_root, e2_root, between_tokens, lemma_to_concept
                )
                if pattern_key:
                    pattern_counts[pattern_key][relation] += 1
                    found_any_pattern = True

        # === FALLBACK: Universal catch-all for remaining samples ===
        if not found_any_pattern:
            # Record: entity POS tags + dependency path length
            path_length = min(abs(e1_root.i - e2_root.i), 10)  # Cap at 10
            pattern_key = ("FALLBACK", e1_root.pos_, e2_root.pos_, path_length)
            pattern_counts[pattern_key][relation] += 1
            fallback_count += 1

    print(f"\\nPattern mining complete!")
    print(f"Total unique patterns: {len(pattern_counts)}")
    print(f"\\nPattern type counts:")
    print(f"  DIRECT (e1->e2): {direct_e1_to_e2}")
    print(f"  DIRECT (e2->e1): {direct_e2_to_e1}")
    print(f"  2-HOP: {two_hop_count}")
    print(f"  SIBLING: {sibling_count}")
    print(f"  FALLBACK: {fallback_count}")

    return pattern_counts


def summarize_patterns(pattern_counts):
    """
    Summarize mined patterns by type.

    Args:
        pattern_counts: Pattern count dictionary

    Returns:
        summary: Dict with statistics
    """
    type_counts = defaultdict(int)
    relation_patterns = defaultdict(int)

    for pattern_key, relation_counts in pattern_counts.items():
        pattern_type = pattern_key[0]
        type_counts[pattern_type] += 1

        for relation, count in relation_counts.items():
            relation_patterns[relation] += 1

    summary = {
        "total_patterns": len(pattern_counts),
        "by_type": dict(type_counts),
        "by_relation": dict(relation_patterns)
    }

    return summary
