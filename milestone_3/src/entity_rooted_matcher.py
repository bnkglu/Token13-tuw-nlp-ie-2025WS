"""
Entity-rooted pattern matching for Milestone 3.

Instead of using DependencyMatcher freely (which finds patterns anywhere in the doc),
this module checks if pattern structures exist specifically at entity root positions.
This guarantees 100% anchoring alignment.
"""


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
        # Check if e2 is child of e1 with correct dep
        return e2_root.head == e1_root and e2_root.dep_ == dep_label
    elif direction == 'e2->e1':
        # Check if e1 is child of e2 with correct dep
        return e1_root.head == e2_root and e1_root.dep_ == dep_label

    return False


def check_triangle_pattern(e1_root, e2_root, pattern):
    """
    Check if TRIANGLE pattern exists at entity roots.

    Pattern: anchor > e1, anchor > e2 with specific deps and lemma/POS.

    Args:
        e1_root: Entity 1 root token
        e2_root: Entity 2 root token
        pattern: Pattern dict with pattern_key

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

    # Find common ancestor
    def get_ancestors(token, max_depth=10):
        """Get all ancestors of a token."""
        ancestors = []
        current = token
        depth = 0
        while current.head != current and depth < max_depth:
            current = current.head
            ancestors.append(current)
            depth += 1
        return ancestors

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

    # Check lemma (handle both literal and concept)
    # For concepts, we'd need concept_clusters, but for now check literal
    if lca.lemma_.lower() != anchor_lemma.lower():
        # Could be a concept - pattern_augmentation.py handles concept expansion
        # For now, skip concept matching (will rely on DependencyMatcher's IN list)
        pass

    # Check dependencies
    # For '>': direct child, For '>>': descendant
    if e1_rel_op == '>':
        if e1_root.head != lca:
            return False
    elif e1_rel_op == '>>':
        if lca not in e1_ancestors:
            return False

    if e2_rel_op == '>':
        if e2_root.head != lca:
            return False
    elif e2_rel_op == '>>':
        if lca not in e2_ancestors:
            return False

    # Check dependency labels
    if e1_root.dep_ != e1_dep:
        return False
    if e2_root.dep_ != e2_dep:
        return False

    return True


def check_bridge_pattern(e1_root, e2_root, pattern):
    """
    Check if BRIDGE pattern exists at entity roots.

    Pattern: e1 > prep > e2

    Args:
        e1_root: Entity 1 root token
        e2_root: Entity 2 root token
        pattern: Pattern dict with pattern_key

    Returns:
        bool: True if pattern matches
    """
    pattern_key = pattern['pattern_key']
    if pattern_key[0] != 'BRIDGE':
        return False

    # Handle both old and new pattern key formats
    if len(pattern_key) == 4:
        _, prep_lemma, third, fourth = pattern_key
        # Check if third is a REL_OP or dependency
        if third in {'>', '>>'}:
            e2_rel_op, e2_dep = third, fourth
        else:
            # Old format: (BRIDGE, prep, e1_dep, e2_dep)
            e2_rel_op, e2_dep = '>', fourth
    else:
        return False

    # Find prepositions that are children of e1
    for child in e1_root.children:
        if child.pos_ != 'ADP':
            continue

        # Check if prep lemma matches (literal check for now)
        if child.lemma_.lower() != prep_lemma.lower():
            # Could be a concept
            continue

        # Check if e2 is related to this prep
        if e2_rel_op == '>':
            # e2 should be direct child of prep
            if e2_root.head == child:
                # Check dependency if specified
                if e2_dep and e2_root.dep_ != e2_dep:
                    # e2_dep might be an IN list in the actual pattern
                    # For now, accept any pobj/nmod/obl type deps
                    if e2_root.dep_ not in ['pobj', 'nmod', 'obl', 'pcomp', 'dobj', 'attr', 'oprd', 'conj']:
                        continue
                return True
        elif e2_rel_op == '>>':
            # e2 should be descendant of prep
            def is_descendant(token, ancestor, max_depth=5):
                current = token
                depth = 0
                while current.head != current and depth < max_depth:
                    if current.head == ancestor:
                        return True
                    current = current.head
                    depth += 1
                return False

            if is_descendant(e2_root, child):
                return True

    return False


def check_linear_pattern(e1_root, e2_root, pattern, doc):
    """
    Check if LINEAR pattern exists at entity roots.

    Pattern: e1 .* token1 .* token2 .* e2 (precedence-based)

    Args:
        e1_root: Entity 1 root token
        e2_root: Entity 2 root token
        pattern: Pattern dict with pattern_key
        doc: spaCy Doc

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

    # Determine order
    if e1_pos < e2_pos:
        start, end = e1_pos, e2_pos
        between = doc[start+1:end]
    else:
        start, end = e2_pos, e1_pos
        between = doc[start+1:end]

    # Check if required tokens appear in order between entities
    last_found_pos = start

    for required_lemma, required_pos in tokens_tuple:
        # Find this token after last_found_pos
        found = False
        for token in doc[last_found_pos+1:end]:
            # Match POS first
            if token.pos_ != required_pos:
                continue

            # For lemma: check literal match
            # If lemma looks like a concept (uppercase), skip lemma check for now
            # (DependencyMatcher handles concept expansion via {IN: [...]})
            if required_lemma.isupper() or '_' in required_lemma:
                # Likely a concept - accept any lemma with matching POS
                last_found_pos = token.i
                found = True
                break
            elif token.lemma_.lower() == required_lemma.lower():
                # Literal lemma match
                last_found_pos = token.i
                found = True
                break

        if not found:
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
      3. Default to "Other" if no match

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
    from tqdm.auto import tqdm

    predictions = []
    directions = []
    explanations = []

    # Statistics
    stats = {
        'total_samples': len(samples),
        'matched': 0,
        'default_other': 0,
        'pattern_usage': {},
        'matches_by_type': {}
    }

    print(f"\nApplying {len(patterns)} patterns (entity-rooted) to {len(samples)} samples...")

    for sample in tqdm(samples, desc="Classifying"):
        doc = sample['doc']
        e1_root = sample['e1_span'].root
        e2_root = sample['e2_span'].root

        matched_pattern = None

        # Try patterns in priority order (already sorted)
        for pattern in patterns:
            pattern_type = pattern['pattern_type']

            # Check if pattern matches at entity roots
            matched = False

            if pattern_type == 'DIRECT':
                matched = check_direct_pattern(e1_root, e2_root, pattern)
            elif pattern_type == 'TRIANGLE':
                matched = check_triangle_pattern(e1_root, e2_root, pattern)
            elif pattern_type == 'BRIDGE':
                matched = check_bridge_pattern(e1_root, e2_root, pattern)
            elif pattern_type == 'LINEAR':
                matched = check_linear_pattern(e1_root, e2_root, pattern, doc)

            if matched:
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
        else:
            predictions.append('Other')
            directions.append(None)
            explanations.append('No pattern matched at entity roots; defaulting to Other.')
            stats['default_other'] += 1

    # Calculate statistics
    stats['match_rate'] = stats['matched'] / stats['total_samples'] if stats['total_samples'] > 0 else 0
    stats['default_rate'] = stats['default_other'] / stats['total_samples'] if stats['total_samples'] > 0 else 0

    print(f"\nClassification complete!")
    print(f"  Matched: {stats['matched']} ({stats['match_rate']:.1%})")
    print(f"  Default to Other: {stats['default_other']} ({stats['default_rate']:.1%})")
    print(f"  Unique patterns used: {len(stats['pattern_usage'])}")
    print(f"  Matches by type: {stats['matches_by_type']}")

    return predictions, directions, explanations, stats
