"""
Execution engine module for Milestone 3.

Compiles unified DependencyMatcher and applies patterns with strict anchoring verification.
"""

from spacy.matcher import DependencyMatcher
from tqdm.auto import tqdm


def compile_dependency_matcher(patterns, nlp):
    """
    Compile all patterns into a single DependencyMatcher.

    Args:
        patterns: Sorted list of pattern dicts (length desc, precision desc)
        nlp: spaCy model

    Returns:
        dep_matcher: DependencyMatcher with all patterns
        pattern_lookup: Dict {match_id -> pattern_dict}
    """
    dep_matcher = DependencyMatcher(nlp.vocab)
    pattern_lookup = {}

    print(f"Compiling {len(patterns)} patterns into DependencyMatcher...")

    for i, pattern in enumerate(patterns):
        match_id = f"pattern_{i}"
        dep_pattern = pattern['dep_matcher_pattern']

        try:
            # Cache node ordering for robust anchoring verification / parsing
            node_order = [node.get("RIGHT_ID") for node in dep_pattern]
            pattern["_node_order"] = node_order
            pattern["_node_index"] = {node_id: j for j, node_id in enumerate(node_order)}

            dep_matcher.add(match_id, [dep_pattern])
            pattern_lookup[match_id] = pattern
        except Exception as e:
            print(f"Warning: Failed to add pattern {pattern['pattern_id']}: {e}")

    print(f"Successfully compiled {len(pattern_lookup)} patterns")

    return dep_matcher, pattern_lookup


def parse_match_indices(token_indices, pattern):
    """
    Parse DependencyMatcher token indices into node_id -> idx mapping.

    Args:
        token_indices: List of token indices from match
        pattern: Pattern dict with dep_matcher_pattern

    Returns:
        match_indices: Dict of {node_id -> token_idx}
    """
    dep_pattern = pattern['dep_matcher_pattern']
    match_indices = {}

    # Prefer cached node order (added in compile_dependency_matcher)
    node_order = pattern.get("_node_order")
    if not node_order:
        node_order = [node.get("RIGHT_ID") for node in dep_pattern]

    for i, node_id in enumerate(node_order):
        if i < len(token_indices):
            match_indices[node_id] = token_indices[i]

    return match_indices


def verify_anchoring(token_indices, pattern, e1_root_idx, e2_root_idx):
    """
    Verify that matched nodes align with entity root positions.

    Args:
        token_indices: List of token indices from DependencyMatcher
        pattern: Pattern dict with cached node index positions
        e1_root_idx: Expected e1 root position
        e2_root_idx: Expected e2 root position

    Returns:
        bool: True if anchoring passed
    """
    idx_map = pattern.get("_node_index")
    if not idx_map:
        # Fallback if compiled elsewhere
        idx_map = {node.get("RIGHT_ID"): i for i, node in enumerate(pattern.get("dep_matcher_pattern", []))}

    e1_pos = idx_map.get("e1")
    e2_pos = idx_map.get("e2")
    if e1_pos is None or e2_pos is None:
        return False
    if e1_pos >= len(token_indices) or e2_pos >= len(token_indices):
        return False

    e1_matched = token_indices[e1_pos]
    e2_matched = token_indices[e2_pos]

    return e1_matched == e1_root_idx and e2_matched == e2_root_idx


def verify_anchoring_relaxed(token_indices, pattern, e1_span, e2_span, max_distance=2):
    """
    Relaxed anchoring: Allow matches within entity spans or nearby tokens.

    This addresses the 94.6% anchoring failure rate by accepting matches
    that are within or near entity boundaries, not just exact root positions.

    Args:
        token_indices: DependencyMatcher result indices
        pattern: Pattern dict with cached node positions
        e1_span: Entity 1 span object (has .start, .end attributes)
        e2_span: Entity 2 span object (has .start, .end attributes)
        max_distance: Max token distance outside span (default: 2)

    Returns:
        bool: True if matched positions are within or near entity spans
    """
    idx_map = pattern.get("_node_index")
    if not idx_map:
        # Fallback if compiled elsewhere
        idx_map = {node.get("RIGHT_ID"): i
                  for i, node in enumerate(pattern.get("dep_matcher_pattern", []))}

    e1_pos = idx_map.get("e1")
    e2_pos = idx_map.get("e2")
    if e1_pos is None or e2_pos is None:
        return False
    if e1_pos >= len(token_indices) or e2_pos >= len(token_indices):
        return False

    e1_matched = token_indices[e1_pos]
    e2_matched = token_indices[e2_pos]

    # Check if within entity span boundaries (with tolerance)
    e1_ok = (e1_span.start - max_distance <= e1_matched <= e1_span.end + max_distance)
    e2_ok = (e2_span.start - max_distance <= e2_matched <= e2_span.end + max_distance)

    return e1_ok and e2_ok


def sort_matches_by_priority(matches, pattern_lookup, nlp):
    """
    Sort matches by pattern priority (length desc, precision desc).

    Args:
        matches: List of (match_id, token_indices) tuples
        pattern_lookup: Dict mapping match_id to pattern
        nlp: spaCy model

    Returns:
        sorted_matches: List of (priority, match_id, token_indices)
    """
    match_with_priority = []

    for match_id_int, token_indices in matches:
        # Convert match_id from int to string
        match_id = nlp.vocab.strings[match_id_int]

        pattern = pattern_lookup.get(match_id)
        if pattern:
            # Enhanced priority with pattern type ranking
            pattern_type = pattern.get('pattern_type', '')
            pattern_length = pattern['length']
            precision = pattern['precision']
            support = pattern.get("support", 0)

            # Type ranking: DIRECT > TRIANGLE > BRIDGE > LINEAR > FALLBACK
            # Lower rank number = higher priority
            type_rank = {'DIRECT': 0, 'DIRECT_2HOP': 1, 'TRIANGLE': 2,
                         'BRIDGE': 3, 'LINEAR': 4, 'DIRECT_SIBLING': 5,
                         'FALLBACK': 10}.get(pattern_type, 99)

            # Priority tuple: (type rank, pattern length desc, precision desc, support desc)
            priority = (type_rank, -pattern_length, -precision, -support)
            match_with_priority.append((priority, match_id, token_indices))

    # Sort by priority
    match_with_priority.sort(key=lambda x: x[0])

    return match_with_priority


def apply_patterns_no_anchoring(samples, dep_matcher, pattern_lookup, nlp):
    """
    Apply patterns to samples WITHOUT anchoring verification.

    Process:
      1. Run dep_matcher(doc) → get all matches
      2. Sort matches by pattern priority
      3. Return first match (no anchoring check)
      4. Default to "Other" if no match

    Args:
        samples: List of processed samples
        dep_matcher: Compiled DependencyMatcher
        pattern_lookup: Match ID to pattern mapping
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
        'default_other': 0,
        'pattern_usage': {}
    }

    print(f"\\nApplying patterns to {len(samples)} samples (NO anchoring verification)...")

    for sample in tqdm(samples, desc="Classifying"):
        doc = sample['doc']

        # Get all matches
        matches = dep_matcher(doc)

        matched_pattern = None

        if matches:
            # Sort matches by priority
            sorted_matches = sort_matches_by_priority(matches, pattern_lookup, nlp)

            # Take first match (highest priority) WITHOUT anchoring check
            if sorted_matches:
                priority, match_id, token_indices = sorted_matches[0]
                matched_pattern = pattern_lookup[match_id]

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
            explanations.append('No pattern matched; defaulting to Other.')
            stats['default_other'] += 1

    # Calculate statistics
    stats['match_rate'] = stats['matched'] / stats['total_samples'] if stats['total_samples'] > 0 else 0
    stats['default_rate'] = stats['default_other'] / stats['total_samples'] if stats['total_samples'] > 0 else 0

    print(f"\\nClassification complete!")
    print(f"  Matched: {stats['matched']} ({stats['match_rate']:.1%})")
    print(f"  Default to Other: {stats['default_other']} ({stats['default_rate']:.1%})")
    print(f"  Unique patterns used: {len(stats['pattern_usage'])}")

    return predictions, directions, explanations, stats


def apply_patterns_with_anchoring(samples, dep_matcher, pattern_lookup, nlp):
    """
    Apply patterns to samples with anchoring verification.

    Process:
      1. Run dep_matcher(doc) → get all matches
      2. Sort matches by pattern priority
      3. For each match, verify anchoring
      4. Return first passing match (decision list)
      5. Default to "Other" if no match

    Args:
        samples: List of processed samples
        dep_matcher: Compiled DependencyMatcher
        pattern_lookup: Match ID to pattern mapping
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
        'default_other': 0,
        'failed_anchoring': 0,
        'match_attempts': 0,
        'pattern_usage': {}
    }
    failed_by_pattern = {}
    failed_by_type = {}

    print(f"\\nApplying patterns to {len(samples)} samples...")

    for sample in tqdm(samples, desc="Classifying"):
        doc = sample['doc']
        e1_root_idx = sample['e1_span'].root.i
        e2_root_idx = sample['e2_span'].root.i

        # Get all matches
        matches = dep_matcher(doc)

        matched_pattern = None

        if matches:
            # Sort matches by priority
            sorted_matches = sort_matches_by_priority(matches, pattern_lookup, nlp)

            # Try patterns in order
            for priority, match_id, token_indices in sorted_matches:
                pattern = pattern_lookup[match_id]
                stats["match_attempts"] += 1

                # Verify anchoring (RELAXED: allows matches within/near entity spans)
                if verify_anchoring_relaxed(token_indices, pattern, sample['e1_span'], sample['e2_span']):
                    matched_pattern = pattern
                    break
                else:
                    stats['failed_anchoring'] += 1
                    pid = pattern.get("pattern_id", match_id)
                    ptype = pattern.get("pattern_type", "UNKNOWN")
                    failed_by_pattern[pid] = failed_by_pattern.get(pid, 0) + 1
                    failed_by_type[ptype] = failed_by_type.get(ptype, 0) + 1

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
            explanations.append('No pattern matched with anchoring; defaulting to Other.')
            stats['default_other'] += 1

    # Calculate statistics
    stats['match_rate'] = stats['matched'] / stats['total_samples'] if stats['total_samples'] > 0 else 0
    stats['default_rate'] = stats['default_other'] / stats['total_samples'] if stats['total_samples'] > 0 else 0
    stats['avg_match_attempts_per_sample'] = stats["match_attempts"] / stats['total_samples'] if stats['total_samples'] > 0 else 0
    stats['failed_anchoring_by_type'] = failed_by_type
    # Keep top 25 patterns causing anchoring failures
    stats['failed_anchoring_top_patterns'] = sorted(
        failed_by_pattern.items(), key=lambda x: x[1], reverse=True
    )[:25]

    print(f"\\nClassification complete!")
    print(f"  Matched: {stats['matched']} ({stats['match_rate']:.1%})")
    print(f"  Default to Other: {stats['default_other']} ({stats['default_rate']:.1%})")
    print(f"  Failed anchoring: {stats['failed_anchoring']}")
    print(f"  Match attempts: {stats['match_attempts']} (avg {stats['avg_match_attempts_per_sample']:.1f}/sample)")
    print(f"  Unique patterns used: {len(stats['pattern_usage'])}")

    return predictions, directions, explanations, stats


def analyze_pattern_usage(stats, patterns):
    """
    Analyze which patterns were used and their coverage.

    Args:
        stats: Statistics from apply_patterns_with_anchoring
        patterns: List of all patterns

    Returns:
        usage_analysis: Dict with pattern usage analysis
    """
    pattern_usage = stats['pattern_usage']

    # Sort by usage
    sorted_usage = sorted(pattern_usage.items(), key=lambda x: x[1], reverse=True)

    # Find pattern details
    pattern_map = {p['pattern_id']: p for p in patterns}

    usage_analysis = {
        'total_patterns': len(patterns),
        'used_patterns': len(pattern_usage),
        'unused_patterns': len(patterns) - len(pattern_usage),
        'coverage': len(pattern_usage) / len(patterns) if patterns else 0,
        'top_patterns': []
    }

    # Top 20 patterns
    for pattern_id, count in sorted_usage[:20]:
        pattern = pattern_map.get(pattern_id)
        if pattern:
            usage_analysis['top_patterns'].append({
                'pattern_id': pattern_id,
                'count': count,
                'type': pattern['pattern_type'],
                'relation': pattern['relation'],
                'precision': pattern['precision'],
                'support': pattern['support']
            })

    return usage_analysis
