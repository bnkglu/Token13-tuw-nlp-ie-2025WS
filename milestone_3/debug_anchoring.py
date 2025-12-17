"""
Debug script to understand why 95% of anchoring checks are failing.

This script will:
1. Load a sample of test data
2. Run pattern matching with detailed logging
3. Analyze anchoring failures
4. Identify root causes
"""

import json
import sys
from pathlib import Path
import spacy
from collections import defaultdict

sys.path.insert(0, str(Path.cwd() / 'src'))

from utils import preprocess_data
from execution_engine import compile_dependency_matcher, parse_match_indices, verify_anchoring

# Load spaCy model
print("Loading spaCy model...")
nlp = spacy.load('en_core_web_lg')

# Load patterns
print("Loading patterns...")
with open('data/patterns_augmented.json', 'r') as f:
    patterns = json.load(f)
print(f"Loaded {len(patterns)} patterns")

# Load test data (just first 100 samples for debugging)
print("\nLoading test data...")
with open('../data/processed/test/test.json', 'r') as f:
    test_data = json.load(f)[:100]  # Only first 100

test_processed = preprocess_data(test_data, nlp)
print(f"Loaded {len(test_processed)} test samples")

# Compile matcher
print("\nCompiling DependencyMatcher...")
dep_matcher, pattern_lookup = compile_dependency_matcher(patterns, nlp)

# Detailed anchoring analysis
print("\n" + "="*80)
print("DETAILED ANCHORING ANALYSIS")
print("="*80)

anchoring_failures = []
anchoring_successes = []
failure_reasons = defaultdict(int)

for sample_idx, sample in enumerate(test_processed[:20]):  # First 20 samples
    doc = sample['doc']
    e1_root_idx = sample['e1_span'].root.i
    e2_root_idx = sample['e2_span'].root.i

    print(f"\n{'='*80}")
    print(f"Sample {sample_idx + 1}: {sample['sentence'][:100]}...")
    print(f"  E1: '{sample['e1_span'].text}' (root token: '{doc[e1_root_idx].text}' at position {e1_root_idx})")
    print(f"  E2: '{sample['e2_span'].text}' (root token: '{doc[e2_root_idx].text}' at position {e2_root_idx})")
    print(f"  True relation: {sample['relation_directed']}")

    # Get all matches
    matches = dep_matcher(doc)
    print(f"\n  Total DependencyMatcher matches: {len(matches)}")

    if len(matches) == 0:
        print("  → No matches found!")
        continue

    # Analyze each match
    for match_idx, (match_id_int, token_indices) in enumerate(matches[:5]):  # First 5 matches
        match_id = nlp.vocab.strings[match_id_int]
        pattern = pattern_lookup.get(match_id)

        if not pattern:
            continue

        print(f"\n  Match {match_idx + 1}:")
        print(f"    Pattern: {pattern['pattern_id']} ({pattern['pattern_type']})")
        print(f"    Predicted relation: {pattern['relation']}")
        print(f"    Precision: {pattern['precision']:.2f}, Support: {pattern['support']}")

        # Parse indices
        match_indices = parse_match_indices(token_indices, pattern)

        print(f"    Matched token indices: {token_indices}")
        print(f"    Parsed match_indices: {match_indices}")

        # Show what tokens were matched
        for node_id, idx in match_indices.items():
            if idx < len(doc):
                token = doc[idx]
                print(f"      {node_id}: token[{idx}] = '{token.text}' (POS={token.pos_}, DEP={token.dep_})")

        # Verify anchoring
        e1_matched = match_indices.get('e1')
        e2_matched = match_indices.get('e2')

        print(f"    Anchoring check:")
        print(f"      Expected e1: {e1_root_idx} ('{doc[e1_root_idx].text}')")
        print(f"      Matched e1:  {e1_matched} ('{doc[e1_matched].text}' if e1_matched else 'None')")
        print(f"      Expected e2: {e2_root_idx} ('{doc[e2_root_idx].text}')")
        print(f"      Matched e2:  {e2_matched} ('{doc[e2_matched].text}' if e2_matched else 'None')")

        anchored = verify_anchoring(match_indices, e1_root_idx, e2_root_idx)

        if anchored:
            print(f"    ✅ ANCHORING PASSED!")
            anchoring_successes.append({
                'sample_idx': sample_idx,
                'pattern_id': pattern['pattern_id'],
                'pattern_type': pattern['pattern_type'],
                'relation': pattern['relation']
            })
        else:
            print(f"    ❌ ANCHORING FAILED!")

            # Determine failure reason
            if e1_matched is None or e2_matched is None:
                reason = "missing_e1_or_e2"
            elif e1_matched != e1_root_idx and e2_matched != e2_root_idx:
                reason = "both_misaligned"
            elif e1_matched != e1_root_idx:
                reason = "e1_misaligned"
            elif e2_matched != e2_root_idx:
                reason = "e2_misaligned"
            else:
                reason = "unknown"

            failure_reasons[reason] += 1

            anchoring_failures.append({
                'sample_idx': sample_idx,
                'pattern_id': pattern['pattern_id'],
                'pattern_type': pattern['pattern_type'],
                'expected_e1': e1_root_idx,
                'matched_e1': e1_matched,
                'expected_e2': e2_root_idx,
                'matched_e2': e2_matched,
                'reason': reason
            })

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nTotal anchoring checks: {len(anchoring_failures) + len(anchoring_successes)}")
print(f"  Successes: {len(anchoring_successes)} ({len(anchoring_successes)/(len(anchoring_failures)+len(anchoring_successes)+0.0001)*100:.1f}%)")
print(f"  Failures: {len(anchoring_failures)} ({len(anchoring_failures)/(len(anchoring_failures)+len(anchoring_successes)+0.0001)*100:.1f}%)")

print(f"\nFailure reasons breakdown:")
for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
    pct = count / len(anchoring_failures) * 100 if anchoring_failures else 0
    print(f"  {reason}: {count} ({pct:.1f}%)")

# Show examples of each failure type
print("\n" + "="*80)
print("EXAMPLE FAILURES BY TYPE")
print("="*80)

for reason in failure_reasons.keys():
    examples = [f for f in anchoring_failures if f['reason'] == reason][:3]
    if examples:
        print(f"\n{reason.upper()} examples:")
        for ex in examples:
            sample = test_processed[ex['sample_idx']]
            print(f"  Sample {ex['sample_idx']}: {sample['sentence'][:80]}...")
            print(f"    Pattern: {ex['pattern_id']} ({ex['pattern_type']})")
            print(f"    Expected: e1={ex['expected_e1']}, e2={ex['expected_e2']}")
            print(f"    Matched:  e1={ex['matched_e1']}, e2={ex['matched_e2']}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
