"""
Test script to compare WITH vs WITHOUT anchoring verification.

This will show the impact of removing anchoring checks.
"""

import json
import sys
from pathlib import Path
import spacy

sys.path.insert(0, str(Path.cwd() / 'src'))

from utils import preprocess_data
from execution_engine import (
    compile_dependency_matcher,
    apply_patterns_with_anchoring,
    apply_patterns_no_anchoring
)

# Load spaCy model
print("Loading spaCy model...")
nlp = spacy.load('en_core_web_lg')

# Load patterns
print("Loading patterns...")
with open('data/patterns_augmented.json', 'r') as f:
    patterns = json.load(f)
print(f"Loaded {len(patterns)} patterns")

# Load test data (first 100 samples for quick test)
print("\nLoading test data...")
with open('../data/processed/test/test.json', 'r') as f:
    test_data = json.load(f)[:100]

test_processed = preprocess_data(test_data, nlp)
print(f"Loaded {len(test_processed)} test samples")

# Compile matcher
print("\nCompiling DependencyMatcher...")
dep_matcher, pattern_lookup = compile_dependency_matcher(patterns, nlp)

# Test 1: WITH anchoring verification (current)
print("\n" + "="*80)
print("TEST 1: WITH Anchoring Verification (Current)")
print("="*80)

preds_with, dirs_with, expls_with, stats_with = apply_patterns_with_anchoring(
    test_processed, dep_matcher, pattern_lookup, nlp
)

print(f"\nResults:")
print(f"  Matched: {stats_with['matched']} ({stats_with['match_rate']:.1%})")
print(f"  Default to Other: {stats_with['default_other']} ({stats_with['default_rate']:.1%})")
print(f"  Failed anchoring: {stats_with['failed_anchoring']}")
print(f"  Match attempts: {stats_with['match_attempts']}")

# Test 2: WITHOUT anchoring verification (new)
print("\n" + "="*80)
print("TEST 2: WITHOUT Anchoring Verification (New)")
print("="*80)

preds_without, dirs_without, expls_without, stats_without = apply_patterns_no_anchoring(
    test_processed, dep_matcher, pattern_lookup, nlp
)

print(f"\nResults:")
print(f"  Matched: {stats_without['matched']} ({stats_without['match_rate']:.1%})")
print(f"  Default to Other: {stats_without['default_other']} ({stats_without['default_rate']:.1%})")

# Compare
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

print(f"\nMatch rate improvement:")
print(f"  WITH anchoring: {stats_with['match_rate']:.1%}")
print(f"  WITHOUT anchoring: {stats_without['match_rate']:.1%}")
print(f"  Improvement: {(stats_without['match_rate'] - stats_with['match_rate']):.1%}")

print(f"\nPrediction agreement:")
same = sum(1 for p1, p2 in zip(preds_with, preds_without) if p1 == p2)
print(f"  Same predictions: {same}/{len(preds_with)} ({same/len(preds_with):.1%})")
print(f"  Different predictions: {len(preds_with)-same}/{len(preds_with)} ({(len(preds_with)-same)/len(preds_with):.1%})")

# Show examples where they differ
print(f"\nExamples where predictions differ:")
diff_count = 0
for i in range(len(test_processed)):
    if preds_with[i] != preds_without[i]:
        if diff_count < 5:  # Show first 5
            print(f"\nSample {i+1}: {test_processed[i]['sentence'][:80]}...")
            print(f"  True label: {test_processed[i]['relation_directed']}")
            print(f"  WITH anchoring: {preds_with[i]}")
            print(f"  WITHOUT anchoring: {preds_without[i]}")
        diff_count += 1

print(f"\n\nTotal differences: {diff_count}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
