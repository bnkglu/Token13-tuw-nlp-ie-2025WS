"""
Test script to compare DependencyMatcher vs Entity-Rooted matching.

This will verify that entity-rooted matching eliminates anchoring failures.
"""

import json
import sys
from pathlib import Path
import spacy

sys.path.insert(0, str(Path.cwd() / 'src'))

from utils import preprocess_data
from execution_engine import compile_dependency_matcher, apply_patterns_with_anchoring
from entity_rooted_matcher import apply_patterns_entity_rooted

# Load spaCy model
print("Loading spaCy model...")
nlp = spacy.load('en_core_web_lg')

# Load patterns
print("Loading patterns...")
with open('data/patterns_augmented.json', 'r') as f:
    patterns = json.load(f)
print(f"Loaded {len(patterns)} patterns")

# Load test data (first 50 samples)
print("\nLoading test data...")
with open('../data/processed/test/test.json', 'r') as f:
    test_data = json.load(f)[:50]

test_processed = preprocess_data(test_data, nlp)
print(f"Loaded {len(test_processed)} test samples")

# Test 1: DependencyMatcher approach (current)
print("\n" + "="*80)
print("TEST 1: DependencyMatcher Approach (Current)")
print("="*80)

dep_matcher, pattern_lookup = compile_dependency_matcher(patterns, nlp)

preds_dm, dirs_dm, expls_dm, stats_dm = apply_patterns_with_anchoring(
    test_processed, dep_matcher, pattern_lookup, nlp
)

print(f"\nResults:")
print(f"  Matched: {stats_dm['matched']}")
print(f"  Failed anchoring: {stats_dm['failed_anchoring']}")
print(f"  Match attempts: {stats_dm['match_attempts']}")
if stats_dm['match_attempts'] > 0:
    fail_rate = stats_dm['failed_anchoring'] / stats_dm['match_attempts'] * 100
    print(f"  Anchoring failure rate: {fail_rate:.1f}%")

# Test 2: Entity-rooted approach (new)
print("\n" + "="*80)
print("TEST 2: Entity-Rooted Approach (New)")
print("="*80)

preds_er, dirs_er, expls_er, stats_er = apply_patterns_entity_rooted(
    test_processed, patterns, nlp
)

print(f"\nResults:")
print(f"  Matched: {stats_er['matched']}")
print(f"  Default to Other: {stats_er['default_other']}")

# Compare predictions
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

print(f"\nMatch rate:")
print(f"  DependencyMatcher: {stats_dm['matched']}/{len(test_processed)} ({stats_dm['match_rate']:.1%})")
print(f"  Entity-Rooted: {stats_er['matched']}/{len(test_processed)} ({stats_er['match_rate']:.1%})")

# Show some example differences
print(f"\nExample predictions:")
for i in range(min(5, len(test_processed))):
    print(f"\nSample {i+1}: {test_processed[i]['sentence'][:80]}...")
    print(f"  DependencyMatcher: {preds_dm[i]}")
    print(f"  Entity-Rooted: {preds_er[i]}")
    if preds_dm[i] != preds_er[i]:
        print(f"  ⚠️  DIFFERENT!")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
