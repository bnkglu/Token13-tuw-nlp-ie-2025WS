"""
Evaluate training set coverage and accuracy.

This script measures how well the mined patterns cover the training data,
which is critical for the maximum coverage approach.
"""

import json
import sys
from pathlib import Path
import spacy

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import preprocess_data
from execution_engine import compile_dependency_matcher, apply_patterns_with_anchoring


def get_directed_label(item):
    """Extract directed relation label from data item."""
    rel_type = item['relation']['type']
    direction = item['relation'].get('direction', '')
    if rel_type == 'Other':
        return 'Other'
    direction = direction.replace('(', '').replace(')', '')
    return f"{rel_type}({direction})"


def main():
    print("=" * 80)
    print("TRAINING SET COVERAGE EVALUATION")
    print("=" * 80)

    # Load spaCy model
    print("\nLoading spaCy model...")
    nlp = spacy.load('en_core_web_lg')

    # Load training data
    print("Loading training data...")
    train_path = Path(__file__).parent.parent.parent / "data" / "processed" / "train" / "train.json"
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    print(f"Loaded {len(train_data)} training samples")

    # Preprocess
    print("Preprocessing samples...")
    train_processed = preprocess_data(train_data, nlp)

    # Load patterns
    print("Loading patterns...")
    patterns_path = Path(__file__).parent.parent / "data" / "patterns_augmented.json"
    with open(patterns_path, 'r') as f:
        patterns = json.load(f)
    print(f"Loaded {len(patterns)} patterns")

    # Compile and apply
    print("\nCompiling DependencyMatcher...")
    dep_matcher, pattern_lookup = compile_dependency_matcher(patterns, nlp)

    print("Applying patterns to training set...")
    preds, dirs, expls, stats = apply_patterns_with_anchoring(
        train_processed, dep_matcher, pattern_lookup, nlp
    )

    # Get true labels
    true_labels = [get_directed_label(item) for item in train_data]

    # Calculate metrics
    correct = sum(1 for t, p in zip(true_labels, preds) if t == p)
    matched = stats['matched']
    total = len(train_data)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\nCoverage Metrics:")
    print(f"  Total samples: {total}")
    print(f"  Matched (found pattern): {matched} ({matched/total:.1%})")
    print(f"  Default to Other: {stats['default_other']} ({stats['default_other']/total:.1%})")

    print(f"\nAccuracy Metrics:")
    print(f"  Correct predictions: {correct} ({correct/total:.1%})")
    print(f"  Training Accuracy: {correct/total:.1%}")

    print(f"\nPattern Usage:")
    print(f"  Unique patterns used: {stats.get('unique_patterns_used', len(stats.get('pattern_usage', {})))}/{len(patterns)}")
    print(f"  Utilization rate: {stats.get('unique_patterns_used', len(stats.get('pattern_usage', {})))/len(patterns):.1%}")

    print(f"\nAnchoring Stats:")
    print(f"  Match attempts: {stats['match_attempts']}")
    print(f"  Failed anchoring: {stats['failed_anchoring']}")
    if stats['match_attempts'] > 0:
        fail_rate = stats['failed_anchoring'] / stats['match_attempts'] * 100
        print(f"  Anchoring failure rate: {fail_rate:.1f}%")

    # Analyze unmatched samples
    if matched < total:
        print(f"\n" + "-" * 80)
        print("UNMATCHED SAMPLES ANALYSIS")
        print("-" * 80)

        unmatched_relations = {}
        for i, (pred, true) in enumerate(zip(preds, true_labels)):
            if pred == 'Other' and true != 'Other':  # Assume Other = no match
                unmatched_relations[true] = unmatched_relations.get(true, 0) + 1

        print(f"\nUnmatched samples by relation (top 10):")
        for rel, count in sorted(unmatched_relations.items(), key=lambda x: -x[1])[:10]:
            print(f"  {rel}: {count}")

        if unmatched_relations:
            total_unmatched = sum(unmatched_relations.values())
            print(f"\nTotal unmatched non-Other samples: {total_unmatched}")

    # Per-relation accuracy
    print(f"\n" + "-" * 80)
    print("PER-RELATION ACCURACY")
    print("-" * 80)

    relation_stats = {}
    for true, pred in zip(true_labels, preds):
        if true not in relation_stats:
            relation_stats[true] = {'total': 0, 'correct': 0}
        relation_stats[true]['total'] += 1
        if true == pred:
            relation_stats[true]['correct'] += 1

    print(f"\n{'Relation':<40} {'Total':<10} {'Correct':<10} {'Accuracy':<10}")
    print("-" * 70)
    for rel in sorted(relation_stats.keys()):
        stats_rel = relation_stats[rel]
        acc = stats_rel['correct'] / stats_rel['total'] * 100
        print(f"{rel:<40} {stats_rel['total']:<10} {stats_rel['correct']:<10} {acc:<10.1f}%")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    target_coverage = 95
    target_accuracy = 90

    print(f"\nTarget Goals:")
    print(f"  Training Coverage: ≥{target_coverage}% (Current: {matched/total:.1%})")
    print(f"  Training Accuracy: ≥{target_accuracy}% (Current: {correct/total:.1%})")

    if matched/total * 100 >= target_coverage and correct/total * 100 >= target_accuracy:
        print(f"\n✅ BOTH TARGETS MET!")
    elif matched/total * 100 >= target_coverage:
        print(f"\n⚠️  Coverage target met, but accuracy below target")
        print(f"   Consider: Pattern precision may be too low, add filtering")
    elif correct/total * 100 >= target_accuracy:
        print(f"\n⚠️  Accuracy target met, but coverage below target")
        print(f"   Consider: Need more fallback patterns or relax constraints")
    else:
        print(f"\n❌ BOTH TARGETS MISSED")
        print(f"   Coverage gap: {target_coverage - matched/total*100:.1f} percentage points")
        print(f"   Accuracy gap: {target_accuracy - correct/total*100:.1f} percentage points")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
