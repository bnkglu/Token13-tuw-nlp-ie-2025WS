"""
Calculate accuracy for WITH vs WITHOUT anchoring on the 100 test samples.
"""

import json
import sys
from pathlib import Path
import spacy

def _find_milestone3_src(start: Path) -> Path:
    start = start.resolve()
    candidates = [
        start / "milestone_3" / "src",
        start / "src" if start.name == "milestone_3" else None
    ]
    for c in candidates:
        if c and c.exists():
            return c
    for p in start.parents:
        c = p / "milestone_3" / "src"
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not locate milestone_3/src from CWD={start}")

src_path = _find_milestone3_src(Path.cwd())
milestone3_dir = src_path.parent
repo_root = milestone3_dir.parent
sys.path.insert(0, str(src_path))

from utils import preprocess_data
from execution_engine import (
    compile_dependency_matcher,
    apply_patterns_with_anchoring,
    apply_patterns_no_anchoring,
)

nlp = spacy.load("en_core_web_lg")

with open(milestone3_dir / "data" / "patterns_augmented.json", "r") as f:
    patterns = json.load(f)

with open(repo_root / "data" / "processed" / "test" / "test.json", "r") as f:
    test_data = json.load(f)[:100]

test_processed = preprocess_data(test_data, nlp)

# Get ground truth
def get_directed_label(item):
    rel_type = item['relation']['type']
    direction = item['relation'].get('direction', '')
    if rel_type == 'Other':
        return 'Other'
    direction = direction.replace('(', '').replace(')', '')
    return f"{rel_type}({direction})"

true_labels = [get_directed_label(item) for item in test_data]

# Compile matcher
dep_matcher, pattern_lookup = compile_dependency_matcher(patterns, nlp)

# WITH anchoring
print("Running WITH anchoring...")
preds_with, _, _, stats_with = apply_patterns_with_anchoring(
    test_processed, dep_matcher, pattern_lookup, nlp
)

# WITHOUT anchoring
print("\nRunning WITHOUT anchoring...")
preds_without, _, _, stats_without = apply_patterns_no_anchoring(
    test_processed, dep_matcher, pattern_lookup, nlp
)

# Calculate accuracy
correct_with = sum(1 for t, p in zip(true_labels, preds_with) if t == p)
correct_without = sum(1 for t, p in zip(true_labels, preds_without) if t == p)

acc_with = correct_with / len(true_labels)
acc_without = correct_without / len(true_labels)

print("\n" + "="*80)
print("ACCURACY COMPARISON")
print("="*80)

print(f"\nWITH Anchoring:")
print(f"  Correct: {correct_with}/100")
print(f"  Accuracy: {acc_with:.1%}")

print(f"\nWITHOUT Anchoring:")
print(f"  Correct: {correct_without}/100")
print(f"  Accuracy: {acc_without:.1%}")

print(f"\nDifference:")
print(f"  {acc_without - acc_with:+.1%} ({correct_without - correct_with:+d} more correct)")

# Analyze the changes
print("\n" + "="*80)
print("DETAILED ANALYSIS OF CHANGED PREDICTIONS")
print("="*80)

better = 0  # Changed from wrong to correct
worse = 0   # Changed from correct to wrong
both_wrong = 0  # Changed but both wrong

for i, (true, p_with, p_without) in enumerate(zip(true_labels, preds_with, preds_without)):
    if p_with != p_without:
        was_correct = (p_with == true)
        is_correct = (p_without == true)

        if not was_correct and is_correct:
            better += 1
        elif was_correct and not is_correct:
            worse += 1
        else:
            both_wrong += 1

print(f"\nOf {sum(1 for p1, p2 in zip(preds_with, preds_without) if p1 != p2)} changed predictions:")
print(f"  Improved (wrong → correct): {better}")
print(f"  Degraded (correct → wrong): {worse}")
print(f"  Still wrong (wrong → different wrong): {both_wrong}")

print(f"\nNet improvement: {better - worse:+d} predictions")

print("\n" + "="*80)
if acc_without > acc_with:
    print("✅ RECOMMENDATION: Use NO ANCHORING approach")
    print(f"   Expected performance gain: {(acc_without - acc_with)*100:.1f} percentage points")
else:
    print("⚠️  WARNING: NO ANCHORING performed worse")
    print(f"   Performance loss: {(acc_with - acc_without)*100:.1f} percentage points")
print("="*80)
