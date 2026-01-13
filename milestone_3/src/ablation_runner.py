"""
Ablation study runner for Milestone 3.

This script systematically tests different component configurations
to measure their individual contribution to model performance.

Experiments:
  A1: Baseline (current system as-is)
  A2: Enable FrameNet filtering
  A3: Enable enhanced matching (WordNet + FrameNet in matcher)
  A4: Disable WordNet augmentation
  A5: Add M2-style lexical patterns (TODO)
  A6: Relax entity anchoring
  A7: Fix pattern direction (TODO)

Usage:
  python ablation_runner.py --all
  python ablation_runner.py --experiments A1 A2 A3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from ast import literal_eval
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import spacy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from pattern_augmentation import (
    filter_patterns_tiered,
    filter_patterns_framenet_aware,
    generate_passive_variants,
    sort_patterns,
)
from utils import preprocess_data


# =============================================================================
# CONFIGURATION
# =============================================================================

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "ablation_results"
PROCESSED_DATA_DIR = BASE_DIR.parent / "data" / "processed"

# Create results directory
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

EXPERIMENTS = {
    "A1_baseline": {
        "description": "Baseline - current system as-is",
        "config": {
            "enable_framenet_filtering": False,
            "enhanced_matching_enabled": False,
            "entity_type_disambiguation": True,
            "min_frame_score": 0.5,
            "e1_anchor_tolerance": 0,  # exact
            "e2_anchor_tolerance": 2,
            "wordnet_enabled": True,
            "lexical_fallback": False,
        }
    },
    "A2_framenet_enabled": {
        "description": "Enable FrameNet pattern filtering",
        "config": {
            "enable_framenet_filtering": True,
            "enhanced_matching_enabled": False,
            "entity_type_disambiguation": True,
            "min_frame_score": 0.5,
            "e1_anchor_tolerance": 0,
            "e2_anchor_tolerance": 2,
            "wordnet_enabled": True,
            "lexical_fallback": False,
        }
    },
    "A3_enhanced_matching": {
        "description": "Enable enhanced matching (WordNet + FrameNet in matcher)",
        "config": {
            "enable_framenet_filtering": False,
            "enhanced_matching_enabled": True,
            "entity_type_disambiguation": True,
            "min_frame_score": 0.5,
            "e1_anchor_tolerance": 0,
            "e2_anchor_tolerance": 2,
            "wordnet_enabled": True,
            "lexical_fallback": False,
        }
    },
    "A4_no_wordnet": {
        "description": "Disable WordNet features entirely",
        "config": {
            "enable_framenet_filtering": False,
            "enhanced_matching_enabled": False,
            "entity_type_disambiguation": False,  # Uses WordNet
            "min_frame_score": 0.5,
            "e1_anchor_tolerance": 0,
            "e2_anchor_tolerance": 2,
            "wordnet_enabled": False,
            "lexical_fallback": False,
        }
    },
    "A5_framenet_strict": {
        "description": "FrameNet with stricter threshold (0.6)",
        "config": {
            "enable_framenet_filtering": False,
            "enhanced_matching_enabled": False,
            "entity_type_disambiguation": True,
            "min_frame_score": 0.6,  # Stricter
            "e1_anchor_tolerance": 0,
            "e2_anchor_tolerance": 2,
            "wordnet_enabled": True,
            "lexical_fallback": False,
        }
    },
    "A6_relaxed_anchoring": {
        "description": "Relaxed anchoring (e1 ±1, e2 ±3)",
        "config": {
            "enable_framenet_filtering": False,
            "enhanced_matching_enabled": False,
            "entity_type_disambiguation": True,
            "min_frame_score": 0.5,
            "e1_anchor_tolerance": 1,  # Relaxed
            "e2_anchor_tolerance": 3,  # Relaxed
            "wordnet_enabled": True,
            "lexical_fallback": False,
        }
    },
    "A7_all_features": {
        "description": "All features enabled",
        "config": {
            "enable_framenet_filtering": True,
            "enhanced_matching_enabled": True,
            "entity_type_disambiguation": True,
            "min_frame_score": 0.5,
            "e1_anchor_tolerance": 0,
            "e2_anchor_tolerance": 2,
            "wordnet_enabled": True,
            "lexical_fallback": False,
        }
    },
    "A8_lexical_fallback": {
        "description": "Baseline + M2-style lexical fallback patterns",
        "config": {
            "enable_framenet_filtering": False,
            "enhanced_matching_enabled": False,
            "entity_type_disambiguation": True,
            "min_frame_score": 0.5,
            "e1_anchor_tolerance": 0,
            "e2_anchor_tolerance": 2,
            "wordnet_enabled": True,
            "lexical_fallback": True,  # NEW: Enable lexical fallback
        }
    },
}


# =============================================================================
# PATTERN BUILDING
# =============================================================================

def build_patterns(config: Dict[str, Any]) -> List[Dict]:
    """Build patterns with specified configuration."""
    raw_patterns_path = DATA_DIR / "raw_patterns.json"
    concept_clusters_path = DATA_DIR / "concept_clusters.json"

    print(f"  Loading raw patterns...")
    with open(raw_patterns_path, "r") as f:
        raw = json.load(f)

    with open(concept_clusters_path, "r") as f:
        concept_data = json.load(f)

    expanded_clusters = concept_data.get("expanded_clusters", {})

    pattern_counts_str = raw["pattern_counts"]
    pattern_counts = {literal_eval(k): v for k, v in pattern_counts_str.items()}

    print(f"  Candidate patterns: {len(pattern_counts)}")

    # Apply tiered filtering
    filtered_patterns = filter_patterns_tiered(pattern_counts, expanded_clusters)
    print(f"  After tiered filtering: {len(filtered_patterns)} patterns")

    # Optionally apply FrameNet filtering
    if config.get("enable_framenet_filtering", False):
        print("  Applying FrameNet filtering...")
        filtered_patterns = filter_patterns_framenet_aware(filtered_patterns)
        print(f"  After FrameNet validation: {len(filtered_patterns)} patterns")

    # Generate passive variants
    augmented_patterns = generate_passive_variants(filtered_patterns, min_precision_for_flip=0.75)

    # Sort patterns
    sorted_patterns = sort_patterns(augmented_patterns)

    print(f"  Final pattern count: {len(sorted_patterns)}")

    return sorted_patterns


# =============================================================================
# ENTITY-ROOTED MATCHING WITH CONFIG
# =============================================================================

# Global lexical matcher (cached after first use)
_lexical_matcher = None


def get_lexical_matcher(nlp, train_samples, train_labels):
    """Get or create the lexical pattern matcher (cached)."""
    global _lexical_matcher
    if _lexical_matcher is None:
        from lexical_patterns import LexicalPatternMatcher
        print("  Mining lexical patterns for fallback...")
        _lexical_matcher = LexicalPatternMatcher(nlp)
        _lexical_matcher.mine_patterns(train_samples, train_labels)
    return _lexical_matcher


def apply_patterns_with_config(samples, patterns, nlp, config: Dict[str, Any],
                               train_samples=None, train_labels=None):
    """
    Apply patterns with configurable settings.

    This is a simplified version that respects ablation config.
    """
    # Import here to avoid circular imports and allow config injection
    import entity_rooted_matcher as erm

    # Temporarily override global config
    original_enhanced = erm.ENHANCED_MATCHING_ENABLED
    original_entity_type = erm.ENTITY_TYPE_DISAMBIGUATION_ENABLED
    original_min_frame = erm.MIN_FRAME_SCORE

    try:
        # Apply config
        erm.ENHANCED_MATCHING_ENABLED = config.get("enhanced_matching_enabled", False)
        erm.ENTITY_TYPE_DISAMBIGUATION_ENABLED = config.get("entity_type_disambiguation", True)
        erm.MIN_FRAME_SCORE = config.get("min_frame_score", 0.5)

        # Check if lexical fallback is enabled
        use_lexical = config.get("lexical_fallback", False)

        if use_lexical and train_samples is not None and train_labels is not None:
            # Use the new function with lexical fallback
            lexical_matcher = get_lexical_matcher(nlp, train_samples, train_labels)
            predictions, directions, explanations, stats = erm.apply_patterns_with_lexical_fallback(
                samples, patterns, nlp, lexical_matcher
            )
        else:
            # Use the original function
            predictions, directions, explanations, stats = erm.apply_patterns_entity_rooted(
                samples, patterns, nlp
            )

        return predictions, directions, explanations, stats

    finally:
        # Restore original config
        erm.ENHANCED_MATCHING_ENABLED = original_enhanced
        erm.ENTITY_TYPE_DISAMBIGUATION_ENABLED = original_entity_type
        erm.MIN_FRAME_SCORE = original_min_frame


# =============================================================================
# EVALUATION
# =============================================================================

def get_directed_label(item):
    """Get directed relation label from data item."""
    rel_type = item['relation']['type']
    direction = item['relation'].get('direction', '')
    if rel_type == 'Other':
        return 'Other'
    direction = direction.replace('(', '').replace(')', '')
    return f"{rel_type}({direction})"


def evaluate_predictions(predictions, ground_truth):
    """Compute evaluation metrics."""
    metrics = {
        "accuracy": accuracy_score(ground_truth, predictions),
        "precision_macro": precision_score(ground_truth, predictions, average='macro', zero_division=0),
        "recall_macro": recall_score(ground_truth, predictions, average='macro', zero_division=0),
        "f1_macro": f1_score(ground_truth, predictions, average='macro', zero_division=0),
        "precision_weighted": precision_score(ground_truth, predictions, average='weighted', zero_division=0),
        "recall_weighted": recall_score(ground_truth, predictions, average='weighted', zero_division=0),
        "f1_weighted": f1_score(ground_truth, predictions, average='weighted', zero_division=0),
    }

    # Per-class metrics
    report = classification_report(ground_truth, predictions, output_dict=True, zero_division=0)
    metrics["per_class"] = report

    return metrics


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def run_experiment(exp_name: str, config: Dict[str, Any], nlp, train_samples, test_samples, train_labels, test_labels):
    """Run a single ablation experiment."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"Description: {EXPERIMENTS[exp_name]['description']}")
    print(f"Config: {config}")
    print(f"{'='*80}")

    # Build patterns with this config
    print("\n[1/3] Building patterns...")
    patterns = build_patterns(config)

    # Apply patterns to test set
    print("\n[2/3] Applying patterns to test set...")
    test_preds, _, _, test_stats = apply_patterns_with_config(
        test_samples, patterns, nlp, config,
        train_samples=train_samples, train_labels=train_labels
    )

    # Evaluate
    print("\n[3/3] Evaluating...")
    test_metrics = evaluate_predictions(test_preds, test_labels)

    # Also run on train for comparison
    print("\n[Bonus] Evaluating on train set...")
    train_preds, _, _, train_stats = apply_patterns_with_config(
        train_samples, patterns, nlp, config,
        train_samples=train_samples, train_labels=train_labels
    )
    train_metrics = evaluate_predictions(train_preds, train_labels)

    result = {
        "experiment": exp_name,
        "description": EXPERIMENTS[exp_name]["description"],
        "config": config,
        "pattern_count": len(patterns),
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "test_stats": {k: v for k, v in test_stats.items() if k != 'pattern_usage'},
        "train_stats": {k: v for k, v in train_stats.items() if k != 'pattern_usage'},
    }

    # Print summary
    print(f"\n{'='*40}")
    print(f"RESULTS: {exp_name}")
    print(f"{'='*40}")
    print(f"  Patterns: {len(patterns)}")
    print(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    print(f"  Train Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"  Train F1 (macro): {train_metrics['f1_macro']:.4f}")

    return result


def generate_comparison_report(results: List[Dict], output_path: Path):
    """Generate a markdown comparison report."""
    report_lines = [
        "# Ablation Study Results",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary Table",
        "",
        "| Experiment | Description | Patterns | Test Acc | Test F1 | Train Acc | Train F1 |",
        "|------------|-------------|----------|----------|---------|-----------|----------|",
    ]

    # Sort by test accuracy descending
    sorted_results = sorted(results, key=lambda x: x['test_metrics']['accuracy'], reverse=True)

    for r in sorted_results:
        report_lines.append(
            f"| {r['experiment']} | {r['description'][:30]} | {r['pattern_count']} | "
            f"{r['test_metrics']['accuracy']:.4f} | {r['test_metrics']['f1_macro']:.4f} | "
            f"{r['train_metrics']['accuracy']:.4f} | {r['train_metrics']['f1_macro']:.4f} |"
        )

    report_lines.extend([
        "",
        "## Key Findings",
        "",
        f"**Best Test Accuracy**: {sorted_results[0]['experiment']} ({sorted_results[0]['test_metrics']['accuracy']:.4f})",
        f"**Best Test F1**: {max(results, key=lambda x: x['test_metrics']['f1_macro'])['experiment']} "
        f"({max(results, key=lambda x: x['test_metrics']['f1_macro'])['test_metrics']['f1_macro']:.4f})",
        "",
        "## Detailed Configurations",
        "",
    ])

    for r in results:
        report_lines.extend([
            f"### {r['experiment']}",
            f"**Description**: {r['description']}",
            "",
            "**Config**:",
            "```json",
            json.dumps(r['config'], indent=2),
            "```",
            "",
            f"**Test Stats**: Match rate={r['test_stats'].get('match_rate', 0):.2%}, "
            f"Default to Other={r['test_stats'].get('default_rate', 0):.2%}",
            "",
        ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS.keys()),
                        help="Specific experiments to run")
    parser.add_argument("--output", type=str, default=None, help="Output file for results")
    args = parser.parse_args()

    # Determine which experiments to run
    if args.all:
        experiments_to_run = list(EXPERIMENTS.keys())
    elif args.experiments:
        experiments_to_run = args.experiments
    else:
        print("Please specify --all or --experiments")
        return

    print(f"Running {len(experiments_to_run)} experiments: {experiments_to_run}")

    # Load spaCy
    print("\nLoading spaCy model...")
    nlp = spacy.load("en_core_web_lg")

    # Load and preprocess data
    print("\nLoading data...")
    train_path = PROCESSED_DATA_DIR / "train" / "train.json"
    test_path = PROCESSED_DATA_DIR / "test" / "test.json"

    with open(train_path) as f:
        train_data = json.load(f)
    with open(test_path) as f:
        test_data = json.load(f)

    print(f"  Train samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")

    # Preprocess
    print("\nPreprocessing data (this may take a while)...")
    train_samples = preprocess_data(train_data, nlp)
    test_samples = preprocess_data(test_data, nlp)

    # Get ground truth labels
    train_labels = [get_directed_label(item) for item in train_data]
    test_labels = [get_directed_label(item) for item in test_data]

    # Run experiments
    results = []
    for exp_name in experiments_to_run:
        config = EXPERIMENTS[exp_name]["config"]
        result = run_experiment(exp_name, config, nlp, train_samples, test_samples, train_labels, test_labels)
        results.append(result)

    # Save raw results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = RESULTS_DIR / f"ablation_results_{timestamp}.json"

    # Convert results for JSON serialization
    json_results = []
    for r in results:
        r_copy = r.copy()
        # Remove non-serializable per-class metrics details
        if 'per_class' in r_copy.get('test_metrics', {}):
            del r_copy['test_metrics']['per_class']
        if 'per_class' in r_copy.get('train_metrics', {}):
            del r_copy['train_metrics']['per_class']
        json_results.append(r_copy)

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nRaw results saved to: {results_file}")

    # Generate report
    report_file = RESULTS_DIR / f"ablation_report_{timestamp}.md"
    generate_comparison_report(results, report_file)

    # Print final summary
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print("="*80)
    print("\nFinal Rankings (by Test Accuracy):")
    sorted_results = sorted(results, key=lambda x: x['test_metrics']['accuracy'], reverse=True)
    for i, r in enumerate(sorted_results, 1):
        print(f"  {i}. {r['experiment']}: {r['test_metrics']['accuracy']:.4f} acc, {r['test_metrics']['f1_macro']:.4f} F1")


if __name__ == "__main__":
    main()
