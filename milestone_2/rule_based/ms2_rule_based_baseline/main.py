#!/usr/bin/env python3
"""
Explainable Rule-Based Relation Extraction
Milestone 2 - SemEval 2010 Task 8

A deterministic, rule-based system for relation extraction that is both
effective and fully explainable. Rules are automatically discovered from
training data using pattern mining.
"""

from pathlib import Path

import spacy
from sklearn.metrics import classification_report

from src.classifier import apply_rule_based_classifier
from src.config import (
    MIN_PRECISION,
    MIN_SUPPORT,
    SPACY_MODEL,
    TOP_RULES_DISPLAY,
    TOP_RULES_PER_RELATION,
    get_project_paths,
)
from src.data_processing import load_datasets, preprocess_data
from src.evaluation import (
    analyze_error_patterns,
    analyze_errors,
    compute_metrics,
    print_metrics_summary,
    print_rule_diagnostics,
    save_all_outputs,
)
from src.pattern_mining import (
    analyze_relation_features,
    extract_candidate_patterns,
    filter_and_rank_patterns,
    generate_patterns_from_analysis,
    visualize_rules_by_relation,
)


def parse_arguments():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(
        description="MS2 Baseline: Rule-based relation extraction"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: local results/)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the rule-based relation extraction system."""
    args = parse_arguments()
    
    # Load spaCy model
    print("Loading spaCy model...")
    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError:
        print(
            f"Model '{SPACY_MODEL}' not found. Please download it using: "
            f"python -m spacy download {SPACY_MODEL}"
        )
        return

    print("Libraries loaded successfully!")
    print(f"spaCy version: {spacy.__version__}")

    # Set up paths
    current_dir = Path(__file__).parent
    paths = get_project_paths(current_dir)

    # Use provided output dir or default to local results folder
    if args.output_dir:
        results_dir = Path(args.output_dir)
    else:
        results_dir = current_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCurrent working directory: {Path.cwd()}")
    print(f"Script directory: {current_dir}")
    print(f"Project root: {paths['project_root']}")
    print(f"Results directory: {results_dir}")

    # Load datasets
    print("\nLoading datasets...")
    try:
        train_data, test_data = load_datasets(paths["data_dir"])
        print(f"Training samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Process data
    print("Processing data...\n")
    train_processed = preprocess_data(train_data, nlp)
    print("\nProcessing test data...")
    test_processed = preprocess_data(test_data, nlp)

    print(f"\nProcessed {len(train_processed)} training samples")
    print(f"Processed {len(test_processed)} test samples")

    # Display sample
    _display_sample(train_processed[0])

    # Generate data-driven patterns
    print("\n" + "=" * 80)
    print("GENERATING DATA-DRIVEN PATTERNS")
    print("Top features per relation extracted from analysis")
    print("=" * 80)

    relation_features = analyze_relation_features(train_processed)
    data_driven_patterns = generate_patterns_from_analysis(relation_features)
    _display_patterns(data_driven_patterns)

    # Mine and rank rules
    print("\nStep 1: Mining patterns from training data...")
    lexical_patterns, dep_patterns = extract_candidate_patterns(train_processed)

    print(f"\nFound {len(lexical_patterns)} unique lexical pattern candidates")
    print(f"Found {len(dep_patterns)} unique dependency pattern candidates")

    print(f"\nStep 2: Filtering by precision >= {MIN_PRECISION} "
          f"and support >= {MIN_SUPPORT}...")
    discovered_rules = filter_and_rank_patterns(
        lexical_patterns, dep_patterns,
        min_precision=MIN_PRECISION, min_support=MIN_SUPPORT
    )

    print(f"\nDiscovered {len(discovered_rules)} high-quality rules")
    _display_top_rules(discovered_rules)
    visualize_rules_by_relation(discovered_rules, top_n=TOP_RULES_PER_RELATION)

    # Test classifier on sample
    print("\nTesting rule-based classifier on sample sentences...")
    _test_classifier_samples(train_processed[:5], discovered_rules, nlp)

    # Evaluate on full datasets
    print("\nEvaluating on training set...")
    train_preds, train_dirs, train_expls = apply_rule_based_classifier(
        train_processed, discovered_rules, nlp
    )
    train_true = [s["relation_directed"] for s in train_processed]

    print("\nEvaluating on test set...")
    test_preds, test_dirs, test_expls = apply_rule_based_classifier(
        test_processed, discovered_rules, nlp
    )
    test_true = [s["relation_directed"] for s in test_processed]

    # Print evaluation results
    _print_evaluation_results(
        train_true, train_preds, test_true, test_preds, discovered_rules
    )

    # Error analysis
    print("\nAnalyzing errors on test set...")
    analyze_errors(test_processed, test_preds, test_true, test_expls, n_samples=15)
    analyze_error_patterns(test_processed, test_preds, test_true)

    # Save outputs to results directory
    save_all_outputs(
        results_dir,
        train_processed, train_preds, train_true,
        test_processed, test_preds, test_true,
        rules=discovered_rules,
        test_explanations=test_expls,
    )

    print(f"\n{'=' * 80}")
    print(f"All results saved to: {results_dir}")
    print("=" * 80)


def _display_sample(sample: dict) -> None:
    """Display a sample output for verification."""
    print("\n" + "=" * 80)
    print("Sample output:")
    print("=" * 80)

    e1_span = sample["e1_span"]
    e2_span = sample["e2_span"]

    print(f"Text: {sample['text']}")
    print(f"Entity 1: {e1_span.text} (POS: {e1_span.root.pos_}, DEP: {e1_span.root.dep_})")
    print(f"Entity 2: {e2_span.text} (POS: {e2_span.root.pos_}, DEP: {e2_span.root.dep_})")
    print(f"Relation: {sample['relation']}")
    print(f"\nDependency path: {sample['dep_path'][:3]}...")
    print(f"Between words: {[w['text'] for w in sample['between_words']]}")


def _display_patterns(patterns: dict) -> None:
    """Display generated patterns for each relation."""
    for relation in sorted(patterns.keys()):
        rel_patterns = patterns[relation]
        print(f"\n{relation}:")
        print(f"  Keywords ({len(rel_patterns['keywords'])}): "
              f"{rel_patterns['keywords'][:10]}")
        print(f"  Verbs ({len(rel_patterns['verb_patterns'])}): "
              f"{rel_patterns['verb_patterns']}")
        print(f"  Preps ({len(rel_patterns['prep_patterns'])}): "
              f"{rel_patterns['prep_patterns']}")
        print(f"  Dep patterns: {len(rel_patterns['dependency_patterns'])} "
              f"patterns extracted")

    print("\n" + "=" * 80)
    print("Data-driven patterns generated successfully!")
    print("These patterns are based on actual frequency analysis of the training data.")
    print("=" * 80)


def _display_top_rules(rules: list[dict]) -> None:
    """Display top discovered rules."""
    print("\nTop 10 rules:")
    print("=" * 100)
    print(f"{'Relation':<25} {'Type':<15} {'Precision':<12} "
          f"{'Support':<10} {'Pattern'}")
    print("-" * 100)

    for rule in rules[:TOP_RULES_DISPLAY]:
        pattern_str = str(rule["pattern_data"])[:40]
        print(f"{rule['relation']:<25} {rule['pattern_type']:<15} "
              f"{rule['precision']:<12.3f} {rule['support']:<10} {pattern_str}")


def _test_classifier_samples(
    samples: list[dict],
    rules: list[dict],
    nlp,
) -> None:
    """Test classifier on a few samples and display results."""
    print("=" * 80)
    preds, dirs, expls = apply_rule_based_classifier(samples, rules, nlp)

    for i, (sample, relation, explanation) in enumerate(zip(samples, preds, expls)):
        print(f"\nSample {i + 1}:")
        print(f"Text: {sample['text'][:100]}...")
        print(f"E1: '{sample['e1_span'].text}' | E2: '{sample['e2_span'].text}'")
        print(f"True: {sample['relation_directed']}")
        print(f"Predicted: {relation}")
        print(f"Explanation: {explanation}")
        match = "✓" if relation == sample["relation_directed"] else "✗"
        print(f"Match: {match}")


def _print_evaluation_results(
    train_true: list[str],
    train_preds: list[str],
    test_true: list[str],
    test_preds: list[str],
    rules: list[dict],
) -> None:
    """Print evaluation results for train and test sets."""
    print("=" * 80)
    print("DETERMINISTIC RULE-BASED SYSTEM EVALUATION")
    print("=" * 80)

    # Training results
    print("\n### TRAINING SET RESULTS ###\n")
    train_metrics = compute_metrics(train_true, train_preds)
    print(f"Accuracy: {train_metrics['accuracy']:.3f}")
    print("\nPer-class metrics:")
    print(classification_report(train_true, train_preds, zero_division=0))
    print("=" * 80)

    # Test results
    print("\n### TEST SET RESULTS ###\n")
    test_metrics = compute_metrics(test_true, test_preds)
    print(f"Accuracy: {test_metrics['accuracy']:.3f}")
    print("\nPer-class metrics:")
    print(classification_report(test_true, test_preds, zero_division=0, digits=3))

    # Diagnostics and summary
    print_rule_diagnostics(rules)
    print_metrics_summary(train_metrics, test_metrics)


if __name__ == "__main__":
    main()
