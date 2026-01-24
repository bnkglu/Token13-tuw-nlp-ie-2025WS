#!/usr/bin/env python3
"""
MS2 Bridge - Configurable Rule-Based Relation Extraction

A bridge between MS2 (pure lexical/dependency patterns) and MS3 (semantic features).
Supports configurable semantic feature extraction and prediction modes.

Usage:
    python main.py                                    # MS2 baseline
    python main.py --use-semantics                    # Enable semantic patterns
    python main.py --use-semantics --prediction-mode priority_based
"""

import argparse
from pathlib import Path

import spacy
from sklearn.metrics import classification_report

from src.classification import apply_rule_based_classifier
from src.utils.config import (
    MIN_PRECISION,
    MIN_SUPPORT,
    SPACY_MODEL,
    TOP_RULES_DISPLAY,
    TOP_RULES_PER_RELATION,
    get_project_paths,
)
from src.data.loader import load_datasets, preprocess_data
from src.analysis.evaluation import (
    analyze_error_patterns,
    analyze_errors,
    compute_metrics,
    print_metrics_summary,
    print_rule_diagnostics,
    save_all_outputs,
)
from src.analysis.mining import (
    analyze_relation_features,
    generate_patterns_from_analysis,
    visualize_rules_by_relation,
)
from src.patterns.extractor import PatternExtractor
from src.utils.semantic import setup_nltk


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MS2 Bridge: Rule-based relation extraction with optional "
                    "MS3 semantic features."
    )
    parser.add_argument(
        "--use-semantics",
        action="store_true",
        help="Enable MS3 semantic patterns (SYNSET, FRAME, HYPERNYM, PREP_STRUCT)",
    )
    parser.add_argument(
        "--prediction-mode",
        choices=["first_match", "priority_based"],
        default="first_match",
        help="first_match: precision & support ranking | priority_based: pattern type tier ranking",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: local results/<config>/)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the Statistical Rule-Based System."""
    args = parse_arguments()

    # Determine operation modes
    USE_SEMANTICS = args.use_semantics
    PREDICTION_MODE = args.prediction_mode

    # Build output directory name based on flags
    sem_suffix = 'fn_wn_on' if USE_SEMANTICS else 'fn_wn_off'
    mode_suffix = PREDICTION_MODE

    print("="*80)
    print("STATISTICAL RULE-BASED SYSTEM - SemEval 2010 Task 8")
    print("="*80)
    print("Configuration:")
    print(f"  --use-semantics:   {USE_SEMANTICS}")
    print(f"  --prediction-mode: {PREDICTION_MODE}")
    print("=" * 80)

    print("\nLoading spaCy model...")
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

    # Download NLTK resources if using semantics
    if USE_SEMANTICS:
        setup_nltk()

    # Set up paths
    current_dir = Path(__file__).parent
    paths = get_project_paths(current_dir)
    
    # Use provided output dir or default to local results folder
    # This structure matches what the user requested: .../statistical_rule_based_system/results/{setting}
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = current_dir / "results" / f"{sem_suffix}_{mode_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCurrent working directory: {Path.cwd()}")
    print(f"Script directory: {current_dir}")
    print(f"Project root: {paths['project_root']}")
    print(f"Output directory: {output_dir}")

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
    # Instantiate modular extractor
    extractor = PatternExtractor(
        min_precision=MIN_PRECISION,
        min_support=MIN_SUPPORT,
        use_semantics=USE_SEMANTICS,
    )
    
    # Extract
    patterns = extractor.extract_patterns(train_processed)
    print("Pattern extraction complete.")

    print(f"\nStep 2: Filtering by precision >= {MIN_PRECISION} "
          f"and support >= {MIN_SUPPORT}...")
    
    # Filter and rank
    discovered_rules_objs = extractor.filter_and_rank(patterns, prediction_mode=PREDICTION_MODE)
    
    # Convert to legacy dictionaries for compatibility
    discovered_rules = [r.to_dict() for r in discovered_rules_objs]

    print(f"\nDiscovered {len(discovered_rules)} high-quality rules")
    _display_top_rules(discovered_rules)
    visualize_rules_by_relation(discovered_rules, top_n=TOP_RULES_PER_RELATION)

    # Test classifier on sample
    print("\nTesting rule-based classifier on sample sentences...")
    _test_classifier_samples(train_processed[:5], discovered_rules, nlp, PREDICTION_MODE)

    # Evaluate on full datasets
    print("\nEvaluating on training set...")
    train_preds, train_dirs, train_expls = apply_rule_based_classifier(
        train_processed, discovered_rules, nlp, PREDICTION_MODE
    )
    train_true = [s["relation_directed"] for s in train_processed]

    print("\nEvaluating on test set...")
    test_preds, test_dirs, test_expls = apply_rule_based_classifier(
        test_processed, discovered_rules, nlp, PREDICTION_MODE
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

    # Save outputs
    save_all_outputs(
        output_dir,
        train_processed, train_preds, train_true,
        test_processed, test_preds, test_true,
        rules=discovered_rules,
        test_explanations=test_expls,
    )

    print(f"\n{'=' * 80}")
    print(f"All results saved to: {output_dir}")
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
    prediction_mode: str,
) -> None:
    """Test classifier on a few samples and display results."""
    print("=" * 80)
    preds, dirs, expls = apply_rule_based_classifier(
        samples, rules, nlp, prediction_mode
    )

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

    print("\n### TRAINING SET RESULTS ###\n")
    train_metrics = compute_metrics(train_true, train_preds)
    print(f"Accuracy: {train_metrics['accuracy']:.3f}")
    print("\nPer-class metrics:")
    print(classification_report(train_true, train_preds, zero_division=0))
    print("=" * 80)

    print("\n### TEST SET RESULTS ###\n")
    test_metrics = compute_metrics(test_true, test_preds)
    print(f"Accuracy: {test_metrics['accuracy']:.3f}")
    print("\nPer-class metrics:")
    print(classification_report(test_true, test_preds, zero_division=0, digits=3))

    print_rule_diagnostics(rules)
    print_metrics_summary(train_metrics, test_metrics)


if __name__ == "__main__":
    main()
