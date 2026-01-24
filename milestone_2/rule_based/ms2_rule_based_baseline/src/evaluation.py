"""Evaluation and error analysis utilities.

This module handles:
- Computing classification metrics
- Analyzing errors
- Generating reports
- Saving predictions and plots
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary of metric names to values.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def print_metrics_summary(
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> None:
    """
    Print a summary table of train vs test metrics.

    Args:
        train_metrics: Training set metrics.
        test_metrics: Test set metrics.
    """
    print(f"\n{'Metric':<30} {'Test Set':<15} {'Train Set':<15}")
    print("-" * 60)
    print(
        f"{'Macro-averaged Precision':<30} "
        f"{test_metrics['macro_precision']:<15.3f} "
        f"{train_metrics['macro_precision']:<15.3f}"
    )
    print(
        f"{'Macro-averaged Recall':<30} "
        f"{test_metrics['macro_recall']:<15.3f} "
        f"{train_metrics['macro_recall']:<15.3f}"
    )
    print(
        f"{'Macro-averaged F1':<30} "
        f"{test_metrics['macro_f1']:<15.3f} "
        f"{train_metrics['macro_f1']:<15.3f}"
    )
    print(
        f"{'Accuracy':<30} "
        f"{test_metrics['accuracy']:<15.3f} "
        f"{train_metrics['accuracy']:<15.3f}"
    )


def analyze_errors(
    samples: list[dict],
    predictions: list[str],
    true_labels: list[str],
    explanations: list[str],
    n_samples: int = 20,
) -> list[dict]:
    """
    Analyze misclassified examples.

    Args:
        samples: List of sample dictionaries.
        predictions: Predicted labels.
        true_labels: True labels.
        explanations: Rule explanations.
        n_samples: Number of error samples to display.

    Returns:
        List of error dictionaries.
    """
    errors = []

    for i, (sample, pred, true) in enumerate(zip(samples, predictions, true_labels)):
        if pred != true:
            errors.append({
                "index": i,
                "sample": sample,
                "predicted": pred,
                "true": true,
                "text": sample["text"],
                "explanation": explanations[i],
            })

    total = len(samples)
    error_rate = len(errors) / total * 100
    print(f"Total errors: {len(errors)} / {total} ({error_rate:.1f}%)")
    print(f"\nShowing first {min(n_samples, len(errors))} errors:\n")
    print("=" * 80)

    for i, error in enumerate(errors[:n_samples]):
        print(f"\nError {i + 1}:")
        print(f"Text: {error['text']}")
        print(f"Entity 1: {error['sample']['e1_span'].text}")
        print(f"Entity 2: {error['sample']['e2_span'].text}")
        print(f"True relation: {error['true']}")
        print(f"Predicted: {error['predicted']}")
        print(f"Rule applied: {error['explanation']}")

        between_words = [w["text"] for w in error["sample"]["between_words"]]
        print(f"Between words: {between_words}")

        dep_path = error["sample"]["dep_path"]
        if dep_path:
            path_str = " -> ".join([f"{d[0]}({d[2]})" for d in dep_path[:5]])
            print(f"Dependency path: {path_str}")

        print("-" * 80)

    return errors


def analyze_error_patterns(
    samples: list[dict],
    predictions: list[str],
    true_labels: list[str],
) -> dict:
    """
    Analyze error patterns by relation type.

    Args:
        samples: List of sample dictionaries.
        predictions: Predicted labels.
        true_labels: True labels.

    Returns:
        Error matrix as nested dictionary.
    """
    error_matrix = defaultdict(lambda: defaultdict(int))

    for sample, pred, true in zip(samples, predictions, true_labels):
        if pred != true:
            error_matrix[true][pred] += 1

    print("\nMost Common Misclassification Patterns:")
    print("=" * 80)
    print(f"{'True Label':<25} {'Predicted As':<25} {'Count':<10}")
    print("-" * 80)

    all_errors = []
    for true_label in error_matrix:
        for pred_label in error_matrix[true_label]:
            count = error_matrix[true_label][pred_label]
            all_errors.append((true_label, pred_label, count))

    all_errors.sort(key=lambda x: x[2], reverse=True)

    for true_label, pred_label, count in all_errors[:15]:
        print(f"{true_label:<25} {pred_label:<25} {count:<10}")

    return error_matrix


def print_rule_diagnostics(rules: list[dict]) -> None:
    """
    Print rule statistics and diagnostics.

    Args:
        rules: List of rule dictionaries.
    """
    print("\n" + "=" * 80)
    print("RULE DIAGNOSTICS")
    print("=" * 80)

    relation_rule_counts = defaultdict(int)
    for rule in rules:
        relation_rule_counts[rule["relation"]] += 1

    print("\nRules discovered per relation:")
    print(f"{'Relation':<30} {'Number of Rules'}")
    print("-" * 50)
    for relation in sorted(relation_rule_counts.keys()):
        print(f"{relation:<30} {relation_rule_counts[relation]}")

    print(f"\nTotal rules: {len(rules)}")
    print(f"Average precision: {np.mean([r['precision'] for r in rules]):.3f}")
    print(f"Average support: {np.mean([r['support'] for r in rules]):.1f}")


def save_predictions_for_scorer(
    predictions: list[str],
    processed_data: list[dict],
    output_file: Path | str,
) -> None:
    """
    Save predictions in official scorer format.

    Args:
        predictions: List of predicted labels.
        processed_data: List of sample dictionaries.
        output_file: Path to output file.
    """
    with open(output_file, "w") as f:
        for pred, sample in zip(predictions, processed_data):
            sample_id = sample["id"]
            f.write(f"{sample_id}\t{pred}\n")

    print(f"Saved {len(predictions)} predictions to {output_file}")


def create_answer_key(
    processed_data: list[dict],
    output_file: Path | str,
) -> None:
    """
    Create answer key file in official format.

    Args:
        processed_data: List of sample dictionaries.
        output_file: Path to output file.
    """
    with open(output_file, "w") as f:
        for sample in processed_data:
            sample_id = sample["id"]
            gold_label = sample["relation_directed"]
            f.write(f"{sample_id}\t{gold_label}\n")

    print(f"Saved {len(processed_data)} gold labels to {output_file}")


def save_results(
    train_report: str,
    test_report: str,
    output_file: Path | str,
) -> None:
    """
    Save classification reports to a text file.

    Args:
        train_report: Training set classification report.
        test_report: Test set classification report.
        output_file: Path to output file.
    """
    with open(output_file, "w") as f:
        f.write("DETERMINISTIC RULE-BASED SYSTEM EVALUATION\n")
        f.write("=" * 80 + "\n")
        f.write("\n### TRAINING SET RESULTS ###\n")
        f.write(train_report)
        f.write("\n" + "=" * 80 + "\n")
        f.write("\n### TEST SET RESULTS ###\n")
        f.write(test_report)
    print(f"Saved evaluation results to {output_file}")


def plot_and_save_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    output_file: Path | str,
    title: str = "Confusion Matrix",
) -> None:
    """
    Generate and save confusion matrix plot.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        output_file: Path to output file.
        title: Plot title.
    """
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved confusion matrix plot to {output_file}")


def save_all_outputs(
    output_dir: Path,
    train_processed: list[dict],
    train_preds: list[str],
    train_true: list[str],
    test_processed: list[dict],
    test_preds: list[str],
    test_true: list[str],
    rules: list[dict] | None = None,
    test_explanations: list[str] | None = None,
) -> None:
    """
    Save all outputs with organized folder structure.

    Creates subfolders:
        - evaluation/: evaluation_results.txt, confusion_matrix_test.png
        - rules/: rules.json, rules_summary.tsv
        - predictions/: test_predictions.json

    Args:
        output_dir: Base output directory.
        train_processed: Processed training samples.
        train_preds: Training predictions.
        train_true: Training ground truth.
        test_processed: Processed test samples.
        test_preds: Test predictions.
        test_true: Test ground truth.
        rules: Discovered rules (optional).
        test_explanations: Explanations for test predictions (optional).
    """
    import json

    # Create organized subfolders
    eval_dir = output_dir / "evaluation"
    rules_dir = output_dir / "rules"
    predictions_dir = output_dir / "predictions"

    for directory in [eval_dir, rules_dir, predictions_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # === 1. Save Rules ===
    if rules:
        print("\nSaving discovered rules...")
        rules_file = rules_dir / "rules.json"
        with open(rules_file, "w") as f:
            json.dump(rules, f, indent=2, default=str)
        print(f"  Saved {len(rules)} rules to {rules_file}")

        # Save rules summary (TSV for easy viewing)
        rules_tsv = rules_dir / "rules_summary.tsv"
        with open(rules_tsv, "w") as f:
            f.write("relation\tpattern_type\tprecision\tsupport\tpattern_data\n")
            for rule in rules:
                f.write(
                    f"{rule['relation']}\t{rule['pattern_type']}\t"
                    f"{rule['precision']:.3f}\t{rule['support']}\t"
                    f"{rule['pattern_data']}\n"
                )
        print(f"  Saved rules summary to {rules_tsv}")

    # === 2. Save Predictions with Triggered Rules ===
    print("\nSaving predictions with triggered rules...")
    test_preds_file = predictions_dir / "test_predictions.json"
    test_predictions_data = []

    for i, (sample, pred, true) in enumerate(
        zip(test_processed, test_preds, test_true)
    ):
        pred_entry = {
            "id": sample.get("id", i),
            "text": sample["text"],
            "e1": sample["e1_span"].text,
            "e2": sample["e2_span"].text,
            "gold_label": true,
            "predicted_label": pred,
            "is_correct": pred == true,
            "triggered_rule": test_explanations[i] if test_explanations else "N/A",
        }
        test_predictions_data.append(pred_entry)

    with open(test_preds_file, "w") as f:
        json.dump(test_predictions_data, f, indent=2, default=str)
    print(f"  Saved {len(test_predictions_data)} test predictions to {test_preds_file}")

    # Quick stats
    correct = sum(1 for p in test_predictions_data if p["is_correct"])
    accuracy = correct / len(test_predictions_data) * 100
    print(f"  Accuracy: {correct}/{len(test_predictions_data)} ({accuracy:.1f}%)")

    # === 3. Save Evaluation Results ===
    print("\nSaving evaluation results and plots...")

    train_report = classification_report(train_true, train_preds, zero_division=0)
    test_report = classification_report(
        test_true, test_preds, zero_division=0, digits=3
    )

    save_results(train_report, test_report, eval_dir / "evaluation_results.txt")
    plot_and_save_confusion_matrix(
        test_true,
        test_preds,
        eval_dir / "confusion_matrix_test.png",
        title="Confusion Matrix (Test Set)",
    )

