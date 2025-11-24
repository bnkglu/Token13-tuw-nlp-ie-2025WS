"""Evaluation pipeline for rule-based relation extraction.

This module provides tools to evaluate rule-based extraction against
the SemEval 2010 Task 8 test set using the official scorer.
"""

import argparse
import json
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import spacy
from tqdm import tqdm

from src.rules.ml_rule_pipeline import MLRulePipeline
from src.rules.rule_matcher import RuleMatcher
from src.rules.sdp_extractor import SDPExtractor


@dataclass
class EvaluationResult:
    """Results from evaluation.

    Attributes
    ----------
    total_examples : int
        Total number of examples
    predictions : Dict[int, str]
        Predictions by sentence ID
    gold_labels : Dict[int, str]
        Gold labels by sentence ID
    correct : int
        Number of correct predictions
    confusion_matrix : Dict[str, Dict[str, int]]
        Confusion matrix (gold -> pred -> count)
    per_relation_metrics : Dict[str, Dict[str, float]]
        Precision, recall, F1 per relation
    """

    total_examples: int = 0
    predictions: Dict[int, str] = field(default_factory=dict)
    gold_labels: Dict[int, str] = field(default_factory=dict)
    correct: int = 0
    confusion_matrix: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    per_relation_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


class RuleEvaluator:
    """Evaluate rule-based relation extraction.

    Parameters
    ----------
    matcher : RuleMatcher
        The rule matcher to evaluate
    extractor : Optional[SDPExtractor]
        SDP extractor for loading test data
    """

    # Path to official SemEval scorer
    SCORER_PATH = Path(__file__).parent.parent.parent.parent / "resources" / \
        "SemEval2010_task8_scorer-v1.2" / "semeval2010_task8_scorer-v1.2.pl"

    def __init__(
        self,
        matcher: RuleMatcher,
        extractor: Optional[SDPExtractor] = None,
    ):
        self.matcher = matcher
        self.extractor = extractor or SDPExtractor()
        self.nlp = matcher.nlp

    def evaluate_conllu(
        self,
        test_conllu_path: str,
        default_relation: str = "Other",
    ) -> EvaluationResult:
        """Evaluate on CoNLL-U formatted test data.

        Parameters
        ----------
        test_conllu_path : str
            Path to test CoNLL-U file
        default_relation : str
            Default relation when no match is found

        Returns
        -------
        EvaluationResult
            Evaluation results
        """
        examples = self.extractor.load_conllu(test_conllu_path)
        result = EvaluationResult()
        result.total_examples = len(examples)

        for ex in tqdm(examples, desc="Evaluating"):
            sent_id = ex["sent_id"]
            gold = ex["relation"]
            result.gold_labels[sent_id] = gold

            # Get entity positions
            e1_id = ex.get("e1_token_id")
            e2_id = ex.get("e2_token_id")

            if e1_id is None or e2_id is None:
                result.predictions[sent_id] = default_relation
                continue

            # Process text with spaCy
            doc = self.nlp(ex["text"])

            # Adjust token indices (CoNLL-U is 1-indexed, spaCy is 0-indexed)
            e1_idx = e1_id - 1
            e2_idx = e2_id - 1

            # Get prediction
            prediction = self.matcher.extract_for_semeval(
                doc, e1_idx, e2_idx, apply_constraints=True
            )

            if prediction is None:
                prediction = default_relation

            result.predictions[sent_id] = prediction

            # Update confusion matrix
            result.confusion_matrix[gold][prediction] += 1

            if prediction == gold:
                result.correct += 1

        # Compute per-relation metrics
        result.per_relation_metrics = self._compute_metrics(result.confusion_matrix)

        return result

    def _compute_metrics(
        self, confusion: Dict[str, Dict[str, int]]
    ) -> Dict[str, Dict[str, float]]:
        """Compute precision, recall, F1 per relation."""
        metrics = {}

        # Get all relation types
        all_relations = set(confusion.keys())
        for preds in confusion.values():
            all_relations.update(preds.keys())

        for rel in all_relations:
            # True positives: predicted rel and gold is rel
            tp = confusion.get(rel, {}).get(rel, 0)

            # False positives: predicted rel but gold is different
            fp = sum(
                confusion.get(g, {}).get(rel, 0)
                for g in all_relations if g != rel
            )

            # False negatives: gold is rel but predicted different
            fn = sum(
                confusion.get(rel, {}).get(p, 0)
                for p in all_relations if p != rel
            )

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            metrics[rel] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": tp + fn,
            }

        return metrics

    def export_predictions(
        self,
        result: EvaluationResult,
        output_path: str,
    ) -> None:
        """Export predictions in SemEval format.

        Parameters
        ----------
        result : EvaluationResult
            Evaluation result with predictions
        output_path : str
            Output file path
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for sent_id in sorted(result.predictions.keys()):
                pred = result.predictions[sent_id]
                f.write(f"{sent_id}\t{pred}\n")

        print(f"Exported predictions to: {output_path}")

    def export_gold(
        self,
        result: EvaluationResult,
        output_path: str,
    ) -> None:
        """Export gold labels in SemEval format.

        Parameters
        ----------
        result : EvaluationResult
            Evaluation result with gold labels
        output_path : str
            Output file path
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for sent_id in sorted(result.gold_labels.keys()):
                gold = result.gold_labels[sent_id]
                f.write(f"{sent_id}\t{gold}\n")

        print(f"Exported gold labels to: {output_path}")

    def run_official_scorer(
        self,
        predictions_path: str,
        gold_path: str,
    ) -> Optional[str]:
        """Run official SemEval scorer.

        Parameters
        ----------
        predictions_path : str
            Path to predictions file
        gold_path : str
            Path to gold labels file

        Returns
        -------
        Optional[str]
            Scorer output, or None if failed
        """
        if not self.SCORER_PATH.exists():
            print(f"Warning: Scorer not found at {self.SCORER_PATH}")
            return None

        try:
            result = subprocess.run(
                ["perl", str(self.SCORER_PATH), predictions_path, gold_path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            return result.stdout + result.stderr
        except Exception as e:
            print(f"Error running scorer: {e}")
            return None

    def print_results(self, result: EvaluationResult) -> None:
        """Print evaluation results summary.

        Parameters
        ----------
        result : EvaluationResult
            Evaluation results
        """
        accuracy = result.correct / result.total_examples if result.total_examples > 0 else 0

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Total examples: {result.total_examples}")
        print(f"Correct: {result.correct}")
        print(f"Accuracy: {accuracy:.2%}")
        print()

        # Macro F1 (excluding Other)
        f1_scores = [
            m["f1"]
            for rel, m in result.per_relation_metrics.items()
            if rel != "Other" and m["support"] > 0
        ]
        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

        print(f"Macro F1 (excl. Other): {macro_f1:.4f}")
        print()

        print("Per-relation metrics:")
        print("-" * 60)
        print(f"{'Relation':<35} {'P':>8} {'R':>8} {'F1':>8} {'Supp':>6}")
        print("-" * 60)

        for rel, m in sorted(result.per_relation_metrics.items()):
            print(
                f"{rel:<35} {m['precision']:>8.2%} {m['recall']:>8.2%} "
                f"{m['f1']:>8.4f} {m['support']:>6}"
            )

        print("-" * 60)

    def analyze_errors(
        self,
        result: EvaluationResult,
        n_examples: int = 10,
    ) -> None:
        """Analyze common errors.

        Parameters
        ----------
        result : EvaluationResult
            Evaluation results
        n_examples : int
            Number of error examples to show per category
        """
        print("\n" + "=" * 60)
        print("ERROR ANALYSIS")
        print("=" * 60)

        # Find most common error types
        errors = []
        for gold, preds in result.confusion_matrix.items():
            for pred, count in preds.items():
                if gold != pred and count > 0:
                    errors.append((gold, pred, count))

        errors.sort(key=lambda x: x[2], reverse=True)

        print("\nMost common errors (gold -> predicted):")
        print("-" * 60)
        for gold, pred, count in errors[:20]:
            print(f"  {gold:<30} -> {pred:<30} ({count})")


def train_and_evaluate(
    train_path: str,
    test_path: str,
    top_n: int = 30,
    model_type: str = "logistic",
    output_dir: Optional[str] = None,
) -> EvaluationResult:
    """Train ML-based pattern discovery and evaluate on test set.

    Parameters
    ----------
    train_path : str
        Path to training CoNLL-U file
    test_path : str
        Path to test CoNLL-U file
    top_n : int
        Number of top patterns per relation
    model_type : str
        ML model type ("logistic" or "random_forest")
    output_dir : Optional[str]
        Directory for output files

    Returns
    -------
    EvaluationResult
        Evaluation results
    """
    print("Training ML-based pattern discovery...")
    pipeline = MLRulePipeline(
        model_type=model_type,
        n_top_patterns=top_n,
    )
    pipeline_result = pipeline.run(train_path)
    matcher = pipeline_result.rule_matcher

    print(f"\nPattern statistics:")
    stats = matcher.get_pattern_statistics()
    print(f"  Total patterns: {stats['total_patterns']}")
    for rel, count in stats["patterns_by_relation"].items():
        print(f"    {rel}: {count}")

    print("\nEvaluating on test set...")
    evaluator = RuleEvaluator(matcher)
    result = evaluator.evaluate_conllu(test_path)

    evaluator.print_results(result)
    evaluator.analyze_errors(result)

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        pred_file = output_path / "predictions.txt"
        gold_file = output_path / "gold.txt"

        evaluator.export_predictions(result, str(pred_file))
        evaluator.export_gold(result, str(gold_file))

        # Export rules
        rules_file = output_path / "ml_patterns.yaml"
        pipeline.export_rules(pipeline_result, str(rules_file))

        # Run official scorer
        print("\nRunning official SemEval scorer...")
        scorer_output = evaluator.run_official_scorer(str(pred_file), str(gold_file))
        if scorer_output:
            print(scorer_output)

            # Save scorer output
            with open(output_path / "scorer_output.txt", "w") as f:
                f.write(scorer_output)

    return result


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate ML-based rule extraction"
    )
    parser.add_argument(
        "--train",
        type=str,
        default="data/processed/train/train.conllu",
        help="Path to training CoNLL-U file",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="data/processed/test/test.conllu",
        help="Path to test CoNLL-U file",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top patterns per relation",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["logistic", "random_forest"],
        default="logistic",
        help="ML model type",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/rule_based",
        help="Output directory",
    )

    args = parser.parse_args()

    train_and_evaluate(
        train_path=args.train,
        test_path=args.test,
        top_n=args.top_n,
        model_type=args.model,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
