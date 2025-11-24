"""Integrated pipeline combining ML pattern discovery with rule generation.

This module provides the full pipeline:
1. Extract SDP features from training data
2. Train ML classifier to find discriminative patterns
3. Export high-weight features as spaCy rules
4. Evaluate on test data
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .feature_extractor import SDPFeatureExtractor, extract_features_from_conllu
from .ml_pattern_learner import MLPatternLearner, MLPatternResult
from .pattern_generator import GeneratedPattern, PatternGenerator
from .rule_matcher import RuleMatcher
from .sdp_extractor import SDPExtractor, SDPSignature


@dataclass
class MLRulePipelineResult:
    """Results from the ML-to-rule pipeline.

    Attributes
    ----------
    ml_result : MLPatternResult
        ML training results
    generated_rules : Dict[str, List[GeneratedPattern]]
        Generated rules per relation
    rule_matcher : RuleMatcher
        Configured rule matcher
    """

    ml_result: MLPatternResult
    generated_rules: Dict[str, List[GeneratedPattern]]
    rule_matcher: RuleMatcher


class MLRulePipeline:
    """Pipeline that uses ML to discover patterns and converts them to rules.

    The pipeline:
    1. Extracts SDP signatures from training data
    2. Trains a classifier on signature features
    3. Identifies high-weight signature features per class
    4. Converts those signatures to DependencyMatcher patterns
    5. Creates a RuleMatcher for deployment

    Parameters
    ----------
    model_type : str
        ML model type ("logistic" or "random_forest")
    n_top_patterns : int
        Number of top patterns to extract per relation
    min_feature_freq : int
        Minimum feature frequency
    use_frequency_boost : bool
        Combine ML importance with frequency for ranking
    """

    def __init__(
        self,
        model_type: str = "logistic",
        n_top_patterns: int = 30,
        min_feature_freq: int = 2,
        use_frequency_boost: bool = True,
    ):
        self.model_type = model_type
        self.n_top_patterns = n_top_patterns
        self.min_feature_freq = min_feature_freq
        self.use_frequency_boost = use_frequency_boost

        self._sdp_extractor = SDPExtractor()
        self._pattern_generator = PatternGenerator()

    def run(self, train_conllu_path: str) -> MLRulePipelineResult:
        """Run the full pipeline.

        Parameters
        ----------
        train_conllu_path : str
            Path to training CoNLL-U file

        Returns
        -------
        MLRulePipelineResult
            Pipeline results with matcher
        """
        print("=" * 70)
        print("ML-TO-RULE PIPELINE")
        print("=" * 70)

        # Step 1: Train ML model
        print("\n[Step 1] Training ML classifier...")
        ml_learner = MLPatternLearner(
            model_type=self.model_type,
            n_top_features=self.n_top_patterns * 2,  # Get more, filter later
            min_feature_freq=self.min_feature_freq,
        )
        ml_result = ml_learner.train(train_conllu_path)

        # Step 2: Extract signature patterns (full SDP patterns)
        print("\n[Step 2] Extracting discriminative signature patterns...")
        signature_patterns = self._extract_signature_patterns(ml_result)

        # Step 3: Optionally boost with frequency
        if self.use_frequency_boost:
            print("\n[Step 3] Boosting patterns with frequency data...")
            signature_patterns = self._boost_with_frequency(
                signature_patterns, train_conllu_path
            )

        # Step 4: Generate DependencyMatcher rules
        print("\n[Step 4] Generating DependencyMatcher rules...")
        generated_rules = self._generate_rules(signature_patterns)

        # Step 5: Create RuleMatcher
        print("\n[Step 5] Creating RuleMatcher...")
        matcher = self._create_matcher(generated_rules)

        print(f"\nPipeline complete! Created {len(matcher._pattern_info)} rules.")

        return MLRulePipelineResult(
            ml_result=ml_result,
            generated_rules=generated_rules,
            rule_matcher=matcher,
        )

    def _extract_signature_patterns(
        self, ml_result: MLPatternResult
    ) -> Dict[str, List[Tuple[SDPSignature, float]]]:
        """Extract signature patterns from ML results."""
        patterns = {}

        # We need to reconstruct SDPSignature from the signature strings
        # This requires parsing the string representation back to components

        for relation, features in ml_result.important_features.items():
            sig_patterns = []

            for feat in features:
                if feat.is_signature_feature():
                    sig_str = feat.get_signature_string()
                    if sig_str:
                        # Parse signature string back to SDPSignature
                        sig = self._parse_signature_string(sig_str)
                        if sig:
                            sig_patterns.append((sig, feat.importance))

            # Sort by importance and take top N
            sig_patterns.sort(key=lambda x: x[1], reverse=True)
            patterns[relation] = sig_patterns[:self.n_top_patterns]

        return patterns

    def _parse_signature_string(self, sig_str: str) -> Optional[SDPSignature]:
        """Parse a signature string back to SDPSignature.

        Format: "NOUN --(dep)--> word:POS --(dep)--> NOUN"
        """
        import re

        # Split by arrows
        parts = re.split(r"\s*--\([^)]+\)-->\s*", sig_str)
        dep_matches = re.findall(r"--\(([^)]+)\)-->", sig_str)

        if not parts:
            return None

        pos_pattern = []
        trigger_words = []

        for part in parts:
            part = part.strip()
            if ":" in part:
                # Has trigger word: "word:POS"
                word, pos = part.rsplit(":", 1)
                pos_pattern.append(pos)
                trigger_words.append(word)
            else:
                # Just POS
                pos_pattern.append(part)
                trigger_words.append("")

        return SDPSignature(
            pos_pattern=tuple(pos_pattern),
            dep_pattern=tuple(dep_matches),
            trigger_words=tuple(trigger_words),
            direction="e1_to_e2",
            path_length=len(pos_pattern),
        )

    def _boost_with_frequency(
        self,
        patterns: Dict[str, List[Tuple[SDPSignature, float]]],
        train_path: str,
    ) -> Dict[str, List[Tuple[SDPSignature, float]]]:
        """Boost pattern scores with frequency information."""
        from collections import Counter

        # Extract all signatures and count frequencies inline
        examples = self._sdp_extractor.extract_all_signatures(train_path)

        freq_lookup: Counter = Counter()
        for ex in examples:
            if ex.signature:
                key = (ex.signature.pos_pattern, ex.signature.dep_pattern)
                freq_lookup[key] += 1

        # Boost scores
        boosted = {}
        for relation, sig_patterns in patterns.items():
            boosted_list = []
            for sig, importance in sig_patterns:
                key = (sig.pos_pattern, sig.dep_pattern)
                freq = freq_lookup.get(key, 1)

                # Combined score: importance * log(frequency + 1)
                boosted_score = importance * np.log1p(freq)
                boosted_list.append((sig, boosted_score))

            # Re-sort by boosted score
            boosted_list.sort(key=lambda x: x[1], reverse=True)
            boosted[relation] = boosted_list

        return boosted

    def _generate_rules(
        self,
        patterns: Dict[str, List[Tuple[SDPSignature, float]]],
    ) -> Dict[str, List[GeneratedPattern]]:
        """Generate DependencyMatcher rules from signatures."""
        rules = {}

        for relation, sig_patterns in patterns.items():
            rules[relation] = []

            for sig, score in sig_patterns:
                try:
                    gen_pattern = self._pattern_generator.signature_to_matcher(
                        sig, relation, confidence=score
                    )
                    rules[relation].append(gen_pattern)
                except Exception as e:
                    print(f"Warning: Could not generate rule for {sig.to_string()}: {e}")

        return rules

    def _create_matcher(
        self, rules: Dict[str, List[GeneratedPattern]]
    ) -> RuleMatcher:
        """Create RuleMatcher from generated rules."""
        matcher = RuleMatcher()

        for relation, gen_patterns in rules.items():
            for gp in gen_patterns:
                try:
                    matcher.add_pattern(
                        gp.pattern_id,
                        relation,
                        gp.matcher_pattern,
                        confidence=gp.confidence,
                    )
                except Exception as e:
                    print(f"Warning: Could not add pattern {gp.pattern_id}: {e}")

        return matcher

    def print_patterns(
        self, result: MLRulePipelineResult, n_per_relation: int = 10
    ) -> None:
        """Print discovered patterns.

        Parameters
        ----------
        result : MLRulePipelineResult
            Pipeline results
        n_per_relation : int
            Number of patterns to print per relation
        """
        print("\n" + "=" * 70)
        print("ML-DISCOVERED PATTERNS")
        print("=" * 70)

        for relation, gen_patterns in result.generated_rules.items():
            print(f"\n{relation}:")
            print("-" * 50)

            for i, gp in enumerate(gen_patterns[:n_per_relation], 1):
                print(f"  {i}. {gp.signature.to_string()}")
                print(f"     Score: {gp.confidence:.4f}")

    def export_rules(
        self, result: MLRulePipelineResult, output_path: str
    ) -> None:
        """Export rules to YAML configuration.

        Parameters
        ----------
        result : MLRulePipelineResult
            Pipeline results
        output_path : str
            Output YAML file path
        """
        self._pattern_generator.export_to_yaml(
            result.generated_rules, output_path
        )


def run_ml_pipeline(
    train_path: str,
    test_path: Optional[str] = None,
    model_type: str = "logistic",
    n_patterns: int = 30,
    output_dir: Optional[str] = None,
) -> MLRulePipelineResult:
    """Run the ML-to-rule pipeline with optional evaluation.

    Parameters
    ----------
    train_path : str
        Training CoNLL-U file
    test_path : Optional[str]
        Test CoNLL-U file for evaluation
    model_type : str
        ML model type
    n_patterns : int
        Number of patterns per relation
    output_dir : Optional[str]
        Output directory

    Returns
    -------
    MLRulePipelineResult
        Pipeline results
    """
    pipeline = MLRulePipeline(
        model_type=model_type,
        n_top_patterns=n_patterns,
    )

    result = pipeline.run(train_path)
    pipeline.print_patterns(result)

    # Export
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Export rules
        rules_path = out_path / "ml_patterns.yaml"
        pipeline.export_rules(result, str(rules_path))

        # Export ML report
        report_path = out_path / "ml_report.txt"
        with open(report_path, "w") as f:
            f.write(f"Model: {model_type}\n")
            f.write(f"CV F1: {result.ml_result.cv_scores['mean_f1']:.4f}\n\n")
            f.write(result.ml_result.classification_report)

    # Evaluate if test path provided
    if test_path:
        print("\n" + "=" * 70)
        print("EVALUATION ON TEST SET")
        print("=" * 70)

        from src.models.rule_based.evaluate_rules import RuleEvaluator

        evaluator = RuleEvaluator(result.rule_matcher)
        eval_result = evaluator.evaluate_conllu(test_path)
        evaluator.print_results(eval_result)

        if output_dir:
            pred_path = out_path / "predictions.txt"
            evaluator.export_predictions(eval_result, str(pred_path))

    return result


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run ML-to-rule pipeline for relation extraction"
    )
    parser.add_argument(
        "--train", "-t",
        type=str,
        default="data/processed/train/train.conllu",
        help="Training CoNLL-U file",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Test CoNLL-U file for evaluation",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["logistic", "random_forest"],
        default="logistic",
        help="ML model type",
    )
    parser.add_argument(
        "--n-patterns",
        type=int,
        default=30,
        help="Number of patterns per relation",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs/ml_rules",
        help="Output directory",
    )

    args = parser.parse_args()

    run_ml_pipeline(
        train_path=args.train,
        test_path=args.test,
        model_type=args.model,
        n_patterns=args.n_patterns,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
