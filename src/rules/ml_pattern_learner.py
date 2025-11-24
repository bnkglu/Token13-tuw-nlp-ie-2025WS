"""ML-based pattern discovery for rule generation.

This module trains classifiers on SDP features and extracts the most
discriminative patterns based on feature importance/coefficients.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from .feature_extractor import (
    FeatureMatrix,
    SDPFeatureExtractor,
    extract_features_from_conllu,
)
from .sdp_extractor import SDPExtractor, SDPSignature


@dataclass
class ImportantFeature:
    """A feature identified as important by the ML model.

    Attributes
    ----------
    feature_name : str
        Name of the feature (e.g., "SIG:NOUN --(prep)--> of:ADP")
    relation : str
        Relation this feature is important for
    importance : float
        Importance score (coefficient or feature importance)
    feature_type : str
        Type of feature (SIG, POS, DEP, TRIG, etc.)
    frequency : int
        How often this feature appears in training data
    """

    feature_name: str
    relation: str
    importance: float
    feature_type: str
    frequency: int = 0

    def is_signature_feature(self) -> bool:
        """Check if this is a full signature feature."""
        return self.feature_name.startswith("SIG:")

    def get_signature_string(self) -> Optional[str]:
        """Extract the signature string if this is a signature feature."""
        if self.is_signature_feature():
            return self.feature_name[4:]  # Remove "SIG:" prefix
        return None


@dataclass
class MLPatternResult:
    """Results from ML-based pattern learning.

    Attributes
    ----------
    model : Any
        Trained classifier
    feature_matrix : FeatureMatrix
        Feature matrix used for training
    important_features : Dict[str, List[ImportantFeature]]
        Important features per relation
    cv_scores : Dict[str, float]
        Cross-validation scores
    classification_report : str
        Full classification report
    """

    model: Any
    feature_matrix: FeatureMatrix
    important_features: Dict[str, List[ImportantFeature]]
    cv_scores: Dict[str, float] = field(default_factory=dict)
    classification_report: str = ""


class MLPatternLearner:
    """Train classifiers to discover discriminative SDP patterns.

    Parameters
    ----------
    model_type : str
        Type of classifier ("logistic", "random_forest")
    n_top_features : int
        Number of top features to extract per class
    min_feature_freq : int
        Minimum feature frequency for extraction
    """

    SUPPORTED_MODELS = {"logistic", "random_forest"}

    def __init__(
        self,
        model_type: str = "logistic",
        n_top_features: int = 50,
        min_feature_freq: int = 2,
    ):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model type must be one of {self.SUPPORTED_MODELS}")

        self.model_type = model_type
        self.n_top_features = n_top_features
        self.min_feature_freq = min_feature_freq

        self._model = None
        self._feature_extractor = None
        self._feature_matrix = None

    def _create_model(self) -> Union[LogisticRegression, RandomForestClassifier]:
        """Create the classifier model."""
        if self.model_type == "logistic":
            return LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )
        else:  # random_forest
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )

    def train(
        self,
        train_conllu_path: str,
        cv_folds: int = 5,
    ) -> MLPatternResult:
        """Train classifier on CoNLL-U data.

        Parameters
        ----------
        train_conllu_path : str
            Path to training CoNLL-U file
        cv_folds : int
            Number of cross-validation folds

        Returns
        -------
        MLPatternResult
            Training results with important features
        """
        # Extract features
        print("Extracting features...")
        self._feature_extractor = SDPFeatureExtractor(
            min_feature_freq=self.min_feature_freq
        )
        self._feature_matrix, _ = extract_features_from_conllu(
            train_conllu_path,
            extractor=self._feature_extractor,
            fit=True,
        )

        X = self._feature_matrix.X
        y = self._feature_matrix.y

        print(f"Training set: {X.shape[0]} examples, {X.shape[1]} features")

        # Create and train model
        print(f"Training {self.model_type} classifier...")
        self._model = self._create_model()

        # Cross-validation
        cv_scores = cross_val_score(
            self._model, X, y,
            cv=cv_folds,
            scoring="f1_macro",
        )

        cv_results = {
            "mean_f1": float(np.mean(cv_scores)),
            "std_f1": float(np.std(cv_scores)),
            "scores": cv_scores.tolist(),
        }
        print(f"CV F1 (macro): {cv_results['mean_f1']:.4f} +/- {cv_results['std_f1']:.4f}")

        # Train on full data
        self._model.fit(X, y)

        # Get predictions for classification report
        y_pred = self._model.predict(X)
        class_names = [
            self._feature_matrix.idx_to_label[i]
            for i in range(len(self._feature_matrix.idx_to_label))
        ]
        report = classification_report(y, y_pred, target_names=class_names)

        # Extract important features
        print("Extracting important features...")
        important_features = self._extract_important_features()

        return MLPatternResult(
            model=self._model,
            feature_matrix=self._feature_matrix,
            important_features=important_features,
            cv_scores=cv_results,
            classification_report=report,
        )

    def _extract_important_features(self) -> Dict[str, List[ImportantFeature]]:
        """Extract important features from trained model."""
        important = {}

        if self.model_type == "logistic":
            # For logistic regression, use coefficients
            coefs = self._model.coef_  # Shape: (n_classes, n_features)

            for class_idx in range(coefs.shape[0]):
                relation = self._feature_matrix.idx_to_label[class_idx]
                class_coefs = coefs[class_idx]

                # Get top features by coefficient magnitude
                top_indices = np.argsort(class_coefs)[::-1][:self.n_top_features]

                features = []
                for idx in top_indices:
                    feat_name = self._feature_extractor.get_feature_name(idx)
                    if feat_name is None:
                        continue

                    feat_type = feat_name.split(":")[0]
                    importance = float(class_coefs[idx])

                    # Skip negative coefficients (they predict against this class)
                    if importance <= 0:
                        continue

                    freq = self._feature_extractor._feature_counts.get(feat_name, 0)

                    features.append(ImportantFeature(
                        feature_name=feat_name,
                        relation=relation,
                        importance=importance,
                        feature_type=feat_type,
                        frequency=freq,
                    ))

                important[relation] = features

        else:  # random_forest
            # For random forest, use feature importances (global)
            importances = self._model.feature_importances_

            # Get top features by importance
            top_indices = np.argsort(importances)[::-1][:self.n_top_features * 5]

            # Assign to classes based on which class they appear most in
            # This is an approximation for multi-class feature importance
            for idx in top_indices:
                feat_name = self._feature_extractor.get_feature_name(idx)
                if feat_name is None:
                    continue

                feat_type = feat_name.split(":")[0]
                importance = float(importances[idx])
                freq = self._feature_extractor._feature_counts.get(feat_name, 0)

                # Find which class this feature is most associated with
                # by looking at the feature's presence per class
                X = self._feature_matrix.X
                y = self._feature_matrix.y

                feature_col = X[:, idx].toarray().flatten()
                class_presence = {}

                for class_idx in range(len(self._feature_matrix.idx_to_label)):
                    class_mask = (y == class_idx)
                    presence = feature_col[class_mask].sum() / class_mask.sum()
                    class_presence[class_idx] = presence

                best_class = max(class_presence, key=class_presence.get)
                relation = self._feature_matrix.idx_to_label[best_class]

                if relation not in important:
                    important[relation] = []

                if len(important[relation]) < self.n_top_features:
                    important[relation].append(ImportantFeature(
                        feature_name=feat_name,
                        relation=relation,
                        importance=importance,
                        feature_type=feat_type,
                        frequency=freq,
                    ))

        return important

    def get_signature_patterns(
        self, result: MLPatternResult
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Extract only full signature patterns from results.

        Parameters
        ----------
        result : MLPatternResult
            Training results

        Returns
        -------
        Dict[str, List[Tuple[str, float]]]
            Relation -> list of (signature_string, importance)
        """
        patterns = {}

        for relation, features in result.important_features.items():
            sig_patterns = []

            for feat in features:
                if feat.is_signature_feature():
                    sig_str = feat.get_signature_string()
                    if sig_str:
                        sig_patterns.append((sig_str, feat.importance))

            patterns[relation] = sig_patterns

        return patterns

    def print_results(self, result: MLPatternResult) -> None:
        """Print training results summary.

        Parameters
        ----------
        result : MLPatternResult
            Training results
        """
        print("\n" + "=" * 70)
        print("ML PATTERN LEARNING RESULTS")
        print("=" * 70)

        print(f"\nModel: {self.model_type}")
        print(f"CV F1 (macro): {result.cv_scores['mean_f1']:.4f} +/- {result.cv_scores['std_f1']:.4f}")

        print("\n" + "-" * 70)
        print("CLASSIFICATION REPORT")
        print("-" * 70)
        print(result.classification_report)

        print("\n" + "-" * 70)
        print("TOP DISCRIMINATIVE PATTERNS PER RELATION")
        print("-" * 70)

        for relation, features in result.important_features.items():
            print(f"\n{relation}:")

            # Show signature features first
            sig_feats = [f for f in features if f.is_signature_feature()]
            if sig_feats:
                print("  Signature patterns:")
                for f in sig_feats[:5]:
                    sig_str = f.get_signature_string()
                    print(f"    - {sig_str}")
                    print(f"      Importance: {f.importance:.4f}, Freq: {f.frequency}")

            # Show other features
            other_feats = [f for f in features if not f.is_signature_feature()]
            if other_feats:
                print("  Other features:")
                for f in other_feats[:5]:
                    print(f"    - {f.feature_name} (imp: {f.importance:.4f})")


def train_and_extract_patterns(
    train_conllu_path: str,
    model_type: str = "logistic",
    n_top_features: int = 50,
    output_path: Optional[str] = None,
) -> MLPatternResult:
    """Train classifier and extract discriminative patterns.

    Parameters
    ----------
    train_conllu_path : str
        Path to training CoNLL-U file
    model_type : str
        Classifier type ("logistic" or "random_forest")
    n_top_features : int
        Number of top features per class
    output_path : Optional[str]
        Path to save results

    Returns
    -------
    MLPatternResult
        Training results
    """
    learner = MLPatternLearner(
        model_type=model_type,
        n_top_features=n_top_features,
    )

    result = learner.train(train_conllu_path)
    learner.print_results(result)

    if output_path:
        import json

        export_data = {
            "model_type": model_type,
            "cv_scores": result.cv_scores,
            "patterns": {},
        }

        for relation, features in result.important_features.items():
            export_data["patterns"][relation] = [
                {
                    "feature": f.feature_name,
                    "importance": f.importance,
                    "type": f.feature_type,
                    "frequency": f.frequency,
                }
                for f in features
            ]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"\nExported results to: {output_path}")

    return result


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train ML models to discover discriminative SDP patterns"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/processed/train/train.conllu",
        help="Training CoNLL-U file",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["logistic", "random_forest"],
        default="logistic",
        help="Model type",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Top N features per class",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file",
    )

    args = parser.parse_args()

    train_and_extract_patterns(
        train_conllu_path=args.input,
        model_type=args.model,
        n_top_features=args.top_n,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
