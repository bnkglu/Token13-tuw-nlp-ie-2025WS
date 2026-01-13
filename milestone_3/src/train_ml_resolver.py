"""Train an ML resolver for relation extraction.

This module provides two approaches:

1. OtherResolver: ML classifier to distinguish Other vs non-Other relations.
   Works as a post-processor to the entity-rooted pattern matcher.

2. DependencyMatcher resolver (legacy): Uses pattern matching signals as
   features to resolve conflicts.

Usage
-----
  # Train OtherResolver
  python milestone_3/src/train_ml_resolver.py --mode other-resolver

  # Legacy DependencyMatcher resolver
  python milestone_3/src/train_ml_resolver.py --mode dep-matcher --limit 2000
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import spacy
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split, cross_val_score

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils import preprocess_data
from execution_engine import (
    compile_dependency_matcher,
    sort_matches_by_priority,
    verify_anchoring_relaxed,
    parse_match_indices,
)

# Import semantic modules for feature extraction
try:
    from framenet_scorer import score_frame_compatibility, RELATION_FRAMES
    FRAMENET_AVAILABLE = True
except ImportError:
    FRAMENET_AVAILABLE = False
    score_frame_compatibility = None
    RELATION_FRAMES = {}

try:
    from wordnet_augmentor import relation_specific_match, RELATION_HYPERNYMS
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False
    relation_specific_match = None
    RELATION_HYPERNYMS = {}

# All relation types for semantic feature extraction
ALL_RELATIONS = [
    "Cause-Effect", "Component-Whole", "Entity-Origin",
    "Instrument-Agency", "Member-Collection", "Content-Container",
    "Entity-Destination", "Product-Producer", "Message-Topic"
]


# =============================================================================
# OtherResolver: ML classifier for Other vs non-Other
# =============================================================================

class OtherResolver:
    """
    ML classifier to detect when prediction should be Other.

    Uses text features (between entities, entity tokens, dependency path)
    to learn patterns that distinguish Other from non-Other relations.

    This is designed to work with the entity-rooted pattern matcher:
    1. Entity-rooted matcher predicts a relation
    2. OtherResolver decides if the prediction should be overridden to "Other"
    """

    def __init__(self, classifier_type: str = "rf"):
        """
        Initialize the resolver.

        Args:
            classifier_type: "rf" for RandomForest, "gb" for GradientBoosting
        """
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95
        )

        if classifier_type == "gb":
            self.classifier = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        else:
            self.classifier = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            )

        self.is_trained = False
        self.threshold = 0.5

    def _get_dep_path(self, t1, t2, max_depth: int = 5) -> List[str]:
        """Get dependency labels on path between two tokens."""
        deps = []
        current = t1
        for _ in range(max_depth):
            if current.head == current:
                break
            deps.append(f"t1_{current.dep_}")
            current = current.head

        current = t2
        for _ in range(max_depth):
            if current.head == current:
                break
            deps.append(f"t2_{current.dep_}")
            current = current.head

        return deps

    def extract_features(self, sample: Dict[str, Any]) -> str:
        """
        Extract text features from a sample.

        Features include:
        - Text between entities
        - Entity tokens
        - Dependency path labels
        - POS tags
        - Entity head lemmas
        """
        doc = sample['doc']
        e1_span = sample['e1_span']
        e2_span = sample['e2_span']

        features = []

        # 1. Text between entities
        e1_end = e1_span.end
        e2_start = e2_span.start
        if e1_end < e2_start:
            between_tokens = doc[e1_end:e2_start]
            between_text = " ".join([t.lemma_.lower() for t in between_tokens])
            features.append(between_text)
        elif e2_span.end < e1_span.start:
            between_tokens = doc[e2_span.end:e1_span.start]
            between_text = " ".join([t.lemma_.lower() for t in between_tokens])
            features.append(between_text)

        # 2. Entity tokens (lemmatized)
        e1_lemmas = " ".join([t.lemma_.lower() for t in e1_span])
        e2_lemmas = " ".join([t.lemma_.lower() for t in e2_span])
        features.append(f"E1_{e1_lemmas}")
        features.append(f"E2_{e2_lemmas}")

        # 3. Entity root POS tags
        e1_root = e1_span.root
        e2_root = e2_span.root
        features.append(f"E1POS_{e1_root.pos_}")
        features.append(f"E2POS_{e2_root.pos_}")

        # 4. Dependency path features
        path_deps = self._get_dep_path(e1_root, e2_root)
        features.extend(path_deps)

        # 5. Entity head lemmas
        if e1_root.head != e1_root:
            features.append(f"E1HEAD_{e1_root.head.lemma_.lower()}")
        if e2_root.head != e2_root:
            features.append(f"E2HEAD_{e2_root.head.lemma_.lower()}")

        # 6. POS sequence between entities (if not too long)
        start_idx = min(e1_span.start, e2_span.start)
        end_idx = max(e1_span.end, e2_span.end)
        if end_idx - start_idx <= 15:
            pos_seq = "_".join([t.pos_ for t in doc[start_idx:end_idx]])
            features.append(f"POSSEQ_{pos_seq}")

        # 7. Distance category
        distance = abs(e1_root.i - e2_root.i)
        if distance <= 3:
            features.append("DIST_close")
        elif distance <= 7:
            features.append("DIST_medium")
        else:
            features.append("DIST_far")

        # 8. Semantic features from FrameNet and WordNet
        # Find anchor token (typically the verb connecting entities)
        anchor = self._find_anchor(e1_root, e2_root)
        if anchor:
            anchor_lemma = anchor.lemma_.lower()
            anchor_pos = anchor.pos_

            # 8a. FrameNet compatibility scores for each relation
            if FRAMENET_AVAILABLE and score_frame_compatibility is not None:
                best_frame_score = 0.0
                best_frame_rel = None
                for rel in ALL_RELATIONS:
                    score = score_frame_compatibility(rel, anchor_lemma, e1_root.dep_, e2_root.dep_)
                    if score > best_frame_score:
                        best_frame_score = score
                        best_frame_rel = rel

                # Binned frame score (0-10 scale)
                features.append(f"FRAME_SCORE_{int(best_frame_score * 10)}")
                if best_frame_rel and best_frame_score >= 0.6:
                    features.append(f"FRAME_BEST_{best_frame_rel}")

            # 8b. WordNet relation-specific matches
            if WORDNET_AVAILABLE and relation_specific_match is not None:
                pos_wn = 'v' if anchor_pos == 'VERB' else 'n'
                for rel in ALL_RELATIONS:
                    matched, conf, group = relation_specific_match(anchor_lemma, rel, pos_wn)
                    if matched:
                        features.append(f"WN_{rel}_{group}")
                        features.append(f"WN_CONF_{int(conf * 10)}")

        return " ".join(features)

    def _find_anchor(self, e1_root, e2_root):
        """
        Find the anchor token (typically LCA verb) connecting two entities.

        Returns the lowest common ancestor that is a VERB, or None.
        """
        # Get ancestors of e1
        e1_ancestors = []
        current = e1_root
        while current.head != current:
            current = current.head
            e1_ancestors.append(current)

        # Find LCA
        current = e2_root
        while current.head != current:
            current = current.head
            if current in e1_ancestors:
                # Found LCA - return if it's a verb
                if current.pos_ == 'VERB':
                    return current
                # If LCA is not a verb, check its head
                if current.head != current and current.head.pos_ == 'VERB':
                    return current.head
                return current

        return None

    def train(
        self,
        samples: List[Dict[str, Any]],
        true_labels: List[str],
        rule_predictions: List[str],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train on samples where rules made Other-related errors.

        Args:
            samples: List of processed samples
            true_labels: Ground truth labels
            rule_predictions: Labels predicted by rule-based system
            verbose: Print training statistics

        Returns:
            Training statistics dict
        """
        X_texts = []
        y = []

        # Build training data from Other-related samples
        for sample, true, pred in zip(samples, true_labels, rule_predictions):
            if true == "Other" or pred == "Other":
                X_texts.append(self.extract_features(sample))
                y.append(1 if true == "Other" else 0)

        if len(X_texts) < 10:
            if verbose:
                print(f"Warning: Only {len(X_texts)} Other-related samples, skipping training")
            return {"status": "skipped", "reason": "insufficient_data"}

        X = self.vectorizer.fit_transform(X_texts)
        y = np.array(y)

        if verbose:
            print(f"\nTraining OtherResolver on {len(X_texts)} samples...")
            print(f"  Class distribution: Other={sum(y)}, non-Other={len(y)-sum(y)}")

        # Cross-validation
        n_splits = min(5, len(y) // 10 + 1)
        if n_splits >= 2:
            cv_scores = cross_val_score(self.classifier, X, y, cv=n_splits)
            if verbose:
                print(f"  Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        else:
            cv_scores = np.array([0.0])

        # Train final model
        self.classifier.fit(X, y)
        self.is_trained = True

        # Find optimal threshold
        train_probs = self.classifier.predict_proba(X)[:, 1]
        best_f1 = 0
        best_threshold = 0.5
        for thresh in np.arange(0.3, 0.8, 0.05):
            preds = (train_probs >= thresh).astype(int)
            tp = np.sum((preds == 1) & (y == 1))
            fp = np.sum((preds == 1) & (y == 0))
            fn = np.sum((preds == 0) & (y == 1))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh

        self.threshold = best_threshold

        if verbose:
            print(f"  Optimal threshold: {self.threshold:.2f} (F1={best_f1:.3f})")

        return {
            "status": "trained",
            "n_samples": len(X_texts),
            "n_other": int(sum(y)),
            "n_non_other": int(len(y) - sum(y)),
            "cv_accuracy": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "threshold": float(self.threshold),
            "best_f1": float(best_f1)
        }

    def predict_is_other(self, sample: Dict[str, Any]) -> bool:
        """Predict if sample should be classified as Other."""
        if not self.is_trained:
            return False

        text = self.extract_features(sample)
        X = self.vectorizer.transform([text])
        proba = self.classifier.predict_proba(X)[0]
        return proba[1] > self.threshold

    def predict_proba_other(self, sample: Dict[str, Any]) -> float:
        """Get probability that sample should be Other."""
        if not self.is_trained:
            return 0.5

        text = self.extract_features(sample)
        X = self.vectorizer.transform([text])
        proba = self.classifier.predict_proba(X)[0]
        return float(proba[1])

    def save(self, path: str):
        """Save trained model to disk."""
        data = {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "is_trained": self.is_trained,
            "threshold": self.threshold
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """Load trained model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.classifier = data["classifier"]
        self.is_trained = data["is_trained"]
        self.threshold = data.get("threshold", 0.5)


def train_other_resolver(args) -> None:
    """Train OtherResolver using entity-rooted pattern predictions."""
    from entity_rooted_matcher import apply_patterns_entity_rooted

    repo_root = Path(__file__).parent.parent.parent
    train_path = repo_root / "data" / "processed" / "train" / "train.json"
    patterns_path = Path(__file__).parent.parent / "data" / "patterns_augmented.json"
    output_path = Path(__file__).parent.parent / "data" / "other_resolver.pkl"

    print("=" * 80)
    print("TRAIN OTHER RESOLVER")
    print("=" * 80)

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_lg")

    print(f"Loading training data: {train_path}")
    train_data = json.load(open(train_path, "r"))
    if args.limit and args.limit > 0:
        train_data = train_data[: args.limit]
        print(f"Limiting to {len(train_data)} samples")

    print("Preprocessing...")
    processed = preprocess_data(train_data, nlp)
    true_labels = [get_directed_label(x) for x in train_data]

    print(f"Loading patterns: {patterns_path}")
    patterns = json.load(open(patterns_path, "r"))

    print("Applying entity-rooted patterns...")
    preds, dirs, expls, stats = apply_patterns_entity_rooted(processed, patterns, nlp)

    print("\nTraining OtherResolver...")
    resolver = OtherResolver()
    train_stats = resolver.train(processed, true_labels, preds)

    if resolver.is_trained:
        resolver.save(str(output_path))
        print(f"\nSaved resolver to {output_path}")

        # Evaluate improvement with CONSERVATIVE approach
        # Only override when ML is very confident
        print("\n" + "=" * 80)
        print("EVALUATING RESOLVER IMPACT (CONSERVATIVE)")
        print("=" * 80)

        # Apply resolver CONSERVATIVELY - only override with high confidence
        improved_preds = []
        overrides_to_other = 0
        high_conf_threshold = 0.98  # Extremely conservative - only override with near-certainty

        for sample, pred in zip(processed, preds):
            other_prob = resolver.predict_proba_other(sample)

            # Only override non-Other to Other if VERY confident
            if pred != "Other" and other_prob > high_conf_threshold:
                improved_preds.append("Other")
                overrides_to_other += 1
            else:
                # Keep original prediction
                improved_preds.append(pred)

        # Calculate metrics
        orig_correct = sum(1 for t, p in zip(true_labels, preds) if t == p)
        new_correct = sum(1 for t, p in zip(true_labels, improved_preds) if t == p)

        print(f"\nOriginal accuracy: {orig_correct}/{len(true_labels)} ({orig_correct/len(true_labels):.1%})")
        print(f"With resolver:     {new_correct}/{len(true_labels)} ({new_correct/len(true_labels):.1%})")
        print(f"Improvement:       {new_correct - orig_correct:+d} samples")
        print(f"Overrides to Other: {overrides_to_other} (threshold={high_conf_threshold})")


# =============================================================================
# Legacy DependencyMatcher resolver (existing code)
# =============================================================================


def get_directed_label(item: Dict[str, Any]) -> str:
    rel_type = item["relation"]["type"]
    direction = item["relation"].get("direction", "")
    if rel_type == "Other":
        return "Other"
    direction = direction.replace("(", "").replace(")", "")
    return f"{rel_type}({direction})"


@dataclass(frozen=True)
class MatchSignal:
    pattern_type: str
    base_relation: str
    relation: str
    direction: Optional[str]
    precision: float
    support: int
    length: int
    e1_inside: bool
    e2_inside: bool


def _span_end_inclusive(span) -> int:
    return span.end - 1


def _inside_span(idx: int, span) -> bool:
    return span.start <= idx <= _span_end_inclusive(span)


def collect_anchored_match_signals(
    sample: Dict[str, Any],
    dep_matcher,
    pattern_lookup: Dict[str, Dict[str, Any]],
    nlp,
    *,
    top_k: int = 50,
    max_distance: int = 2,
) -> List[MatchSignal]:
    doc = sample["doc"]
    e1_span = sample["e1_span"]
    e2_span = sample["e2_span"]

    matches = dep_matcher(doc)
    if not matches:
        return []

    sorted_matches = sort_matches_by_priority(matches, pattern_lookup, nlp)

    signals: List[MatchSignal] = []
    for _priority, match_id, token_indices in sorted_matches:
        pattern = pattern_lookup.get(match_id)
        if not pattern:
            continue

        if not verify_anchoring_relaxed(token_indices, pattern, e1_span, e2_span, max_distance=max_distance):
            continue

        match_indices = parse_match_indices(token_indices, pattern)
        e1_idx = match_indices.get("e1")
        e2_idx = match_indices.get("e2")
        if e1_idx is None or e2_idx is None:
            continue

        signals.append(
            MatchSignal(
                pattern_type=pattern.get("pattern_type", ""),
                base_relation=pattern.get("base_relation", pattern.get("relation", "")),
                relation=pattern.get("relation", ""),
                direction=pattern.get("direction"),
                precision=float(pattern.get("precision", 0.0)),
                support=int(pattern.get("support", 0)),
                length=int(pattern.get("length", 0)),
                e1_inside=_inside_span(e1_idx, e1_span),
                e2_inside=_inside_span(e2_idx, e2_span),
            )
        )

        if len(signals) >= top_k:
            break

    return signals


def featurize_sample(
    sample: Dict[str, Any],
    signals: List[MatchSignal],
    *,
    top_pattern_ids: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """Turn a set of match signals into an explainable numeric feature dict."""

    feats: Dict[str, float] = {}

    # Coverage / counts
    feats["has_match"] = 1.0 if signals else 0.0
    feats["num_matches"] = float(len(signals))

    # Basic sentence/entity geometry
    e1 = sample["e1_span"]
    e2 = sample["e2_span"]
    feats["entity_token_distance"] = float(abs(e1.root.i - e2.root.i))
    feats["entity_span_distance"] = float(abs(e1.start - e2.start))

    # Aggregate by pattern type
    by_type: Dict[str, List[MatchSignal]] = defaultdict(list)
    for s in signals:
        by_type[s.pattern_type].append(s)

    for t, items in by_type.items():
        feats[f"count_type_{t}"] = float(len(items))
        feats[f"max_precision_type_{t}"] = float(max(x.precision for x in items))
        feats[f"max_support_type_{t}"] = float(max(x.support for x in items))
        feats[f"max_length_type_{t}"] = float(max(x.length for x in items))

        # Anchoring quality
        feats[f"frac_inside_both_type_{t}"] = float(sum(1 for x in items if x.e1_inside and x.e2_inside)) / float(len(items))

    # Global best signals
    if signals:
        feats["max_precision"] = float(max(s.precision for s in signals))
        feats["max_support"] = float(max(s.support for s in signals))
        feats["max_length"] = float(max(s.length for s in signals))
        feats["frac_inside_both"] = float(sum(1 for s in signals if s.e1_inside and s.e2_inside)) / float(len(signals))

        # Directional cues
        dir_counts = Counter(s.direction or "" for s in signals)
        feats["count_dir_e1e2"] = float(dir_counts.get("e1,e2", 0))
        feats["count_dir_e2e1"] = float(dir_counts.get("e2,e1", 0))

        # Other dominance
        other_count = sum(1 for s in signals if s.base_relation == "Other" or s.relation == "Other")
        feats["frac_other_signals"] = float(other_count) / float(len(signals))

    # Optional: top-N pattern_id one-hot features (disabled by default)
    if top_pattern_ids:
        for s in signals:
            pid = s.relation + "|" + s.pattern_type  # stable-ish id without hashing
            idx = top_pattern_ids.get(pid)
            if idx is not None:
                feats[f"pid_{idx}"] = 1.0

    return feats


def dicts_to_matrix(feature_dicts: List[Dict[str, float]]) -> Tuple[np.ndarray, List[str]]:
    """Convert list of sparse dicts to dense matrix with stable column order."""

    all_keys = sorted({k for d in feature_dicts for k in d.keys()})
    key_to_col = {k: i for i, k in enumerate(all_keys)}

    X = np.zeros((len(feature_dicts), len(all_keys)), dtype=np.float32)
    for r, d in enumerate(feature_dicts):
        for k, v in d.items():
            X[r, key_to_col[k]] = float(v)

    return X, all_keys


def main_dep_matcher(args) -> None:
    """Legacy DependencyMatcher resolver training."""
    repo_root = Path(__file__).parent.parent.parent
    train_path = repo_root / "data" / "processed" / "train" / "train.json"
    patterns_path = Path(__file__).parent.parent / "data" / "patterns_augmented.json"

    print("=" * 80)
    print("TRAIN ML RESOLVER")
    print("=" * 80)

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_lg")

    print(f"Loading training data: {train_path}")
    train_data = json.load(open(train_path, "r"))
    if args.limit and args.limit > 0:
        train_data = train_data[: args.limit]
        print(f"Limiting to {len(train_data)} samples")

    print("Preprocessing...")
    processed = preprocess_data(train_data, nlp)
    y = np.array([get_directed_label(x) for x in train_data])

    print(f"Loading patterns: {patterns_path}")
    patterns = json.load(open(patterns_path, "r"))

    print("Compiling DependencyMatcher...")
    dep_matcher, pattern_lookup = compile_dependency_matcher(patterns, nlp)

    # Optional: build top-N pattern-id-like features from training frequency
    top_pid_map: Optional[Dict[str, int]] = None
    if args.top_pattern_features and args.top_pattern_features > 0:
        pid_counter = Counter()
        for p in patterns:
            pid_counter[p.get("relation", "") + "|" + p.get("pattern_type", "")] += 1
        most_common = [k for k, _v in pid_counter.most_common(args.top_pattern_features)]
        top_pid_map = {k: i for i, k in enumerate(most_common)}
        print(f"Enabled {len(top_pid_map)} top pattern-id-like features")

    print("Collecting match signals + building features...")
    feat_dicts: List[Dict[str, float]] = []
    coverage = 0

    for s in processed:
        signals = collect_anchored_match_signals(
            s,
            dep_matcher,
            pattern_lookup,
            nlp,
            top_k=args.top_k,
            max_distance=args.max_distance,
        )
        if signals:
            coverage += 1
        feat_dicts.append(featurize_sample(s, signals, top_pattern_ids=top_pid_map))

    print(f"Candidate coverage (>=1 anchored match): {coverage}/{len(processed)} ({coverage/len(processed):.1%})")

    X, cols = dicts_to_matrix(feat_dicts)
    print(f"Feature matrix: X={X.shape} (features={len(cols)})")

    X_train, X_dev, y_train, y_dev = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    # Model A: Logistic Regression (strong baseline, explainable)
    print("\nTraining LogisticRegression...")
    lr = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        class_weight="balanced",
        multi_class="auto",
    )
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_dev)

    # Model B: Random Forest (non-linear, still inspectable)
    print("Training RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_dev)

    def report(name: str, pred: np.ndarray) -> None:
        acc = accuracy_score(y_dev, pred)
        f1m = f1_score(y_dev, pred, average="macro")
        print(f"\n{name}:")
        print(f"  Dev accuracy: {acc:.3f}")
        print(f"  Dev macro-F1: {f1m:.3f}")
        print(classification_report(y_dev, pred, digits=3))

    report("LogReg", pred_lr)
    report("RandomForest", pred_rf)

    # Explainability quick peek
    print("\nTop features (LogReg, by max |weight| over classes):")
    if hasattr(lr, "coef_"):
        coef = np.abs(lr.coef_)
        max_abs = coef.max(axis=0)
        top_idx = np.argsort(-max_abs)[:20]
        for i in top_idx:
            print(f"  {cols[i]}: {max_abs[i]:.4f}")

    print("\nTop features (RF, impurity importance):")
    imp = rf.feature_importances_
    top_idx = np.argsort(-imp)[:20]
    for i in top_idx:
        print(f"  {cols[i]}: {imp[i]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="other-resolver",
                        choices=["other-resolver", "dep-matcher"],
                        help="Training mode: 'other-resolver' for OtherResolver, 'dep-matcher' for legacy.")
    parser.add_argument("--limit", type=int, default=0, help="If set, only use first N samples (0=all).")
    parser.add_argument("--top-k", type=int, default=50, help="Max anchored matches per sample (dep-matcher mode).")
    parser.add_argument("--max-distance", type=int, default=2, help="Anchoring tolerance (dep-matcher mode).")
    parser.add_argument("--test-size", type=float, default=0.1, help="Dev split size (dep-matcher mode).")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top-pattern-features", type=int, default=0, help="Enable top-N pattern-id features.")
    args = parser.parse_args()

    if args.mode == "other-resolver":
        train_other_resolver(args)
    else:
        main_dep_matcher(args)
