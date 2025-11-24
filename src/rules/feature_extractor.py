"""Feature extraction for ML-based pattern learning.

This module converts SDP signatures into feature vectors suitable for
training classifiers to discover discriminative patterns.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from .sdp_extractor import SDPExample, SDPExtractor, SDPSignature


@dataclass
class FeatureVocab:
    """Vocabulary for mapping features to indices.

    Attributes
    ----------
    feature_to_idx : Dict[str, int]
        Feature string to index mapping
    idx_to_feature : Dict[int, str]
        Index to feature string mapping
    """

    feature_to_idx: Dict[str, int] = field(default_factory=dict)
    idx_to_feature: Dict[int, str] = field(default_factory=dict)

    def add(self, feature: str) -> int:
        """Add feature to vocabulary and return its index."""
        if feature not in self.feature_to_idx:
            idx = len(self.feature_to_idx)
            self.feature_to_idx[feature] = idx
            self.idx_to_feature[idx] = feature
        return self.feature_to_idx[feature]

    def get(self, feature: str) -> Optional[int]:
        """Get index for feature, or None if not found."""
        return self.feature_to_idx.get(feature)

    def __len__(self) -> int:
        return len(self.feature_to_idx)


@dataclass
class FeatureMatrix:
    """Feature matrix with vocabulary and label mapping.

    Attributes
    ----------
    X : csr_matrix
        Sparse feature matrix (n_samples x n_features)
    y : np.ndarray
        Label array (n_samples,)
    vocab : FeatureVocab
        Feature vocabulary
    label_to_idx : Dict[str, int]
        Label to index mapping
    idx_to_label : Dict[int, str]
        Index to label mapping
    example_ids : List[int]
        Original sentence IDs
    """

    X: csr_matrix
    y: np.ndarray
    vocab: FeatureVocab
    label_to_idx: Dict[str, int]
    idx_to_label: Dict[int, str]
    example_ids: List[int]


class SDPFeatureExtractor:
    """Extract features from SDP signatures for ML training.

    Features extracted:
    - Full signature string (hash of entire path)
    - POS pattern (sequence of POS tags)
    - Dependency pattern (sequence of dep relations)
    - Trigger word features (specific prepositions, verbs)
    - Path length feature
    - Bigram features (pairs of consecutive elements)
    - Entity position features

    Parameters
    ----------
    use_full_signature : bool
        Include full signature as a feature
    use_pos_pattern : bool
        Include POS pattern as a feature
    use_dep_pattern : bool
        Include dependency pattern as a feature
    use_trigger_words : bool
        Include individual trigger words as features
    use_bigrams : bool
        Include bigram features
    use_path_length : bool
        Include path length as a feature
    min_feature_freq : int
        Minimum frequency for a feature to be included
    """

    def __init__(
        self,
        use_full_signature: bool = True,
        use_pos_pattern: bool = True,
        use_dep_pattern: bool = True,
        use_trigger_words: bool = True,
        use_bigrams: bool = True,
        use_path_length: bool = True,
        min_feature_freq: int = 2,
    ):
        self.use_full_signature = use_full_signature
        self.use_pos_pattern = use_pos_pattern
        self.use_dep_pattern = use_dep_pattern
        self.use_trigger_words = use_trigger_words
        self.use_bigrams = use_bigrams
        self.use_path_length = use_path_length
        self.min_feature_freq = min_feature_freq

        self.vocab = FeatureVocab()
        self._feature_counts: Dict[str, int] = defaultdict(int)
        self._fitted = False

    def extract_features_from_signature(
        self, sig: SDPSignature
    ) -> Dict[str, float]:
        """Extract feature dictionary from a signature.

        Parameters
        ----------
        sig : SDPSignature
            The signature to extract features from

        Returns
        -------
        Dict[str, float]
            Feature name to value mapping
        """
        features = {}

        # Full signature feature
        if self.use_full_signature:
            sig_str = f"SIG:{sig.to_string()}"
            features[sig_str] = 1.0

        # POS pattern feature
        if self.use_pos_pattern:
            pos_str = f"POS:{'-'.join(sig.pos_pattern)}"
            features[pos_str] = 1.0

        # Dependency pattern feature
        if self.use_dep_pattern:
            dep_str = f"DEP:{'-'.join(sig.dep_pattern)}"
            features[dep_str] = 1.0

        # Individual trigger words
        if self.use_trigger_words:
            for i, (pos, trigger) in enumerate(
                zip(sig.pos_pattern, sig.trigger_words)
            ):
                if trigger:
                    features[f"TRIG:{trigger}"] = 1.0
                    features[f"TRIG_POS:{trigger}:{pos}"] = 1.0

        # Bigram features
        if self.use_bigrams:
            # POS bigrams
            for i in range(len(sig.pos_pattern) - 1):
                bigram = f"POS_BI:{sig.pos_pattern[i]}-{sig.pos_pattern[i+1]}"
                features[bigram] = 1.0

            # DEP bigrams
            for i in range(len(sig.dep_pattern) - 1):
                bigram = f"DEP_BI:{sig.dep_pattern[i]}-{sig.dep_pattern[i+1]}"
                features[bigram] = 1.0

            # Mixed POS-DEP bigrams
            for i in range(len(sig.dep_pattern)):
                mixed = f"POS_DEP:{sig.pos_pattern[i]}-{sig.dep_pattern[i]}"
                features[mixed] = 1.0

        # Path length feature
        if self.use_path_length:
            features[f"LEN:{sig.path_length}"] = 1.0

        return features

    def fit(self, examples: List[SDPExample]) -> "SDPFeatureExtractor":
        """Fit the feature extractor on training examples.

        This builds the vocabulary and filters rare features.

        Parameters
        ----------
        examples : List[SDPExample]
            Training examples

        Returns
        -------
        SDPFeatureExtractor
            Self
        """
        # Count all features
        for ex in examples:
            if ex.signature is None:
                continue

            features = self.extract_features_from_signature(ex.signature)
            for feat in features:
                self._feature_counts[feat] += 1

        # Build vocabulary with frequent features only
        for feat, count in self._feature_counts.items():
            if count >= self.min_feature_freq:
                self.vocab.add(feat)

        self._fitted = True
        return self

    def transform(
        self, examples: List[SDPExample]
    ) -> FeatureMatrix:
        """Transform examples into feature matrix.

        Parameters
        ----------
        examples : List[SDPExample]
            Examples to transform

        Returns
        -------
        FeatureMatrix
            Feature matrix with labels
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform()")

        # Build label mapping
        label_set = set(ex.relation_type for ex in examples if ex.signature)
        label_to_idx = {label: i for i, label in enumerate(sorted(label_set))}
        idx_to_label = {i: label for label, i in label_to_idx.items()}

        # Build sparse matrix
        rows, cols, data = [], [], []
        labels = []
        example_ids = []

        for row_idx, ex in enumerate(examples):
            if ex.signature is None:
                continue

            features = self.extract_features_from_signature(ex.signature)

            for feat, val in features.items():
                col_idx = self.vocab.get(feat)
                if col_idx is not None:
                    rows.append(row_idx)
                    cols.append(col_idx)
                    data.append(val)

            labels.append(label_to_idx[ex.relation_type])
            example_ids.append(ex.sent_id)

        # Create sparse matrix
        n_samples = len(labels)
        n_features = len(self.vocab)

        X = csr_matrix(
            (data, (rows, cols)),
            shape=(n_samples, n_features),
            dtype=np.float32,
        )

        y = np.array(labels, dtype=np.int32)

        return FeatureMatrix(
            X=X,
            y=y,
            vocab=self.vocab,
            label_to_idx=label_to_idx,
            idx_to_label=idx_to_label,
            example_ids=example_ids,
        )

    def fit_transform(self, examples: List[SDPExample]) -> FeatureMatrix:
        """Fit and transform in one step.

        Parameters
        ----------
        examples : List[SDPExample]
            Training examples

        Returns
        -------
        FeatureMatrix
            Feature matrix with labels
        """
        self.fit(examples)
        return self.transform(examples)

    def get_feature_name(self, idx: int) -> Optional[str]:
        """Get feature name by index."""
        return self.vocab.idx_to_feature.get(idx)

    def get_feature_stats(self) -> Dict[str, int]:
        """Get statistics about extracted features."""
        stats = {
            "total_features": len(self.vocab),
            "feature_types": defaultdict(int),
        }

        for feat in self.vocab.feature_to_idx:
            prefix = feat.split(":")[0]
            stats["feature_types"][prefix] += 1

        return dict(stats)


def extract_features_from_conllu(
    conllu_path: str,
    extractor: Optional[SDPFeatureExtractor] = None,
    fit: bool = True,
) -> Tuple[FeatureMatrix, SDPFeatureExtractor]:
    """Extract features from a CoNLL-U file.

    Parameters
    ----------
    conllu_path : str
        Path to CoNLL-U file
    extractor : Optional[SDPFeatureExtractor]
        Feature extractor (creates new if None)
    fit : bool
        Whether to fit the extractor

    Returns
    -------
    Tuple[FeatureMatrix, SDPFeatureExtractor]
        Feature matrix and extractor
    """
    # Load examples
    sdp_extractor = SDPExtractor()
    examples = sdp_extractor.extract_all_signatures(conllu_path)

    # Create feature extractor if needed
    if extractor is None:
        extractor = SDPFeatureExtractor()

    # Fit and/or transform
    if fit:
        matrix = extractor.fit_transform(examples)
    else:
        matrix = extractor.transform(examples)

    return matrix, extractor
