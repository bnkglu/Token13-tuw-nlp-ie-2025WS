"""
Lexical pattern extraction and matching for Milestone 3.

Implements M2-style LEMMA and PREP patterns as fallbacks when
entity-rooted dependency patterns fail to match.

Key features:
- Extract LEMMA, PREP, BIGRAM patterns from between-entity spans
- Filter by precision >= 0.60 and support >= 2
- Apply as fallback after main pattern matching

Usage:
    from lexical_patterns import LexicalPatternMatcher

    matcher = LexicalPatternMatcher(nlp)
    matcher.mine_patterns(train_samples, train_labels)
    prediction, direction, explanation = matcher.match(sample)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from spacy.matcher import Matcher, PhraseMatcher


class LexicalPatternMatcher:
    """
    M2-style lexical pattern matcher for fallback predictions.

    Extracts and applies simple lexical patterns based on tokens
    appearing between entities.
    """

    def __init__(self, nlp, min_precision: float = 0.60, min_support: int = 2):
        """
        Initialize the lexical pattern matcher.

        Args:
            nlp: spaCy language model
            min_precision: Minimum precision threshold for pattern acceptance
            min_support: Minimum support (occurrence count) threshold
        """
        self.nlp = nlp
        self.min_precision = min_precision
        self.min_support = min_support

        # Pattern storage
        self.lexical_patterns: Dict[Tuple, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.rules: List[Dict] = []

        # spaCy matchers (initialized on compile)
        self.token_matcher: Optional[Matcher] = None
        self.phrase_matcher: Optional[PhraseMatcher] = None
        self.rule_lookup: Dict[str, Dict] = {}

        self.is_compiled = False

    def get_between_span(self, doc, e1_span, e2_span):
        """Get the span of tokens between two entities."""
        if e1_span.start < e2_span.start:
            return doc[e1_span.end:e2_span.start]
        else:
            return doc[e2_span.end:e1_span.start]

    def get_directed_label(self, item: Dict) -> str:
        """Get directed relation label from data item."""
        rel_type = item.get('relation', {}).get('type', 'Other')
        direction = item.get('relation', {}).get('direction', '')
        if rel_type == 'Other':
            return 'Other'
        direction = direction.replace('(', '').replace(')', '')
        return f"{rel_type}({direction})"

    def mine_patterns(self, samples: List[Dict], labels: Optional[List[str]] = None):
        """
        Mine lexical patterns from training data.

        Args:
            samples: List of preprocessed samples with 'doc', 'e1_span', 'e2_span'
            labels: Optional list of ground truth labels. If None, extracted from samples.
        """
        print(f"Mining lexical patterns from {len(samples)} samples...")

        self.lexical_patterns = defaultdict(lambda: defaultdict(int))

        for i, sample in enumerate(samples):
            doc = sample['doc']
            e1_span = sample['e1_span']
            e2_span = sample['e2_span']

            # Get label
            if labels:
                relation = labels[i]
            elif 'relation' in sample:
                relation = self.get_directed_label(sample)
            else:
                continue

            # Get between span
            between_span = self.get_between_span(doc, e1_span, e2_span)

            if len(between_span) == 0:
                continue

            # Extract LEMMA patterns (single lemmas)
            between_lemmas = [t.lemma_.lower() for t in between_span if not t.is_punct]
            for lemma in between_lemmas:
                if len(lemma) > 2:  # Skip very short lemmas
                    pattern_key = ('LEMMA', lemma)
                    self.lexical_patterns[pattern_key][relation] += 1

            # Extract PREP patterns (prepositions)
            for token in between_span:
                if token.pos_ == 'ADP':
                    pattern_key = ('PREP', token.lemma_.lower())
                    self.lexical_patterns[pattern_key][relation] += 1

            # Extract BIGRAM patterns (consecutive lemma pairs)
            for j in range(len(between_lemmas) - 1):
                bigram = (between_lemmas[j], between_lemmas[j + 1])
                pattern_key = ('BIGRAM', bigram)
                self.lexical_patterns[pattern_key][relation] += 1

        print(f"  Mined {len(self.lexical_patterns)} candidate patterns")

        # Filter and rank patterns
        self._filter_and_rank_patterns()

        # Compile matchers
        self._compile_matchers()

    def _filter_and_rank_patterns(self):
        """Filter patterns by precision and support, then rank them."""
        self.rules = []

        for pattern_key, relation_counts in self.lexical_patterns.items():
            total_count = sum(relation_counts.values())

            if total_count < self.min_support:
                continue

            # Find dominant relation
            best_relation = max(relation_counts, key=relation_counts.get)
            best_count = relation_counts[best_relation]
            precision = best_count / total_count

            if precision < self.min_precision:
                continue

            pattern_type = pattern_key[0]
            pattern_data = pattern_key[1:]

            # Parse direction from relation
            if '(' in best_relation and best_relation != 'Other':
                base_rel = best_relation.split('(')[0]
                direction = best_relation.split('(')[1].rstrip(')')
            else:
                base_rel = best_relation
                direction = None

            rule = {
                'name': f"{best_relation}_{pattern_type}_{hash(pattern_key) % 10000}",
                'relation': best_relation,
                'base_relation': base_rel,
                'direction': direction,
                'pattern_type': pattern_type,
                'pattern_data': pattern_data,
                'precision': precision,
                'support': best_count,
                'total_count': total_count,
                'explanation': f"{pattern_type} pattern: {pattern_data} (precision={precision:.2f}, support={best_count})"
            }

            self.rules.append(rule)

        # Sort by precision (desc), then support (desc)
        self.rules.sort(key=lambda r: (-r['precision'], -r['support']))

        print(f"  Filtered to {len(self.rules)} high-quality rules")

        # Statistics
        by_type = defaultdict(int)
        for rule in self.rules:
            by_type[rule['pattern_type']] += 1
        print(f"  By type: {dict(by_type)}")

    def _compile_matchers(self):
        """Compile spaCy matchers for efficient pattern application."""
        self.token_matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LEMMA")
        self.rule_lookup = {}

        for i, rule in enumerate(self.rules):
            match_id = f"rule_{i}"
            pattern_type = rule['pattern_type']
            pattern_data = rule['pattern_data']

            self.rule_lookup[match_id] = rule

            if pattern_type == 'LEMMA':
                # Use PhraseMatcher for single lemmas
                lemma = pattern_data[0] if isinstance(pattern_data, tuple) else pattern_data
                try:
                    pattern_doc = self.nlp.make_doc(lemma)
                    self.phrase_matcher.add(match_id, [pattern_doc])
                except Exception:
                    pass

            elif pattern_type == 'PREP':
                # Use Matcher for preposition patterns
                prep = pattern_data[0] if isinstance(pattern_data, tuple) else pattern_data
                pattern = [{"LEMMA": prep, "POS": "ADP"}]
                self.token_matcher.add(match_id, [pattern])

            elif pattern_type == 'BIGRAM':
                # Use Matcher for bigram patterns
                bigram = pattern_data[0] if len(pattern_data) == 1 else pattern_data
                if isinstance(bigram, tuple) and len(bigram) == 2:
                    pattern = [{"LEMMA": bigram[0]}, {"LEMMA": bigram[1]}]
                    self.token_matcher.add(match_id, [pattern])

        self.is_compiled = True
        print(f"  Compiled matchers with {len(self.rule_lookup)} rules")

    def match(self, sample: Dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Try to match lexical patterns for a sample.

        Args:
            sample: Preprocessed sample with 'doc', 'e1_span', 'e2_span'

        Returns:
            Tuple of (relation, direction, explanation) or (None, None, None) if no match
        """
        if not self.is_compiled or not self.rules:
            return None, None, None

        doc = sample['doc']
        e1_span = sample['e1_span']
        e2_span = sample['e2_span']

        between_span = self.get_between_span(doc, e1_span, e2_span)

        if len(between_span) == 0:
            return None, None, None

        # Try rules in order (decision list - first match wins)
        for rule in self.rules:
            pattern_type = rule['pattern_type']
            match_id = f"rule_{self.rules.index(rule)}"

            matched = False

            if pattern_type == 'LEMMA':
                matches = self.phrase_matcher(between_span)
                if any(self.nlp.vocab.strings[m[0]] == match_id for m in matches):
                    matched = True

            elif pattern_type in ['PREP', 'BIGRAM']:
                matches = self.token_matcher(between_span)
                if any(self.nlp.vocab.strings[m[0]] == match_id for m in matches):
                    matched = True

            if matched:
                return rule['relation'], rule['direction'], f"Lexical fallback: {rule['explanation']}"

        return None, None, None

    def save(self, path: Path):
        """Save mined rules to JSON file."""
        # Convert rules to JSON-serializable format
        json_rules = []
        for rule in self.rules:
            r = rule.copy()
            # Convert tuple pattern_data to list
            if isinstance(r['pattern_data'], tuple):
                r['pattern_data'] = list(r['pattern_data'])
            json_rules.append(r)

        with open(path, 'w') as f:
            json.dump({
                'min_precision': self.min_precision,
                'min_support': self.min_support,
                'rules': json_rules
            }, f, indent=2)

        print(f"Saved {len(self.rules)} rules to {path}")

    def load(self, path: Path):
        """Load rules from JSON file."""
        with open(path) as f:
            data = json.load(f)

        self.min_precision = data.get('min_precision', 0.60)
        self.min_support = data.get('min_support', 2)

        self.rules = []
        for r in data['rules']:
            # Convert list pattern_data back to tuple for hashability
            if isinstance(r['pattern_data'], list):
                r['pattern_data'] = tuple(r['pattern_data'])
            self.rules.append(r)

        self._compile_matchers()
        print(f"Loaded {len(self.rules)} rules from {path}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the mined patterns."""
        by_type = defaultdict(int)
        by_relation = defaultdict(int)
        precision_sum = 0
        support_sum = 0

        for rule in self.rules:
            by_type[rule['pattern_type']] += 1
            by_relation[rule['base_relation']] += 1
            precision_sum += rule['precision']
            support_sum += rule['support']

        return {
            'total_rules': len(self.rules),
            'by_type': dict(by_type),
            'by_relation': dict(by_relation),
            'avg_precision': precision_sum / len(self.rules) if self.rules else 0,
            'avg_support': support_sum / len(self.rules) if self.rules else 0,
        }


def mine_and_save_lexical_patterns(nlp, train_samples, train_labels, output_path: Path):
    """
    Convenience function to mine and save lexical patterns.

    Args:
        nlp: spaCy language model
        train_samples: Preprocessed training samples
        train_labels: Training labels
        output_path: Path to save the rules

    Returns:
        LexicalPatternMatcher instance
    """
    matcher = LexicalPatternMatcher(nlp)
    matcher.mine_patterns(train_samples, train_labels)
    matcher.save(output_path)

    stats = matcher.get_stats()
    print(f"\nLexical Pattern Statistics:")
    print(f"  Total rules: {stats['total_rules']}")
    print(f"  By type: {stats['by_type']}")
    print(f"  Avg precision: {stats['avg_precision']:.2f}")
    print(f"  Avg support: {stats['avg_support']:.1f}")

    return matcher


if __name__ == "__main__":
    import spacy
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import preprocess_data

    # Load data
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir.parent / "data" / "processed"

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_lg")

    print("Loading training data...")
    with open(data_dir / "train" / "train.json") as f:
        train_data = json.load(f)

    print("Preprocessing...")
    train_samples = preprocess_data(train_data, nlp)

    # Get labels
    def get_directed_label(item):
        rel_type = item['relation']['type']
        direction = item['relation'].get('direction', '')
        if rel_type == 'Other':
            return 'Other'
        direction = direction.replace('(', '').replace(')', '')
        return f"{rel_type}({direction})"

    train_labels = [get_directed_label(item) for item in train_data]

    # Mine and save
    output_path = base_dir / "data" / "lexical_rules.json"
    matcher = mine_and_save_lexical_patterns(nlp, train_samples, train_labels, output_path)
