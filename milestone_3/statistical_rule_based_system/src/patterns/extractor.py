"""
Main pattern extractor that combines all pattern types.

Uses modular extractors from submodules and handles:
- Pattern extraction from all types
- Rule filtering by precision/support
- Rule ranking by priority tiers
"""

from collections import defaultdict

from tqdm import tqdm

from ..utils.semantic import FrameMapper, compute_rule_priority
from .base import Rule, split_directed_label
from .dependency import DependencyPatternExtractor
from .lexical import LexicalPatternExtractor
from .preposition import PrepositionPatternExtractor
from .semantic_patterns import SemanticPatternExtractor


class PatternExtractor:
    """
    Combined pattern extractor using modular sub-extractors.

    Extracts patterns from:
    - Lexical features (LEMMA, BIGRAM, PREP, etc.)
    - Dependency features (DEP_VERB, DEP_LABELS)
    - Semantic features (SYNSET, LEXNAME, HYPERNYM, FRAME)
    - Preposition structures (PREP_STRUCT, PREP_GOV_LEX, PREP_ROLES)
    """

    # Pattern types by category for ablation
    LEXICAL_PATTERN_TYPES = {
        'LEMMA', 'BIGRAM', 'PREP', 'BEFORE_E1', 'AFTER_E2', 'ENTITY_POS',
        'DEP_VERB', 'DEP_LABELS',
    }
    SEMANTIC_PATTERN_TYPES = {
        'SYNSET', 'LEXNAME', 'HYPERNYM', 'FRAME',
        'PREP_STRUCT', 'PREP_GOV_LEX', 'PREP_ROLES',
        'PREP_STRUCT_LEXNAME', 'PREP_STRUCT_HYPERNYM',
    }

    def __init__(
        self,
        min_precision: float = 0.6,
        min_support: int = 2,
        use_semantics: bool = True,
    ):
        """
        Initialize extractor with filtering thresholds.

        Args:
            min_precision: Minimum precision to accept a rule
            min_support: Minimum support count to accept a rule
            use_semantics: If True, include semantic patterns.
                          If False, only use lexical patterns (ablation mode).
        """
        self.min_precision = min_precision
        self.min_support = min_support
        self.use_semantics = use_semantics

        # Initialize sub-extractors
        self.lexical = LexicalPatternExtractor()
        self.dependency = DependencyPatternExtractor()
        self.semantic = SemanticPatternExtractor()
        self.preposition = PrepositionPatternExtractor()

        # Select extractors based on use_semantics flag
        if use_semantics:
            self.extractors = [
                self.lexical,
                self.dependency,
                self.semantic,
                self.preposition,
            ]
        else:
            # Basic mode: only lexical and dependency
            self.extractors = [
                self.lexical,
                self.dependency,
            ]

        # For frame mapper training
        self.frame_mapper = FrameMapper()
        self.rules: list[Rule] = []

    def get_all_pattern_types(self) -> list[str]:
        """Return all pattern types from active extractors."""
        types = []
        for extractor in self.extractors:
            types.extend(extractor.pattern_types)
        return types

    def get_active_pattern_types(self) -> set[str]:
        """Return set of active pattern types based on use_semantics."""
        if self.use_semantics:
            return self.LEXICAL_PATTERN_TYPES | self.SEMANTIC_PATTERN_TYPES
        return self.LEXICAL_PATTERN_TYPES

    def extract_patterns(self, processed_data: list[dict]) -> dict:
        """
        Extract all pattern types from processed training data.

        Args:
            processed_data: List of samples with doc, e1_span, e2_span,
                           relation_directed, etc.

        Returns:
            Dictionary of pattern counts by type
        """
        # Initialize pattern storage
        patterns = {
            pattern_type: defaultdict(lambda: defaultdict(int))
            for pattern_type in self.get_all_pattern_types()
        }

        # Reset semantic extractor for fresh collection
        if self.use_semantics:
            self.semantic.reset()

        # Extract patterns from each sample
        for sample in tqdm(processed_data, desc="Extracting patterns"):
            doc = sample['doc']
            e1_span = sample['e1_span']
            e2_span = sample['e2_span']
            relation = sample['relation_directed']

            # Run all active extractors
            for extractor in self.extractors:
                extractor.extract(doc, e1_span, e2_span, relation, patterns)

        # Train frame mapper with collected verb-relation pairs (only if using semantics)
        if self.use_semantics:
            verb_lemmas, relations = self.semantic.get_verb_relation_pairs()
            if verb_lemmas:
                self.frame_mapper.learn_from_data(verb_lemmas, relations)

        return patterns

    def filter_and_rank(self, patterns: dict, prediction_mode: str = "first_match") -> list[Rule]:
        """
        Filter patterns by precision/support and create ranked rules.

        Args:
            patterns: Output from extract_patterns()
            prediction_mode: 'priority_based' or 'first_match'
                            (Determines if semantic tiers are used for ranking)

        Returns:
            List of Rule objects sorted by priority
        """
        rules = []
        rule_counter = 0

        for pattern_type, pattern_counts in patterns.items():
            print(f"DEBUG: Processing type {pattern_type}, {len(pattern_counts)} candidates")
            accepted_count = 0
            for pattern_key, relation_counts in pattern_counts.items():
                total = sum(relation_counts.values())
                if total < self.min_support:
                    continue

                best_relation = max(relation_counts, key=relation_counts.get)
                support = relation_counts[best_relation]
                precision = support / total

                if precision < self.min_precision:
                    continue

                base_rel, direction = split_directed_label(best_relation)

                # Compute priority based on pattern type
                # has_synset/has_frame are no longer used for priority but kept for compatibility
                priority = compute_rule_priority(
                    pattern_type, precision, support,
                    rule_key=pattern_key
                )

                rule = Rule(
                    name=f"{best_relation}_{pattern_type}_{rule_counter}",
                    relation=best_relation,
                    base_relation=base_rel,
                    direction=direction,
                    pattern_type=pattern_type,
                    pattern_data=pattern_key[1:],  # Skip type prefix
                    precision=precision,
                    support=support,
                    priority=priority,
                    explanation=f"{pattern_type}: {pattern_key[1:]}",
                )
                rules.append(rule)
                rule_counter += 1
                accepted_count += 1
            print(f"DEBUG: Type {pattern_type}: Kept {accepted_count} rules")

        # Sort based on PREDICTION_MODE
        if prediction_mode == 'priority_based':
            # Priority score includes tier + precision + tie-break
            rules.sort(key=lambda r: -r.priority)
        else:
            # First match baseline: pure Precision then Support
            rules.sort(key=lambda r: (-r.precision, -r.support))
            
        self.rules = rules

        print(f"Created {len(rules)} rules after filtering")
        return rules

    def get_rules_by_tier(self) -> dict[str, list[Rule]]:
        """Group rules by priority tier for analysis."""
        tiers = {
            'TIER_1 (Synset+Frame)': [],
            'TIER_2 (Prep-Structure)': [],
            'TIER_3 (Lexname+Hypernym)': [],
            'TIER_4 (Bigram+Dep)': [],
            'TIER_5 (Lexical)': [],
        }

        for rule in self.rules:
            if rule.priority >= 100:
                tiers['TIER_1 (Synset+Frame)'].append(rule)
            elif rule.priority >= 80:
                tiers['TIER_2 (Prep-Structure)'].append(rule)
            elif rule.priority >= 60:
                tiers['TIER_3 (Lexname+Hypernym)'].append(rule)
            elif rule.priority >= 40:
                tiers['TIER_4 (Bigram+Dep)'].append(rule)
            else:
                tiers['TIER_5 (Lexical)'].append(rule)

        return tiers

    def get_rules_by_type(self) -> dict[str, list[Rule]]:
        """Group rules by pattern type for analysis."""
        by_type = defaultdict(list)
        for rule in self.rules:
            by_type[rule.pattern_type].append(rule)
        return dict(by_type)

    def save_rules(self, output_path, format: str = 'json') -> None:
        """
        Save extracted rules to file for analysis.

        Args:
            output_path: Path to save rules
            format: 'json' or 'csv'
        """
        from pathlib import Path
        import json
        import csv

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            rules_data = [rule.to_dict() for rule in self.rules]
            with open(output_path, 'w') as f:
                json.dump(rules_data, f, indent=2)
            print(f"Saved {len(self.rules)} rules to {output_path}")

        elif format == 'csv':
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'name', 'relation', 'base_relation', 'direction',
                    'pattern_type', 'pattern_data', 'precision', 'support',
                    'priority', 'explanation'
                ])
                for rule in self.rules:
                    writer.writerow([
                        rule.name,
                        rule.relation,
                        rule.base_relation,
                        rule.direction,
                        rule.pattern_type,
                        str(rule.pattern_data),
                        f"{rule.precision:.3f}",
                        rule.support,
                        f"{rule.priority:.2f}",
                        rule.explanation,
                    ])
            print(f"Saved {len(self.rules)} rules to {output_path}")
