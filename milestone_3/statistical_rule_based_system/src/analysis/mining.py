"""Pattern mining and rule discovery from training data.

This module handles:
- Extracting candidate patterns (lexical, dependency, semantic)
- Filtering and ranking patterns by precision/support
- Generating data-driven patterns from analysis
"""

from collections import Counter, defaultdict
from typing import Optional

from ..utils.config import (
    MIN_LEMMA_LENGTH,
    MIN_PRECISION,
    MIN_SUPPORT,
    TOP_N_DEP_PATHS,
    TOP_N_KEYWORDS,
    TOP_N_PREPS,
    TOP_N_VERBS,
)


def analyze_relation_features(processed_data: list[dict]) -> dict[str, dict]:
    """
    Analyze linguistic features for each directed relation type.

    Args:
        processed_data: List of processed samples.

    Returns:
        Dictionary mapping relations to feature frequency dictionaries.
    """
    relation_groups = defaultdict(list)
    for sample in processed_data:
        relation_groups[sample["relation_directed"]].append(sample)

    relation_analysis = {}

    for relation, samples in relation_groups.items():
        all_lemmas = []
        all_verbs = []
        all_preps = []
        all_dep_paths = []
        all_between_words = []

        for sample in samples:
            doc = sample["doc"]
            e1_tokens = set(range(sample["e1_span"].start, sample["e1_span"].end))
            e2_tokens = set(range(sample["e2_span"].start, sample["e2_span"].end))

            for token in doc:
                if token.i not in e1_tokens and token.i not in e2_tokens:
                    lemma = token.lemma_.lower()

                    if (token.pos_ == "VERB" and not token.is_punct
                            and len(lemma) > MIN_LEMMA_LENGTH):
                        all_verbs.append(lemma)

                    if token.pos_ == "ADP" and not token.is_punct:
                        all_preps.append(lemma)

                    if (not token.is_stop and not token.is_punct
                            and len(lemma) > MIN_LEMMA_LENGTH):
                        all_lemmas.append(lemma)

            if sample["dep_path"]:
                path_deps = tuple([d[0] for d in sample["dep_path"]])
                all_dep_paths.append(path_deps)

            for word in sample["between_words"]:
                if word["text"].strip() and len(word["lemma"]) > MIN_LEMMA_LENGTH:
                    all_between_words.append(word["lemma"].lower())

        relation_analysis[relation] = {
            "count": len(samples),
            "top_lemmas": Counter(all_lemmas).most_common(TOP_N_KEYWORDS),
            "top_verbs": Counter(all_verbs).most_common(TOP_N_VERBS),
            "top_preps": Counter(all_preps).most_common(TOP_N_PREPS),
            "top_dep_paths": Counter(all_dep_paths).most_common(10),
            "top_between_words": Counter(all_between_words).most_common(20),
        }

    return relation_analysis


def generate_patterns_from_analysis(
    relation_features: dict[str, dict],
    top_n_keywords: int = TOP_N_KEYWORDS,
    top_n_verbs: int = TOP_N_VERBS,
    top_n_preps: int = TOP_N_PREPS,
) -> dict[str, dict]:
    """
    Generate RELATION_PATTERNS dictionary from data analysis.

    Args:
        relation_features: Dictionary of relation -> feature frequencies.
        top_n_keywords: Number of top keywords to extract.
        top_n_verbs: Number of top verbs to extract.
        top_n_preps: Number of top prepositions to extract.

    Returns:
        Dictionary mapping relations to pattern dictionaries.
    """
    generated_patterns = {}

    for relation, features in relation_features.items():
        keywords = [lemma for lemma, _ in features["top_lemmas"][:top_n_keywords]]
        verbs = [verb for verb, _ in features["top_verbs"][:top_n_verbs]]
        preps = [prep for prep, _ in features["top_preps"][:top_n_preps]]

        dep_patterns = []
        for path, _ in features["top_dep_paths"][:TOP_N_DEP_PATHS]:
            if len(path) >= 2:
                dep_patterns.append(list(path[:3]))

        generated_patterns[relation] = {
            "keywords": keywords,
            "prep_patterns": preps,
            "verb_patterns": verbs,
            "dependency_patterns": dep_patterns,
        }

    return generated_patterns


# Legacy extraction functions removed (Refactored to src.patterns.extractor)


def visualize_rules_by_relation(rules: list[dict], top_n: int = 5) -> None:
    """
    Display the top-N highest precision rules for each relation.

    Args:
        rules: List of rule dictionaries.
        top_n: Number of top rules to display per relation.
    """
    relation_rules = defaultdict(list)
    for rule in rules:
        relation_rules[rule["relation"]].append(rule)

    print("\n" + "=" * 100)
    print("TOP RULES BY RELATION TYPE (for Explainability)")
    print("=" * 100)

    for relation in sorted(relation_rules.keys()):
        rules_list = relation_rules[relation][:top_n]
        print(f"\n{'=' * 100}")
        print(f"Relation: {relation} ({len(relation_rules[relation])} total rules)")
        print("=" * 100)

        for i, rule in enumerate(rules_list, 1):
            print(f"\n  Rule {i}: {rule['name']}")
            print(f"    Type: {rule['pattern_type']}")
            print(f"    Precision: {rule['precision']:.3f} | Support: {rule['support']}")
            print(f"    Pattern Data: {rule['pattern_data']}")
