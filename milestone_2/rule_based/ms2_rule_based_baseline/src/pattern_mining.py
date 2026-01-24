"""Pattern mining and rule discovery from training data.

This module handles:
- Extracting candidate patterns (lexical, dependency)
- Filtering and ranking patterns by precision/support
- Generating data-driven patterns from analysis
"""

from collections import Counter, defaultdict
from typing import Any

from .config import (
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


def extract_candidate_patterns(
    processed_data: list[dict],
) -> tuple[dict, dict]:
    """
    Mine candidate lexical and dependency patterns from labeled training data.

    Args:
        processed_data: List of processed samples.

    Returns:
        Tuple of (lexical_patterns, dep_patterns) dictionaries.
    """
    relation_groups = defaultdict(list)
    for sample in processed_data:
        relation_groups[sample["relation_directed"]].append(sample)

    lexical_patterns = defaultdict(lambda: defaultdict(int))
    dep_patterns = defaultdict(lambda: defaultdict(int))

    print("Mining candidate patterns from training data...")
    print("=" * 80)

    for relation, samples in relation_groups.items():
        print(f"\n{relation}: {len(samples)} samples")

        for sample in samples:
            doc = sample["doc"]
            e1_span = sample["e1_span"]
            e2_span = sample["e2_span"]

            # Extract between-span
            if e1_span.start < e2_span.start:
                between_span = doc[e1_span.end:e2_span.start]
            else:
                between_span = doc[e2_span.end:e1_span.start]

            # Lexical patterns
            between_lemmas = [
                t.lemma_.lower() for t in between_span if not t.is_punct
            ]

            for lemma in between_lemmas:
                if len(lemma) > MIN_LEMMA_LENGTH:
                    lexical_patterns[("LEMMA", lemma)][relation] += 1

            for i in range(len(between_lemmas) - 1):
                bigram = (between_lemmas[i], between_lemmas[i + 1])
                lexical_patterns[("BIGRAM", bigram)][relation] += 1

            for token in between_span:
                if token.pos_ == "ADP":
                    lexical_patterns[("PREP", token.lemma_.lower())][relation] += 1

            if e1_span.start > 0:
                before_e1 = doc[e1_span.start - 1]
                if not before_e1.is_punct and len(before_e1.lemma_) > MIN_LEMMA_LENGTH:
                    key = ("BEFORE_E1", before_e1.lemma_.lower())
                    lexical_patterns[key][relation] += 1

            if e2_span.end < len(doc):
                after_e2 = doc[e2_span.end]
                if not after_e2.is_punct and len(after_e2.lemma_) > MIN_LEMMA_LENGTH:
                    key = ("AFTER_E2", after_e2.lemma_.lower())
                    lexical_patterns[key][relation] += 1

            e1_pos = e1_span.root.pos_
            e2_pos = e2_span.root.pos_
            lexical_patterns[("ENTITY_POS", e1_pos, e2_pos)][relation] += 1

            # Dependency patterns
            e1_head = e1_span.root
            e2_head = e2_span.root

            for token in doc:
                if token.pos_ == "VERB":
                    e1_dep_to_verb = None
                    e2_dep_to_verb = None

                    if e1_head.head == token:
                        e1_dep_to_verb = e1_head.dep_
                    elif e1_head == token:
                        e1_dep_to_verb = "VERB_IS_E1"

                    if e2_head.head == token:
                        e2_dep_to_verb = e2_head.dep_
                    elif e2_head == token:
                        e2_dep_to_verb = "VERB_IS_E2"

                    if e1_dep_to_verb and e2_dep_to_verb:
                        verb_lemma = token.lemma_.lower()
                        key = ("DEP_VERB", verb_lemma, e1_dep_to_verb, e2_dep_to_verb)
                        dep_patterns[key][relation] += 1

            dep_patterns[("DEP_LABELS", e1_head.dep_, e2_head.dep_)][relation] += 1

    return lexical_patterns, dep_patterns


def split_relation_and_direction(rel_directed: str) -> tuple[str, str | None]:
    """
    Split directed relation label into base relation and direction.

    Args:
        rel_directed: Directed relation label, e.g., 'Cause-Effect(e1,e2)'.

    Returns:
        Tuple of (base_relation, direction) or (relation, None) for 'Other'.
    """
    if "(" in rel_directed and rel_directed.endswith(")"):
        base, dir_part = rel_directed.split("(", 1)
        direction = dir_part[:-1]
        return base.strip(), direction.strip()
    return rel_directed, None


def filter_and_rank_patterns(
    lexical_patterns: dict,
    dep_patterns: dict,
    min_precision: float = MIN_PRECISION,
    min_support: int = MIN_SUPPORT,
) -> list[dict]:
    """
    Filter patterns by precision and support, then rank them.

    Args:
        lexical_patterns: Dictionary of lexical pattern counts.
        dep_patterns: Dictionary of dependency pattern counts.
        min_precision: Minimum precision threshold.
        min_support: Minimum support threshold.

    Returns:
        Ordered list of rule dictionaries sorted by precision and support.
    """
    rules = []

    def process_patterns(patterns_dict: dict, matcher_type: str) -> None:
        for pattern_key, relation_counts in patterns_dict.items():
            total_count = sum(relation_counts.values())
            if total_count < min_support:
                continue

            best_relation = max(relation_counts, key=relation_counts.get)
            best_count = relation_counts[best_relation]
            precision = best_count / total_count

            if precision >= min_precision:
                pattern_type, *pattern_data = pattern_key
                base_rel, direction = split_relation_and_direction(best_relation)

                rule = {
                    "name": f"{best_relation}_{pattern_type}_{hash(pattern_key) % 10000}",
                    "relation": best_relation,
                    "base_relation": base_rel,
                    "direction": direction,
                    "matcher_type": matcher_type,
                    "pattern_type": pattern_type,
                    "pattern_data": pattern_data,
                    "precision": precision,
                    "support": best_count,
                    "explanation": f"{pattern_type}: {pattern_data}",
                }
                rules.append(rule)

    process_patterns(lexical_patterns, "lexical")
    process_patterns(dep_patterns, "dependency")

    rules.sort(key=lambda r: (-r["precision"], -r["support"]))
    return rules


def visualize_rules_by_relation(
    rules: list[dict],
    top_n: int = 5,
) -> None:
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
