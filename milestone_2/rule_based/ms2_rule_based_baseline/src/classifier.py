"""Rule-based classifier using spaCy matchers.

This module handles:
- Compiling rules into spaCy matchers
- Applying rules to classify samples
- Generating explanations for predictions
"""

from collections.abc import Sequence
from typing import Any

from spacy.matcher import DependencyMatcher, Matcher, PhraseMatcher
from tqdm.auto import tqdm


def compile_matchers(
    rules: list[dict],
    nlp,
) -> tuple[Matcher, PhraseMatcher, DependencyMatcher]:
    """
    Pre-compile all rule patterns into spaCy matchers.

    Args:
        rules: List of rule dictionaries.
        nlp: spaCy language model.

    Returns:
        Tuple of (token_matcher, phrase_matcher, dep_matcher).
    """
    token_matcher = Matcher(nlp.vocab)
    phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
    dep_matcher = DependencyMatcher(nlp.vocab)

    # Compile token patterns (BIGRAM, PREP)
    for i, rule in enumerate(rules):
        match_id = f"rule_{i}"

        if rule["matcher_type"] == "lexical":
            pattern_type = rule["pattern_type"]
            pattern_data = rule["pattern_data"]

            if pattern_type == "BIGRAM":
                pattern = [
                    {"LEMMA": pattern_data[0][0]},
                    {"LEMMA": pattern_data[0][1]},
                ]
                token_matcher.add(match_id, [pattern])

            elif pattern_type == "PREP":
                pattern = [{"LEMMA": pattern_data[0], "POS": "ADP"}]
                token_matcher.add(match_id, [pattern])

    # Compile phrase patterns (LEMMA)
    lemma_rules = [
        (i, r) for i, r in enumerate(rules)
        if r["matcher_type"] == "lexical" and r["pattern_type"] == "LEMMA"
    ]

    if lemma_rules:
        patterns = [nlp(r["pattern_data"][0]) for _, r in lemma_rules]
        match_ids = [f"rule_{i}" for i, _ in lemma_rules]

        for match_id, pattern in zip(match_ids, patterns):
            phrase_matcher.add(match_id, [pattern])

    # Compile dependency patterns (DEP_VERB)
    for i, rule in enumerate(rules):
        if rule["pattern_type"] == "DEP_VERB":
            match_id = f"rule_{i}"
            verb_lemma, e1_dep, e2_dep = rule["pattern_data"]

            pattern = [
                {
                    "RIGHT_ID": "verb",
                    "RIGHT_ATTRS": {"LEMMA": verb_lemma, "POS": "VERB"},
                },
                {
                    "LEFT_ID": "verb",
                    "REL_OP": ">",
                    "RIGHT_ID": "e1",
                    "RIGHT_ATTRS": {"DEP": e1_dep},
                },
                {
                    "LEFT_ID": "verb",
                    "REL_OP": ">",
                    "RIGHT_ID": "e2",
                    "RIGHT_ATTRS": {"DEP": e2_dep},
                },
            ]
            dep_matcher.add(match_id, [pattern])

    return token_matcher, phrase_matcher, dep_matcher


def check_rule_match(
    rule: dict,
    rule_index: int,
    doc,
    between_span,
    e1_span,
    e2_span,
    e1_head,
    e2_head,
    token_matcher: Matcher,
    phrase_matcher: PhraseMatcher,
    dep_matcher: DependencyMatcher,
    nlp,
) -> bool:
    """
    Check if a rule matches the given sample.

    Args:
        rule: Rule dictionary.
        rule_index: Index of the rule in the rules list.
        doc: spaCy Doc.
        between_span: Span between entities.
        e1_span: Entity 1 span.
        e2_span: Entity 2 span.
        e1_head: Root token of entity 1.
        e2_head: Root token of entity 2.
        token_matcher: Compiled token matcher.
        phrase_matcher: Compiled phrase matcher.
        dep_matcher: Compiled dependency matcher.
        nlp: spaCy language model.

    Returns:
        True if the rule matches, False otherwise.
    """
    match_id = f"rule_{rule_index}"
    pattern_type = rule["pattern_type"]
    pattern_data = rule["pattern_data"]

    if pattern_type in ["BIGRAM", "PREP"]:
        matches = token_matcher(between_span)
        if any(nlp.vocab.strings[m[0]] == match_id for m in matches):
            return True

    elif pattern_type == "LEMMA":
        matches = phrase_matcher(between_span)
        if any(nlp.vocab.strings[m[0]] == match_id for m in matches):
            return True

    elif pattern_type == "DEP_VERB":
        matches = dep_matcher(doc)
        for match_id_found, token_ids in matches:
            if nlp.vocab.strings[match_id_found] == match_id:
                e1_in_match = any(
                    t in range(e1_span.start, e1_span.end) for t in token_ids
                )
                e2_in_match = any(
                    t in range(e2_span.start, e2_span.end) for t in token_ids
                )
                if e1_in_match or e2_in_match:
                    return True

    elif pattern_type == "BEFORE_E1" and e1_span.start > 0:
        if doc[e1_span.start - 1].lemma_.lower() == pattern_data[0]:
            return True

    elif pattern_type == "AFTER_E2" and e2_span.end < len(doc):
        if doc[e2_span.end].lemma_.lower() == pattern_data[0]:
            return True

    elif pattern_type == "ENTITY_POS":
        if e1_head.pos_ == pattern_data[0] and e2_head.pos_ == pattern_data[1]:
            return True

    elif pattern_type == "DEP_LABELS":
        if e1_head.dep_ == pattern_data[0] and e2_head.dep_ == pattern_data[1]:
            return True

    return False


def apply_rule_based_classifier(
    samples: list[dict],
    rules: list[dict],
    nlp,
) -> tuple[list[str], list[str | None], list[str]]:
    """
    Apply rule-based classification to samples.

    Args:
        samples: List of processed sample dictionaries.
        rules: List of rule dictionaries.
        nlp: spaCy language model.

    Returns:
        Tuple of (predictions, directions, explanations) lists.
    """
    token_matcher, phrase_matcher, dep_matcher = compile_matchers(rules, nlp)

    predictions = []
    directions = []
    explanations = []

    for sample in tqdm(samples, desc="Classifying"):
        doc = sample["doc"]
        e1_span = sample["e1_span"]
        e2_span = sample["e2_span"]

        if e1_span.start < e2_span.start:
            between_span = doc[e1_span.end:e2_span.start]
        else:
            between_span = doc[e2_span.end:e1_span.start]

        e1_head = e1_span.root
        e2_head = e2_span.root

        matched_rule = None

        # Apply rules in order (first match wins)
        for i, rule in enumerate(rules):
            if check_rule_match(
                rule, i, doc, between_span, e1_span, e2_span,
                e1_head, e2_head, token_matcher, phrase_matcher,
                dep_matcher, nlp
            ):
                matched_rule = rule
                break

        if matched_rule:
            predictions.append(matched_rule["relation"])
            directions.append(matched_rule["direction"])
            explanation = (
                f"Rule {matched_rule['name']}: {matched_rule['explanation']} "
                f"(precision={matched_rule['precision']:.2f}, "
                f"support={matched_rule['support']})"
            )
            explanations.append(explanation)
        else:
            predictions.append("Other")
            directions.append(None)
            explanations.append("No high-precision rule matched; defaulting to Other.")

    return predictions, directions, explanations
