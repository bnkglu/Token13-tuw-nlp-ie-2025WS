"""Main classification engine."""

from typing import Optional

from tqdm.auto import tqdm

from .manual_checks import check_rule_match_manual
from .matcher_utils import compile_matchers


def apply_rule_based_classifier(
    samples: list[dict],
    rules: list[dict],
    nlp,
    prediction_mode: str = "first_match",
) -> tuple[list[str], list[Optional[str]], list[str]]:
    """
    Apply rule-based classification to samples.

    Args:
        samples: List of processed sample dictionaries.
        rules: List of rule dictionaries.
        nlp: spaCy language model.
        prediction_mode: 'first_match' or 'priority_based'.

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

        # Optimization: Run matchers ONCE per sample
        matched_rule_indices = set()
        
        # Lexical matches
        if len(between_span) > 0:
            for match_id, start, end in token_matcher(between_span):
                matched_rule_indices.add(nlp.vocab.strings[match_id])
            for match_id, start, end in phrase_matcher(between_span):
                matched_rule_indices.add(nlp.vocab.strings[match_id])
        
        # Dependency matches
        for match_id, token_ids in dep_matcher(doc):
            match_str = nlp.vocab.strings[match_id]
            # Verify entities are involved in dependency match
            e1_in_match = any(t in range(e1_span.start, e1_span.end) for t in token_ids)
            e2_in_match = any(t in range(e2_span.start, e2_span.end) for t in token_ids)
            if e1_in_match and e2_in_match:
                matched_rule_indices.add(match_str)

        matched_rule = None

        if prediction_mode == "first_match":
            for i, rule in enumerate(rules):
                match_id = f"rule_{i}"
                pattern_type = rule["pattern_type"]
                is_match = False

                # Dispatch check based on pattern type
                if pattern_type in {'BIGRAM', 'PREP', 'LEMMA', 'DEP_VERB'}:
                    # O(1) check using pre-computed matches
                    if match_id in matched_rule_indices:
                        is_match = True
                else:
                    # Manual check for everything else (Semantic + Structural)
                    if check_rule_match_manual(
                        rule, doc, between_span, e1_span, e2_span,
                        e1_head, e2_head
                    ):
                        is_match = True

                if is_match:
                    matched_rule = rule
                    break

        elif prediction_mode == "priority_based":
            matching_rules = []
            for i, rule in enumerate(rules):
                match_id = f"rule_{i}"
                pattern_type = rule["pattern_type"]
                is_match = False
                
                if pattern_type in {'BIGRAM', 'PREP', 'LEMMA', 'DEP_VERB'}:
                    if match_id in matched_rule_indices:
                        is_match = True
                else:
                    if check_rule_match_manual(
                        rule, doc, between_span, e1_span, e2_span,
                        e1_head, e2_head
                    ):
                        is_match = True
                
                if is_match:
                    matching_rules.append(rule)

            if matching_rules:
                matched_rule = max(matching_rules, key=lambda r: r.get("priority", 0))

        if matched_rule:
            predictions.append(matched_rule["relation"])
            directions.append(matched_rule["direction"])
            explanation = (
                f"Rule {matched_rule['name']}: {matched_rule['explanation']} "
                f"(precision={matched_rule['precision']:.2f}, "
                f"support={matched_rule['support']}, "
                f"priority={matched_rule.get('priority', 0):.0f})"
            )
            explanations.append(explanation)
        else:
            predictions.append("Other")
            directions.append(None)
            explanations.append("No high-precision rule matched; defaulting to Other.")

    return predictions, directions, explanations
