"""Utilities for compiling spaCy matchers."""

from spacy.matcher import DependencyMatcher, Matcher, PhraseMatcher


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

    for i, rule in enumerate(rules):
        match_id = f"rule_{i}"

        if rule.get("matcher_type") == "lexical":
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

    lemma_rules = [
        (i, r) for i, r in enumerate(rules)
        if r.get("matcher_type") == "lexical" and r["pattern_type"] == "LEMMA"
    ]

    if lemma_rules:
        patterns = [nlp(r["pattern_data"][0]) for _, r in lemma_rules]
        match_ids = [f"rule_{i}" for i, _ in lemma_rules]

        for match_id, pattern in zip(match_ids, patterns):
            phrase_matcher.add(match_id, [pattern])

    for i, rule in enumerate(rules):
        if rule["pattern_type"] == "DEP_VERB":
            match_id = f"rule_{i}"
            verb_lemma, e1_dep, e2_dep = rule["pattern_data"]

            pattern = [
                {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": verb_lemma, "POS": "VERB"}},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "e1", "RIGHT_ATTRS": {"DEP": e1_dep}},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "e2", "RIGHT_ATTRS": {"DEP": e2_dep}},
            ]
            dep_matcher.add(match_id, [pattern])

    return token_matcher, phrase_matcher, dep_matcher
