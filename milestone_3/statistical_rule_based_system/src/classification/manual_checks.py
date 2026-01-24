"""Manual rule verification logic."""

from ..utils.semantic import (
    extract_prep_structure_features,
    get_frames_for_verb,
    get_lexname,
    get_shallow_hypernym,
    get_synset_name,
)


def check_rule_match_manual(
    rule: dict,
    doc,
    between_span,
    e1_span,
    e2_span,
    e1_head,
    e2_head,
) -> bool:
    """
    Check if a rule matches the given sample (manual checks).

    Args:
        rule: Rule dictionary.
        doc: spaCy Doc.
        between_span: Span between entities.
        e1_span: Entity 1 span.
        e2_span: Entity 2 span.
        e1_head: Root token of entity 1.
        e2_head: Root token of entity 2.

    Returns:
        True if the rule matches.
    """
    pattern_type = rule["pattern_type"]
    pattern_data = rule["pattern_data"]

    # Manual structural patterns
    # 1. BEFORE_E1
    if pattern_type == "BEFORE_E1" and e1_span.start > 0:
        token = doc[e1_span.start - 1]
        if not token.is_punct and len(token.lemma_) > 2:
            if token.lemma_.lower() == pattern_data[0]:
                return True

    # 2. AFTER_E2
    if pattern_type == "AFTER_E2" and e2_span.end < len(doc):
        token = doc[e2_span.end]
        if not token.is_punct and len(token.lemma_) > 2:
            if token.lemma_.lower() == pattern_data[0]:
                return True

    elif pattern_type == "ENTITY_POS":
        if e1_head.pos_ == pattern_data[0] and e2_head.pos_ == pattern_data[1]:
            return True

    elif pattern_type == "DEP_LABELS":
        if e1_head.dep_ == pattern_data[0] and e2_head.dep_ == pattern_data[1]:
            return True

    # Semantic patterns
    elif pattern_type == "SYNSET":
        for token in between_span:
            if token.pos_ == "VERB":
                synset = get_synset_name(token.lemma_.lower(), "v")
                if synset == pattern_data[0]:
                    return True

    elif pattern_type == "LEXNAME":
        for token in between_span:
            if token.pos_ == "VERB":
                lexname = get_lexname(token.lemma_.lower(), "v")
                if lexname == pattern_data[0]:
                    return True

    elif pattern_type == "FRAME":
        for token in between_span:
            if token.pos_ == "VERB":
                frames = get_frames_for_verb(token.lemma_.lower())
                if pattern_data[0] in frames:
                    return True

    elif pattern_type == "HYPERNYM":
        e1_hyp = get_shallow_hypernym(e1_span.root.lemma_, "n")
        e2_hyp = get_shallow_hypernym(e2_span.root.lemma_, "n")
        if e1_hyp == pattern_data[0] and e2_hyp == pattern_data[1]:
            return True

    elif pattern_type in ("PREP_STRUCT", "PREP_GOV_LEX", "PREP_ROLES", "PREP_STRUCT_LEXNAME", "PREP_STRUCT_HYPERNYM"):
        prep_feats = extract_prep_structure_features(doc, e1_span, e2_span)
        if prep_feats:
            # Base data from structure
            struct_data = list(prep_feats.to_pattern_key("PREP_STRUCT")[1:])
            
            if pattern_type == "PREP_STRUCT_LEXNAME":
                e1_lex = get_lexname(e1_span.root.lemma_, 'n')
                e2_lex = get_lexname(e2_span.root.lemma_, 'n')
                if e1_lex and e2_lex:
                    candidate_data = struct_data + [e1_lex, e2_lex]
                    if tuple(candidate_data) == tuple(pattern_data):
                        return True
            
            elif pattern_type == "PREP_STRUCT_HYPERNYM":
                e1_hyp = get_shallow_hypernym(e1_span.root.lemma_, 'n')
                e2_hyp = get_shallow_hypernym(e2_span.root.lemma_, 'n')
                if e1_hyp and e2_hyp:
                    candidate_data = struct_data + [e1_hyp, e2_hyp]
                    if tuple(candidate_data) == tuple(pattern_data):
                        return True
            
            else:
                # Standard Prep patterns
                key = prep_feats.to_pattern_key(pattern_type)
                if key[1:] == tuple(pattern_data):
                    return True

    return False
