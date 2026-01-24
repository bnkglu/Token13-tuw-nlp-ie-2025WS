"""
Semantic pattern extractors using WordNet and FrameNet.

Pattern types new in milestone_3:
- SYNSET: WordNet synset names for verbs
- LEXNAME: WordNet lexicographer categories
- HYPERNYM: Shallow hypernyms of entity heads
- FRAME: FrameNet frames evoked by verbs
"""

from ..utils.semantic import (
    extract_prep_structure_features,
    get_frames_for_verb,
    get_lexname,
    get_shallow_hypernym,
    get_synset_name,
)
from .base import PatternExtractorBase


class SemanticPatternExtractor(PatternExtractorBase):
    """Extract semantic patterns using WordNet and FrameNet."""

    def __init__(self):
        """Initialize with empty verb/relation tracking for frame mapper."""
        self.verb_lemmas: list[str] = []
        self.relations: list[str] = []

    @property
    def pattern_types(self) -> list[str]:
        """Pattern types handled by this extractor."""
        return ['SYNSET', 'LEXNAME', 'HYPERNYM', 'FRAME', 
                'PREP_STRUCT_LEXNAME', 'PREP_STRUCT_HYPERNYM']

    def extract(
        self,
        doc,
        e1_span,
        e2_span,
        relation: str,
        patterns: dict,
    ) -> None:
        """Extract semantic patterns."""
        # Get between-span
        if e1_span.start < e2_span.start:
            between = doc[e1_span.end:e2_span.start]
        else:
            between = doc[e2_span.end:e1_span.start]

        # SYNSET, LEXNAME, FRAME from verbs
        for token in between:
            if token.pos_ == 'VERB':
                verb_lemma = token.lemma_.lower()

                # SYNSET pattern
                synset = get_synset_name(verb_lemma, 'v')
                if synset:
                    key = ('SYNSET', synset)
                    patterns['SYNSET'][key][relation] += 1
                    self.verb_lemmas.append(verb_lemma)
                    self.relations.append(relation)

                # LEXNAME pattern
                lexname = get_lexname(verb_lemma, 'v')
                if lexname:
                    key = ('LEXNAME', lexname)
                    patterns['LEXNAME'][key][relation] += 1

                # FRAME patterns
                frames = get_frames_for_verb(verb_lemma)
                for frame in frames:
                    key = ('FRAME', frame)
                    patterns['FRAME'][key][relation] += 1

        # HYPERNYM pattern from entity heads
        e1_hyp = get_shallow_hypernym(e1_span.root.lemma_, 'n')
        e2_hyp = get_shallow_hypernym(e2_span.root.lemma_, 'n')
        if e1_hyp and e2_hyp:
            key = ('HYPERNYM', e1_hyp, e2_hyp)
            patterns['HYPERNYM'][key][relation] += 1

        # PREP_STRUCT_LEXNAME & PREP_STRUCT_HYPERNYM
        # Combines prepositional structure with entity semantics
        prep_feats = extract_prep_structure_features(doc, e1_span, e2_span)
        if prep_feats:
            # 1. PREP_STRUCT_LEXNAME
            e1_lex = get_lexname(e1_span.root.lemma_, 'n')
            e2_lex = get_lexname(e2_span.root.lemma_, 'n')
            if e1_lex and e2_lex:
                # Key: (prep, gov, e1_role, e2_role, e1_lex, e2_lex)
                data = list(prep_feats.to_pattern_key("PREP_STRUCT")[1:])
                data.extend([e1_lex, e2_lex])
                key = ('PREP_STRUCT_LEXNAME', *data)
                patterns['PREP_STRUCT_LEXNAME'][key][relation] += 1

            # 2. PREP_STRUCT_HYPERNYM
            if e1_hyp and e2_hyp:
                # Key: (prep, gov, e1_role, e2_role, e1_hyp, e2_hyp)
                data = list(prep_feats.to_pattern_key("PREP_STRUCT")[1:])
                data.extend([e1_hyp, e2_hyp])
                key = ('PREP_STRUCT_HYPERNYM', *data)
                patterns['PREP_STRUCT_HYPERNYM'][key][relation] += 1

    def get_verb_relation_pairs(self) -> tuple[list[str], list[str]]:
        """Return collected verb-relation pairs for frame mapper training."""
        return self.verb_lemmas, self.relations

    def reset(self) -> None:
        """Reset collected data for new extraction run."""
        self.verb_lemmas = []
        self.relations = []
