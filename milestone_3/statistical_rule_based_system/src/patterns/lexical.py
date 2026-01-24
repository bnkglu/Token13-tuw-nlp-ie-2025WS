"""
Lexical pattern extractors.

Pattern types from milestone_2:
- LEMMA: Single lemmas between entities
- BIGRAM: Consecutive lemma pairs
- PREP: Simple preposition lemmas
- BEFORE_E1: Context word before entity 1
- AFTER_E2: Context word after entity 2
- ENTITY_POS: POS tag pair of entity roots
"""

from collections import defaultdict

from .base import PatternExtractorBase


class LexicalPatternExtractor(PatternExtractorBase):
    """Extract lexical patterns from between-entity span and context."""

    @property
    def pattern_types(self) -> list[str]:
        """Pattern types handled by this extractor."""
        return ['LEMMA', 'BIGRAM', 'PREP', 'BEFORE_E1', 'AFTER_E2', 'ENTITY_POS']

    def extract(
        self,
        doc,
        e1_span,
        e2_span,
        relation: str,
        patterns: dict,
    ) -> None:
        """Extract all lexical pattern types. Following patterns are extracted:
        - LEMMA: Single lemmas between entities
        - BIGRAM: Consecutive lemma pairs
        - PREP: Simple preposition lemmas
        - BEFORE_E1: Context word before entity 1
        - AFTER_E2: Context word after entity 2
        - ENTITY_POS: POS tag pair of entity roots
        Args:
            doc: spaCy Doc
            e1_span: Entity 1 span
            e2_span: Entity 2 span
            relation: Relation type
            patterns: Dictionary to store patterns
        """
        # Get between-span
        if e1_span.start < e2_span.start:
            between = doc[e1_span.end:e2_span.start]
        else:
            between = doc[e2_span.end:e1_span.start]

        # Milestone 2 alignment:
        # - between_lemmas for general structure (bigrams) includes short words
        # - single lemma rules only use words with length > 2
        all_lemmas = [t.lemma_.lower() for t in between if not t.is_punct]
        
        # LEMMA patterns (only length > 2)
        for lemma in all_lemmas:
            if len(lemma) > 2:
                key = ('LEMMA', lemma)
                patterns['LEMMA'][key][relation] += 1

        # BIGRAM patterns (all lengths)
        for i in range(len(all_lemmas) - 1):
            bigram = (all_lemmas[i], all_lemmas[i + 1])
            key = ('BIGRAM', bigram)
            patterns['BIGRAM'][key][relation] += 1

        # PREP patterns
        for token in between:
            if token.pos_ == 'ADP':
                key = ('PREP', token.lemma_.lower())
                patterns['PREP'][key][relation] += 1

        # BEFORE_E1 patterns
        if e1_span.start > 0:
            before_e1 = doc[e1_span.start - 1]
            if not before_e1.is_punct and len(before_e1.lemma_) > 2:
                key = ('BEFORE_E1', before_e1.lemma_.lower())
                patterns['BEFORE_E1'][key][relation] += 1

        # AFTER_E2 patterns
        if e2_span.end < len(doc):
            after_e2 = doc[e2_span.end]
            if not after_e2.is_punct and len(after_e2.lemma_) > 2:
                key = ('AFTER_E2', after_e2.lemma_.lower())
                patterns['AFTER_E2'][key][relation] += 1

        # ENTITY_POS patterns
        key = ('ENTITY_POS', e1_span.root.pos_, e2_span.root.pos_)
        patterns['ENTITY_POS'][key][relation] += 1
