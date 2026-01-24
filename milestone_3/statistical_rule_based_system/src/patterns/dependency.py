"""
Dependency pattern extractors.

Pattern types from milestone_2:
- DEP_VERB: Verb lemma + entity dependency roles to that verb
- DEP_LABELS: Simple dependency labels of entity heads
"""

from .base import PatternExtractorBase


class DependencyPatternExtractor(PatternExtractorBase):
    """Extract dependency-based patterns from syntactic structure."""

    @property
    def pattern_types(self) -> list[str]:
        """Pattern types handled by this extractor."""
        return ['DEP_VERB', 'DEP_LABELS']

    def extract(
        self,
        doc,
        e1_span,
        e2_span,
        relation: str,
        patterns: dict,
    ) -> None:
        """Extract dependency patterns."""
        e1_head = e1_span.root
        e2_head = e2_span.root

        # DEP_VERB: Find verbs that connect both entities
        for token in doc:
            if token.pos_ == 'VERB':
                e1_dep_to_verb = None
                e2_dep_to_verb = None

                # Check e1 relation to verb
                if e1_head.head == token:
                    e1_dep_to_verb = e1_head.dep_
                elif e1_head == token:
                    e1_dep_to_verb = 'VERB_IS_E1'

                # Check e2 relation to verb
                if e2_head.head == token:
                    e2_dep_to_verb = e2_head.dep_
                elif e2_head == token:
                    e2_dep_to_verb = 'VERB_IS_E2'

                if e1_dep_to_verb and e2_dep_to_verb:
                    verb_lemma = token.lemma_.lower()
                    key = ('DEP_VERB', verb_lemma, e1_dep_to_verb, e2_dep_to_verb)
                    patterns['DEP_VERB'][key][relation] += 1

        # DEP_LABELS: Simple entity head dependency labels
        key = ('DEP_LABELS', e1_head.dep_, e2_head.dep_)
        patterns['DEP_LABELS'][key][relation] += 1
