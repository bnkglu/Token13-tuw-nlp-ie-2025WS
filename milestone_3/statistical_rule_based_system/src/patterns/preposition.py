"""
Preposition-structure pattern extractors.

Pattern types new in milestone_3 (data-driven):
- PREP_STRUCT: (prep, gov_lemma, e1_role, e2_role)
- PREP_GOV_LEX: (prep, gov_lexname)
- PREP_ROLES: (prep, e1_role, e2_role)
"""

from ..utils.semantic import extract_prep_structure_features
from .base import PatternExtractorBase


class PrepositionPatternExtractor(PatternExtractorBase):
    """Extract data-driven preposition-structure patterns."""

    @property
    def pattern_types(self) -> list[str]:
        """Pattern types handled by this extractor."""
        return ['PREP_STRUCT', 'PREP_GOV_LEX', 'PREP_ROLES']

    def extract(
        self,
        doc,
        e1_span,
        e2_span,
        relation: str,
        patterns: dict,
    ) -> None:
        """Extract preposition-structure patterns."""
        prep_feats = extract_prep_structure_features(doc, e1_span, e2_span)

        if prep_feats is None:
            return

        # PREP_STRUCT: (prep, gov_lemma, e1_role, e2_role)
        key = prep_feats.to_pattern_key('PREP_STRUCT')
        patterns['PREP_STRUCT'][key][relation] += 1

        # PREP_GOV_LEX: (prep, gov_lexname)
        if prep_feats.gov_lexname:
            key = prep_feats.to_pattern_key('PREP_GOV_LEX')
            patterns['PREP_GOV_LEX'][key][relation] += 1

        # PREP_ROLES: (prep, e1_role, e2_role)
        key = prep_feats.to_pattern_key('PREP_ROLES')
        patterns['PREP_ROLES'][key][relation] += 1
