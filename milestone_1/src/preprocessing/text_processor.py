"""Text processing module using spaCy."""

import spacy
from spacy.tokens import Doc, Span
from typing import List, Dict
import warnings


class TextProcessor:
    """Process text using spaCy for linguistic annotation."""

    def __init__(self, model_name: str = "en_core_web_lg"):
        """Initialize the text processor with a spaCy model."""
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            warnings.warn(f"Model {model_name} not found. Run: python -m spacy download {model_name}")
            raise

        # Register custom extensions (idempotent)
        for ext, default in [("entity_id", None)]:
            if not Span.has_extension(ext):
                Span.set_extension(ext, default=default)
        for ext, default in [("relation", None), ("example_id", None), ("entity_spans", []), ("comment", None)]:
            if not Doc.has_extension(ext):
                Doc.set_extension(ext, default=default)

    def process(self, example: Dict) -> Doc:
        """Process a single example with spaCy and align entities."""
        doc = self.nlp(example['clean_sentence'])
        doc._.example_id = example['id']
        doc._.relation = example.get('relation')
        doc._.comment = example.get('comment')
        doc._.entity_spans = self._align_entities(doc, example['entities'])
        return doc

    def _align_entities(self, doc: Doc, entities: Dict[str, Dict]) -> List[Span]:
        """Align entity spans with token boundaries using spaCy's char_span."""
        entity_spans = []
        for entity_id, entity_info in entities.items():
            # Use char_span with expand mode for robust alignment
            span = doc.char_span(entity_info['start_char'], entity_info['end_char'], alignment_mode="expand")
            if span:
                span._.entity_id = entity_id
                entity_spans.append(span)
        return entity_spans

    def extract_features(self, doc: Doc) -> Dict:
        """Extract linguistic features from processed document."""
        features = {
            'tokens': [t.text for t in doc],
            'pos_tags': [t.pos_ for t in doc],
            'tags': [t.tag_ for t in doc],
            'lemmas': [t.lemma_ for t in doc],
            'dependencies': [
                {'token_id': t.i, 'dep': t.dep_, 'head_id': t.head.i, 'head_text': t.head.text}
                for t in doc
            ],
            'entities': [
                {
                    'entity_id': span._.entity_id,
                    'text': span.text,
                    'token_ids': list(range(span.start, span.end)),
                    'start_token': span.start,
                    'end_token': span.end,
                    'start_char': span.start_char,
                    'end_char': span.end_char
                }
                for span in getattr(doc._, 'entity_spans', [])
            ]
        }
        return features

    def get_dependency_path(self, doc: Doc, entity1_span: Span, entity2_span: Span) -> List[int]:
        """Get shortest dependency path between two entities using networkx-style approach."""
        from collections import deque
        
        token1, token2 = entity1_span.root, entity2_span.root
        
        # Build paths to root for both tokens
        path1 = self._path_to_root(token1)
        path2 = self._path_to_root(token2)
        
        # Find lowest common ancestor (LCA)
        path1_set = set(t.i for t in path1)
        lca = next((t for t in path2 if t.i in path1_set), None)
        
        if not lca:
            return []
        
        # Build path: token1 -> lca -> token2
        path1_to_lca = [t.i for t in path1 if t != lca][:path1.index(lca)]
        path2_to_lca = [t.i for t in path2 if t != lca][:path2.index(lca)]
        
        return path1_to_lca + [lca.i] + list(reversed(path2_to_lca))

    def _path_to_root(self, token) -> List:
        """Get path from token to root with cycle detection."""
        path, visited = [], set()
        while token.dep_ != "ROOT" and token.i not in visited:
            path.append(token)
            visited.add(token.i)
            if token.head.i == token.i:
                break
            token = token.head
        path.append(token)
        return path

    def process_batch(self, examples: List[Dict], batch_size: int = 32) -> List[Doc]:
        """Process multiple examples efficiently using spaCy's pipe."""
        docs = []
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            texts = [ex['clean_sentence'] for ex in batch]
            
            for doc, example in zip(self.nlp.pipe(texts), batch):
                doc._.example_id = example['id']
                doc._.relation = example.get('relation')
                doc._.comment = example.get('comment')
                doc._.entity_spans = self._align_entities(doc, example['entities'])
                docs.append(doc)
        return docs
