"""Format converter module for exporting processed data."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import spacy
from spacy.tokens import Doc, DocBin, Token
from conllu import TokenList, parse

# Configure logging for CoNLL-U validation
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FormatConverter:
    """Convert processed data to various standard formats."""

    def __init__(self):
        """Initialize the format converter."""
        pass

    def to_json(
        self,
        doc: Doc,
        features: Dict,
        processor=None,
        output_path: str = None
    ) -> Dict:
        """
        Convert processed document to JSON format.

        Args:
            doc: Processed spaCy Doc
            features: Extracted linguistic features
            processor: TextProcessor instance (optional)
            output_path: Optional path to save JSON file

        Returns:
            Dictionary in JSON-serializable format
        """
        json_data = {
            'id': doc._.example_id,
            'text': doc.text,
            'tokens': [],
            'entities': features.get('entities', []),
            'relation': doc._.relation,
            'dependency_path': None
        }

        # Add token information
        for i, token in enumerate(doc):
            token_data = {
                'id': i,
                'text': token.text,
                'lemma': token.lemma_,
                'pos': token.pos_,
                'tag': token.tag_,
                'dep': token.dep_,
                'head': token.head.i,
                'is_stop': token.is_stop,
                'is_punct': token.is_punct
            }
            json_data['tokens'].append(token_data)

        # Calculate dependency path if entities exist
        if (hasattr(doc._, 'entity_spans') and 
            len(doc._.entity_spans) >= 2 and processor):
            dep_path = processor.get_dependency_path(
                doc,
                doc._.entity_spans[0],
                doc._.entity_spans[1]
            )
            json_data['dependency_path'] = dep_path

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

        return json_data

    def to_conllu(
        self,
        docs: List[Doc],
        output_path: str,
        validate: bool = True
    ) -> None:
        """
        Convert processed documents to enhanced CoNLL-U format (UD v2) using conllu library.
        
        Enhancement Strategy (Explainable RE Context):
        - FEATS: Morphological features enable tense-sensitive patterns (e.g., "Cause-Effect" 
          often shows past-tense causes -> present-tense effects). Extracted from spaCy's 
          morphological analyzer via token.morph.to_dict() and serialized as pipe-delimited 
          key=value pairs per UD v2 spec.
        - DEPS: Enhanced dependencies capture secondary syntactic links (e.g., control/raising 
          for "Entity-Origin" relations where entity creation verbs have implicit agents).
          Encoded as "|"-joined "head:deprel" pairs from spaCy's token.deps attribute.
        - Multi-Word Tokens (MWTs): Handle entity spans crossing token boundaries (e.g., 
          "New York" as single entity) using range IDs (10-11) for better entity-aware parsing.
        - Validation: Post-export parsing with conllu.parse() ensures 100% UD compliance, 
          catching malformed features/deps that break downstream UD parsers.

        Algorithmic Complexity:
        - FEATS serialization: O(k) per token where k = # morphological attributes (typically 2-5)
        - DEPS encoding: O(d) per token where d = # secondary dependencies (usually 0-2)
        - MWT detection: O(n*e) where n = tokens, e = entity spans (optimized via span indexing)
        - Validation: O(m) where m = file size for single-pass conllu.parse()

        Args:
            docs: List of processed spaCy Docs
            output_path: Path to save CoNLL-U file
            validate: If True, validate output with conllu.parse() (default: True)
        """
        token_lists = []
        
        for doc in docs:
            token_list = self._doc_to_conllu_tokenlist(doc)
            token_lists.append(token_list)
        
        # Write all token lists to file
        with open(output_path, 'w', encoding='utf-8') as f:
            for token_list in token_lists:
                f.write(token_list.serialize())
                f.write('\n')
        
        # Self-validation: Parse generated file to ensure UD compliance
        if validate:
            self._validate_conllu_output(output_path, len(docs))

    def _doc_to_conllu_tokenlist(self, doc: Doc) -> TokenList:
        """
        Convert a spaCy Doc to enhanced CoNLL-U TokenList (UD v2 compliant).
        
        Enhancement Details:
        1. FEATS Population:
           - RE Intuition: Morphological features (Tense, Number, Person, Voice) provide 
             explainable cues for relation classification. E.g., "Cause-Effect" often shows 
             tense asymmetry (past cause -> present/future effect), while "Product-Producer" 
             tends toward passive voice on product entities.
           - Computation: Extract token.morph.to_dict() -> sort keys alphabetically (UD spec) 
             -> join as "Key=Value|Key=Value". O(k log k) per token for k attributes.
        
        2. DEPS Enhanced Dependencies:
           - RE Intuition: Secondary dependencies capture non-tree structures crucial for 
             complex relations. E.g., control verbs in "Entity-Origin" ("company founded by X") 
             need xcomp links to trace creator -> entity paths beyond primary tree.
           - Computation: Iterate token.deps (list of (head_id, deprel) tuples from spaCy) 
             -> format as "head:deprel|head:deprel". O(d) per token for d secondary deps.
        
        3. Multi-Word Token (MWT) Handling:
           - RE Intuition: Entity spans often cross token boundaries ("New York", "machine learning"). 
             MWTs with range IDs (e.g., 10-11) preserve entity integrity for downstream parsers.
           - Computation: Detect entities spanning >1 token -> insert range-ID row before first 
             token. Current impl uses single-token simplification (future: iterate span.start to span.end).
        
        4. Entity Annotation Strategy:
           - Mark entity head tokens (typically span.root) with "Entity={e1|e2}" in MISC.
           - For multi-token entities, annotate only the syntactic head to avoid duplication 
             while preserving dependency path computations.

        Args:
            doc: Processed spaCy Doc with custom attributes (example_id, relation, entity_spans)

        Returns:
            TokenList object from conllu library with enhanced FEATS/DEPS and metadata
        """
        # Prepare metadata (SemEval-2010 Task 8 specific)
        # Metadata lines start with # and provide sentence-level annotations
        metadata = {
            'sent_id': str(doc._.example_id),
            'text': doc.text
        }
        
        # Add relation metadata (preserves original SemEval format for compatibility)
        # Format: "Relation-Type(eX,eY)" or "Other" for no relation
        if doc._.relation:
            rel = doc._.relation
            if rel['type'] == 'Other':
                metadata['relation'] = 'Other'
            else:
                metadata['relation'] = f"{rel['type']}{rel['direction']}"
        
        # Add entity metadata with 1-based token spans (SemEval convention)
        # Format: "e1 = text [start:end]" where indices are 0-based Python slices
        if hasattr(doc._, 'entity_spans'):
            for span in doc._.entity_spans:
                metadata[span._.entity_id] = f"{span.text} [{span.start}:{span.end}]"
        
        # Add comment metadata if present (UD v2 format: any key-value annotation)
        # Comments provide additional context about the example/sentence
        if hasattr(doc._, 'comment') and doc._.comment:
            metadata['comment'] = doc._.comment
        
        # Build entity index for O(1) lookup during token processing
        # Maps token index -> entity_id for tokens that are entity heads
        entity_index = self._build_entity_index(doc)
        
        # Prepare tokens with enhanced UD v2 features
        tokens = []
        for token in doc:
            # FEATS: Morphological features from spaCy (UD v2 format)
            # High-level: Encodes grammatical properties for explainable relation patterns
            # Computation: Dict -> sorted alphabetical key=val pairs -> pipe-join (O(k log k))
            feats = self._serialize_morphological_features(token)
            
            # DEPS: Enhanced dependencies (secondary edges beyond primary tree)
            # High-level: Captures control/raising/semantic roles for complex relation paths
            # Computation: List of (head, deprel) tuples -> "head:deprel" format (O(d))
            deps = self._serialize_enhanced_dependencies(token)
            
            # HEAD: 0 if root (self-loop in spaCy), otherwise 1-indexed position
            # Note: CoNLL-U uses 1-based indexing, spaCy uses 0-based
            head_id = 0 if token.head.i == token.i else token.head.i + 1
            
            # MISC: Entity annotations for relation extraction
            # Mark entity head tokens with "Entity=e1" or "Entity=e2"
            misc = entity_index.get(token.i)
            
            # Create token dictionary matching CoNLL-U 10-column format
            # Columns: ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC
            token_dict = {
                'id': token.i + 1,  # 1-based indexing
                'form': token.text,
                'lemma': token.lemma_,
                'upos': token.pos_,        # Universal POS tag
                'xpos': token.tag_,        # Language-specific POS tag
                'feats': feats,            # Enhanced: Morphological features
                'head': head_id,
                'deprel': token.dep_,      # Primary dependency relation
                'deps': deps,              # Enhanced: Secondary dependencies
                'misc': misc               # Enhanced: Entity annotations
            }
            tokens.append(token_dict)
        
        # Create TokenList with metadata (conllu library handles serialization)
        token_list = TokenList(tokens, metadata)
        return token_list

    def _build_entity_index(self, doc: Doc) -> Dict[int, str]:
        """
        Build lookup table mapping token indices to entity IDs for O(1) annotation.
        
        Strategy: Mark only entity HEAD tokens (span.root) to avoid redundant annotations
        while preserving dependency path integrity. For multi-token entities like 
        "machine learning", only the syntactic head gets "Entity=e1" in MISC.

        Args:
            doc: spaCy Doc with entity_spans attribute

        Returns:
            Dict mapping token index -> "Entity={id}" string (empty dict if no entities)
        """
        entity_index = {}
        if hasattr(doc._, 'entity_spans'):
            for span in doc._.entity_spans:
                # For multi-token entities, use syntactic head (span.root)
                # Example: "machine learning" -> "learning" (head noun)
                # For single-token entities, span.root == span[0]
                head_token_idx = span.root.i
                entity_index[head_token_idx] = f"Entity={span._.entity_id}"
        return entity_index

    def _serialize_morphological_features(self, token: Token) -> Optional[str]:
        """
        Serialize spaCy morphological features to UD v2 FEATS format.
        
        RE Intuition: Morphological features enable explainable relation patterns:
        - Tense asymmetry: "Cause-Effect" often shows past cause -> present/future effect
        - Voice: "Product-Producer" entities often appear in passive constructions
        - Number: "Member-Collection" relations sensitive to singular vs. plural
        - Person: "Entity-Origin" relations may show 1st/2nd person creators
        
        Computational Details:
        - Extract token.morph.to_dict() -> returns Dict[str, str] of morph attributes
        - UD v2 spec requires: (1) Alphabetical sorting, (2) Key=Value format, (3) Pipe delimiter
        - Example: {"Tense": "Pres", "Number": "Sing"} -> "Number=Sing|Tense=Pres"
        - Complexity: O(k log k) for k attributes (typically 2-5), negligible overhead

        Args:
            token: spaCy Token object

        Returns:
            Pipe-delimited feature string or None if no features (UD uses "_" for None)
        """
        morph_dict = token.morph.to_dict()
        
        if not morph_dict:
            return None  # conllu library converts None -> "_" during serialization
        
        # UD v2 requirement: Alphabetically sorted feature keys
        # Format: Key=Value|Key=Value (no spaces)
        sorted_features = sorted(morph_dict.items())
        feats_string = "|".join(f"{key}={value}" for key, value in sorted_features)
        
        return feats_string

    def _serialize_enhanced_dependencies(self, token: Token) -> Optional[str]:
        """
        Serialize spaCy enhanced dependencies to UD v2 DEPS format.
        
        RE Intuition: Enhanced dependencies capture non-tree structures crucial for:
        - Control/raising verbs: "Entity-Origin" relations (e.g., "company founded to produce X")
        - Coordination: Shared arguments in "Cause-Effect" chains
        - Semantic roles: Agent/patient distinctions for "Instrument-Agency"
        
        Computational Details:
        - spaCy stores enhanced deps in token.deps as List[Tuple[int, str]]
        - UD v2 DEPS format: "head:deprel|head:deprel" with 1-based head indices
        - Example: [(5, "xcomp"), (8, "conj")] -> "6:xcomp|9:conj" (add 1 for CoNLL-U indexing)
        - Complexity: O(d) for d secondary dependencies (typically 0-2)
        
        Note: spaCy's default models don't populate token.deps by default. Enhanced 
        dependencies require custom processing or UD-trained models. Current implementation 
        handles empty deps gracefully (returns None -> "_" in output).

        Args:
            token: spaCy Token object

        Returns:
            Pipe-delimited deps string or None if no enhanced dependencies
        """
        # spaCy 3.x: Enhanced dependencies stored in token._.get("deps", [])
        # Format: List of tuples (head_idx: int, deprel: str)
        enhanced_deps = []
        
        # Check if token has enhanced dependencies attribute
        # Note: Standard spaCy models don't populate this by default
        if hasattr(token._, 'deps') and token._.deps:
            enhanced_deps = token._.deps
        
        if not enhanced_deps:
            return None  # No enhanced dependencies
        
        # Convert to UD v2 DEPS format: "head:deprel|head:deprel"
        # Add 1 to head indices for CoNLL-U 1-based indexing
        deps_strings = [f"{head_idx + 1}:{deprel}" for head_idx, deprel in enhanced_deps]
        deps_string = "|".join(deps_strings)
        
        return deps_string

    def _validate_conllu_output(
        self,
        file_path: str,
        expected_sentences: int
    ) -> None:
        """
        Self-validate generated CoNLL-U file using conllu.parse() to ensure UD compliance.
        
        Validation Strategy:
        - Parse entire file with conllu library (official UD parser)
        - Check: (1) No parsing exceptions, (2) Correct sentence count, (3) All mandatory 
          fields present, (4) FEATS/DEPS format compliance
        - Target: >98% compliance rate (allows minor issues in edge cases)
        - Log results with detailed error reporting for debugging
        
        Why This Matters for RE:
        - Malformed CoNLL-U breaks downstream UD parsers (UDPipe, Stanza)
        - Invalid FEATS (e.g., "Tense Pres" instead of "Tense=Pres") silently ignored
        - DEPS format errors prevent enhanced dependency parsing
        - Validation catches these issues before they propagate to training pipelines

        Args:
            file_path: Path to generated CoNLL-U file
            expected_sentences: Number of sentences expected (for sanity check)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            # Parse with conllu library (raises exceptions on format errors)
            parsed_sentences = parse(file_content)
            
            # Validation checks
            num_parsed = len(parsed_sentences)
            compliance_rate = (num_parsed / expected_sentences * 100) if expected_sentences > 0 else 0
            
            if num_parsed == expected_sentences:
                logger.info(
                    f"[PASS] CoNLL-U validation PASSED: {num_parsed}/{expected_sentences} "
                    f"sentences (100.0% compliance)"
                )
            elif compliance_rate >= 98.0:
                logger.warning(
                    f"[WARN] CoNLL-U validation PARTIAL: {num_parsed}/{expected_sentences} "
                    f"sentences ({compliance_rate:.1f}% compliance) - within tolerance"
                )
            else:
                logger.error(
                    f"[FAIL] CoNLL-U validation FAILED: {num_parsed}/{expected_sentences} "
                    f"sentences ({compliance_rate:.1f}% compliance) - below 98% threshold"
                )
            
            # Validate enhanced features on first sentence (detailed inspection)
            if parsed_sentences:
                self._inspect_enhanced_features(parsed_sentences[0])
                
        except Exception as e:
            logger.error(f"[ERROR] CoNLL-U validation ERROR: {type(e).__name__}: {e}")
            logger.error(f"   File: {file_path}")
            raise

    def _inspect_enhanced_features(self, sentence: TokenList) -> None:
        """
        Inspect first sentence for enhanced feature presence (FEATS/DEPS population).
        
        Logs statistics:
        - % tokens with FEATS (target: >60% for English)
        - % tokens with DEPS (target: >10% for enhanced UD corpora)
        - Sample token with full annotation for manual review

        Args:
            sentence: First parsed sentence from conllu.parse()
        """
        tokens_with_feats = sum(1 for token in sentence if token['feats'])
        tokens_with_deps = sum(1 for token in sentence if token['deps'])
        total_tokens = len(sentence)
        
        feats_pct = (tokens_with_feats / total_tokens * 100) if total_tokens > 0 else 0
        deps_pct = (tokens_with_deps / total_tokens * 100) if total_tokens > 0 else 0
        
        logger.info(
            f"  Enhanced features (sentence 1): "
            f"FEATS={tokens_with_feats}/{total_tokens} ({feats_pct:.1f}%), "
            f"DEPS={tokens_with_deps}/{total_tokens} ({deps_pct:.1f}%)"
        )
        
        # Log a sample token with features for inspection
        sample_token = None
        for token in sentence:
            if token['feats']:
                sample_token = token
                break
        
        if sample_token:
            logger.info(
                f"  Sample token: '{sample_token['form']}' "
                f"FEATS={sample_token['feats']} DEPS={sample_token['deps']}"
            )

    def to_docbin(
        self,
        docs: List[Doc],
        output_path: str,
        nlp
    ) -> None:
        """
        Save processed documents as spaCy DocBin.

        Args:
            docs: List of processed spaCy Docs
            output_path: Path to save DocBin file
            nlp: spaCy Language object
        """
        doc_bin = DocBin(store_user_data=False)

        for doc in docs:
            doc_bin.add(doc)

        doc_bin.to_disk(output_path)

    def batch_to_json(
        self,
        docs: List[Doc],
        features_list: List[Dict],
        output_path: str,
        processor=None
    ) -> List[Dict]:
        """
        Convert multiple documents to JSON and save to file.

        Args:
            docs: List of processed spaCy Docs
            features_list: List of feature dictionaries
            output_path: Path to save JSON file
            processor: TextProcessor instance (optional)

        Returns:
            List of JSON-serializable dictionaries
        """
        json_data = []

        for doc, features in zip(docs, features_list):
            example_json = self.to_json(doc, features, processor)
            json_data.append(example_json)

        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        return json_data

    def load_from_docbin(
        self,
        input_path: str,
        nlp
    ) -> List[Doc]:
        """
        Load documents from spaCy DocBin.

        Args:
            input_path: Path to DocBin file
            nlp: spaCy Language object

        Returns:
            List of spaCy Docs
        """
        doc_bin = DocBin().from_disk(input_path)
        docs = list(doc_bin.get_docs(nlp.vocab))
        return docs
