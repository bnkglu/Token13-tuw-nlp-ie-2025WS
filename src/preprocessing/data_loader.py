"""Data loader module for SemEval-2010 Task 8 dataset."""

import re
from typing import List, Dict, Optional


class SemEvalDataLoader:
    """Load and parse SemEval-2010 Task 8 data files."""

    def __init__(self):
        """Initialize the data loader with compiled regex patterns."""
        self.entity_pattern = re.compile(r'<e([12])>(.*?)</e\1>')
        self.quoted_text_pattern = re.compile(r'^\d+\s+"(.+)"$')
        self.relation_pattern = re.compile(r'([A-Za-z\-]+)\((e[12]),(e[12])\)')
        self.tag_pattern = re.compile(r'</?e[12]>')  # For cleaning tags

    def load_file(self, file_path: str, has_labels: bool = True) -> List[Dict]:
        """Load data from a SemEval format file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        examples, i = [], 0
        while i < len(lines):
            sentence_match = self.quoted_text_pattern.match(lines[i])
            if not sentence_match:
                i += 1
                continue
            
            sentence_id = lines[i].split('\t')[0]
            raw_sentence = sentence_match.group(1)
            
            example = {
                'id': int(sentence_id),
                'raw_sentence': raw_sentence,
                'entities': self._extract_entities(raw_sentence),
                'clean_sentence': self.entity_pattern.sub(r'\2', raw_sentence),  # Clean inline
                'relation': None,
                'comment': None
            }
            
            # Parse relation label if present
            if has_labels and i + 1 < len(lines) and not lines[i + 1].startswith('Comment:'):
                example['relation'] = self._parse_relation(lines[i + 1])
                i += 1
            
            # Parse comment if present
            if i + 1 < len(lines) and lines[i + 1].startswith('Comment:'):
                example['comment'] = lines[i + 1].replace('Comment:', '').strip()
                i += 1
            
            examples.append(example)
            i += 1
        
        return examples

    def _extract_entities(self, sentence: str) -> Dict[str, Dict]:
        """Extract entity mentions from tagged sentence."""
        entities = {}
        clean_pos = 0
        
        for match in self.entity_pattern.finditer(sentence):
            entity_id = f"e{match.group(1)}"
            entity_text = match.group(2)
            
            # Calculate position by removing all tags before this match
            prefix = sentence[:match.start()]
            clean_start = len(self.tag_pattern.sub('', prefix))
            
            entities[entity_id] = {
                'text': entity_text,
                'start_char': clean_start,
                'end_char': clean_start + len(entity_text),
                'original_start': match.start(),
                'original_end': match.end()
            }
        
        return entities

    def _parse_relation(self, relation_line: str) -> Optional[Dict]:
        """Parse relation label line."""
        relation_line = relation_line.strip()
        
        if relation_line == "Other":
            return {'type': 'Other', 'direction': None, 'arg1': None, 'arg2': None}
        
        match = self.relation_pattern.match(relation_line)
        if match:
            return {
                'type': match.group(1),
                'direction': f"({match.group(2)},{match.group(3)})",
                'arg1': match.group(2),
                'arg2': match.group(3)
            }
        return None

    def load_train_data(self, file_path: str) -> List[Dict]:
        """Load training data with labels."""
        return self.load_file(file_path, has_labels=True)

    def load_test_data(self, file_path: str) -> List[Dict]:
        """Load test data without labels."""
        return self.load_file(file_path, has_labels=False)

    def load_test_labels(self, file_path: str) -> Dict[int, str]:
        """Load test labels from key file."""
        labels = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    labels[int(parts[0])] = self._parse_relation(parts[1])
        return labels

    def extract_relation_types(self, examples: List[Dict]) -> List[str]:
        """Extract unique relation types from examples."""
        relation_types = {ex['relation']['type'] for ex in examples if ex.get('relation')}
        sorted_types = sorted(rt for rt in relation_types if rt != 'Other')
        return sorted_types + (['Other'] if 'Other' in relation_types else [])

    def extract_all_labels(self, examples: List[Dict]) -> List[str]:
        """Extract all unique relation labels (including directionality)."""
        labels = set()
        for ex in examples:
            if ex.get('relation'):
                rel = ex['relation']
                labels.add('Other' if rel['type'] == 'Other' else f"{rel['type']}{rel['direction']}")
        
        sorted_labels = ['Other'] if 'Other' in labels else []
        sorted_labels.extend(sorted(lbl for lbl in labels if lbl != 'Other'))
        return sorted_labels
