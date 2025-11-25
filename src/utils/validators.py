"""Validation utilities for data quality checks."""

from typing import List, Dict


class DataValidator:
    """Validate preprocessed data for quality and correctness."""

    def __init__(self, valid_relations: List[str]):
        """Initialize the validator with valid relation types."""
        self.valid_relations = set(valid_relations)

    def validate_example(self, example: Dict) -> Dict:
        """Validate a single example."""
        issues = []
        
        # Check required fields
        for field in ['id', 'clean_sentence', 'entities']:
            if field not in example:
                issues.append(f"Missing required field: {field}")
        
        # Check entities
        if 'entities' in example:
            for entity_id in ['e1', 'e2']:
                if entity_id not in example['entities']:
                    issues.append(f"Missing entity {entity_id}")
                elif not all(k in example['entities'][entity_id] for k in ['start_char', 'end_char']):
                    issues.append(f"Entity {entity_id} missing position fields")
        
        # Check relation
        if example.get('relation') and example['relation']['type'] not in ['Other'] + list(self.valid_relations):
            issues.append(f"Invalid relation type: {example['relation']['type']}")
        
        return {'valid': len(issues) == 0, 'issues': issues}

    def validate_dataset(self, examples: List[Dict]) -> Dict:
        """Validate entire dataset."""
        results = [self.validate_example(ex) for ex in examples]
        valid_count = sum(1 for r in results if r['valid'])
        invalid_examples = [
            {'id': ex.get('id'), 'issues': r['issues']} 
            for ex, r in zip(examples, results) if not r['valid']
        ]
        
        return {
            'total_examples': len(examples),
            'valid_examples': valid_count,
            'invalid_examples': len(examples) - valid_count,
            'validation_rate': valid_count / len(examples) if examples else 0,
            'issues': invalid_examples
        }

    def check_entity_alignment(self, example: Dict, tokens: List[str]) -> bool:
        """Check if entities align properly with sentence text."""
        sentence = example['clean_sentence']
        return all(
            sentence[ent['start_char']:ent['end_char']] == ent['text']
            for ent in example['entities'].values()
        )
