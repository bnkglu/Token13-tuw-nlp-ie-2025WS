"""Statistics module for dataset analysis."""

import json
from collections import Counter
from typing import List, Dict


class DatasetStatistics:
    """Compute dataset statistics and save to JSON."""

    def compute_statistics(self, examples: List[Dict]) -> Dict:
        """Compute comprehensive dataset statistics using Counter and list comprehensions."""
        relations_directed = [
            'Other' if ex['relation']['type'] == 'Other' 
            else f"{ex['relation']['type']}{ex['relation']['direction']}"
            for ex in examples if ex.get('relation')
        ]
        relations_undirected = [ex['relation']['type'] for ex in examples if ex.get('relation')]
        
        lengths_chars = [len(ex['clean_sentence']) for ex in examples]
        lengths_words = [len(ex['clean_sentence'].split()) for ex in examples]
        entity_lengths = [
            len(ent['text']) 
            for ex in examples 
            for ent in ex['entities'].values()
        ]
        
        return {
            'total_examples': len(examples),
            'relation_distribution_directed': dict(Counter(relations_directed)),
            'relation_distribution_undirected': dict(Counter(relations_undirected)),
            'sentence_length_stats': {
                'avg_chars': sum(lengths_chars) / len(lengths_chars),
                'min_chars': min(lengths_chars),
                'max_chars': max(lengths_chars),
                'avg_words': sum(lengths_words) / len(lengths_words),
                'min_words': min(lengths_words),
                'max_words': max(lengths_words)
            },
            'entity_stats': {
                'avg_entity_length': sum(entity_lengths) / len(entity_lengths) if entity_lengths else 0,
                'min_entity_length': min(entity_lengths) if entity_lengths else 0,
                'max_entity_length': max(entity_lengths) if entity_lengths else 0
            },
            'token_stats': {}
        }

    def save_statistics(self, stats: Dict, output_path: str) -> None:
        """Save statistics to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    def print_summary(self, stats: Dict) -> None:
        """Print a formatted summary of statistics."""
        print(f"\n{'='*60}\nDATASET STATISTICS SUMMARY\n{'='*60}")
        print(f"\nTotal Examples: {stats['total_examples']}")

        sent = stats['sentence_length_stats']
        print(f"\nSentence Length Statistics:")
        print(f"  Average characters: {sent['avg_chars']:.2f}")
        print(f"  Average words: {sent['avg_words']:.2f}")
        print(f"  Range: {sent['min_words']}-{sent['max_words']} words")

        ent = stats['entity_stats']
        print(f"\nEntity Statistics:")
        print(f"  Average entity length: {ent['avg_entity_length']:.2f} chars")

        total = stats['total_examples']
        print("\nRelation Distribution (Undirected):")
        for rel, count in sorted(stats['relation_distribution_undirected'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {rel}: {count} ({count/total*100:.2f}%)")

        print("\nRelation Distribution (Directed - with argument order):")
        for rel, count in sorted(stats['relation_distribution_directed'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {rel}: {count} ({count/total*100:.2f}%)")

        print(f"\n{'='*60}\n")
