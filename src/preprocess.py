"""Main preprocessing script for SemEval-2010 Task 8 dataset."""

import sys
from pathlib import Path
import argparse
from tqdm import tqdm

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    RAW_TRAIN_FILE, RAW_TEST_FILE, TEST_KEY_FILE,
    TRAIN_OUTPUT_DIR, TEST_OUTPUT_DIR, STATS_OUTPUT_DIR,
    SPACY_MODEL
)
from preprocessing.data_loader import SemEvalDataLoader
from preprocessing.text_processor import TextProcessor
from preprocessing.format_converters import FormatConverter
from utils.validators import DataValidator
from utils.statistics import DatasetStatistics


def preprocess_dataset(
    input_file: str,
    output_dir: Path,
    split_name: str,
    has_labels: bool = True,
    label_file: str = None
):
    """
    Preprocess a dataset split.

    Args:
        input_file: Path to input data file
        output_dir: Directory for output files
        split_name: Name of the split (train/test)
        has_labels: Whether the file contains labels
        label_file: Optional path to label file
    """
    print(f"\n{'=' * 60}")
    print(f"Processing {split_name} split")
    print(f"{'=' * 60}\n")

    # Initialize components
    loader = SemEvalDataLoader()
    processor = TextProcessor(model_name=SPACY_MODEL)
    converter = FormatConverter()
    stats_calculator = DatasetStatistics()

    # Load data
    print(f"Loading data from {input_file}...")
    if has_labels:
        examples = loader.load_file(input_file, has_labels=True)
    else:
        examples = loader.load_file(input_file, has_labels=False)
        if label_file:
            print(f"Loading labels from {label_file}...")
            labels = loader.load_test_labels(label_file)
            # Merge labels with examples
            for example in examples:
                if example['id'] in labels:
                    example['relation'] = labels[example['id']]

    print(f"Loaded {len(examples)} examples")

    # Extract relation types dynamically from the data
    relation_types = loader.extract_relation_types(examples)
    all_labels = loader.extract_all_labels(examples)
    
    print(f"Found {len(relation_types)} unique relation types:")
    for rel_type in relation_types:
        print(f"  - {rel_type}")
    print(f"Total labels (with directionality): {len(all_labels)}")

    # Initialize validator with extracted relation types
    validator = DataValidator(valid_relations=relation_types)

    # Validate data
    print("\nValidating data...")
    validation_results = validator.validate_dataset(examples)
    print(
        f"Valid examples: {validation_results['valid_examples']}/"
        f"{validation_results['total_examples']}"
    )

    if validation_results['invalid_examples'] > 0:
        print(f"Warning: {validation_results['invalid_examples']} "
              f"invalid examples found")

    # Process with spaCy
    print("\nProcessing text with spaCy...")
    docs = []
    features_list = []

    for example in tqdm(examples, desc="Processing"):
        doc = processor.process(example)
        features = processor.extract_features(doc)
        docs.append(doc)
        features_list.append(features)

    # Export to different formats
    print("\nExporting to multiple formats...")

    # JSON
    json_path = output_dir / f"{split_name}.json"
    print(f"  Saving JSON to {json_path}")
    converter.batch_to_json(docs, features_list, str(json_path), processor)

    # CoNLL-U
    conllu_path = output_dir / f"{split_name}.conllu"
    print(f"  Saving CoNLL-U to {conllu_path}")
    converter.to_conllu(docs, str(conllu_path))

    # spaCy DocBin
    docbin_path = output_dir / f"{split_name}.spacy"
    print(f"  Saving DocBin to {docbin_path}")
    converter.to_docbin(docs, str(docbin_path), processor.nlp)

    # Compute and save statistics
    print("\nComputing statistics...")
    stats = stats_calculator.compute_statistics(examples)
    
    # Add relation type information to statistics
    stats['relation_types'] = relation_types
    stats['all_labels'] = all_labels
    stats['num_relation_types'] = len(relation_types)
    stats['num_labels'] = len(all_labels)

    stats_path = STATS_OUTPUT_DIR / f"{split_name}_statistics.json"
    stats_calculator.save_statistics(stats, str(stats_path))
    print(f"  Saved statistics to {stats_path}")

    # Print summary
    stats_calculator.print_summary(stats)

    print(f"\n{split_name.capitalize()} preprocessing complete!")


def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description='Preprocess SemEval-2010 Task 8 dataset'
    )
    parser.add_argument(
        '--split',
        choices=['train', 'test', 'both'],
        default='both',
        help='Which split to process'
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("SemEval-2010 Task 8 Data Preprocessing Pipeline")
    print("=" * 60)

    # Process training data
    if args.split in ['train', 'both']:
        preprocess_dataset(
            input_file=str(RAW_TRAIN_FILE),
            output_dir=TRAIN_OUTPUT_DIR,
            split_name='train',
            has_labels=True
        )

    # Process test data
    if args.split in ['test', 'both']:
        preprocess_dataset(
            input_file=str(RAW_TEST_FILE),
            output_dir=TEST_OUTPUT_DIR,
            split_name='test',
            has_labels=True
        )

    print("\n" + "=" * 60)
    print("ALL PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\nProcessed data saved to:")
    print(f"  Training: {TRAIN_OUTPUT_DIR}")
    print(f"  Test: {TEST_OUTPUT_DIR}")
    print(f"  Statistics: {STATS_OUTPUT_DIR}")
    print("\n")


if __name__ == '__main__':
    main()
