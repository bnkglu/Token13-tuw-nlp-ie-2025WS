# Preprocessing Source Code

This directory contains the implementation of the data preprocessing pipeline for SemEval-2010 Task 8.

## Usage

Run the main preprocessing script:

```bash
python src/preprocess.py --split both
```

Options:
- `--split train` - Process only training data
- `--split test` - Process only test data
- `--split both` - Process both splits (default)

## Required Libraries

Dependencies are listed in the repository root requirements.txt. See `../requirements.txt` for the full list and install with:

```bash
pip install -r ../requirements.txt
python3 -m spacy download en_core_web_lg
```

## Module Overview

### Directory Structure

```
src/
├── config.py                      # Configuration and paths
├── preprocess.py                  # Main preprocessing script
├── preprocessing/
│   ├── __init__.py
│   ├── data_loader.py             # SemEval data loader
│   ├── text_processor.py          # spaCy processing
│   └── format_converters.py       # Export to formats
└── utils/
    ├── __init__.py
    ├── validators.py              # Data validation
    └── statistics.py              # Statistics computation
```

### Outputs (generated in data/processed/):

### Core Modules

- **config.py** - Configuration settings, file paths, and constants
- **preprocess.py** - Main script that orchestrates the entire pipeline

### preprocessing/

Contains modules for loading and processing the data:

- **data_loader.py** - Parses SemEval format files, extracts entities and relations
- **text_processor.py** - spaCy-based linguistic processing (tokenization, POS, dependencies, lemmatization)
- **format_converters.py** - Exports data to JSON, CoNLL-U (UD v2 with FEATS), and spaCy DocBin formats

### utils/

Helper utilities for validation and analysis:

- **validators.py** - Data quality checks and validation
- **statistics.py** - Dataset statistics computation and visualization

## Features

### Enhanced CoNLL-U Export (UD v2 Compliant)
- **FEATS**: Morphological features extracted from spaCy (Tense, Number, Person, etc.)
- **DEPS**: Enhanced dependencies (currently unpopulated)
- **MISC**: Entity annotations (Entity=e1, Entity=e2) marked on entity head tokens
- **Metadata**: Sentence ID, text, relation, entities, and comments

### Output Formats
1. **JSON** - Complete annotations with all features
2. **CoNLL-U** - Universal Dependencies v2 format with morphological features and comments
3. **DocBin** - spaCy binary format for efficient loading

## How It Works

### Pipeline Flow

1. **Data Loading** (`data_loader.py`)
   - Reads SemEval-2010 Task 8 formatted text files
   - Parses sentences with entity markers (`<e1>`, `</e1>`, `<e2>`, `</e2>`)
   - Extracts entity positions, text, and relation labels
   - Returns structured data with clean sentences and entity metadata

2. **Text Processing** (`text_processor.py`)
   - Processes sentences using spaCy's NLP pipeline
   - Aligns extracted entities with spaCy tokens using character spans
   - Extracts linguistic features: POS tags, dependencies, lemmas, morphological features
   - Computes dependency paths between entity pairs
   - Stores all annotations in spaCy Doc objects with custom extensions

3. **Format Conversion** (`format_converters.py`)
   - Exports processed data to multiple formats:
     - **JSON**: Human-readable format with all annotations
     - **CoNLL-U**: UD v2 format with FEATS, DEPS, MISC columns
     - **DocBin**: Efficient spaCy binary format for model training
   - Adds metadata (relation, entities, comments) to CoNLL-U files

4. **Validation & Statistics** (`validators.py`, `statistics.py`)
   - Validates entity alignment and relation labels
   - Computes dataset statistics (relation distribution, sentence lengths)
   - Outputs statistics to JSON files for analysis

### File Details

#### config.py
- **Purpose**: Central configuration management
- **Contents**: File paths, spaCy model name, relation types, output directories
- **Usage**: Imported by all other modules to access shared settings

#### preprocess.py
- **Purpose**: Main orchestration script
- **Workflow**:
  1. Loads raw data using `DataLoader`
  2. Processes sentences with `TextProcessor` (batch processing)
  3. Validates processed data with `DataValidator`
  4. Exports to all formats using `FormatConverter`
  5. Computes and saves statistics
- **Arguments**: `--split` to choose train/test/both

#### preprocessing/data_loader.py
- **Purpose**: Parse SemEval format files
- **Key Methods**:
  - `load_file()`: Reads and parses raw text files
  - `_extract_entities()`: Finds entity positions and text using regex
  - `_parse_relation()`: Extracts relation labels (e.g., "Cause-Effect(e1,e2)")
  - `extract_relation_types()`: Gets unique relation labels from dataset
- **Output**: Dictionary with `id`, `sentence`, `e1`, `e2`, `relation`, `comment`

#### preprocessing/text_processor.py
- **Purpose**: Apply spaCy NLP pipeline
- **Key Methods**:
  - `process()`: Processes single sentence, aligns entities
  - `_align_entities()`: Maps entity char spans to spaCy token indices
  - `extract_features()`: Extracts POS, lemma, morph features per token
  - `get_dependency_path()`: Computes shortest path between entities in dependency tree
  - `process_batch()`: Efficient batch processing using spaCy's pipe
- **Custom Extensions**: Adds `e1`, `e2`, `relation`, `comment` to Doc objects

#### preprocessing/format_converters.py
- **Purpose**: Export processed data to standard formats
- **Key Methods**:
  - `to_json()`: Converts Doc objects to JSON with all features
  - `to_conllu()`: Exports to Universal Dependencies format with metadata
  - `to_docbin()`: Saves spaCy DocBin for efficient loading
  - `_doc_to_conllu_tokenlist()`: Builds CoNLL-U TokenList with FEATS, MISC
- **CoNLL-U Metadata**: Includes `# sent_id`, `# text`, `# relation`, `# e1`, `# e2`, `# comment`

#### utils/validators.py
- **Purpose**: Data quality assurance
- **Key Methods**:
  - `validate_example()`: Checks single example for entity alignment issues
  - `validate_dataset()`: Validates entire dataset, logs failures
  - `check_entity_alignment()`: Verifies entities align with spaCy tokens
- **Checks**: Relation validity, entity presence, token alignment

#### utils/statistics.py
- **Purpose**: Dataset analysis and visualization
- **Key Methods**:
  - `compute_statistics()`: Calculates relation distributions, sentence lengths
  - `save_statistics()`: Exports statistics to JSON files
  - `print_summary()`: Displays statistics to console
- **Outputs**: JSON statistics files