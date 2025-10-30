# Data Preprocessing Documentation

## Overview

This document describes the preprocessing pipeline implemented for the SemEval-2010 Task 8 dataset (Multi-Way Classification of Semantic Relations Between Pairs of Nominals). The preprocessing pipeline transforms raw annotated text into multiple standard formats suitable for relation extraction tasks.

## What Was Done

### 1. Data Loading and Parsing

The raw SemEval data format consists of sentences with entity markers and relation labels:

```
1    "The system has its greatest application in an arrayed <e1>configuration</e1> of antenna <e2>elements</e2>."
Component-Whole(e2,e1)
Comment: Not a collection: there is structure here, organisation.
```

**Data Sources:**
- Training data: `SemEval2010_task8_training/TRAIN_FILE.TXT` (8,000 examples)
- Test data: `SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT` (2,717 examples)

**Note:** We use `TEST_FILE_FULL.TXT` instead of `TEST_FILE.txt` because it includes relation labels and comments in the same format as the training data, enabling comprehensive statistics and analysis.

**Implementation:**
- Regular expression-based parsing to extract:
  - Sentence ID and text
  - Entity mentions (e1 and e2) with their character positions
  - Relation labels with directionality
  - Optional comments
- Entity tags removed from text while preserving character-level alignment
- Both training and test data have the same format with full annotations

**Module:** `src/preprocessing/data_loader.py`

### 2. Text Processing with spaCy

Each sentence was processed using the spaCy `en_core_web_lg` model to extract linguistic features:

**Linguistic Annotations:**
- Tokenization (word-level segmentation)
- Part-of-Speech (POS) tagging
- Morphological analysis (fine-grained tags)
- Dependency parsing (syntactic structure)
- Lemmatization (base word forms)

**Entity Alignment:**
- Entities (e1 and e2) are pre-annotated in the dataset with XML-style tags (e.g., `<e1>configuration</e1>`)
- Entity tags are extracted and removed from the text during data loading
- Entity spans are aligned with spaCy token boundaries using character positions
- Token indices stored for each entity mention
- Dependency paths computed between entity pairs

**Note:** Named Entity Recognition (NER) was not performed, as entities are already annotated in the SemEval-2010 Task 8 dataset.

**Module:** `src/preprocessing/text_processor.py`

### 3. Data Validation

Quality checks performed on the dataset:
- Verification of required fields (ID, sentence, entities, relations)
- Entity presence validation (both e1 and e2)
- Character position correctness
- Relation label validity
- Entity-token alignment verification

**Module:** `src/utils/validators.py`

### 4. Output Formats

The preprocessed data is exported in three standard formats:

#### a. JSON Format
Human-readable and easily parseable format containing:
- Sentence ID and text
- Token-level information (text, lemma, POS, tag, dependency)
- Entity mentions with token indices and character spans
- Relation label with directionality
- Dependency path between entities

**Files:**
- `data/processed/train/train.json`
- `data/processed/test/test.json`

#### b. CoNLL-U Format (UD v2 Compliant)
Standard format for Universal Dependencies with enhanced features:
- One token per line with 10 tab-separated columns
- Metadata comments for sentence ID, text, relation, entities, and comments
- Columns: ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC
- **FEATS**: Morphological features extracted from spaCy (Tense, Number, Person, etc.)
- **DEPS**: Enhanced dependencies (currently unpopulated)
- **MISC**: Entity annotations (Entity=e1, Entity=e2) marked on entity head tokens

**Files:**
- `data/processed/train/train.conllu` (8,000 sentences, 2,604 with comments)
- `data/processed/test/test.conllu` (2,717 sentences, 932 with comments)

#### c. spaCy DocBin Format
Binary format for efficient loading in spaCy pipelines:
- Stores all linguistic annotations
- Fast deserialization
- Compatible with spaCy 3.x

**Files:**
- `data/processed/train/train.spacy`
- `data/processed/test/test.spacy`

**Module:** `src/preprocessing/format_converters.py`

### 5. Statistics and Analysis

Statistics computed for both splits:

**Metrics:**
- Total number of examples
- Sentence length statistics (characters and words)
- Entity length statistics
- Relation distribution (directed - with argument order)
- Relation distribution (undirected - relation types only)
- Dynamically extracted relation types and labels
- Class balance analysis

**Outputs:**
- `data/processed/statistics/train_statistics.json`
- `data/processed/statistics/test_statistics.json`
- Console output with formatted statistics summary

**Module:** `src/utils/statistics.py`

## Dataset Statistics

### Training Data
- Total examples: 8,000
- Average sentence length: 17.20 words (101.79 characters)
- Sentence length range: 3-85 words
- Average entity length: 7.02 characters
- **Relation types: 10** (9 semantic relations + Other)
- **Labels with directionality: 19** (18 directed + Other)

**Undirected Relation Distribution:**
- Other: 1,410 (17.62%)
- Cause-Effect: 1,003 (12.54%)
- Component-Whole: 941 (11.76%)
- Entity-Destination: 845 (10.56%)
- Product-Producer: 717 (8.96%)
- Entity-Origin: 716 (8.95%)
- Member-Collection: 690 (8.62%)
- Message-Topic: 634 (7.92%)
- Content-Container: 540 (6.75%)
- Instrument-Agency: 504 (6.30%)

### Test Data
- Total examples: 2,717
- Average sentence length: 17.25 words (101.42 characters)
- Sentence length range: 4-60 words
- Average entity length: 6.86 characters
- **Relation types: 10** (9 semantic relations + Other)
- **Labels with directionality: 19** (18 directed + Other)

**Undirected Relation Distribution:**
- Other: 454 (16.71%)
- Cause-Effect: 328 (12.07%)
- Component-Whole: 312 (11.48%)
- Entity-Destination: 292 (10.75%)
- Message-Topic: 261 (9.61%)
- Entity-Origin: 258 (9.50%)
- Member-Collection: 233 (8.58%)
- Product-Producer: 231 (8.50%)
- Content-Container: 192 (7.07%)
- Instrument-Agency: 156 (5.74%)

**Note:** Test data is processed from `TEST_FILE_FULL.TXT` which includes relation labels and comments, maintaining the same format as the training data for comprehensive analysis. Both datasets show similar distribution patterns, confirming good train-test split quality.

## How to Use

### Prerequisites

#### Quick Setup
The automated setup script can be run:
```bash
./setup.sh
```

#### Manual Setup
1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Download spaCy model:
```bash
python3 -m spacy download en_core_web_lg
```

**Required Libraries:**
- **spacy** (>=3.7.0) - Industrial-strength NLP library for tokenization, POS tagging, dependency parsing, lemmatization, and morphological features
- **conllu** (>=5.0.0) - Universal Dependencies CoNLL-U parser/serializer for reading and writing CoNLL-U format files
- **tqdm** (>=4.66.0) - Progress bar library for displaying processing progress

### Running the Preprocessing Pipeline

#### Process Both Splits
```bash
python src/preprocess.py --split both
```

#### Process Training Data Only
```bash
python src/preprocess.py --split train
```

#### Process Test Data Only
```bash
python src/preprocess.py --split test
```

### Loading Preprocessed Data

#### Load JSON Format
```python
import json

with open('data/processed/train/train.json', 'r') as f:
    data = json.load(f)

# Access first example
example = data[0]
print(f"ID: {example['id']}")
print(f"Text: {example['text']}")
print(f"Relation: {example['relation']}")
print(f"Entities: {example['entities']}")
```

#### Load CoNLL-U Format
```python
from conllu import parse

# Using the official conllu library (recommended)
with open('data/processed/train/train.conllu', 'r', encoding='utf-8') as f:
    sentences = parse(f.read())

# Access first sentence
sentence = sentences[0]
print(f"Metadata: {sentence.metadata}")
print(f"Tokens: {len(sentence)}")

# Access token information
for token in sentence:
    print(f"{token['form']}\t{token['lemma']}\t{token['upos']}\t{token['feats']}")

# Access metadata
sent_id = sentence.metadata.get('sent_id')
relation = sentence.metadata.get('relation')
comment = sentence.metadata.get('comment')
```

#### Load spaCy DocBin Format
```python
import spacy
from spacy.tokens import DocBin

# Load spaCy model
nlp = spacy.load('en_core_web_lg')

# Load DocBin
doc_bin = DocBin().from_disk('data/processed/train/train.spacy')
docs = list(doc_bin.get_docs(nlp.vocab))

# Access first document
doc = docs[0]
for token in doc:
    print(f"{token.text}\t{token.lemma_}\t{token.pos_}\t{token.dep_}")

# Access relation and entity attributes
print(f"Relation: {doc._.relation}")
print(f"Entity 1: {doc._.e1}")
print(f"Entity 2: {doc._.e2}")
```

## Project Structure

```
Token13-tuw-nlp-ie-2025WS/
├── src/
│   ├── config.py                      # Configuration and paths
│   ├── preprocess.py                  # Main preprocessing script
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── data_loader.py             # SemEval data loader
│   │   ├── text_processor.py          # spaCy processing
│   │   └── format_converters.py       # Export to formats
│   └── utils/
│       ├── __init__.py
│       ├── validators.py              # Data validation
│       └── statistics.py              # Statistics computation
├── data/
│   └── processed/
│       ├── train/                     # Training data outputs
│       ├── test/                      # Test data outputs
│       └── statistics/                # Statistics and plots
├── resources/                         # Raw SemEval data
├── requirements.txt                   # Python dependencies
└── PREPROCESSING.md                   # This document
```