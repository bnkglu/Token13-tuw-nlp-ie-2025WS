# Pattern Extraction Module

This module handles the extraction of linguistic patterns from training data. It uses a modular architecture where specific feature extractors are plugged into a central orchestrator.

## Components

### 1. Extractor Orchestrator (`extractor.py`)
- **`PatternExtractor`**: The main class that coordinates the mining process.
- **Responsibilities**:
  - Initializes sub-extractors.
  - Aggregates pattern counts from all sources.
  - Filters patterns by **Precision** and **Support**.
  - Assigns **Priority Scores** (Tiers) to discovered rules.

### 2. Feature Sub-Modules

| File | Pattern Types | Description |
|------|--------------|-------------|
| **`lexical.py`** | `LEMMA`, `BIGRAM`, `PREP`, `ENTITY_POS`, `BEFORE_E1`, `AFTER_E2` | Basic surface-level features. Replicates MS2 baseline logic. |
| **`dependency.py`** | `DEP_VERB`, `DEP_LABELS` | Syntactic dependency paths and edge labels. |
| **`semantic_patterns.py`** | `SYNSET`, `FRAME`, `LEXNAME`, `HYPERNYM` | **(New in MS3)** Abstract semantic features using WordNet and FrameNet. Also handles `PREP_STRUCT_LEXNAME`. |
| **`preposition.py`** | `PREP_STRUCT`, `PREP_ROLES` | Complex prepositional attachment patterns. |

### 3. Base Class (`base.py`)
- Defines the `PatternExtractorBase` interface that all sub-modules must implement.
- Defines the `Rule` dataclass structure.

## Usage Example

```python
from src.patterns.extractor import PatternExtractor

# Initialize
extractor = PatternExtractor(
    min_precision=0.6, 
    min_support=2, 
    use_semantics=True
)

# Mine patterns from processed data
patterns = extractor.extract_patterns(train_data)

# Convert to ranked rules
rules = extractor.filter_and_rank(patterns, prediction_mode='priority_based')
```
