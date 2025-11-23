# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **Token13's repository** for the TU Wien NLP Information Extraction course (Winter Semester 2025). The project focuses on **Topic 8: Explainable Relation Extraction** using the **SemEval 2010 Task 8** dataset.

### Project Goals
- Develop rule-based and ML-based systems for relation extraction
- Build explainable models that can identify semantic relationships between entity pairs
- Evaluate methods both quantitatively (precision/recall) and qualitatively

### Key Milestones
- **Milestone 1** (Nov 2, 2025): Dataset preprocessing and storage in standard format
- **Milestone 2** (Nov 30, 2025): Multiple baseline implementations (ML + rule-based)
- **Final Submission** (Jan 25, 2026): Complete solution with code, management summary, and presentation

## Dataset

**Primary Dataset**: SemEval 2010 Task 8 - Multi-Way Classification of Semantic Relations Between Pairs of Nominals

**Location**: `resources/` directory contains the original SemEval data and official scorer
- `resources/SemEval2010_task8_all_data 2/`
- `resources/SemEval2010_task8_data_release 2/`
- `resources/SemEval2010_task8_scorer-v1.2/` - Official evaluation scripts

**Data Organization**:
- `data/raw/` - Original unprocessed data
- `data/preprocessed/` - Cleaned and processed data
- `src/data_preparation/` - Preprocessing scripts

## Code Architecture

### Current Structure

```
src/
└── data_preparation/
    └── data_preprocessing.ipynb

data/
├── raw/          # Original dataset files
└── preprocessed/ # Processed data ready for modeling

resources/
└── SemEval2010_task8_*  # Official task data and evaluation tools
```

### Notebook-Based Workflow

The project currently uses Jupyter notebooks for experimentation:
- `rel_ext_01_task.ipynb` - Distant supervision relation extraction task definition (reference implementation from CS224u)
- `rel_ext_02_experiments.ipynb` - ML experiments with classifiers and feature engineering (reference implementation from CS224u)
- `src/data_preparation/data_preprocessing.ipynb` - Dataset preprocessing pipeline

**Important**: Per project requirements, extensive use of Jupyter notebooks is **discouraged**. The codebase should be migrated to properly organized Python modules with clear separation of concerns.

### Expected Architecture (To Be Implemented)

The codebase should evolve toward:
```
src/
├── data_preparation/   # Data loading and preprocessing
├── models/            # ML and rule-based models
│   ├── baselines/     # Baseline implementations
│   ├── ml_models/     # Machine learning approaches
│   └── rule_based/    # Rule-based extraction systems
├── features/          # Feature extraction utilities
├── evaluation/        # Evaluation metrics and analysis
└── utils/             # Shared utilities
```

## Development Commands

### Data Preprocessing
Currently implemented in notebooks - should be migrated to scripts.

### Testing and Evaluation

**Official SemEval Scorer** (Perl script):
```bash
# Check output format
perl resources/SemEval2010_task8_scorer-v1.2/semeval2010_task8_format_checker.pl <PROPOSED_ANSWERS>

# Score predictions
perl resources/SemEval2010_task8_scorer-v1.2/semeval2010_task8_scorer-v1.2.pl <PROPOSED_ANSWERS> <ANSWER_KEY> > results.txt
```

**Expected format**: Tab-separated `<SENT_ID>\t<RELATION>` per line
- Example: `1\tComponent-Whole(e2,e1)`
- Official metric: Macro-averaged F1-score for 9+1 way classification with directionality

### Linting and Formatting
```bash
# Check code style
flake8 src/
ruff check src/

# Format code
black src/
ruff format src/

# Type checking
mypy src/
```

## Code Style Requirements

Per `/init` command specifications:

### Python Style
- **Line length**: 100 characters
- **Python version**: 3.11+
- **Imports**: Absolute imports, PEP-compliant with isort (profile=black)
- **Strings**: Double quotes, f-strings for interpolation
- **Docstrings**: NumPy style with type hints
- **Type hints**: Required for all function parameters and return types
- **Spacing**: 4 spaces, no trailing whitespace, spaces around operators

### Code Organization
- **KISS principle**: Keep it simple, avoid over-engineering
- **Separation of concerns**: Separate business logic from CLI code
- **Module structure**: Create folders for new modules with `__init__.py` automatically
- **Import location**: Always at top of file, never inside functions/classes

### Testing
- **Framework**: pytest (not unittest classes)
- **Style**: Function-based tests, not class-based
- **Fixtures**: Use pytest fixtures for setup/teardown and data
- **Parametrization**: Use `pytest.mark.parametrize` for parameterized tests

### Special Behaviors
- **Auto-fix linting**: Fix flake8/ruff/black/isort/mypy errors without asking
- **Linear flow**: Limit helper functions in scripts; prefer linear data flow
- **Help strings**: Never include default values in help text
- **Commit messages**: Straightforward summaries without extraneous details

### Alerts (macOS)
When a long task completes:
```bash
# Single beep
echo -ne '\007'

# Dialog notification
osascript -e 'tell application "System Events" to display dialog "Done with task X"'
```

## Project-Specific Considerations

### Relation Extraction Approach

The reference notebooks demonstrate **distant supervision** methodology:
1. Combine corpus (entity mentions in text) with knowledge base (known relations)
2. Use KB to generate labels, corpus to generate features
3. Formulate as multi-label binary classification (one classifier per relation)
4. Feature engineering from text between entity mentions

### Key Technical Concepts

**Evaluation Metrics**:
- Primary: F₀.₅-score (weights precision 2x more than recall)
- Aggregate: Macro-averaged across relations
- Precision favored over recall (quality over quantity for KB augmentation)

**Data Splitting Strategy**:
- Split entities first, then KB triples, then corpus examples
- Minimize information leakage between splits
- Standard splits: tiny (1%), train (74%), dev (25%)

**Negative Sampling**:
- Positive instances from KB
- Negative instances from unrelated entity pairs in corpus
- Balance via downsampling (far more negatives than positives)

### External Tools and Libraries

Based on project description, consider:
- **spaCy**: Dependency matching and pattern building
- **POTATO**: Graph pattern extraction for text classification
- Any open-source rule-based systems for baseline implementations

## Development Workflow

1. **Dataset Preprocessing** (Milestone 1 - Due Nov 2)
   - Load SemEval 2010 Task 8 data
   - Extract entities and relations
   - Store in standard format (likely JSON/TSV)
   - Ensure reproducibility

2. **Baseline Implementations** (Milestone 2 - Due Nov 30)
   - Implement ML baselines (likely using sklearn, transformers)
   - Implement rule-based baselines (spaCy patterns, POTATO, or custom)
   - Quantitative evaluation (precision, recall, F1)
   - Qualitative analysis (error analysis, pattern inspection)

3. **Final Solution** (Due Jan 25, 2026)
   - Optimized and explainable models
   - Comprehensive documentation
   - 2-page management summary (non-technical)
   - Presentation slides

## Important Notes

### Repository Management
- Team repository on GitHub with mentor as collaborator
- Use version control throughout (not bulk commits before deadlines)
- Clean, organized codebase structure
- README should describe high-level structure and reproduction steps

### Evaluation Philosophy
- Both quantitative AND qualitative evaluation essential
- Higher metrics don't always mean better solutions for some tasks
- Manual analysis of system output necessary to understand strengths/limitations
- Explainability is a key requirement for this topic

### Team Collaboration
- Individual contributions should be clearly documented
- Final presentation: each member presents their own contributions (15 min + 5 min Q&A)
- Management summary must describe each member's contributions and any issues

## Quick Reference

**Project Type**: Academic research project - Explainable Relation Extraction
**Institution**: TU Wien, NLP Information Extraction (Winter 2025)
**Team**: Token13
**Instructor**: Gábor Recski
**Review Meeting**: Dec 19, 2025, 9-13h
**Final Presentation**: Jan 16, 2026
