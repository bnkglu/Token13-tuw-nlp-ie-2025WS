# Milestone 3: Implementation Summary

## Complete Implementation Overview

All files and notebooks for Milestone 3 have been successfully created and are ready for execution.

## Created Files

### Python Modules (`src/`)

1. **`utils.py`**
   - Imports reusable functions from Milestone 2
   - Functions: `doc_from_json`, `get_dependency_path`, `get_between_span`, `preprocess_data`

2. **`pattern_miner.py`**
   - Implements all 4 pattern extraction types:
     - Type A (Triangle): LCA-based verb patterns
     - Type B (Bridge): Prepositional chains
     - Type C (Linear): Sequence patterns with precedence operators
     - Type D (Direct): Noun compounds and modifiers
   - Key function: `extract_all_patterns()`

3. **`pattern_augmentation.py`**
   - Tiered filtering logic based on pattern complexity
   - Passive voice generation from active patterns
   - Key functions: `filter_patterns_tiered()`, `generate_passive_variants()`, `sort_patterns()`

4. **`execution_engine.py`**
   - Unified DependencyMatcher compilation
   - Anchoring verification
   - Pattern application with statistics
   - Key functions: `compile_dependency_matcher()`, `apply_patterns_with_anchoring()`

### Jupyter Notebooks (`notebooks/`)

1. **`1_concept_abstraction.ipynb`**
   - Defines manual seed clusters (VERB, NOUN, PREP)
   - Auto-expands VERB and NOUN clusters using word embeddings
   - Manual prepositions only (no expansion)
   - Output: `../data/concept_clusters.json`

2. **`2_unified_pattern_mining.ipynb`**
   - Loads concept clusters
   - Preprocesses training data
   - Extracts all 4 pattern types following priority order
   - Output: `../data/raw_patterns.json`

3. **`3_pattern_refinement.ipynb`**
   - Applies tiered filtering:
     - Complex (len > 3): precision >= 0.60, support >= 1
     - Simple (len <= 3): precision >= 0.60, support >= 3
     - "Other": precision >= 0.90, support >= 3
   - Generates passive voice variants
   - Sorts by (length desc, precision desc)
   - Output: `../data/patterns_augmented.json`

4. **`4_execution_engine.ipynb`**
   - Compiles unified DependencyMatcher
   - Applies patterns with anchoring verification
   - Generates predictions for train and test sets
   - Output: `../data/predictions/train_predictions.json`, `test_predictions.json`

5. **`5_evaluation_analysis.ipynb`**
   - Computes quantitative metrics (Precision, Recall, F1, Accuracy)
   - Compares M2 vs M3 performance
   - Detailed classification report
   - Shows improvements over baseline

## Execution Instructions

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model (if not already installed)
python -m spacy download en_core_web_lg
```

### Running the Pipeline

Execute notebooks in sequence:

```bash
cd milestone_3/notebooks

# 1. Concept Abstraction
jupyter notebook 1_concept_abstraction.ipynb

# 2. Pattern Mining
jupyter notebook 2_unified_pattern_mining.ipynb

# 3. Pattern Refinement
jupyter notebook 3_pattern_refinement.ipynb

# 4. Execution Engine
jupyter notebook 4_execution_engine.ipynb

# 5. Evaluation
jupyter notebook 5_evaluation_analysis.ipynb
```

**OR** run all notebooks programmatically:

```bash
cd milestone_3/notebooks
jupyter nbconvert --to notebook --execute 1_concept_abstraction.ipynb
jupyter nbconvert --to notebook --execute 2_unified_pattern_mining.ipynb
jupyter nbconvert --to notebook --execute 3_pattern_refinement.ipynb
jupyter nbconvert --to notebook --execute 4_execution_engine.ipynb
jupyter nbconvert --to notebook --execute 5_evaluation_analysis.ipynb
```

## Key Implementation Details

### 1. Four Pattern Types

All patterns are converted to DependencyMatcher format:

- **Type A (Triangle):** `anchor > e1, anchor > e2`
- **Type B (Bridge):** `e1 > prep > e2`
- **Type C (Linear):** `e1 .* token .* e2` (precedence)
- **Type D (Direct):** `e2 > e1` (parent-child)

### 2. Mining Priority Order

1. Type D - Check first, extract and CONTINUE
2. Type A - Extract and STOP
3. Type B - Extract and STOP
4. Type C - Fallback if no A/B found

Type D can coexist with A/B/C; A/B/C are mutually exclusive.

### 3. Concept Abstraction

- **Auto-expanded:** VERB and NOUN clusters (similarity > 0.75)
- **Manual only:** PREP clusters (avoid noise)
- Concepts replaced with `{"IN": [...]}` in patterns

### 4. Tiered Thresholding

| Pattern Type | Precision | Support |
|--------------|-----------|---------|
| Complex (len > 3) | >= 0.60 | >= 1 |
| Simple (len <= 3) | >= 0.60 | >= 3 |
| "Other" | >= 0.90 | >= 3 |

### 5. Passive Voice Augmentation

- Generated from active patterns with precision > 0.75
- Dependencies: `nsubj → nsubjpass`, `dobj → agent/pobj`
- Direction flipped: `(e1,e2) ↔ (e2,e1)`

### 6. Anchoring Verification

Ensures matched pattern nodes align with actual entity positions:
```python
match_indices['e1'] == sample.e1_span.root.i
match_indices['e2'] == sample.e2_span.root.i
```

## Expected Results

### Target Performance vs M2 Baseline

| Metric | M2 Baseline | M3 Target | Improvement |
|--------|-------------|-----------|-------------|
| Test Accuracy | 49.7% | 55-60% | +5-10% |
| Macro Recall | 40.2% | 55%+ | +15% |
| Macro F1 | 43.0% | 55%+ | +12% |

### Key Improvements

1. **Concept Abstraction** → Reduces over-specificity
2. **Type D Patterns** → Captures noun compounds (high-frequency)
3. **Passive Augmentation** → Addresses directionality imbalance
4. **Tiered Thresholding** → Trusts complex patterns more
5. **Strict "Other"** → Prevents "Other sink" problem

## Data Flow

```
train.json
    ↓
[Notebook 1] → concept_clusters.json
    ↓
[Notebook 2] → raw_patterns.json
    ↓
[Notebook 3] → patterns_augmented.json
    ↓
[Notebook 4] → train_predictions.json, test_predictions.json
    ↓
[Notebook 5] → Evaluation metrics & M2 vs M3 comparison
```

## Troubleshooting

### Issue: Import errors from Milestone 2

**Solution:** Ensure `milestone_2/rule_based/rule_based_directed.py` exists. The path is hardcoded in `src/utils.py`.

### Issue: spaCy model not found

**Solution:**
```bash
python -m spacy download en_core_web_lg
```

### Issue: Out of memory during pattern mining

**Solution:** Reduce `top_n` parameter in concept expansion or filter training data.

### Issue: DependencyMatcher pattern errors

**Solution:** Check pattern format in `pattern_augmentation.py` - ensure all patterns have valid `RIGHT_ID`, `LEFT_ID`, `REL_OP` structure.

## File Locations

```
milestone_3/
├── README.md                           # Project overview
├── IMPLEMENTATION_SUMMARY.md           # This file
├── requirements.txt                    # Dependencies
├── notebooks/
│   ├── 1_concept_abstraction.ipynb
│   ├── 2_unified_pattern_mining.ipynb
│   ├── 3_pattern_refinement.ipynb
│   ├── 4_execution_engine.ipynb
│   └── 5_evaluation_analysis.ipynb
├── src/
│   ├── utils.py
│   ├── pattern_miner.py
│   ├── pattern_augmentation.py
│   └── execution_engine.py
└── data/
    ├── concept_clusters.json           (created by Notebook 1)
    ├── raw_patterns.json               (created by Notebook 2)
    ├── patterns_augmented.json         (created by Notebook 3)
    └── predictions/
        ├── train_predictions.json      (created by Notebook 4)
        └── test_predictions.json       (created by Notebook 4)
```

## Next Steps

1. **Run Notebooks 1-5** in sequence
2. **Review results** in Notebook 5
3. **Analyze errors** to identify improvement opportunities
4. **Tune hyperparameters** if needed:
   - Similarity threshold (default: 0.75)
   - Tiered thresholds (complex: 1, simple: 3, Other: 3)
   - Passive precision threshold (default: 0.75)

## Implementation Complete! ✓

All code, notebooks, and documentation have been created. The system is ready for execution.

For questions or issues, refer to the plan file:
`~/.claude/plans/streamed-scribbling-sutton.md`
