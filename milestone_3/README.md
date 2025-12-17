# Milestone 3: Adaptive Neuro-Symbolic Relation Extraction

Expanding the Milestone 2 rule-based system with neuro-symbolic techniques to improve recall from 40.2% to 55%+ while maintaining precision.

## Key Innovations

1. **Concept Abstraction** - Generalize verbs/nouns using word embeddings (manual prepositions)
2. **Unified DependencyMatcher** - ALL 4 pattern types compiled into single matcher
3. **Tiered Thresholding** - Trust complex patterns more (support >= 1), simple patterns less (support >= 3)
4. **Passive Voice Augmentation** - Auto-generate passive variants for directionality
5. **Anchoring Verification** - Strict token-level verification of entity alignment
6. **Strict "Other" Rules** - Only keep "Other" patterns with precision > 90%

## Implementation Pipeline

### Notebook 1: Concept Abstraction
- Manual seed clusters for VERB, NOUN, PREP concepts
- Auto-expand VERB and NOUN clusters using word embeddings (similarity > 0.75)
- Manual prepositions only (no expansion)
- Output: `data/concept_clusters.json`

### Notebook 2: Unified Pattern Mining
Extract ALL 4 pattern types as DependencyMatcher patterns:
- **Type A (Triangle):** Event-driven via LCA (verb anchors)
- **Type B (Bridge):** Prepositional chains
- **Type C (Linear):** Sequence fallback with precedence operators
- **Type D (Direct):** Noun compounds and modifiers (NEW!)
- Output: `data/raw_patterns.json`

### Notebook 3: Pattern Refinement & Augmentation
- Tiered filtering (complex >= 1, simple >= 3, "Other" >= 90%)
- Passive voice generation from high-precision active patterns
- Sort by (length desc, precision desc)
- Output: `data/patterns_augmented.json`

### Notebook 4: Execution Engine
- Compile unified DependencyMatcher for all pattern types
- Apply with strict anchoring verification
- Output: Predictions for train and test sets

### Notebook 5: Evaluation & Analysis
- Quantitative metrics (Precision, Recall, F1)
- M2 vs M3 comparison
- Ablation studies (concept abstraction, passive augmentation, anchoring)
- Success case analysis

## Directory Structure

```
milestone_3/
├── notebooks/
│   ├── 1_concept_abstraction.ipynb
│   ├── 2_unified_pattern_mining.ipynb
│   ├── 3_pattern_refinement.ipynb
│   ├── 4_execution_engine.ipynb
│   └── 5_evaluation_analysis.ipynb
├── src/
│   ├── concept_clusters.py
│   ├── pattern_miner.py
│   ├── pattern_augmentation.py
│   ├── execution_engine.py
│   └── utils.py
├── data/
│   ├── concept_clusters.json
│   ├── raw_patterns.json
│   ├── patterns_augmented.json
│   └── predictions/
└── README.md
```

## Expected Performance

| Metric | M2 Baseline | M3 Target | Improvement |
|--------|-------------|-----------|-------------|
| Test Accuracy | 49.7% | 55-60% | +5-10% |
| Macro Recall | 40.2% | 55%+ | +15% |
| Macro F1 | 43.0% | 55%+ | +12% |

## Critical Architectural Decisions

1. **Unified DependencyMatcher** - No hybrid approach, all patterns use DependencyMatcher
2. **Four Pattern Types** - Including new Type D for noun compounds
3. **Tiered Thresholding** - Different thresholds based on pattern complexity
4. **Manual Prepositions** - PREP clusters not auto-expanded (avoid noise)
5. **Strict "Other"** - Only precision > 90% for "Other" patterns

## Dependencies

- Python 3.8+
- spaCy 3.x with en_core_web_lg model
- NumPy, Pandas, scikit-learn
- tqdm

## Usage

Run notebooks in sequence from 1 to 5. Each notebook saves its output for the next stage.

## References

- Milestone 2: Rule-Based Relation Extraction (baseline)
- Plan file: `~/.claude/plans/streamed-scribbling-sutton.md`
