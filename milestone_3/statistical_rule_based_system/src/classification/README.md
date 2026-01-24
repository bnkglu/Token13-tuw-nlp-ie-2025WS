# Classification Module

This module handles the application of extracted rules to classify relation samples.

## Components

- **`classifier.py`**: The main `RuleBasedClassifier` engine. It orchestrates the classification process using an optimized hybrid approach (O(1) matcher lookup + manual fallbacks).
- **`matcher_utils.py`**: Utilities for compiling spaCy matchers (TokenMatcher, PhraseMatcher, DependencyMatcher) from rule definitions.
- **`manual_checks.py`**: Implementation of manual rule verification logic for patterns that cannot be expressed purely in spaCy matchers (e.g., Semantic rules, complex relative positioning).

## Usage

```python
from src.classification import apply_rule_based_classifier

predictions, directions, explanations = apply_rule_based_classifier(
    samples, rules, nlp, prediction_mode='priority_based'
)
```
