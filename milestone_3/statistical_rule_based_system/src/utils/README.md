# Utilities Module

Shared configuration and helper logic.

## Components

- **`config.py`**: System-wide constants (thresholds, model names) and path resolution helpers.
- **`semantic.py`**: Core semantic logic, including NLTK integration (WordNet, FrameNet), priority calculation logic, and unified semantic feature extraction.

## Usage

```python
from src.utils.config import MIN_PRECISION
from src.utils.semantic import compute_rule_priority
```
