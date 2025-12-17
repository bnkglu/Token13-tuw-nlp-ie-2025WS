"""
Utility functions reused from Milestone 2.

Imports key functions from the Milestone 2 rule-based implementation:
- doc_from_json: Reconstruct spaCy Doc from JSON
- get_dependency_path: LCA-based dependency path extraction
- get_between_span: Extract tokens between entities
- preprocess_data: Full preprocessing pipeline
- _split_relation_and_direction: Parse directed relation labels
"""

import sys
from pathlib import Path

# Add Milestone 2 path
m2_path = Path(__file__).parent.parent.parent / "milestone_2" / "rule_based"
sys.path.insert(0, str(m2_path))

# Import functions from Milestone 2
try:
    # IMPORTANT: use the import-safe API module (no top-level notebook execution)
    from rule_based_directed_api import (
        doc_from_json,
        get_dependency_path,
        get_between_span,
        preprocess_data,
        _split_relation_and_direction,
        save_predictions_for_scorer,
        create_answer_key,
        analyze_errors,
        analyze_error_patterns
    )
    print(f"Successfully imported functions from Milestone 2: {m2_path}")
except ImportError as e:
    print(f"Error importing from Milestone 2: {e}")
    print(f"Please ensure Milestone 2 code exists at: {m2_path}")
    raise

# Re-export for convenience
__all__ = [
    'doc_from_json',
    'get_dependency_path',
    'get_between_span',
    'preprocess_data',
    '_split_relation_and_direction',
    'save_predictions_for_scorer',
    'create_answer_key',
    'analyze_errors',
    'analyze_error_patterns'
]
