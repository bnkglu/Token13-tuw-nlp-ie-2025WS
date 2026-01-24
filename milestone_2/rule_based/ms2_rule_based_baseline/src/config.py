"""Configuration constants for the rule-based relation extraction system."""

from pathlib import Path

# Model settings
SPACY_MODEL = "en_core_web_lg"

# Rule mining thresholds
MIN_PRECISION = 0.60
MIN_SUPPORT = 2

# Feature extraction settings
TOP_N_KEYWORDS = 30
TOP_N_VERBS = 15
TOP_N_PREPS = 10
TOP_N_DEP_PATHS = 5

# Analysis settings
MIN_LEMMA_LENGTH = 2

# Output settings
DEFAULT_ERROR_SAMPLES = 20
TOP_RULES_DISPLAY = 10
TOP_RULES_PER_RELATION = 3


def get_project_paths(script_dir: Path) -> dict:
    """
    Get project paths based on script location.

    Args:
        script_dir: Path to the directory containing the main script.

    Returns:
        Dictionary containing project root and data directory paths.
    """
    project_root = script_dir.parent.parent.parent
    data_dir = project_root / "data"
    return {
        "project_root": project_root,
        "data_dir": data_dir,
        "train_file": data_dir / "processed/train/train.json",
        "test_file": data_dir / "processed/test/test.json",
    }
