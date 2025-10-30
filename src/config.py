"""Configuration file for the SemEval-2010 Task 8 preprocessing pipeline."""

import os
from pathlib import Path


# Base directories
BASE_DIR = Path(__file__).parent.parent
RESOURCES_DIR = BASE_DIR / "resources"
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Raw data paths
RAW_TRAIN_FILE = (
    RESOURCES_DIR / "SemEval2010_task8_all_data 2" /
    "SemEval2010_task8_training" / "TRAIN_FILE.TXT"
)
RAW_TEST_FILE = (
    RESOURCES_DIR / "SemEval2010_task8_all_data 2" /
    "SemEval2010_task8_testing_keys" / "TEST_FILE_FULL.TXT"
)
TEST_KEY_FILE = (
    RESOURCES_DIR / "SemEval2010_task8_all_data 2" /
    "SemEval2010_task8_testing_keys" / "TEST_FILE_KEY.TXT"
)

# Processed data paths
TRAIN_OUTPUT_DIR = PROCESSED_DIR / "train"
TEST_OUTPUT_DIR = PROCESSED_DIR / "test"
STATS_OUTPUT_DIR = PROCESSED_DIR / "statistics"

# Create output directories
for directory in [TRAIN_OUTPUT_DIR, TEST_OUTPUT_DIR, STATS_OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# spaCy configuration
SPACY_MODEL = "en_core_web_lg"

# Note: Relation types and labels are now extracted dynamically from the data
# See src/preprocessing/data_loader.py methods:
# - extract_relation_types() for unique relation types
# - extract_all_labels() for all labels including directionality
