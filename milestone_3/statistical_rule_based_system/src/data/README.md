# Data Module

Handles data ingestion and preprocessing.

## Components

- **`loader.py`**: Functions to load raw JSON datasets and preprocess them with spaCy (tokenization, entity marking) for the pipeline.

## Usage

```python
from src.data.loader import load_datasets, preprocess_data

train_raw, test_raw = load_datasets(data_dir)
train_processed = preprocess_data(train_raw, nlp)
```
