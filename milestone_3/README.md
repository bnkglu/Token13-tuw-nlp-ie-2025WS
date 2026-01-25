# Milestone 3: Semantic-Enriched Relation Extraction

This directory contains the codebase for Milestone 3, which enhances a rule-based relation extraction system with semantic features (WordNet, FrameNet) and priority-based matching.

## Key Improvements over Milestone 2



*   **Semantic Generalization**: Extended beyond surface lexical patterns by incorporating **WordNet** (synsets, hypernyms, lexical categories) and **FrameNet** (semantic frames), enabling abstraction over verb meaning and entity types.
*   **Priority-Based Conflict Resolution**: Implemented a tiered scoring system where specific semantic rules (Tier 1-3) take precedence over generic lexical rules (Tier 4-5), effectively resolving rule conflicts.
*   **Modular Architecture**: Refactored monolithic scripts into a clean, extensible design with dedicated modules for `patterns` (extraction), `classification` (matching), and `analysis`.
*   **Performance Optimization**: Re-engineered the matching engine to use pre-compiled spaCy matchers, drastically improving classification speed.
*   **Automated Ablation Study**: A robust testing framework (`run_ablation.sh`) to systematically compare Baseline vs. Semantic and First-Match vs. Priority-Based strategies.

## Installation & Setup

### 1. Create Environment
We recommend using a clean python environment (Python 3.10+):

```bash
python -m venv .venv
source venv/bin/activate
```

### 2. Install Dependencies
Install all required packages from `requirements.txt`:

```bash
cd statistical_rule_based_system
pip install -r requirements.txt
```

### 3. Download Model Resources
The system requires the spaCy English model:

```bash
python -m spacy download en_core_web_lg
```
*(Note: NLTK resources will be downloaded automatically on first run)*

---

## Usage

### Automated Ablation Study (Recommended)
To run the full benchmark comparing all configurations (Baseline vs Semantic, First Match vs Priority), use the provided shell script:

```bash
./run_ablation.sh --all
```

This will:
1.  Run 4 experiments (Combinations of Semantic On/Off + Priority/FirstMatch).
2.  Generate a summary report in `results/ABLATION_SUMMARY.md`.
3.  Save full logs and outputs to timestamped folders in `results/`.

### Manual Execution
You can run the system manually to test specific configurations. 

**Go to the system directory:**
```bash
cd statistical_rule_based_system
```

**Run with specific flags:**

*   **Baseline**:
    ```bash
    python main.py --prediction-mode first_match
    ```
*   **Semantic + Priority Based**:
    ```bash
    python main.py --use-semantics --prediction-mode priority_based
    ```
*   **Custom Output Directory**:
    ```bash
    python main.py --use-semantics --output_dir my_experiment_results
    ```

For more details on the inner workings, see [src/README.md](statistical_rule_based_system/src/classification/README.md).

---

## Directory Structure

```text
milestone_3/
├── run_ablation.sh                 # Main automation script
├── statistical_rule_based_system/  # Core System Code
│   ├── main.py                     # Entry point
│   ├── src/                        # Source code
│   │   ├── analysis/               # Evaluation & Mining
│   │   ├── classification/         # Matchers & Logic
│   │   ├── patterns/               # Rule Extraction (Lexical, Semantic)
│   │   └── utils/                  # Helpers (config.py, semantic.py)
│   └── results/                    # Default output directory for manual runs
└── results/                        # Output directory for ablation runs
```
