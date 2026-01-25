# Statistical Rule-Based System

A configurable rule-based relation extraction system that combines the MS2 notebook baseline with optional MS3 semantic enhancements.

## Usage

```bash
# MS2 Baseline with semantics disabled explicitly
python main.py  # Results in: results/semantics_off_first_match/

# MS3 mode: Enable semantic patterns (SYNSET, FRAME, HYPERNYM, PREP_STRUCT)
python main.py --use-semantics  # Results in: results/semantics_on_first_match/

# MS3 Full: Semantics + priority tiers for priority_based selection
python main.py --use-semantics --prediction-mode=priority_based
# Results in: results/semantics_on_priority_based/
```

## Automated Ablation Study

To run the complete ablation study (4 configurations) with a single command, use the provided shell script from the parent directory:

```bash
# Run all experiments and generate summary
../run_ablation.sh --all

# Or run interactively
../run_ablation.sh
```


## Flags

| Flag | Description |
|------|-------------|
| `--use-semantics` | Enable MS3 semantic patterns (SYNSET, LEXNAME, FRAME, HYPERNYM, PREP_STRUCT) |
| `--prediction-mode` | `first_match` (default, MS2 style) or `priority_based` (priority-based) |
| `--use-tiers` | Apply semantic priority tiers (only affects `priority_based` mode) |

## Output

Results are saved to `results/<semantics>_<mode>[_tiers]/`:
- `evaluation_results.txt` - Classification reports
- `confusion_matrix_test.png` - Confusion matrix visualization
- `rb_*_predictions_directed.txt` - Predictions for official scorer
- `rb_*_answer_key_directed.txt` - Gold labels for scoring

## Ablation Configurations

| Config | Flags | Description |
|--------|-------|-------------|
| A | (none) | MS2 baseline |
| B | `--use-semantics` | MS2 + semantic patterns |
| C | `--prediction-mode=priority_based --use-tiers` | Priority tiers, no semantics |
| D | `--use-semantics --prediction-mode=priority_based --use-tiers` | Full MS3 |

---

## Semantics Off vs. On

### Semantics OFF (MS2 Baseline)
Extracts only **surface-level lexical and syntactic patterns**:

| Pattern Type | Description | Example |
|--------------|-------------|---------|
| `LEMMA` | Single lemma in between-span | `"contain"` |
| `BIGRAM` | Two consecutive lemmas | `("is", "part")` |
| `PREP` | Preposition in between-span | `"of"`, `"in"` |
| `BEFORE_E1` | Lemma immediately before E1 | `"the"` |
| `AFTER_E2` | Lemma immediately after E2 | `"was"` |
| `DEP_VERB` | Verb connecting entities via deps | `("contain", "nsubj", "dobj")` |
| `DEP_LABELS` | Dependency labels of E1 and E2 | `("nsubj", "pobj")` |
| `ENTITY_POS` | POS tags of entity roots | `("NOUN", "NOUN")` |

### Semantics ON (MS3 Enhancement)
Adds **WordNet and FrameNet-based generalization patterns**:

| Pattern Type | Source | Description | Example |
|--------------|--------|-------------|---------|
| `SYNSET` | WordNet | First synset of verb lemma | `"contain.v.01"` |
| `LEXNAME` | WordNet | Lexicographer filename (semantic domain) | `"verb.stative"` |
| `FRAME` | FrameNet | Semantic frame triggered by verb | `"Contacting"` |
| `HYPERNYM` | WordNet | First hypernym of entity head nouns | `("artifact.n.01", "whole.n.02")` |
| `PREP_STRUCT` | Syntax+WN | Preposition + governor + entity roles | `("of", "part", "pobj", "pobj")` |
| `PREP_GOV_LEX` | Syntax+WN | Preposition + governor's lexname | `("of", "noun.artifact")` |
| `PREP_ROLES` | Syntax | Preposition + entity dependency roles | `("of", "pobj", "pobj")` |

---

## How Semantic Patterns Are Extracted

During **training** (rule discovery phase):

1. **SYNSET Extraction**: For each verb in the between-span, query WordNet for synsets matching the lemma. Use the first synset's name (e.g., `"cause.v.01"`).

2. **LEXNAME Extraction**: From the same synset, extract the lexicographer filename which groups words by semantic domain (e.g., `"verb.change"`, `"noun.artifact"`).

3. **FRAME Extraction**: Query NLTK's FrameNet for lexical units matching the verb lemma. Extract the frame name (e.g., `"Causation"`, `"Containing"`). A `FrameMapper` is trained to associate frames with relations.

4. **HYPERNYM Extraction**: For entity head nouns, get the first hypernym from WordNet's taxonomy. Creates patterns like `(entity1_hypernym, entity2_hypernym)`.

5. **PREP_STRUCT Extraction**: When a preposition links entities, extract `(prep, gov_lemma, e1_role, e2_role)` - the preposition lemma, governing word's lemma, and dependency roles of both entities to the governor.

6. **PREP_GOV_LEX Extraction**: From the same prepositional structure, extract `(prep, gov_lexname)` - the preposition and the governing word's WordNet lexname (semantic category).

7. **PREP_ROLES Extraction**: From the same prepositional structure, extract `(prep, e1_role, e2_role)` - the preposition and dependency roles only, without the governor lemma.

---

## How Semantic Patterns Are Used During Classification

During **prediction** (test phase), the classifier checks if a test sample matches learned semantic patterns:

### Pattern Matching Logic

1. **SYNSET Matching**: For each verb in the between-span of the test sample, compute its synset and check if it matches any learned SYNSET pattern.

2. **LEXNAME Matching**: Similarly, extract the lexname of verbs in the between-span and match against learned LEXNAME patterns.

3. **FRAME Matching**: Query FrameNet for frames triggered by verbs in the between-span. If any frame matches a learned FRAME pattern, the rule fires.

4. **HYPERNYM Matching**: Extract hypernyms of both entity heads and check if the pair `(e1_hypernym, e2_hypernym)` matches a learned HYPERNYM pattern.

5. **PREP_STRUCT Matching**: If a preposition exists between entities, extract its structure features and compare against learned PREP_STRUCT, PREP_GOV_LEX, or PREP_ROLES patterns.

### Selection Strategy

- **`first_match` mode**: Returns the first rule that matches (MS2 style, sorted by precision/support)
- **`priority_based` mode**: Collects ALL matching rules, then selects the one with highest priority score (MS3 style, semantic tiers)
