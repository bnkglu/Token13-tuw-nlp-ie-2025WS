# Semantic Resources: WordNet and FrameNet Integration

This document explains how WordNet and FrameNet are used in the Statistical Rule-Based semantic pattern extraction system.

---

## WordNet Integration

WordNet is a lexical database that groups English words into sets of cognitive synonyms (synsets), each expressing a distinct concept. We use NLTK's WordNet interface.

### 1. SYNSET Extraction

**Purpose**: Generalize verb lemmas to their first synset for semantic abstraction.

**How it works**:
```python
from nltk.corpus import wordnet as wn

# For a verb lemma like "contain"
synsets = wn.synsets("contain", pos=wn.VERB)
if synsets:
    synset_name = synsets[0].name()  # e.g., "contain.v.01"
```

**Example**:
- Lemma: `"cause"` → Synset: `"cause.v.01"`
- Lemma: `"produce"` → Synset: `"produce.v.01"`

**Why this helps**: Different verbs with similar meanings map to the same or related synsets, allowing the system to generalize beyond exact lexical matches.

---

### 2. LEXNAME Extraction

**Purpose**: Extract the lexicographer filename (semantic domain) of a word.

**How it works**:
```python
# From the same synset
synset = wn.synsets("contain", pos=wn.VERB)[0]
lexname = synset.lexname()  # e.g., "verb.stative"
```

**Lexname Categories** (examples):
- `verb.change` - verbs of change
- `verb.stative` - verbs of being/having
- `verb.communication` - verbs of communication
- `noun.artifact` - nouns for man-made objects
- `noun.person` - nouns for people

**Example**:
- `"cause"` → `"verb.change"`
- `"contain"` → `"verb.stative"`
- `"artifact"` → `"noun.artifact"`

**Why this helps**: Groups semantically related words into broader categories, providing mid-level abstraction between lemmas and very general concepts.

---

### 3. HYPERNYM Extraction

**Purpose**: Find the immediate parent concept (hypernym) of entity nouns in WordNet's taxonomy.

**How it works**:
```python
# For entity head nouns
synsets = wn.synsets("car", pos=wn.NOUN)
if synsets:
    hypernyms = synsets[0].hypernyms()
    if hypernyms:
        hypernym_name = hypernyms[0].name()  # e.g., "motor_vehicle.n.01"
```

**Example Taxonomy**:
- `"car"` → `"motor_vehicle.n.01"` → `"vehicle.n.01"` → `"conveyance.n.03"`
- `"book"` → `"publication.n.01"` → `"work.n.02"` → `"product.n.02"`

**Pattern Creation**:
- For a relation between entities E1="car" and E2="garage"
- Extract: `(hypernym_e1, hypernym_e2)` = `("motor_vehicle.n.01", "structure.n.01")`

**Why this helps**: Captures semantic relationships at a more abstract level (e.g., "vehicle-structure" instead of "car-garage").

---

### 4. PREP_GOV_LEX (WordNet for Prepositions)

**Purpose**: Use lexnames to generalize the governing word in prepositional structures.

**How it works**:
```python
# In a phrase like "part of machine"
gov_lemma = "part"  # governing word
gov_pos = 'n'  # it's a noun

synsets = wn.synsets(gov_lemma, pos=wn.NOUN)
if synsets:
    gov_lexname = synsets[0].lexname()  # e.g., "noun.artifact"
```

**Pattern**: `(prep, gov_lexname)` = `("of", "noun.artifact")`

**Why this helps**: Generalizes specific governors to semantic categories, e.g., "part", "component", "piece" all map to similar lexnames.

---

## FrameNet Integration

FrameNet is a lexical database based on Frame Semantics. It defines semantic frames (conceptual structures) and lexical units (words) that evoke these frames.

### FRAME Extraction

**Purpose**: Map verbs to semantic frames they evoke.

**How it works**:
```python
from nltk.corpus import framenet as fn
import re

# For a verb lemma like "cause"
pattern = r'(?i)^' + re.escape("cause") + r'\.'
lexical_units = fn.lus(pattern)

frames = []
for lu in lexical_units:
    frame_name = lu.frame.name  # e.g., "Causation"
    if frame_name not in frames:
        frames.append(frame_name)
```

**Example Mappings**:
- `"cause"` → `"Causation"` frame
- `"contain"` → `"Containing"` frame
- `"produce"` → `"Manufacturing"`, `"Creating"` frames
- `"send"` → `"Sending"` frame

**Frame Definitions** (examples):
- **Causation**: An Agent or Cause causes an Effect
- **Containing**: A Container holds Contents
- **Manufacturing**: A Producer creates a Product

---

### FrameMapper Training

**Purpose**: Learn which frames are most predictive of which relations.

**How it works**:

1. **Collection Phase** (during training):
```python
# For each training sample with a verb
verb_lemmas = []
relations = []

for sample in training_data:
    for verb in between_span:
        frames = get_frames_for_verb(verb.lemma_)
        for frame in frames:
            verb_lemmas.append(verb.lemma_)
            relations.append(sample.relation)
```

2. **Learning Phase**:
```python
# Count frame-relation co-occurrences
frame_relation_counts = defaultdict(lambda: defaultdict(int))
for lemma, relation in zip(verb_lemmas, relations):
    frames = get_frames_for_verb(lemma)
    for frame in frames:
        frame_relation_counts[frame][relation] += 1

# Select frames with high precision (>= 0.6) and support (>= 3)
for frame, rel_counts in frame_relation_counts.items():
    total = sum(rel_counts.values())
    if total >= 3:  # min_support
        best_rel = max(rel_counts, key=rel_counts.get)
        precision = rel_counts[best_rel] / total
        if precision >= 0.6:  # min_precision
            frame_to_relation[frame] = best_rel
```

3. **Prediction Phase**:
```python
# During classification
for verb in between_span:
    frames = get_frames_for_verb(verb.lemma_)
    for frame in frames:
        if frame in learned_frame_patterns:
            # This frame matches a learned pattern
            return matched_rule
```

**Why this helps**: Frames provide semantic abstraction beyond individual words. Multiple verbs can evoke the same frame, allowing generalization across semantically similar verbs.

---

## Summary: Abstraction Hierarchy

The semantic resources provide multiple levels of abstraction:

1. **Lexical** (most specific): `"cause"`, `"produce"`, `"create"`
2. **Synset**: `"cause.v.01"`, `"produce.v.01"`, `"create.v.01"`
3. **Lexname**: `"verb.change"` (all three map here)
4. **Frame**: `"Causation"`, `"Manufacturing"`, `"Creating"`
5. **Hypernym** (for nouns): `"artifact.n.01"`, `"entity.n.01"`

This hierarchy allows the system to match at different levels of specificity, improving generalization while maintaining precision.
