# %% [markdown]
# # Explainable Rule-Based Relation Extraction
# ## Milestone 2 - SemEval 2010 Task 8
# 
# **Objective:** Implement and evaluate a deterministic, rule-based system for relation extraction that is both effective and fully explainable.
# 
# This notebook details the process of building a relation extraction system using spaCy. The core of this approach is an automatic rule discovery mechanism that mines patterns from training data, filters them based on statistical quality (precision and support), and applies them using spaCy's efficient matchers.
# 
# **Key Goals for Milestone 2:**
# 1.  **Implement a Baseline:** Develop a rule-based system from scratch.
# 2.  **Quantitative Evaluation:** Measure performance using metrics like accuracy, precision, recall, and F1-score.
# 3.  **Qualitative Analysis:** Analyze the system's behavior, understand its strengths through explainability, and investigate its weaknesses through error analysis.
# 
# This notebook will walk through each of these steps, from data preparation to the final analysis.

# %%
## 1. Setup: Libraries and Data Loading

# === 1.1 Import Libraries ===
import json
import pandas as pd
import numpy as np
import spacy
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score, classification_report
from tqdm.auto import tqdm
import os
from pathlib import Path

# === 1.2 Load spaCy Model ===
nlp = spacy.load("en_core_web_lg")

print("Libraries loaded successfully!")
print(f"spaCy version: {spacy.__version__}")

# === 1.3 Load Datasets ===
# Set working directory to the project root for consistent paths
# Assumes the notebook is run from the root of the project
print(f"Current working directory: {Path.cwd()}")

print("\nLoading datasets...")
try:
    with open('../data/processed/train/train.json', 'r') as f:
        train_data = json.load(f)

    with open('../data/processed/test/test.json', 'r') as f:
        test_data = json.load(f)

    bert_high_conf_preds = pd.read_csv("../data/predictions/bert_high_confidence_predictions-doublesided.csv")
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"BERT high-confidence predictions loaded: {bert_high_conf_preds.shape[0]} samples")
    
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure you have run the preprocessing scripts and that the data files exist at the specified paths.")


# %% [markdown]
# ## 2. Data Processing and Feature Extraction
# 
# To build reliable rule-based patterns, we first transform each annotated sample into a structured linguistic representation. This preprocessing stage provides all the features our rule induction and matching steps depend on.
# 
# 1. **Reconstructing spaCy `Doc` Objects**
# 
#     We rebuild spaCy `Doc` objects directly from the pre-tokenized JSON annotations.
#     This gives us access to tokens, lemmas, POS tags, and dependency heads **without** running the spaCy NLP pipeline again.
#     Each `Doc` is therefore lightweight but still fully compatible with spaCy’s token and dependency operations.
# 
# 2. **Identifying Entity Spans**
# 
#     For each sample, we use the token indices of the annotated entities (`e1` and `e2`) to recover their corresponding `Span` objects inside the reconstructed `Doc`.
#     These spans give us the entity roots, their heads, and their token ranges.
# 
# 3. **Extracting Linguistic Features**
# 
#     We compute two core features for rule construction:
# 
#     * **Dependency Path**:
#     Instead of using spaCy’s LCA matrix, we compute the dependency path between the entity roots by traversing their ancestor chains and locating the first common ancestor manually.
#     This method is simple, deterministic, and works cleanly with our reconstructed dependency trees.
# 
#     * **Between-Entity Tokens**:
#     We extract the exact token span between `e1` and `e2`, capturing intermediate lemmas, POS tags, and dependency labels.
#     These between-words often encode strong relational cues (e.g., “caused by”, “part of”, “located in”).
#         - **Span**: the continuous sequence of tokens lying strictly between the two entity spans.
#             *Example*: In “A binds to B”, the span between `e1 = A` and `e2 = B` is the tokens “binds to”.
# 
# Together, these features give a compact but expressive description of how the two entities relate within the sentence.
# 
# The following functions implement this preprocessing pipeline.

# %%
from spacy.tokens import Doc

def doc_from_json(item, nlp):
    """
    Create a spaCy Doc from pre-computed JSON annotations.
    """
    tokens_data = item['tokens']
    
    # Extract token attributes
    words = [t['text'] for t in tokens_data]
    spaces = [i < len(words) - 1 for i in range(len(words))]
    
    # Create Doc with words and spaces
    doc = Doc(nlp.vocab, words=words, spaces=spaces)
    
    # Set linguistic attributes from pre-computed data
    for token, token_data in zip(doc, tokens_data):
        token.lemma_ = token_data['lemma']
        token.pos_ = token_data['pos']
        token.tag_ = token_data['tag']
        token.dep_ = token_data['dep']
        
        # Set head (dependency parent)
        head_id = token_data['head']
        if head_id != token.i:
            token.head = doc[head_id]
    
    return doc


def get_dependency_path(doc, e1_span, e2_span):
    """Extract dependency path between entity roots via LCA (no matrix)."""
    e1_root = e1_span.root
    e2_root = e2_span.root
    
    # Collect ancestors from e1_root to the root
    ancestors_e1 = []
    cur = e1_root
    while True:
        ancestors_e1.append(cur)
        if cur.head == cur:  # reached root
            break
        cur = cur.head
    
    # Walk up from e2_root until we hit something in ancestors_e1
    path_down_nodes = []
    cur = e2_root
    while cur not in ancestors_e1:
        path_down_nodes.append(cur)
        if cur.head == cur:  # fallback, no intersection (shouldn't happen in a tree)
            break
        cur = cur.head
    
    lca = cur
    # nodes from e1_root up to LCA (exclusive)
    path_up_nodes = []
    cur = e1_root
    while cur != lca:
        path_up_nodes.append(cur)
        cur = cur.head
    
    # Build features
    path_up = [(t.dep_, t.pos_, t.lemma_) for t in path_up_nodes]
    lca_feat = (lca.dep_, lca.pos_, lca.lemma_)
    path_down = [(t.dep_, t.pos_, t.lemma_) for t in reversed(path_down_nodes)]

    return path_up + [lca_feat] + path_down


def get_between_span(doc, e1_span, e2_span):
    """Get span between entities using Doc slicing."""
    if e1_span.start < e2_span.start:
        return doc[e1_span.end:e2_span.start]
    return doc[e2_span.end:e1_span.start]


def preprocess_data(data_list, nlp):
    """
    Process data using pre-computed annotations from JSON.
    """
    processed = []
    
    for item in tqdm(data_list, desc="Processing"):
        # Create Doc from pre-computed annotations
        doc = doc_from_json(item, nlp)
        
        e1_info = item['entities'][0]
        e2_info = item['entities'][1]
        
        # Create spans using token indices
        e1_token_ids = e1_info['token_ids']
        e2_token_ids = e2_info['token_ids']
        e1_span = doc[min(e1_token_ids):max(e1_token_ids)+1]
        e2_span = doc[min(e2_token_ids):max(e2_token_ids)+1]
        
        # Extract features
        dep_path = get_dependency_path(doc, e1_span, e2_span)
        between_span = get_between_span(doc, e1_span, e2_span)
        
        between_words = [
            {'text': t.text, 'lemma': t.lemma_, 'pos': t.pos_, 'dep': t.dep_}
            for t in between_span
        ]
        
        # 4) Labels (directed)
        rel_type = item['relation']['type']           # e.g. "Cause-Effect" or "Other"
        direction = item['relation'].get('direction', '') or ''
        direction = direction.replace('(', '').replace(')', '')
        if not direction:
            direction = 'e1,e2'
        
        # SemEval convention: "Other" is undirected, keep as plain "Other"
        if rel_type == "Other":
            rel_directed = "Other"
        else:
            rel_directed = f"{rel_type}({direction})"   # e.g. "Cause-Effect(e1,e2)"
        
        # 5) Store processed sample
        processed.append({
            'id': item['id'],
            'text': item['text'],
            'doc': doc,
            'e1_span': e1_span,
            'e2_span': e2_span,
            'relation': rel_type,               # undirected type (10 classes)
            'relation_directed': rel_directed,  # directed label (19, with Other undirected)
            'direction': direction,             # "e1,e2" or "e2,e1"
            'dep_path': dep_path,
            'between_words': between_words
        })
    
    return processed


# %%
# Process train and test data
print("Processing data...")
print()

train_processed = preprocess_data(train_data, nlp)
print("\nProcessing test data...")
test_processed = preprocess_data(test_data, nlp)

print(f"\nProcessed {len(train_processed)} training samples")
print(f"Processed {len(test_processed)} test samples")

# Display sample
print("\n" + "="*80)
print("Sample output:")
print("="*80)
sample = train_processed[0]
doc = sample['doc']
e1_span = sample['e1_span']
e2_span = sample['e2_span']

print(f"Text: {sample['text']}")
print(f"Entity 1: {e1_span.text} (POS: {e1_span.root.pos_}, DEP: {e1_span.root.dep_})")
print(f"Entity 2: {e2_span.text} (POS: {e2_span.root.pos_}, DEP: {e2_span.root.dep_})")
print(f"Relation: {sample['relation']}")
print(f"\nDependency path: {sample['dep_path'][:3]}...")
print(f"Between words: {[w['text'] for w in sample['between_words']]}")


# %% [markdown]
# ## 3.5 Exploratory Data Analysis - Extract Patterns from Data
# 
# Before defining rules manually, let's analyze the actual dataset to discover:
# 1. Most frequent words/lemmas per relation type
# 2. Common verbs and prepositions for each relation
# 3. Dependency patterns extracted from the shortest path between entity roots
# 4. Discriminative features that distinguish relations
# 
# ---
# 
# **Why These Default Values?**
# 
# **Keywords: 30** — Open-class words (nouns, adjectives) have high variety; need more examples to capture diverse expressions  
# **Verbs: 15** — Medium-sized vocabulary; syntactic backbone of relations  
# **Prepositions: 10** — Small closed-class set (~70 in English); highly discriminative
# 
# These balance **coverage** (capture enough patterns) vs. **precision** (avoid noise).
# 

# %%
def generate_patterns_from_analysis(relation_features, top_n_keywords=30, top_n_verbs=15, top_n_preps=10):
    """
    Generate RELATION_PATTERNS dictionary from data analysis.
    Extract most frequent and distinctive features per relation.
    """
    generated_patterns = {}
    
    for relation, features in relation_features.items():
        # Extract top keywords (lemmas)
        keywords = [lemma for lemma, count in features['top_lemmas'][:top_n_keywords]]
        
        # Extract top verbs
        verbs = [verb for verb, count in features['top_verbs'][:top_n_verbs]]
        
        # Extract top prepositions
        preps = [prep for prep, count in features['top_preps'][:top_n_preps]]
        
        # Extract dependency patterns (convert tuples back to lists)
        dep_patterns = []
        for path, count in features['top_dep_paths'][:5]:
            if len(path) >= 2:  # At least 2 dependencies
                dep_patterns.append(list(path[:3]))  # Take first 3 deps
        
        generated_patterns[relation] = {
            'keywords': keywords,
            'prep_patterns': preps,
            'verb_patterns': verbs,
            'dependency_patterns': dep_patterns
        }
    
    return generated_patterns

# %%
def analyze_relation_features(processed_data):
    """
    Analyze linguistic features for each relation type.
    Returns dictionaries of feature frequencies per relation.
    """
    # Group by relation type
    relation_groups = defaultdict(list)
    for sample in processed_data:
        relation_groups[sample['relation_directed']].append(sample)
    
    # Analyze each relation
    relation_analysis = {}
    
    for relation, samples in relation_groups.items():
        # Collect features from all samples of this relation
        all_lemmas = []
        all_verbs = []
        all_preps = []
        all_dep_paths = []
        all_between_words = []
        
        for sample in samples:
            doc = sample['doc']
            
            # Collect lemmas (excluding entities)
            e1_tokens = set(range(sample['e1_span'].start, sample['e1_span'].end))
            e2_tokens = set(range(sample['e2_span'].start, sample['e2_span'].end))
            
            for token in doc:
                if token.i not in e1_tokens and token.i not in e2_tokens:
                    lemma = token.lemma_.lower()
                    
                    # Collect verbs (don't filter stopwords for verbs)
                    if token.pos_ == 'VERB' and not token.is_punct and len(lemma) > 2:
                        all_verbs.append(lemma)
                    
                    # Collect prepositions (INCLUDE stopwords like "of", "in", "at")
                    if token.pos_ == 'ADP' and not token.is_punct:
                        all_preps.append(lemma)
                    
                    # Collect other lemmas (filter stopwords for general keywords)
                    if not token.is_stop and not token.is_punct and len(lemma) > 2:
                        all_lemmas.append(lemma)
            
            # Collect dependency paths (sequence of dependency labels along shortest path)
            if sample['dep_path']:
                path_deps = tuple([d[0] for d in sample['dep_path']])
                all_dep_paths.append(path_deps)
            
            # Between words (fixed: should be "if word['text'].strip()" not "if not")
            for word in sample['between_words']:
                if word['text'].strip() and len(word['lemma']) > 2:
                    all_between_words.append(word['lemma'].lower())
        
        # Count frequencies
        lemma_freq = Counter(all_lemmas).most_common(30)
        verb_freq = Counter(all_verbs).most_common(15)
        prep_freq = Counter(all_preps).most_common(10)
        dep_path_freq = Counter(all_dep_paths).most_common(10)
        between_freq = Counter(all_between_words).most_common(20)
        
        relation_analysis[relation] = {
            'count': len(samples),
            'top_lemmas': lemma_freq,
            'top_verbs': verb_freq,
            'top_preps': prep_freq,
            'top_dep_paths': dep_path_freq,
            'top_between_words': between_freq
        }
    
    return relation_analysis

# %%
# Generate data-driven patterns
print("\n" + "="*80)
print("GENERATING DATA-DRIVEN PATTERNS")
print("Top features per relation extracted from analysis")
print("="*80)

# First, analyze the training data to get relation features
relation_features = analyze_relation_features(train_processed)
data_driven_patterns = generate_patterns_from_analysis(relation_features)

# Display generated patterns
for relation in sorted(data_driven_patterns.keys()):
    patterns = data_driven_patterns[relation]
    print(f"\n{relation}:")
    print(f"  Keywords ({len(patterns['keywords'])}): {patterns['keywords'][:10]}")
    print(f"  Verbs ({len(patterns['verb_patterns'])}): {patterns['verb_patterns']}")
    print(f"  Preps ({len(patterns['prep_patterns'])}): {patterns['prep_patterns']}")
    print(f"  Dep patterns: {len(patterns['dependency_patterns'])} patterns extracted")

print("\n" + "="*80)
print("Data-driven patterns generated successfully!")
print("These patterns are based on actual frequency analysis of the training data.")
print("="*80)

# %% [markdown]
# ## 4. Automatic Rule Discovery from Training Data
# 
# We build a deterministic and fully explainable **directed** rule-based relation
# classifier. Rules are mined from the training set and converted into spaCy
# matchers (Matcher, PhraseMatcher, DependencyMatcher).
# Each discovered rule is associated with:
# 
# * a **directed relation label** (e.g., `Cause-Effect(e1,e2)`),
# * the **base relation type** (e.g., `Cause-Effect`),
# * the **direction** (`e1,e2` or `e2,e1`; for `Other` this is `None`),
# * a precision score,
# * a support count,
# * and a short human-readable explanation.
# 
# Rules are globally ranked by `(precision, support)` and applied in a deterministic
# decision list: **the first matching rule wins**.
# This yields an efficient, interpretable, and data-driven rule-based system.
# 
# <br>
# 
# ---
# 
# ## Explainability
# 
# A helper module plots or prints the top rules per directed relation, showing:
# 
# * the pattern type (LEMMA, PREP, DEP_VERB, …)
# * precision
# * support
# * the explanation used by the classifier
# 
# Every prediction is explainable: we always know **which rule fired and why**.
# 
# ---
# 
# ## How Patterns Are Scored and Converted into Directed Rules
# 
# ### 1. Pattern Mining (Directed)
# 
# For each training example we extract lexical and dependency features relative to
# the *two entities* in their **true direction** (`e1 -> e2`):
# 
# #### Lexical patterns
# 
# * `LEMMA`: single lemmas in the surface region between entities
# * `BIGRAM`: lemma pairs between entities
# * `PREP`: prepositions between entities
# * `BEFORE_E1` / `AFTER_E2`: context window tokens
# * `ENTITY_POS`: `(POS(E1), POS(E2))` pair
# 
# #### Dependency patterns
# 
# * `DEP_VERB`:
#   verb lemma + dependency role of **E1 relative to that verb** + dependency role of **E2 relative to the same verb**
#   (fully directed)
# * `DEP_LABELS`: `(dep(E1), dep(E2))` pair for entity heads
# 
# For each pattern we accumulate a **directed frequency table**:
# 
# ```
# pattern_counts[pattern][relation_directed] = count
# ```
# 
# Examples of directed labels:
# 
# * `Cause-Effect(e1,e2)`
# * `Message-Topic(e2,e1)`
# * `Entity-Origin(e2,e1)`
# * `Other` (undirected)
# 
# ---
# 
# ### 2. Precision and Support Computation
# 
# In `filter_and_rank_patterns`, the system evaluates each pattern by:
# 
# 1. **total_count**
#    Total frequency across all directed relations.
# 
# 2. **best_relation**
#    The directed relation with the highest count.
# 
# 3. **support**
#    The frequency for the dominant directed relation.
# 
# 4. **precision**
#    `precision = support / total_count`
# 
# A rule is accepted if:
# 
# * `precision ≥ 0.60`
# * `support ≥ 2`
# 
# This guarantees that rules consistently signal a **specific directed label**, not
# just the undirected relation type.
# 
# ---
# 
# ### 3. Rule Representation (Directed)
# 
# During rule creation, the directed label is parsed into:
# 
# * `relation`: full directed label, e.g. `"Cause-Effect(e2,e1)"`
# * `base_relation`: e.g. `"Cause-Effect"`
# * `direction`: `"e2,e1"` (or `None` for `Other`)
# 
# Example:
# 
# ```python
# {
#     "name": "Cause-Effect(e2,e1)_PREP_5821",
#     "relation": "Cause-Effect(e2,e1)",
#     "base_relation": "Cause-Effect",
#     "direction": "e2,e1",
#     "matcher_type": "lexical",
#     "pattern_type": "PREP",
#     "pattern_data": ["from"],
#     "precision": 0.81,
#     "support": 27,
#     "explanation": "PREP pattern: ['from']"
# }
# ```
# 
# Rules are sorted by:
# 
# 1. precision (descending)
# 2. support (descending)
# 
# This ensures high-quality directed signals fire first.
# 
# ---
# 
# ### 4. Why This Directed Rule-Based Approach Works
# 
# * **Fully directed**: rules distinguish `e1->e2` vs `e2->e1`, improving accuracy.
# * **Data-driven**: everything is mined automatically from the training data.
# * **Precise**: rule selection uses frequency-based precision scoring.
# * **Interpretable**: every prediction is grounded in an explicit linguistic pattern.
# * **Deterministic**: the same input always yields the same decision.
# * **Model-free inference**: no training/inference cost, extremely fast.
# 

# %%
def extract_candidate_patterns(processed_data):
    """
    Mine candidate lexical and dependency patterns from labeled training data.

    Args:
        processed_data: iterable of samples, each with:
            - 'relation': gold relation label
            - 'doc': spaCy Doc
            - 'e1_span', 'e2_span': entity spans
            - 'dep_path': dependency path between entities (optional)

    Returns:
        lexical_patterns: dict[pattern_key][relation] -> count
        dep_patterns: dict[pattern_key][relation] -> count
    """
    # Group samples by relation
    relation_groups = defaultdict(list)
    for sample in processed_data:
        relation_groups[sample['relation_directed']].append(sample)
    
    # Track pattern occurrences: pattern_key -> {relation -> count}
    lexical_patterns = defaultdict(lambda: defaultdict(int))
    dep_patterns = defaultdict(lambda: defaultdict(int))
    
    print("Mining candidate patterns from training data...")
    print("="*80)
    
    for relation, samples in relation_groups.items():
        print(f"\n{relation}: {len(samples)} samples")
        
        for sample in samples:
            doc = sample['doc']
            e1_span = sample['e1_span']
            e2_span = sample['e2_span']
            
            # Extract between-span features
            if e1_span.start < e2_span.start:
                between_span = doc[e1_span.end:e2_span.start]
            else:
                between_span = doc[e2_span.end:e1_span.start]
            
            # 1. LEXICAL PATTERNS: Between-span lemmas and bigrams
            between_lemmas = [t.lemma_.lower() for t in between_span if not t.is_punct]
            
            # Single lemmas
            for lemma in between_lemmas:
                if len(lemma) > 2:
                    pattern_key = ('LEMMA', lemma)
                    lexical_patterns[pattern_key][relation] += 1
            
            # Bigrams
            for i in range(len(between_lemmas) - 1):
                bigram = (between_lemmas[i], between_lemmas[i+1])
                pattern_key = ('BIGRAM', bigram)
                lexical_patterns[pattern_key][relation] += 1
            
            # Prepositions (very important)
            for token in between_span:
                if token.pos_ == 'ADP':
                    pattern_key = ('PREP', token.lemma_.lower())
                    lexical_patterns[pattern_key][relation] += 1
            
            # Context window: word before e1 and word after e2
            if e1_span.start > 0:
                before_e1 = doc[e1_span.start - 1]
                if not before_e1.is_punct and len(before_e1.lemma_) > 2:
                    pattern_key = ('BEFORE_E1', before_e1.lemma_.lower())
                    lexical_patterns[pattern_key][relation] += 1
            
            if e2_span.end < len(doc):
                after_e2 = doc[e2_span.end]
                if not after_e2.is_punct and len(after_e2.lemma_) > 2:
                    pattern_key = ('AFTER_E2', after_e2.lemma_.lower())
                    lexical_patterns[pattern_key][relation] += 1
            
            # Entity POS tag pattern
            pattern_key = ('ENTITY_POS', e1_span.root.pos_, e2_span.root.pos_)
            lexical_patterns[pattern_key][relation] += 1
            
            # 2. DEPENDENCY PATTERNS: e1 and e2 roles + verb
            e1_head = e1_span.root
            e2_head = e2_span.root
            
            # Find connecting verb (if any)
            dep_path = sample['dep_path']
            path_lemmas = [d[2] for d in dep_path] if dep_path else []
            path_deps = [d[0] for d in dep_path] if dep_path else []
            
            # Look for verb in path
            for token in doc:
                if token.pos_ == 'VERB':
                    # Check if this verb connects e1 and e2
                    e1_dep_to_verb = None
                    e2_dep_to_verb = None
                    
                    # Check e1 relation to verb
                    if e1_head.head == token:
                        e1_dep_to_verb = e1_head.dep_
                    elif e1_head == token:
                        e1_dep_to_verb = 'VERB_IS_E1'
                    
                    # Check e2 relation to verb
                    if e2_head.head == token:
                        e2_dep_to_verb = e2_head.dep_
                    elif e2_head == token:
                        e2_dep_to_verb = 'VERB_IS_E2'
                    
                    if e1_dep_to_verb and e2_dep_to_verb:
                        verb_lemma = token.lemma_.lower()
                        pattern_key = ('DEP_VERB', verb_lemma, e1_dep_to_verb, e2_dep_to_verb)
                        dep_patterns[pattern_key][relation] += 1
            
            # Simpler: just e1 and e2 dependency labels
            pattern_key = ('DEP_LABELS', e1_head.dep_, e2_head.dep_)
            dep_patterns[pattern_key][relation] += 1
    
    return lexical_patterns, dep_patterns

# %%
def _split_relation_and_direction(rel_directed):
    """
    Helper: split 'Cause-Effect(e1,e2)' -> ('Cause-Effect', 'e1,e2')
    Keeps 'Other' as ('Other', None).
    """
    if '(' in rel_directed and rel_directed.endswith(')'):
        base, dir_part = rel_directed.split('(', 1)
        direction = dir_part[:-1]  # strip trailing ')'
        base = base.strip()
        direction = direction.strip()
        return base, direction
    else:
        return rel_directed, None  # e.g. 'Other'

def filter_and_rank_patterns(lexical_patterns, dep_patterns, min_precision=0.60, min_support=2):
    """
    Filter patterns by precision and support, then rank them.
    Lower thresholds (precision=0.60, support=2) for better coverage.
    Returns: ordered list of rule dicts
    """
    rules = []
    
    # Process lexical patterns
    for pattern_key, relation_counts in lexical_patterns.items():
        total_count = sum(relation_counts.values())
        if total_count < min_support:
            continue
        
        # Find dominant (DIRECTED) relation
        best_relation = max(relation_counts, key=relation_counts.get)  # e.g. 'Cause-Effect(e2,e1)'
        best_count = relation_counts[best_relation]
        precision = best_count / total_count
        
        if precision >= min_precision:
            pattern_type, *pattern_data = pattern_key
            base_rel, direction = _split_relation_and_direction(best_relation)
            
            rule = {
                'name': f"{best_relation}_{pattern_type}_{hash(pattern_key) % 10000}",
                'relation': best_relation,        # full directed label
                'base_relation': base_rel,        # optional: undirected type
                'direction': direction,           # 'e1,e2' / 'e2,e1' / None for Other
                'matcher_type': 'lexical',
                'pattern_type': pattern_type,
                'pattern_data': pattern_data,
                'precision': precision,
                'support': best_count,
                'explanation': f"{pattern_type} pattern: {pattern_data}"
            }
            rules.append(rule)
    
    # Process dependency patterns  
    for pattern_key, relation_counts in dep_patterns.items():
        total_count = sum(relation_counts.values())
        if total_count < min_support:
            continue
        
        best_relation = max(relation_counts, key=relation_counts.get)
        best_count = relation_counts[best_relation]
        precision = best_count / total_count
        
        if precision >= min_precision:
            pattern_type, *pattern_data = pattern_key
            base_rel, direction = _split_relation_and_direction(best_relation)
            
            rule = {
                'name': f"{best_relation}_{pattern_type}_{hash(pattern_key) % 10000}",
                'relation': best_relation,
                'base_relation': base_rel,
                'direction': direction,
                'matcher_type': 'dependency',
                'pattern_type': pattern_type,
                'pattern_data': pattern_data,
                'precision': precision,
                'support': best_count,
                'explanation': f"{pattern_type}: {pattern_data}"
            }
            rules.append(rule)
    
    # Sort by precision (descending), then support (descending)
    rules.sort(key=lambda r: (-r['precision'], -r['support']))
    
    return rules

# %%
# Mine patterns from training data
print("\nStep 1: Mining patterns from training data...")
lexical_patterns, dep_patterns = extract_candidate_patterns(train_processed)

print(f"\nFound {len(lexical_patterns)} unique lexical pattern candidates")
print(f"Found {len(dep_patterns)} unique dependency pattern candidates")

# Filter and rank patterns
print("\nStep 2: Filtering by precision ≥ 0.60 and support ≥ 2...")
DISCOVERED_RULES = filter_and_rank_patterns(lexical_patterns, dep_patterns, 
                                             min_precision=0.60, min_support=2)

print(f"\nDiscovered {len(DISCOVERED_RULES)} high-quality rules")
print("\nTop 10 rules:")
print("="*100)
print(f"{'Relation':<25} {'Type':<15} {'Precision':<12} {'Support':<10} {'Pattern'}")
print("-"*100)
for rule in DISCOVERED_RULES[:10]:
    print(f"{rule['relation']:<25} {rule['pattern_type']:<15} {rule['precision']:<12.3f} {rule['support']:<10} {str(rule['pattern_data'])[:40]}")

# %%
# Visualize rules by relation for explainability
def visualize_rules_by_relation(rules, top_n=5):
    """
    Display the top-N highest precision rules for each relation.

    Note:
        `rules` is already be sorted globally by (precision desc, support desc) as produced by `filter_and_rank_patterns()`.
        
        Because of this, taking the first N rules in each relation group
        truly reflects the strongest patterns for that relation.

    This function does not re-sort; it only groups and prints the top rules.
    """
    relation_rules = defaultdict(list)
    
    for rule in rules:
        relation_rules[rule['relation']].append(rule)
    
    print("\n" + "="*100)
    print("TOP RULES BY RELATION TYPE (for Explainability)")
    print("="*100)
    
    for relation in sorted(relation_rules.keys()):
        rules_list = relation_rules[relation][:top_n]
        print(f"\n{'='*100}")
        print(f"Relation: {relation} ({len(relation_rules[relation])} total rules)")
        print(f"{'='*100}")
        
        for i, rule in enumerate(rules_list, 1):
            print(f"\n  Rule {i}: {rule['name']}")
            print(f"    Type: {rule['pattern_type']}")
            print(f"    Precision: {rule['precision']:.3f} | Support: {rule['support']}")
            
            # Convert to spaCy Matcher syntax
            pattern_type = rule['pattern_type']
            pattern_data = rule['pattern_data']
            
            if pattern_type == 'LEMMA':
                spacy_pattern = f'[{{"LEMMA": "{pattern_data[0]}"}}]'
            elif pattern_type == 'BIGRAM':
                spacy_pattern = f'[{{"LEMMA": "{pattern_data[0][0]}"}}, {{"LEMMA": "{pattern_data[0][1]}"}}]'
            elif pattern_type == 'PREP':
                spacy_pattern = f'[{{"LEMMA": "{pattern_data[0]}", "POS": "ADP"}}]'
            elif pattern_type == 'BEFORE_E1':
                spacy_pattern = f'Word before E1: {{"LEMMA": "{pattern_data[0]}"}}'
            elif pattern_type == 'AFTER_E2':
                spacy_pattern = f'Word after E2: {{"LEMMA": "{pattern_data[0]}"}}'
            elif pattern_type == 'ENTITY_POS':
                spacy_pattern = f'E1.pos_=="{pattern_data[0]}" AND E2.pos_=="{pattern_data[1]}"'
            elif pattern_type == 'DEP_VERB':
                verb, e1_dep, e2_dep = pattern_data
                # Show structured DependencyMatcher pattern
                # REL_OPs are ">" indicating head relations
                spacy_pattern = f'''
            DependencyMatcher Pattern:
            [
                {{
                    "RIGHT_ID": "verb",
                    "RIGHT_ATTRS": {{"LEMMA": "{verb}", "POS": "VERB"}}
                }},
                {{
                    "LEFT_ID": "verb",
                    "REL_OP": ">",  
                    "RIGHT_ID": "e1",
                    "RIGHT_ATTRS": {{"DEP": "{e1_dep}"}}
                }},
                {{
                    "LEFT_ID": "verb",
                    "REL_OP": ">",  # verb is head of e2
                    "RIGHT_ID": "e2",
                    "RIGHT_ATTRS": {{"DEP": "{e2_dep}"}}
                }}
            ]'''
            elif pattern_type == 'DEP_LABELS':
                spacy_pattern = f'E1.dep_=="{pattern_data[0]}" AND E2.dep_=="{pattern_data[1]}"'
            else:
                spacy_pattern = str(pattern_data)
            
            print(f"    spaCy Pattern: {spacy_pattern}")

visualize_rules_by_relation(DISCOVERED_RULES, top_n=3)


# %%
def analyze_relation_features(processed_data):
    """
    Analyze linguistic features for each *directed* relation type.

    Each sample in `processed_data` is expected to contain:
        - 'relation_directed': gold directed relation label (str), e.g. "Cause-Effect(e1,e2)" or "Other"
        - 'doc': spaCy Doc
        - 'e1_span', 'e2_span': spaCy spans for the two entities
        - 'dep_path': list of (dep_label, ...) along shortest path (optional)
        - 'between_words': list of dicts with 'text' and 'lemma'

    Returns:
        dict[relation_directed] -> {
            'count': int,
            'top_lemmas': [(lemma, freq)],
            'top_verbs': [(lemma, freq)],
            'top_preps': [(lemma, freq)],
            'top_dep_paths': [(path_tuple, freq)],
            'top_between_words': [(lemma, freq)]
        }
    """
    # Group by DIRECTED relation type
    relation_groups = defaultdict(list)
    for sample in processed_data:
        relation_groups[sample['relation_directed']].append(sample)
    
    # Analyze each relation
    relation_analysis = {}
    
    for relation, samples in relation_groups.items():
        all_lemmas = []
        all_verbs = []
        all_preps = []
        all_dep_paths = []
        all_between_words = []
        
        for sample in samples:
            doc = sample['doc']
            
            # Collect lemmas (excluding entities)
            e1_tokens = set(range(sample['e1_span'].start, sample['e1_span'].end))
            e2_tokens = set(range(sample['e2_span'].start, sample['e2_span'].end))
            
            for token in doc:
                if token.i not in e1_tokens and token.i not in e2_tokens:
                    lemma = token.lemma_.lower()
                    
                    # Collect verbs (don't filter stopwords for verbs)
                    if token.pos_ == 'VERB' and not token.is_punct and len(lemma) > 2:
                        all_verbs.append(lemma)
                    
                    # Collect prepositions (INCLUDE stopwords like "of", "in", "at")
                    if token.pos_ == 'ADP' and not token.is_punct:
                        all_preps.append(lemma)
                    
                    # Collect other lemmas (filter stopwords for general keywords)
                    if not token.is_stop and not token.is_punct and len(lemma) > 2:
                        all_lemmas.append(lemma)
            
            # Collect dependency paths
            if sample['dep_path']:
                path_deps = tuple([d[0] for d in sample['dep_path']])
                all_dep_paths.append(path_deps)
            
            # Between words
            for word in sample['between_words']:
                if word['text'].strip() and len(word['lemma']) > 2:
                    all_between_words.append(word['lemma'].lower())
        
        # Count frequencies
        lemma_freq = Counter(all_lemmas).most_common(30)
        verb_freq = Counter(all_verbs).most_common(15)
        prep_freq = Counter(all_preps).most_common(10)
        dep_path_freq = Counter(all_dep_paths).most_common(10)
        between_freq = Counter(all_between_words).most_common(20)
        
        relation_analysis[relation] = {
            'count': len(samples),
            'top_lemmas': lemma_freq,
            'top_verbs': verb_freq,
            'top_preps': prep_freq,
            'top_dep_paths': dep_path_freq,
            'top_between_words': between_freq
        }
    
    return relation_analysis

# %% [markdown]
# ## 5. Deterministic Rule Application Engine
# 
# We implement a deterministic, fully explainable **directed** rule application
# engine using:
# 
# * **spaCy Matcher** – for token-level patterns (BIGRAM, PREP)
# * **spaCy PhraseMatcher** – for efficient lemma patterns (LEMMA)
# * **spaCy DependencyMatcher** – for verb–entity dependency structures (DEP_VERB)
# 
# This follows spaCy’s recommended pattern-matching workflow.
# 
# ---
# 
# ### How does it work?
# 
# Rules have already been discovered and ranked by `(precision desc, support desc)`
# using `filter_and_rank_patterns()`.
# Each rule carries:
# 
# * a **directed relation label**, e.g. `Cause-Effect(e1,e2)`
# * the base relation (e.g. `Cause-Effect`)
# * the predicted direction (`e1,e2` or `e2,e1`, or `None` for `Other`)
# * a pattern type and pattern data
# * precision and support statistics
# 
# The classifier iterates over rules in ranked order:
# **the first matching rule determines the final directed prediction**.
# 
# ---
# 
# ### Applying the Rules
# 
# `apply_rule_based_classifier` pre-compiles all patterns into matcher objects:
# 
# * **Matcher**
# * **PhraseMatcher**
# * **DependencyMatcher**
# 
# These matchers operate on the spaCy `Doc` and make rule evaluation fast and deterministic.
# 
# ---
# 
# ### Why We Use Multiple Pattern Types
# 
# Each pattern type captures a different, complementary signal:
# 
# * **Lexical & Bigrams:**
#   Catch frequent surface cues between entities (robust, high coverage, no dependency errors).
# 
# * **Prepositions:**
#   Encode strong relation markers (`in`, `of`, `from`, `to`) and directionality.
# 
# * **Dependency (DEP_VERB):**
#   Capture grammatical roles of entities around a shared verb (high-precision structural cues).
# 
# Using only dependency patterns would be brittle and overly specific; using only lexical patterns would miss structural information.
# Combining all gives **both coverage and precision** while keeping rules interpretable.
# 
# ---
# 
# ### Classification Process
# 
# For each sample:
# 
# 1. All matchers are applied in the global rule order.
# 2. Matching is directed:
# 
#    * lexical patterns are matched in the **between-entity span**
#    * dependency patterns are matched over the **entire sentence**
# 3. Context patterns (`BEFORE_E1`, `AFTER_E2`) and entity POS/DEP patterns
#    are checked with simple Python conditions.
# 
# Once a rule matches:
# 
# * its **directed relation label** is emitted
#   (e.g. `Entity-Destination(e2,e1)`)
# * its stored direction (`rule['direction']`) is returned
# * an explanation is recorded with the rule name, pattern, precision, and support
# 
# If **no rule** fires:
# 
# * the system returns `"Other"`
# * with **no assigned direction**, ensuring correct behavior for SemEval’s undirected “Other”
# 
# ---
# 
# ### Why this design works
# 
# * **Deterministic** — the same sentence always yields the same directed label.
# * **Explainable** — every prediction is tied to a human-readable rule.
# * **Directed-aware** — rules distinguish `e1 -> e2` vs `e2 -> e1` patterns.
# * **Efficient** — no model inference; only spaCy pattern matching.
# * **Modular** — lexical, syntactic, and contextual features cooperate cleanly.

# %%
from spacy.matcher import Matcher, PhraseMatcher, DependencyMatcher

def apply_rule_based_classifier(samples, rules, nlp):
    """
    Apply rule-based classification using spaCy's proper matchers:
    - Matcher: for token sequences (LEMMA, BIGRAM, PREP)
    - PhraseMatcher: for efficient phrase matching (LEMMA lists)
    - DependencyMatcher: for dependency patterns (DEP_VERB)
    
    This follows spaCy documentation best practices.
    """
    # Pre-compile all matchers
    token_matcher = Matcher(nlp.vocab)
    phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
    dep_matcher = DependencyMatcher(nlp.vocab)
    
    # # Map match IDs back to rules
    # rule_lookup = {}  # match_id -> rule
    
    # === 1. Compile Token Patterns (BIGRAM, PREP) ===
    for i, rule in enumerate(rules):
        match_id = f"rule_{i}"
        # rule_lookup[match_id] = rule
        
        if rule['matcher_type'] == 'lexical':
            pattern_type = rule['pattern_type']
            pattern_data = rule['pattern_data']
            
            if pattern_type == 'BIGRAM':
                pattern = [{"LEMMA": pattern_data[0][0]}, {"LEMMA": pattern_data[0][1]}]
                token_matcher.add(match_id, [pattern])
            
            elif pattern_type == 'PREP':
                pattern = [{"LEMMA": pattern_data[0], "POS": "ADP"}]
                token_matcher.add(match_id, [pattern])
    
    # === 2. Compile Phrase Patterns (LEMMA) - More efficient ===
    lemma_rules = [(i, r) for i, r in enumerate(rules) 
                   if r['matcher_type'] == 'lexical' and r['pattern_type'] == 'LEMMA']
    
    if lemma_rules:
        # Create Doc patterns with full pipeline to get lemmas
        # Use nlp() instead of nlp.make_doc() when attr="LEMMA" is needed
        patterns = [nlp(r['pattern_data'][0]) for _, r in lemma_rules]
        match_ids = [f"rule_{i}" for i, _ in lemma_rules]
        
        for match_id, pattern in zip(match_ids, patterns):
            phrase_matcher.add(match_id, [pattern])
    
    # === 3. Compile Dependency Patterns (DEP_VERB) ===
    for i, rule in enumerate(rules):
        if rule['pattern_type'] == 'DEP_VERB':
            match_id = f"rule_{i}"
            verb_lemma, e1_dep, e2_dep = rule['pattern_data']
            
            # Build DependencyMatcher pattern
            pattern = [
                # Anchor: the verb
                {
                    "RIGHT_ID": "verb",
                    "RIGHT_ATTRS": {"LEMMA": verb_lemma, "POS": "VERB"}
                },
                # E1 connected to verb
                {
                    "LEFT_ID": "verb",
                    "REL_OP": ">",  # verb is head of e1
                    "RIGHT_ID": "e1",
                    "RIGHT_ATTRS": {"DEP": e1_dep}
                },
                # E2 connected to verb
                {
                    "LEFT_ID": "verb",
                    "REL_OP": ">",  # verb is head of e2
                    "RIGHT_ID": "e2",
                    "RIGHT_ATTRS": {"DEP": e2_dep}
                }
            ]
            
            dep_matcher.add(match_id, [pattern])
    
    # === 4. Apply Matchers to Samples ===
    predictions, directions, explanations = [], [], []
    
    for sample in tqdm(samples, desc="Classifying"):
        doc = sample['doc']
        e1_span, e2_span = sample['e1_span'], sample['e2_span']
        between_span = doc[e1_span.end:e2_span.start] if e1_span.start < e2_span.start else doc[e2_span.end:e1_span.start]
        e1_head, e2_head = e1_span.root, e2_span.root
        
        matched_rule = None
        
        # Apply rules in order (iterate through rules to maintain priority)
        for i, rule in enumerate(rules):
            match_id = f"rule_{i}"
            pattern_type = rule['pattern_type']
            pattern_data = rule['pattern_data']
            
            # === Token Matcher (BIGRAM, PREP) ===
            if pattern_type in ['BIGRAM', 'PREP']:
                matches = token_matcher(between_span)
                if any(nlp.vocab.strings[m[0]] == match_id for m in matches):
                    matched_rule = rule
                    break
            
            # === Phrase Matcher (LEMMA) ===
            elif pattern_type == 'LEMMA':
                matches = phrase_matcher(between_span)
                if any(nlp.vocab.strings[m[0]] == match_id for m in matches):
                    matched_rule = rule
                    break
            
            # === Dependency Matcher (DEP_VERB) ===
            elif pattern_type == 'DEP_VERB':
                matches = dep_matcher(doc)
                # Check if match involves our entities
                for match_id_found, token_ids in matches:
                    if nlp.vocab.strings[match_id_found] == match_id:
                        # Verify entities are involved in match
                        e1_in_match = any(t in range(e1_span.start, e1_span.end) for t in token_ids)
                        e2_in_match = any(t in range(e2_span.start, e2_span.end) for t in token_ids)
                        if e1_in_match or e2_in_match:
                            matched_rule = rule
                            break
                if matched_rule:
                    break
            
            # === Context Patterns (Manual checks) ===
            elif pattern_type == 'BEFORE_E1' and e1_span.start > 0:
                if doc[e1_span.start - 1].lemma_.lower() == pattern_data[0]:
                    matched_rule = rule
                    break
            
            elif pattern_type == 'AFTER_E2' and e2_span.end < len(doc):
                if doc[e2_span.end].lemma_.lower() == pattern_data[0]:
                    matched_rule = rule
                    break
            
            # === Entity POS Pattern ===
            elif pattern_type == 'ENTITY_POS':
                if e1_head.pos_ == pattern_data[0] and e2_head.pos_ == pattern_data[1]:
                    matched_rule = rule
                    break
            
            # === Simple Dependency Labels ===
            elif pattern_type == 'DEP_LABELS':
                if e1_head.dep_ == pattern_data[0] and e2_head.dep_ == pattern_data[1]:
                    matched_rule = rule
                    break
        
        # Record prediction
        if matched_rule:
            predictions.append(matched_rule['relation'])
            directions.append(matched_rule['direction'])
            explanations.append(f"Rule {matched_rule['name']}: {matched_rule['explanation']} (precision={matched_rule['precision']:.2f}, support={matched_rule['support']})")
        else:
            predictions.append('Other')      # undirected "Other"
            directions.append(None)          # or '' – but NOT "e1,e2"
            explanations.append('No high-precision rule matched; defaulting to Other.')
    
    return predictions, directions, explanations

# %%
# Test the rule-based classifier on a few samples
print("Testing rule-based classifier on sample sentences...")
print("="*80)

# Quick test on 5 samples
test_samples = train_processed[:5]
test_preds, test_dirs, test_expls = apply_rule_based_classifier(test_samples, DISCOVERED_RULES, nlp)

for i, (sample, relation, explanation) in enumerate(zip(test_samples, test_preds, test_expls)):
    print(f"\nSample {i+1}:")
    print(f"Text: {sample['text'][:100]}...")
    print(f"E1: '{sample['e1_span'].text}' | E2: '{sample['e2_span'].text}'")
    print(f"True: {sample['relation_directed']}")
    print(f"Predicted: {relation}")
    print(f"Explanation: {explanation}")
    print(f"Match: {'✓' if relation == sample['relation_directed'] else '✗'}")

# %% [markdown]
# **Generate Predictions Using the Rule-Based Classifier**
# 
# We evaluate the discovered rules by applying the deterministic rule engine
# to the processed datasets. The function `apply_rule_based_classifier`:
# 
# 1. Pre-compiles all rules into spaCy matchers (token, phrase, dependency).
# 2. For each sample, checks the rules in ranked order (by precision and support).
# 3. Stops at the first matching rule (“decision list” behavior).
# 4. Returns the **directed predicted relation**
#    (e.g. `Cause-Effect(e1,e2)` or `Component-Whole(e2,e1)`),
#    the predicted direction (when applicable),
#    and a human-readable explanation of the rule that fired.
# 
# This produces **fully interpretable, deterministic, and direction-aware**
# predictions for every instance.

# %%
print("Evaluating on training set...")
train_predictions, train_directions, train_explanations = apply_rule_based_classifier(
    train_processed, DISCOVERED_RULES, nlp
)
train_true = [s['relation_directed'] for s in train_processed]

print("\nEvaluating on test set...")
test_predictions, test_directions, test_explanations = apply_rule_based_classifier(
    test_processed, DISCOVERED_RULES, nlp
)
test_true = [s['relation_directed'] for s in test_processed]

# %% [markdown]
# ## 6. Evaluation with Deterministic Rules
# 
# ---
# 
# We now evaluate the performance of the deterministic rule-based classifier on
# both the training and test sets. For each split, we compute:
# 
# - **Accuracy** – overall correctness of predictions.
# - **Per-class precision, recall, and F1-score** – to show how well the rules
#   capture each relation category.
# - **Support** – number of samples per class.
# 
# This gives a clear picture of how well the learned rules generalize to unseen
# data and which relations are easy or difficult for the rule-based approach.

# %%
# Comprehensive evaluation metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("="*80)
print("DETERMINISTIC RULE-BASED SYSTEM EVALUATION")
print("="*80)

# Training set evaluation
print("\n### TRAINING SET RESULTS ###\n")
train_accuracy = accuracy_score(train_true, train_predictions)
print(f"Accuracy: {train_accuracy:.3f}")

print("\nPer-class metrics:")
print(classification_report(train_true, train_predictions, zero_division=0))
print("="*80)

# Test set evaluation
print("\n### TEST SET RESULTS ###\n")
test_accuracy = accuracy_score(test_true, test_predictions)
print(f"Accuracy: {test_accuracy:.3f}")

print("\nPer-class metrics:")
print(classification_report(test_true, test_predictions, zero_division=0, digits=3))

# %%
# Generate data-driven patterns
print("\n" + "="*80)
print("GENERATING DATA-DRIVEN PATTERNS")
print("Top features per relation extracted from analysis")
print("="*80)

# First, analyze the training data to get relation features
relation_features = analyze_relation_features(train_processed)
data_driven_patterns = generate_patterns_from_analysis(relation_features)

# Display generated patterns
for relation in sorted(data_driven_patterns.keys()):
    patterns = data_driven_patterns[relation]
    print(f"\n{relation}:")
    print(f"  Keywords ({len(patterns['keywords'])}): {patterns['keywords'][:10]}")
    print(f"  Verbs ({len(patterns['verb_patterns'])}): {patterns['verb_patterns']}")
    print(f"  Preps ({len(patterns['prep_patterns'])}): {patterns['prep_patterns']}")
    print(f"  Dep patterns: {len(patterns['dependency_patterns'])} patterns extracted")

print("\n" + "="*80)
print("Data-driven patterns generated successfully!")
print("These patterns are based on actual frequency analysis of the training data.")
print("="*80)

# %% [markdown]
# **Rule Diagnostics and Summary Statistics**
# 
# To better understand the behavior of the discovered rules, we compute several
# diagnostic statistics:
# 
# - **Number of rules per relation** – shows how many high-precision patterns were
#   learned for each class.
# - **Average precision and support** – summarize the overall quality of the
#   rule set. Higher precision indicates more reliable rules; higher support
#   indicates patterns that appear frequently in training data.
# - **Macro-averaged F1 and accuracy** – provide a global summary of system
#   performance on both training and test sets.
# 
# These diagnostics help identify relations that are well-covered by rules and
# those that may need additional patterns or refinement.
# 

# %%
# Rule statistics and diagnostics
print("\n" + "="*80)
print("RULE DIAGNOSTICS")
print("="*80)

# Count rules per relation
relation_rule_counts = defaultdict(int)
for rule in DISCOVERED_RULES:
    relation_rule_counts[rule['relation']] += 1

print("\nRules discovered per relation:")
print(f"{'Relation':<30} {'Number of Rules'}")
print("-"*50)
for relation in sorted(relation_rule_counts.keys()):
    print(f"{relation:<30} {relation_rule_counts[relation]}")

print(f"\nTotal rules: {len(DISCOVERED_RULES)}")
print(f"Average precision: {np.mean([r['precision'] for r in DISCOVERED_RULES]):.3f}")
print(f"Average support: {np.mean([r['support'] for r in DISCOVERED_RULES]):.1f}")

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

# ========= Macro-averaged metrics =========
test_macro_precision  = precision_score(test_true,  test_predictions,
                                        average='macro', zero_division=0)
train_macro_precision = precision_score(train_true, train_predictions,
                                        average='macro', zero_division=0)

test_macro_recall  = recall_score(test_true,  test_predictions,
                                  average='macro', zero_division=0)
train_macro_recall = recall_score(train_true, train_predictions,
                                  average='macro', zero_division=0)

test_macro_f1  = f1_score(test_true,  test_predictions,
                          average='macro', zero_division=0)
train_macro_f1 = f1_score(train_true, train_predictions,
                          average='macro', zero_division=0)

# ========= Summary table (Train vs Test) =========
print(f"\n{'Metric':<30} {'Test Set':<15} {'Train Set':<15}")
print("-" * 60)
print(f"{'Macro-averaged Precision':<30} {test_macro_precision:<15.3f} {train_macro_precision:<15.3f}")
print(f"{'Macro-averaged Recall':<27}    {test_macro_recall:<15.3f} {train_macro_recall:<15.3f}")
print(f"{'Macro-averaged F1':<23}        {test_macro_f1:<15.3f} {train_macro_f1:<15.3f}")
print(f"{'Accuracy':<13}                  {test_accuracy:<15.3f} {train_accuracy:<15.3f}")

# %%
from collections import defaultdict

print("\n" + "="*80)
print("EXAMPLE RULE FIRINGS (Test Set)")
print("="*80)

# Group test samples by *directed* relation where prediction is correct
test_by_relation = defaultdict(list)
for i, sample in enumerate(test_processed):
    gold_dir = sample['relation_directed']          # e.g. "Cause-Effect(e1,e2)"
    pred_dir = test_predictions[i]                  # directed prediction

    # Only keep correctly classified, non-Other examples
    if pred_dir == gold_dir and gold_dir != 'Other':
        test_by_relation[gold_dir].append((sample, i))

# Show 1–2 examples per directed relation
for relation in sorted(test_by_relation.keys())[:5]:  # First 5 directed relations
    examples = test_by_relation[relation][:2]         # Up to 2 examples each
    
    print(f"\n### {relation} ###")
    for sample, idx in examples:
        print(f"\nText: {sample['text']}")
        print(f"E1: '{sample['e1_span'].text}' | E2: '{sample['e2_span'].text}'")
        
        explanation = test_explanations[idx]
        print(f"Rule fired: {explanation}")
        
        # If it's a DEP_VERB rule, show structured DependencyMatcher pattern
        if 'DEP_VERB' in explanation:
            rule_name = explanation.split(':')[0].replace('Rule ', '')
            for rule in DISCOVERED_RULES:
                if rule['name'] == rule_name and rule['pattern_type'] == 'DEP_VERB':
                    verb, e1_dep, e2_dep = rule['pattern_data']
                    print(f"\n  DependencyMatcher Pattern:")
                    print(f"  [")
                    print(f"      {{")
                    print(f"          \"RIGHT_ID\": \"verb\",")
                    print(f"          \"RIGHT_ATTRS\": {{\"LEMMA\": \"{verb}\", \"POS\": \"VERB\"}}")
                    print(f"      }},")
                    print(f"      {{")
                    print(f"          \"LEFT_ID\": \"verb\",")
                    print(f"          \"REL_OP\": \">\",  # verb is head of e1")
                    print(f"          \"RIGHT_ID\": \"e1\",")
                    print(f"          \"RIGHT_ATTRS\": {{\"DEP\": \"{e1_dep}\"}}")
                    print(f"      }},")
                    print(f"      {{")
                    print(f"          \"LEFT_ID\": \"verb\",")
                    print(f"          \"REL_OP\": \">\",  # verb is head of e2")
                    print(f"          \"RIGHT_ID\": \"e2\",")
                    print(f"          \"RIGHT_ATTRS\": {{\"DEP\": \"{e2_dep}\"}}")
                    print(f"      }}")
                    print(f"  ]")
                    break
        
        print("-" * 80)


# %% [markdown]
# ## 7. Save Predictions for Official Scorer (Optional)
# 
# The files are saved for potential offline evaluation with the official Perl scorer.
# Note: The Perl scorer can be slow. Use the sklearn metrics above for quick evaluation.

# %%
def save_predictions_for_scorer(predictions, processed_data, output_file):
    """
    Save predictions in official scorer format:
        ID\tRelationLabel
    where RelationLabel is already a full label like 'Cause-Effect(e1,e2)' or 'Other'.
    """
    with open(output_file, 'w') as f:
        for pred, sample in zip(predictions, processed_data):
            sample_id = sample['id']
            f.write(f"{sample_id}\t{pred}\n")
    
    print(f"Saved {len(predictions)} predictions to {output_file}")



def create_answer_key(processed_data, output_file):
    """
    Create answer key file from processed data in official format:
        ID\tRelationLabel
    where RelationLabel is the *directed* gold label, e.g. 'Cause-Effect(e1,e2)' or 'Other'.
    """
    with open(output_file, 'w') as f:
        for sample in processed_data:
            sample_id = sample['id']
            gold_label = sample['relation_directed']   # already 'Other' or 'RelType(e1,e2)'
            f.write(f"{sample_id}\t{gold_label}\n")
    
    print(f"Saved {len(processed_data)} gold labels to {output_file}")



print("Preparing files for official scorer...")

save_predictions_for_scorer(train_predictions, train_processed, 'rb_train_predictions_directed.txt')
create_answer_key(train_processed, 'rb_train_answer_key_directed.txt')

save_predictions_for_scorer(test_predictions, test_processed, 'rb_test_predictions_directed.txt')
create_answer_key(test_processed, 'rb_test_answer_key_directed.txt')

# %% [markdown]
# ## 8. Error Analysis
# 
# Analyze misclassifications to understand system limitations.
# 

# %%
# Analyze misclassifications
def analyze_errors(samples, predictions, true_labels, explanations, n_samples=20):
    """Analyze misclassified examples."""
    errors = []
    
    for i, (sample, pred, true) in enumerate(zip(samples, predictions, true_labels)):
        if pred != true:
            errors.append({
                'index': i,
                'sample': sample,
                'predicted': pred,
                'true': true,
                'text': sample['text'],
                'explanation': explanations[i]
            })
    
    print(f"Total errors: {len(errors)} / {len(samples)} ({len(errors)/len(samples)*100:.1f}%)")
    print(f"\nShowing first {min(n_samples, len(errors))} errors:\n")
    print("="*80)
    
    for i, error in enumerate(errors[:n_samples]):
        print(f"\nError {i+1}:")
        print(f"Text: {error['text']}")
        print(f"Entity 1: {error['sample']['e1_span'].text}")
        print(f"Entity 2: {error['sample']['e2_span'].text}")
        print(f"True relation: {error['true']}")
        print(f"Predicted: {error['predicted']}")
        print(f"Rule applied: {error['explanation']}")
        
        # Show between words and dependency info
        between_words = [w['text'] for w in error['sample']['between_words']]
        print(f"Between words: {between_words}")
        
        # Show dependency path
        dep_path = error['sample']['dep_path']
        if dep_path:
            path_str = ' -> '.join([f"{d[0]}({d[2]})" for d in dep_path[:5]])
            print(f"Dependency path: {path_str}")
        
        print("-" * 80)
    
    return errors

# %%
# Error distribution by relation type
def analyze_error_patterns(samples, predictions, true_labels):
    """Analyze error patterns by relation type."""
    error_matrix = defaultdict(lambda: defaultdict(int))
    
    for sample, pred, true in zip(samples, predictions, true_labels):
        if pred != true:
            error_matrix[true][pred] += 1
    
    print("\nMost Common Misclassification Patterns:")
    print("="*80)
    print(f"{'True Label':<25} {'Predicted As':<25} {'Count':<10}")
    print("-"*80)
    
    # Sort by count
    all_errors = []
    for true_label in error_matrix:
        for pred_label in error_matrix[true_label]:
            count = error_matrix[true_label][pred_label]
            all_errors.append((true_label, pred_label, count))
    
    all_errors.sort(key=lambda x: x[2], reverse=True)
    
    for true_label, pred_label, count in all_errors[:15]:
        print(f"{true_label:<25} {pred_label:<25} {count:<10}")
    
    return error_matrix

error_patterns = analyze_error_patterns(test_processed, test_predictions, test_true)

# %%
# Analyze test set errors
print("Analyzing errors on test set...")
test_errors = analyze_errors(test_processed, test_predictions, test_true, test_explanations, n_samples=15)

# %% [markdown]
# # Quantitative, Qualitative and Error Analysis

# %% [markdown]
# ## Confusion Matrix
# ---
# <img src="images/confusion_matrix_test__rb_system_directed.png" alt="Confusion Matrix of Training Data Set (Directed)" width="500" />
# 
# ---
# <img src="images/confusion_matrix_test__rb_system_directed.png" alt="Confusion Matrix of Test Data Set (Directed)" width="500" />

# %% [markdown]
# ## 9. Quantitative and Qualitative Interpretation of Results
# 
# The deterministic rule-based system achieves an accuracy of **0.578 on the training set** and **0.497 on the test set**. The macro-F1 score decreases from **0.53 (train)** to **0.43 (test)**. This performance profile is characteristic of precision-oriented rule-based systems, where surface patterns captured during training do not fully cover the linguistic variability found in unseen test data.
# 
# 
# 
# 
# 
# 
# 
# ### **Precision–Recall Behaviour**
# 
# 
# 
# The system exhibits a distinct **high-precision / low-recall** trade-off for semantically clear relations, while struggling with ambiguous ones.
# 
# 
# 
# * **Precision is generally distinct:** When a rule triggers, it is often correct. For example, *Content-Container(e2,e1)* achieves **96% precision** in training and **86%** in testing.
# 
# * **Recall is the bottleneck:** The system relies on specific lexical triggers. If a sentence lacks a mined pattern (e.g., a specific preposition or verb structure), the system fails to classify it, defaulting to "Other." This results in low recall for specific classes (e.g., *Component-Whole(e1,e2)* recall is only **3.7%** in the test set).
# 
# * **The "Other" Class Sink:** The *Other* category shows high recall (**0.65 Train / 0.48 Test**) but very low precision (**0.29 Train / 0.21 Test**). This confirms that the system defaults to *Other* far too often, absorbing many valid relation instances that lacked recognizable triggers.
# 
# 
# 
# ---
# 
# 
# 
# ### **Strong Performing Relations**
# 
# 
# 
# Specific relations maintain robust performance across both sets due to highly reliable lexical markers:
# 
# 
# 
# * **Cause-Effect:** Consistently strong (Test F1 ~0.82 for *e1,e2*). Verbs like "caused," "generated," and "triggered" are unambiguous markers that generalize well.
# 
# * **Entity-Destination(e1,e2):** Achieves high scores (Test F1 0.78) driven by directional prepositions like "into" and "to." However, note the complete failure of *Entity-Destination(e2,e1)* (0.00 F1), likely due to extreme data sparsity (only 1 support example).
# 
# * **Content-Container:** Performs well when the container follows the content (e.g., "apples in the basket"), utilizing strong prepositional cues like "in" or "inside."
# 
# 
# 
# ---
# 
# 
# 
# ### **Challenging Relations and Generalization Gaps**
# 
# 
# 
# The drop in performance from Train to Test is most visible in relations that rely on ambiguous prepositions or semantic world knowledge rather than syntactic structure.
# 
# 
# 
# * **Component-Whole:** This is the system's weakest point on unseen data.
# 
#     * *Train F1:* 0.25 (e1,e2)
# 
#     * *Test F1:* **0.067** (e1,e2)
# 
#     * *Analysis:* This relation frequently uses the preposition "of" (e.g., "handle of the door"), which is statistically overloaded and used in almost every other relation type. The rules likely overfitted to specific training nouns and failed to generalize to new vocabulary in the test set.
# 
# * **Member-Collection:** Performance is poor (Test F1 ~0.19). Determining if an entity is a "member" of a group (e.g., "student-class" vs. "tree-forest") often requires semantic knowledge bases rather than simple surface patterns.
# 
# * **Product-Producer:** While decent in training (F1 ~0.56 for e2,e1), it degrades in testing (F1 ~0.43). The diversity of verbs indicating creation (manufactured, built, cooked, wrote) makes it difficult for a finite rule list to achieve high coverage.
# 
# 
# 
# ---
# 
# ### **Summary of System Bias**
# 
# 
# 
# The quantitative results highlight a clear bias in the deterministic approach:
# 
# 
# 
# 1.  **Over-specificity:** The system learns patterns that are too specific to the training vocabulary, leading to a **~14% drop in accuracy** on the test set.
# 
# 2.  **Directionality Issues:** The system struggles to generalize directional subtypes when the support is unbalanced. For example, *Entity-Origin(e2,e1)* (Train F1 0.27) lags significantly behind *Entity-Origin(e1,e2)* (Train F1 0.77) simply because fewer passive constructions appear in the text to generate rules.
# 
# 3.  **Default Class Dominance:** The low precision of the *Other* class indicates that a significant number of predictions for *Other* are actually false negatives—valid relations that the rules failed to detect.
# 
# 
# 
# In conclusion, the system functions as a high-precision filter for distinct relations (Cause, Destination) but lacks the soft-matching capability required to handle the ambiguity of Component-Whole or Member-Collection relations effectively.
# 

# %% [markdown]
# ## Rule Diagnostics Overview
# 
# <img src="images/rule_diagnostics.png" alt="Rule Diagnostics Summary (Directed)" width="500" />
# 
# The rule-mining procedure extracted a total of **1651 distinct rules** across all relation types. The metrics associated with these rules provide insight into the system's underlying logic: high specificity and reliance on exact surface patterns.
# 
# 
# 
# ### **Rule Quality and Granularity**
# 
# 
# 
# * **High Precision (0.868 average):** The high average precision indicates that the extracted patterns are reliable. When a rule fires, it is overwhelmingly likely to be correct. The system prioritizes "safe" bets over broad generalizations.
# 
# * **Low Support (6.1 average):** The low average support suggests that the rules are highly granular. Rather than finding a few general rules (e.g., *Subject + verb + Object*), the system has learned hundreds of specific lexical variations. This "long-tail" distribution explains why precision is high but recall remains limited.
# 
# 
# 
# ### **Distribution of Rules per Relation**
# 
# 
# 
# The volume of rules discovered varies significantly by semantic type, reflecting the linguistic diversity of each relation:
# 
# 
# 
# 1.  **High-Variety Relations:**
# 
#     * **Message-Topic (274 combined rules):** This relation employs a vast array of communication verbs (e.g., *discussed, explained, wrote about, mentioned*), requiring many distinct rules to cover the lexical space.
# 
#     * **Entity-Destination (221 rules for e1,e2):** The movement of entities is described using diverse prepositions and motion verbs, leading to a high rule count.
# 
#     * **Other (238 rules):** Since "Other" encompasses all non-target relations, it naturally contains the highest diversity of linguistic structures.
# 
# 
# 
# 2.  **Sparse or Asymmetric Relations:**
# 
#     * **Directional Imbalance:** There is a stark contrast in directional subtypes. For instance, **Content-Container(e1,e2)** has 48 rules, while its reverse **(e2,e1)** has only 5. This reflects natural language usage: we frequently say *"the apples in the box"* (e1,e2) but rarely use the passive or inverse formulations that would generate (e2,e1) patterns.
# 
#     * **Instrument-Agency(e1,e2):** With only 11 rules, this subtype is extremely difficult to capture, likely due to the rarity of the "Agent uses Instrument" phrasing compared to the "Instrument used by Agent" (e2,e1) phrasing, which generated 155 rules.
# 
# 
# 
# ### **Generalization Performance**
# 
# 
# 
# The comparison between Training and Test set metrics highlights the generalization gap inherent in deterministic systems.
# 
# 
# 
# * **The Precision Drop:** The drop in Macro Precision (0.710 $\to$ 0.563) indicates that some rules learned from the training data are slightly overfitted—capturing coincidental patterns in the training text that do not hold up as universals in the test set.
# 
# * **The Recall Floor:** The Macro Recall is low in both sets (0.48 vs 0.40). This confirms that the rule set—while large (1651 rules)—is not exhaustive. It fails to trigger for sentences that use paraphrases or syntactic structures not explicitly seen during training.
# 
# 
# 
# ### **Conclusion**
# 
# 
# 
# The diagnostics depict a system that operates as a **precise, high-granularity filter**. It succeeds by memorizing specific, reliable contexts (demonstrated by the 0.868 average precision) but struggles to generalize to the broad variability of natural language (demonstrated by the sub-50% recall). The system is heavily dependent on the *quantity* of rules to achieve coverage, meaning performance is directly tied to the presence of specific lexical triggers in the input text.

# %% [markdown]
# ## Error Analysis
# 
# <img src="images/error_analysis__most_common_missclf_patterns_directed.png" alt="Error Analysis of Rule-Based System (Directed)" width="500" />
# 
# The error profile indicates that the system acts as a "conservative" classifier. The overwhelming majority of errors are **False Negatives**, where the system fails to find a matching rule for a valid relation and defaults to the "Other" class.
# 
# 
# 
# ### **The "False Negative" Sinkhole (Loss to 'Other')**
# 
# 
# 
# The most distinct error pattern is the mass assignment of valid relations to **"Other."** This occurs when the specific lexical trigger required by a rule is absent or slightly modified in the test set.
# 
# 
# 
# * **Member-Collection(e2,e1) $\rightarrow$ Other (158 errors):** This is the single most frequent error. Relationships like *player-team* or *tree-forest* often appear in nominal compounds or list structures that lack the explicit verbs or prepositions required by the rules.
# 
# * **Component-Whole (Combined 206 errors):** Both directions of this relation suffer heavily (*e1,e2*: 137, *e2,e1*: 69). This confirms that the generic preposition "of" (the primary cue for this relation) is too noisy to generate high-precision rules, causing the system to miss the majority of valid instances.
# 
# * **Product-Producer (Combined 114 errors):** Despite having specific verbs (manufacture, build), the system fails to capture the full breadth of creative acts, defaulting to Other.
# 
# 
# 
# ### **The "False Positive" Leak (Aggressive Rules)**
# 
# 
# 
# While the system is generally conservative, certain rules are overly aggressive, pulling non-relation sentences (True Label: Other) into specific classes.
# 
# 
# 
# * **Other $\rightarrow$ Entity-Destination(e1,e2) (57 errors):** This suggests that rules relying on directional prepositions like "to," "into," or "towards" are firing on non-destination contexts (e.g., temporal changes or abstract shifts).
# 
# * **Other $\rightarrow$ Content-Container / Message-Topic (30 errors each):** Common prepositions like "in" (Container) and "about" (Topic) are triggering on abstract usages that do not represent physical containment or communication topics.
# 
# 
# 
# ### **Semantic Confusion (Inter-Class Errors)**
# 
# 
# 
# True confusion between two active semantic classes is rare, but one specific overlap stands out:
# 
# 
# 
# * **Product-Producer(e1,e2) $\rightarrow$ Cause-Effect(e2,e1) (21 errors):**
# 
#     * *Reason:* There is a semantic overlap between "creating a product" and "causing an effect." Verbs like *generate, yield, or produce* can apply to both.
# 
#     * *Example:* "The factory **produced** toxic waste." Is "waste" a *Product* or an *Effect*? The system struggles to distinguish the nuance, leading to misclassification.
# 
# 
# 
# ### **Summary of Failure Modes**
# 
# 
# 
# 1.  **Recall Failure:** The system misses 150+ instances of *Member-Collection* because it lacks semantic knowledge of groups.
# 
# 2.  **Context Failure:** It misses 200+ instances of *Component-Whole* because it cannot distinguish the relevant use of "of" from the irrelevant ones.
# 
# 3.  **Ambiguity Failure:** It confuses *Creation* (Producer) with *Causation* (Effect) due to shared verbs.


