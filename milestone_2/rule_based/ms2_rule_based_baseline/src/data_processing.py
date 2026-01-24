"""Data loading and preprocessing for relation extraction.

This module handles:
- Loading spaCy model
- Loading JSON datasets
- Creating spaCy Docs from pre-computed annotations
- Extracting dependency paths and features
"""

import json
from typing import Any

from spacy.tokens import Doc
from tqdm.auto import tqdm


def doc_from_json(item: dict[str, Any], nlp) -> Doc:
    """
    Create a spaCy Doc from pre-computed JSON annotations.

    Args:
        item: Dictionary containing token annotations.
        nlp: spaCy language model.

    Returns:
        spaCy Doc with linguistic attributes set.
    """
    tokens_data = item["tokens"]

    words = [t["text"] for t in tokens_data]
    spaces = [i < len(words) - 1 for i in range(len(words))]

    doc = Doc(nlp.vocab, words=words, spaces=spaces)

    for token, token_data in zip(doc, tokens_data):
        token.lemma_ = token_data["lemma"]
        token.pos_ = token_data["pos"]
        token.tag_ = token_data["tag"]
        token.dep_ = token_data["dep"]

        head_id = token_data["head"]
        if head_id != token.i:
            token.head = doc[head_id]

    return doc


def get_dependency_path(doc: Doc, e1_span, e2_span) -> list[tuple[str, str, str]]:
    """
    Extract dependency path between entity roots via LCA (Lowest Common Ancestor).

    Args:
        doc: spaCy Doc.
        e1_span: Entity 1 span.
        e2_span: Entity 2 span.

    Returns:
        List of (dep_label, pos_tag, lemma) tuples along the path.
    """
    e1_root = e1_span.root
    e2_root = e2_span.root

    # Collect ancestors from e1_root to the root
    ancestors_e1 = []
    cur = e1_root
    while True:
        ancestors_e1.append(cur)
        if cur.head == cur:
            break
        cur = cur.head

    # Walk up from e2_root until we hit something in ancestors_e1
    path_down_nodes = []
    cur = e2_root
    while cur not in ancestors_e1:
        path_down_nodes.append(cur)
        if cur.head == cur:
            break
        cur = cur.head

    lca = cur

    # Nodes from e1_root up to LCA (exclusive)
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


def get_between_span(doc: Doc, e1_span, e2_span):
    """
    Get span between entities using Doc slicing.

    Args:
        doc: spaCy Doc.
        e1_span: Entity 1 span.
        e2_span: Entity 2 span.

    Returns:
        Span between the two entities.
    """
    if e1_span.start < e2_span.start:
        return doc[e1_span.end:e2_span.start]
    return doc[e2_span.end:e1_span.start]


def preprocess_data(data_list: list[dict], nlp) -> list[dict]:
    """
    Process raw data into feature-rich samples.

    Args:
        data_list: List of raw data dictionaries from JSON.
        nlp: spaCy language model.

    Returns:
        List of processed sample dictionaries with extracted features.
    """
    processed = []

    for item in tqdm(data_list, desc="Processing"):
        doc = doc_from_json(item, nlp)

        e1_info = item["entities"][0]
        e2_info = item["entities"][1]

        e1_token_ids = e1_info["token_ids"]
        e2_token_ids = e2_info["token_ids"]
        e1_span = doc[min(e1_token_ids):max(e1_token_ids) + 1]
        e2_span = doc[min(e2_token_ids):max(e2_token_ids) + 1]

        dep_path = get_dependency_path(doc, e1_span, e2_span)
        between_span = get_between_span(doc, e1_span, e2_span)

        between_words = [
            {"text": t.text, "lemma": t.lemma_, "pos": t.pos_, "dep": t.dep_}
            for t in between_span
        ]

        # Labels (directed)
        rel_type = item["relation"]["type"]
        direction = item["relation"].get("direction", "") or ""
        direction = direction.replace("(", "").replace(")", "")
        if not direction:
            direction = "e1,e2"

        # SemEval convention: "Other" is undirected
        if rel_type == "Other":
            rel_directed = "Other"
        else:
            rel_directed = f"{rel_type}({direction})"

        processed.append({
            "id": item["id"],
            "text": item["text"],
            "doc": doc,
            "e1_span": e1_span,
            "e2_span": e2_span,
            "relation": rel_type,
            "relation_directed": rel_directed,
            "direction": direction,
            "dep_path": dep_path,
            "between_words": between_words,
        })

    return processed


def load_datasets(data_dir) -> tuple[list[dict], list[dict]]:
    """
    Load train and test datasets from JSON files.

    Args:
        data_dir: Path to the data directory.

    Returns:
        Tuple of (train_data, test_data) lists.

    Raises:
        FileNotFoundError: If dataset files don't exist.
    """
    train_path = data_dir / "processed/train/train.json"
    test_path = data_dir / "processed/test/test.json"

    with open(train_path, "r") as f:
        train_data = json.load(f)

    with open(test_path, "r") as f:
        test_data = json.load(f)

    return train_data, test_data
