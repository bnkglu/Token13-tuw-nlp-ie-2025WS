"""
Import-safe API for the Milestone 2 directed rule-based system.

The original `rule_based_directed.py` was exported from a notebook and contains
top-level execution (model loading, dataset loading, evaluations, etc.). That
behavior is great for an interactive notebook, but it breaks downstream imports
in Milestone 3 (it executes on import and can fail depending on cwd).

This module contains ONLY the reusable functions needed by Milestone 3, with
no top-level side effects.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from spacy.tokens import Doc
from tqdm.auto import tqdm


def doc_from_json(item: Dict[str, Any], nlp) -> Doc:
    """
    Create a spaCy Doc from pre-computed JSON annotations.
    Expects token-level fields: text, lemma, pos, tag, dep, head.
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


def get_dependency_path(doc: Doc, e1_span, e2_span) -> List[Tuple[str, str, str]]:
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

    # Nodes from e1_root up to LCA (exclusive)
    path_up_nodes = []
    cur = e1_root
    while cur != lca:
        path_up_nodes.append(cur)
        cur = cur.head

    path_up = [(t.dep_, t.pos_, t.lemma_) for t in path_up_nodes]
    lca_feat = (lca.dep_, lca.pos_, lca.lemma_)
    path_down = [(t.dep_, t.pos_, t.lemma_) for t in reversed(path_down_nodes)]

    return path_up + [lca_feat] + path_down


def get_between_span(doc: Doc, e1_span, e2_span):
    """Get span between entities using Doc slicing."""
    if e1_span.start < e2_span.start:
        return doc[e1_span.end : e2_span.start]
    return doc[e2_span.end : e1_span.start]


def preprocess_data(data_list: List[Dict[str, Any]], nlp) -> List[Dict[str, Any]]:
    """
    Process data using pre-computed annotations from JSON.

    Returns list of dicts with:
      - doc, e1_span, e2_span
      - relation (undirected type)
      - relation_directed (directed label, or "Other")
      - direction ("e1,e2" / "e2,e1")
      - dep_path, between_words
    """
    processed: List[Dict[str, Any]] = []

    for item in tqdm(data_list, desc="Processing"):
        doc = doc_from_json(item, nlp)

        e1_info = item["entities"][0]
        e2_info = item["entities"][1]

        e1_token_ids = e1_info["token_ids"]
        e2_token_ids = e2_info["token_ids"]
        e1_span = doc[min(e1_token_ids) : max(e1_token_ids) + 1]
        e2_span = doc[min(e2_token_ids) : max(e2_token_ids) + 1]

        dep_path = get_dependency_path(doc, e1_span, e2_span)
        between_span = get_between_span(doc, e1_span, e2_span)

        between_words = [
            {"text": t.text, "lemma": t.lemma_, "pos": t.pos_, "dep": t.dep_}
            for t in between_span
        ]

        rel_type = item["relation"]["type"]  # e.g. "Cause-Effect" or "Other"
        direction = item["relation"].get("direction", "") or ""
        direction = direction.replace("(", "").replace(")", "")
        if not direction:
            direction = "e1,e2"

        if rel_type == "Other":
            rel_directed = "Other"
        else:
            rel_directed = f"{rel_type}({direction})"

        processed.append(
            {
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
            }
        )

    return processed


def _split_relation_and_direction(rel_directed: str) -> Tuple[str, Optional[str]]:
    """
    Split 'Cause-Effect(e1,e2)' -> ('Cause-Effect', 'e1,e2')
    Keeps 'Other' as ('Other', None).
    """
    if "(" in rel_directed and rel_directed.endswith(")"):
        base, dir_part = rel_directed.split("(", 1)
        direction = dir_part[:-1]  # strip trailing ')'
        return base.strip(), direction.strip()
    return rel_directed, None


def save_predictions_for_scorer(predictions: Iterable[str], processed_data: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save predictions in official scorer format:
        ID\\tRelationLabel
    where RelationLabel is already a full label like 'Cause-Effect(e1,e2)' or 'Other'.
    """
    preds = list(predictions)
    with open(output_file, "w", encoding="utf8") as f:
        for pred, sample in zip(preds, processed_data):
            f.write(f"{sample['id']}\t{pred}\n")

    print(f"Saved {len(preds)} predictions to {output_file}")


def create_answer_key(processed_data: List[Dict[str, Any]], output_file: str) -> None:
    """
    Create answer key file from processed data in official format:
        ID\\tRelationLabel
    where RelationLabel is the directed gold label, e.g. 'Cause-Effect(e1,e2)' or 'Other'.
    """
    with open(output_file, "w", encoding="utf8") as f:
        for sample in processed_data:
            f.write(f"{sample['id']}\t{sample['relation_directed']}\n")

    print(f"Saved {len(processed_data)} gold labels to {output_file}")


def analyze_errors(
    samples: List[Dict[str, Any]],
    predictions: List[str],
    true_labels: List[str],
    explanations: List[str],
    n_samples: int = 20,
) -> List[Dict[str, Any]]:
    """Analyze misclassified examples."""
    errors: List[Dict[str, Any]] = []

    for i, (sample, pred, true) in enumerate(zip(samples, predictions, true_labels)):
        if pred != true:
            errors.append(
                {
                    "index": i,
                    "sample": sample,
                    "predicted": pred,
                    "true": true,
                    "text": sample["text"],
                    "explanation": explanations[i] if i < len(explanations) else None,
                }
            )

    if samples:
        print(
            f"Total errors: {len(errors)} / {len(samples)} ({len(errors)/len(samples)*100:.1f}%)"
        )
    else:
        print("Total errors: 0 / 0 (0.0%)")

    print(f"\nShowing first {min(n_samples, len(errors))} errors:\n")
    print("=" * 80)

    for i, error in enumerate(errors[:n_samples]):
        print(f"\nError {i+1}:")
        print(f"Text: {error['text']}")
        print(f"Entity 1: {error['sample']['e1_span'].text}")
        print(f"Entity 2: {error['sample']['e2_span'].text}")
        print(f"True relation: {error['true']}")
        print(f"Predicted: {error['predicted']}")
        print(f"Rule applied: {error['explanation']}")

        between_words = [w["text"] for w in error["sample"]["between_words"]]
        print(f"Between words: {between_words}")

        dep_path = error["sample"]["dep_path"]
        if dep_path:
            path_str = " -> ".join([f"{d[0]}({d[2]})" for d in dep_path[:5]])
            print(f"Dependency path: {path_str}")

        print("-" * 80)

    return errors


def analyze_error_patterns(
    samples: List[Dict[str, Any]],
    predictions: List[str],
    true_labels: List[str],
) -> Dict[str, Dict[str, int]]:
    """Analyze error patterns by relation type."""
    error_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for _sample, pred, true in zip(samples, predictions, true_labels):
        if pred != true:
            error_matrix[true][pred] += 1

    print("\nMost Common Misclassification Patterns:")
    print("=" * 80)
    print(f"{'True Label':<25} {'Predicted As':<25} {'Count':<10}")
    print("-" * 80)

    all_errors = []
    for true_label in error_matrix:
        for pred_label in error_matrix[true_label]:
            all_errors.append((true_label, pred_label, error_matrix[true_label][pred_label]))

    all_errors.sort(key=lambda x: x[2], reverse=True)

    for true_label, pred_label, count in all_errors[:15]:
        print(f"{true_label:<25} {pred_label:<25} {count:<10}")

    return error_matrix


__all__ = [
    "doc_from_json",
    "get_dependency_path",
    "get_between_span",
    "preprocess_data",
    "_split_relation_and_direction",
    "save_predictions_for_scorer",
    "create_answer_key",
    "analyze_errors",
    "analyze_error_patterns",
]

