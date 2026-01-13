"""Build patterns_augmented.json from raw_patterns.json.

This is a non-notebook equivalent of Notebook 3 (pattern refinement).
It exists to make iteration faster and reproducible from the CLI.

Outputs:
  milestone_3/data/patterns_augmented.json

Usage:
  .venv/bin/python milestone_3/src/build_patterns_augmented.py
"""

from __future__ import annotations

import json
from ast import literal_eval
from pathlib import Path

from pattern_augmentation import (
    filter_patterns_tiered,
    filter_patterns_framenet_aware,
    generate_passive_variants,
    sort_patterns
)


def main() -> None:
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"

    raw_patterns_path = data_dir / "raw_patterns.json"
    concept_clusters_path = data_dir / "concept_clusters.json"
    output_path = data_dir / "patterns_augmented.json"

    print("=" * 80)
    print("BUILDING patterns_augmented.json")
    print("=" * 80)

    print(f"Loading raw patterns: {raw_patterns_path}")
    with open(raw_patterns_path, "r") as f:
        raw = json.load(f)

    print(f"Loading concept clusters: {concept_clusters_path}")
    with open(concept_clusters_path, "r") as f:
        concept_data = json.load(f)

    expanded_clusters = concept_data.get("expanded_clusters", {})

    pattern_counts_str = raw["pattern_counts"]
    # Safer than eval(): only allows Python literals.
    pattern_counts = {literal_eval(k): v for k, v in pattern_counts_str.items()}

    print(f"Candidate patterns: {len(pattern_counts)}")

    filtered_patterns = filter_patterns_tiered(pattern_counts, expanded_clusters)
    print(f"After tiered filtering: {len(filtered_patterns)} patterns")

    # FrameNet filtering DISABLED - was hurting accuracy
    # framenet_filtered = filter_patterns_framenet_aware(filtered_patterns)
    # print(f"After FrameNet validation: {len(framenet_filtered)} patterns")

    # Optional passive variants to improve recall on alternations.
    augmented_patterns = generate_passive_variants(filtered_patterns, min_precision_for_flip=0.75)

    # Sort for stable downstream behavior.
    sorted_patterns = sort_patterns(augmented_patterns)

    print(f"Writing: {output_path}")
    with open(output_path, "w") as f:
        json.dump(sorted_patterns, f, indent=2)

    print(f"Saved {len(sorted_patterns)} patterns")


if __name__ == "__main__":
    main()
