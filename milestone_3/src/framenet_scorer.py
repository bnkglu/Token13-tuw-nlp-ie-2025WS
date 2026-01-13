"""
FrameNet-based semantic scoring for relation extraction patterns.

Uses frame semantics to validate relation predictions by checking
if the syntactic structure matches expected semantic roles.

This module provides:
1. Frame-to-relation mappings
2. Semantic role compatibility scoring
3. Pattern validation based on frame expectations

Note: Full FrameNet integration requires nltk.corpus.framenet, but this
module provides curated frame mappings that work without external data.
"""

from typing import Dict, List, Tuple, Optional


# Frame-to-relation mappings
# Maps relation types to relevant FrameNet frames (or frame-like categories)
RELATION_FRAMES: Dict[str, List[str]] = {
    "Cause-Effect": [
        "Causation",
        "Cause_change",
        "Cause_motion",
        "Creating",
        "Killing",
        "Damaging",
        "Destroying",
    ],
    "Component-Whole": [
        "Part_whole",
        "Part_piece",
        "Inclusion",
        "Ingredients",
        "Containing",
    ],
    "Entity-Origin": [
        "Origin",
        "Source_of_getting",
        "Creating",
        "Emanating",
        "Coming_to_be",
    ],
    "Instrument-Agency": [
        "Using",
        "Tool_purpose",
        "Cause_motion",
        "Manipulation",
        "Operating_a_system",
    ],
    "Member-Collection": [
        "Membership",
        "Aggregate",
        "Type",
        "Instance",
        "Categorization",
    ],
    "Content-Container": [
        "Containing",
        "Filling",
        "Storing",
        "Placing",
        "Surrounding",
    ],
    "Entity-Destination": [
        "Motion",
        "Cause_motion",
        "Goal",
        "Arriving",
        "Sending",
    ],
    "Product-Producer": [
        "Manufacturing",
        "Creating",
        "Building",
        "Intentionally_create",
    ],
    "Message-Topic": [
        "Communication",
        "Statement",
        "Telling",
        "Reporting",
        "Topic",
    ],
}


# Expected syntactic roles for each relation based on frame semantics
# Maps relation to expected (e1_dep, e2_dep) patterns with scores
ROLE_EXPECTATIONS: Dict[str, List[Tuple[str, str, float]]] = {
    "Cause-Effect": [
        ("nsubj", "dobj", 0.9),      # Agent causes Patient
        ("nsubj", "nsubjpass", 0.8), # Active/passive causation
        ("pobj", "nsubj", 0.7),      # Effect from cause (reversed)
        ("nmod", "nsubj", 0.6),      # Nominal modification
    ],
    "Component-Whole": [
        ("pobj", "nsubj", 0.9),      # Part of Whole
        ("nmod", "nsubj", 0.8),      # Nominal modifier (part of)
        ("dobj", "nsubj", 0.7),      # Has part
        ("pobj", "poss", 0.8),       # Possession relationship
        ("compound", "nsubj", 0.6),  # Compound noun
    ],
    "Entity-Origin": [
        ("nsubj", "pobj", 0.9),      # Entity from Origin
        ("nsubjpass", "pobj", 0.85), # Entity derived from Origin
        ("nmod", "pobj", 0.7),       # Nominal modification
        ("dobj", "pobj", 0.6),       # Object from source
    ],
    "Instrument-Agency": [
        ("pobj", "nsubj", 0.9),      # Instrument used by Agent
        ("nmod", "nsubj", 0.8),      # Nominal modifier
        ("dobj", "nsubj", 0.7),      # Tool operates something
    ],
    "Member-Collection": [
        ("pobj", "nsubj", 0.9),      # Member of Collection
        ("nmod", "nsubj", 0.8),      # Nominal modifier
        ("appos", "nsubj", 0.7),     # Apposition
    ],
    "Content-Container": [
        ("nsubj", "pobj", 0.9),      # Content in Container
        ("dobj", "pobj", 0.8),       # Put content in container
        ("nmod", "pobj", 0.7),       # Nominal modification
    ],
    "Entity-Destination": [
        ("nsubj", "pobj", 0.9),      # Entity to Destination
        ("dobj", "pobj", 0.8),       # Move entity to destination
        ("nmod", "pobj", 0.7),       # Nominal modification
    ],
    "Product-Producer": [
        ("dobj", "nsubj", 0.9),      # Product made by Producer
        ("nsubjpass", "pobj", 0.85), # Product produced by Producer
        ("nmod", "nsubj", 0.7),      # Nominal modification
    ],
    "Message-Topic": [
        ("dobj", "pobj", 0.9),       # Message about Topic
        ("nsubj", "pobj", 0.8),      # Message concerning Topic
        ("nmod", "nsubj", 0.7),      # Nominal modification
    ],
}


# Verb-to-frame mappings for common relation verbs
# Expanded with 30+ additional verbs for better coverage
VERB_FRAMES: Dict[str, List[str]] = {
    # Causation verbs (expanded)
    "cause": ["Causation"],
    "trigger": ["Causation"],
    "create": ["Creating", "Causation"],
    "produce": ["Creating", "Manufacturing"],
    "generate": ["Creating", "Causation"],
    "lead": ["Causation"],
    "result": ["Causation"],
    "bring": ["Causation", "Cause_motion"],
    "affect": ["Objective_influence", "Causation"],
    "impact": ["Objective_influence", "Causation"],
    "influence": ["Objective_influence", "Causation"],
    "prevent": ["Preventing", "Thwarting"],
    "stop": ["Preventing", "Activity_stop"],
    "enable": ["Causation", "Assistance"],
    "force": ["Causation", "Cause_motion"],
    "accelerate": ["Causation", "Change_position_on_a_scale"],
    "hinder": ["Hindering", "Thwarting"],
    "inhibit": ["Hindering", "Preventing"],
    "suppress": ["Hindering", "Preventing"],
    "promote": ["Causation", "Assistance"],
    "facilitate": ["Assistance", "Causation"],
    "induce": ["Causation"],
    "spark": ["Causation"],
    "block": ["Preventing", "Hindering"],

    # Containment/Part-Whole verbs (expanded)
    "contain": ["Containing", "Part_whole"],
    "include": ["Inclusion", "Part_whole"],
    "hold": ["Containing"],
    "store": ["Storing", "Containing"],
    "comprise": ["Part_whole", "Inclusion"],
    "consist": ["Part_whole"],
    "constitute": ["Part_whole"],
    "form": ["Part_whole", "Creating"],
    "encompass": ["Inclusion", "Part_whole"],
    "embed": ["Containing", "Part_whole"],
    "partition": ["Part_whole"],
    "segment": ["Part_whole"],

    # Origin verbs (expanded)
    "derive": ["Origin", "Source_of_getting"],
    "come": ["Origin", "Motion"],
    "originate": ["Origin", "Coming_to_be"],
    "stem": ["Origin"],
    "descend": ["Origin", "Kinship"],
    "evolve": ["Origin", "Progress"],
    "develop": ["Origin", "Progress", "Creating"],
    "spring": ["Origin"],
    "emerge": ["Origin", "Coming_to_be"],
    "arise": ["Origin", "Coming_to_be"],

    # Movement/destination verbs
    "move": ["Motion", "Cause_motion"],
    "go": ["Motion"],
    "send": ["Sending", "Cause_motion"],
    "travel": ["Motion"],
    "transport": ["Cause_motion", "Bringing"],
    "deliver": ["Bringing", "Sending"],
    "ship": ["Sending", "Bringing"],

    # Communication/Message-Topic verbs (expanded)
    "say": ["Statement", "Communication"],
    "tell": ["Telling", "Communication"],
    "describe": ["Communication", "Statement"],
    "report": ["Reporting", "Communication"],
    "discuss": ["Discussion", "Communication", "Topic"],
    "analyze": ["Scrutiny", "Communication"],
    "address": ["Communication", "Topic"],
    "mention": ["Statement", "Communication"],
    "explain": ["Statement", "Communication"],
    "explore": ["Scrutiny", "Research"],
    "examine": ["Scrutiny", "Research"],
    "investigate": ["Scrutiny", "Research"],
    "review": ["Scrutiny", "Communication"],
    "treat": ["Topic", "Communication"],
    "cover": ["Topic", "Communication"],

    # Instrument-Agency verbs (new)
    "use": ["Using", "Tool_purpose"],
    "apply": ["Using", "Cause_to_make_progress"],
    "employ": ["Using", "Employing"],
    "operate": ["Operating_a_system", "Using"],
    "activate": ["Change_operational_state", "Causation"],
    "handle": ["Manipulation", "Using"],
    "utilize": ["Using"],
    "wield": ["Using", "Manipulation"],

    # Product-Producer verbs (new)
    "make": ["Manufacturing", "Creating"],
    "manufacture": ["Manufacturing"],
    "build": ["Building", "Creating"],
    "construct": ["Building", "Creating"],
    "design": ["Intentionally_create", "Creating"],
    "invent": ["Intentionally_create", "Creating"],
    "write": ["Text_creation", "Creating"],
    "compose": ["Text_creation", "Creating"],

    # Member-Collection verbs (new)
    "belong": ["Membership"],
    "join": ["Becoming_a_member"],
    "collect": ["Gathering_up"],
    "gather": ["Gathering_up"],
    "group": ["Categorization"],
    "assemble": ["Gathering_up", "Building"],
}


def get_relation_frames(relation: str) -> List[str]:
    """
    Get FrameNet frames associated with a relation.

    Args:
        relation: Relation type (e.g., "Cause-Effect(e1,e2)")

    Returns:
        List of frame names
    """
    # Extract base relation without direction
    base_relation = relation.split('(')[0] if '(' in relation else relation
    return RELATION_FRAMES.get(base_relation, [])


def score_frame_compatibility(
    relation: str,
    anchor_lemma: str,
    e1_dep: str,
    e2_dep: str
) -> float:
    """
    Score how well the pattern matches expected frame semantics.

    Args:
        relation: Predicted relation type
        anchor_lemma: The anchor verb/noun lemma
        e1_dep: Dependency label of e1
        e2_dep: Dependency label of e2

    Returns:
        Score 0.0 to 1.0 indicating frame compatibility
    """
    base_relation = relation.split('(')[0] if '(' in relation else relation

    # Base score
    score = 0.5

    # Check if anchor verb evokes expected frames
    if anchor_lemma.lower() in VERB_FRAMES:
        verb_frames = VERB_FRAMES[anchor_lemma.lower()]
        relation_frames = RELATION_FRAMES.get(base_relation, [])

        # Boost score if there's frame overlap
        if any(vf in relation_frames for vf in verb_frames):
            score += 0.2

    # Check if dependency roles match expectations
    role_expectations = ROLE_EXPECTATIONS.get(base_relation, [])
    for expected_e1, expected_e2, role_score in role_expectations:
        if e1_dep == expected_e1 and e2_dep == expected_e2:
            score = max(score, role_score)
            break
        # Partial match
        elif e1_dep == expected_e1 or e2_dep == expected_e2:
            score = max(score, role_score * 0.6)

    return min(score, 1.0)


def validate_relation_semantics(
    relation: str,
    anchor_lemma: str,
    anchor_pos: str,
    e1_dep: str,
    e2_dep: str
) -> Tuple[bool, float, str]:
    """
    Validate if a relation prediction makes semantic sense.

    Args:
        relation: Predicted relation
        anchor_lemma: Anchor word lemma
        anchor_pos: Anchor word POS tag
        e1_dep: E1 dependency label
        e2_dep: E2 dependency label

    Returns:
        Tuple of (is_valid, confidence, reason)
    """
    base_relation = relation.split('(')[0] if '(' in relation else relation

    # Score the frame compatibility
    frame_score = score_frame_compatibility(relation, anchor_lemma, e1_dep, e2_dep)

    # Determine validity threshold
    if frame_score >= 0.7:
        return True, frame_score, "High frame compatibility"
    elif frame_score >= 0.5:
        return True, frame_score, "Moderate frame compatibility"
    else:
        return False, frame_score, "Low frame compatibility - pattern may be spurious"


def get_best_relation_for_pattern(
    anchor_lemma: str,
    anchor_pos: str,
    e1_dep: str,
    e2_dep: str,
    candidate_relations: Optional[List[str]] = None
) -> Tuple[str, float]:
    """
    Suggest the best relation for a given pattern based on frame semantics.

    Args:
        anchor_lemma: Anchor word lemma
        anchor_pos: Anchor word POS
        e1_dep: E1 dependency label
        e2_dep: E2 dependency label
        candidate_relations: Optional list of relations to choose from

    Returns:
        Tuple of (best_relation, confidence)
    """
    if candidate_relations is None:
        candidate_relations = list(RELATION_FRAMES.keys())

    best_relation = "Other"
    best_score = 0.0

    for relation in candidate_relations:
        score = score_frame_compatibility(relation, anchor_lemma, e1_dep, e2_dep)
        if score > best_score:
            best_score = score
            best_relation = relation

    return best_relation, best_score


# Pre-computed semantic role patterns for quick lookup
SEMANTIC_ROLE_PATTERNS: Dict[str, Dict[str, str]] = {
    "nsubj_dobj": {
        "VERB": "Cause-Effect",  # Agent VERB Patient
    },
    "pobj_nsubj": {
        "VERB": "Component-Whole",  # X of Y where X is part
        "ADP": "Component-Whole",
    },
    "nsubj_pobj": {
        "VERB": "Entity-Origin",  # Entity from Source
        "ADP": "Content-Container",  # Content in Container
    },
}


def quick_semantic_check(
    e1_dep: str,
    e2_dep: str,
    anchor_pos: str
) -> Optional[str]:
    """
    Quick semantic role check for common patterns.

    Args:
        e1_dep: E1 dependency label
        e2_dep: E2 dependency label
        anchor_pos: Anchor POS tag

    Returns:
        Suggested relation or None
    """
    pattern_key = f"{e1_dep}_{e2_dep}"
    if pattern_key in SEMANTIC_ROLE_PATTERNS:
        pos_map = SEMANTIC_ROLE_PATTERNS[pattern_key]
        return pos_map.get(anchor_pos)
    return None


if __name__ == "__main__":
    # Test the module
    print("FrameNet Scorer Module")
    print("=" * 50)

    # Test frame compatibility scoring
    test_cases = [
        ("Cause-Effect", "cause", "nsubj", "dobj"),
        ("Component-Whole", "contain", "pobj", "nsubj"),
        ("Entity-Origin", "derive", "nsubj", "pobj"),
        ("Instrument-Agency", "use", "pobj", "nsubj"),
    ]

    print("\nFrame compatibility tests:")
    for relation, verb, e1_dep, e2_dep in test_cases:
        score = score_frame_compatibility(relation, verb, e1_dep, e2_dep)
        print(f"  {relation} + '{verb}' ({e1_dep}, {e2_dep}): {score:.2f}")

    # Test validation
    print("\nValidation tests:")
    for relation, verb, e1_dep, e2_dep in test_cases:
        valid, conf, reason = validate_relation_semantics(
            relation, verb, "VERB", e1_dep, e2_dep
        )
        print(f"  {relation}: valid={valid}, conf={conf:.2f}, reason='{reason}'")
