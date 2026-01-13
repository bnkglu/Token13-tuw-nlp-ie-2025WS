"""
WordNet-based pattern augmentation for relation extraction.

Uses WordNet hypernym hierarchies to generalize patterns and improve
matching while maintaining precision through relation-specific constraints.

Usage:
    from wordnet_augmentor import hypernym_matches, RELATION_HYPERNYMS
"""

from functools import lru_cache
from typing import Set, Optional, List, Dict

try:
    from nltk.corpus import wordnet as wn
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False
    wn = None


# Relation-specific hypernym groups for targeted matching
# These are curated verb groups that commonly appear in each relation type
RELATION_HYPERNYMS: Dict[str, Dict[str, List[str]]] = {
    "Cause-Effect": {
        "cause_verbs": [
            "cause", "induce", "produce", "create", "generate", "trigger",
            "lead", "result", "bring", "provoke", "elicit", "prompt",
            "spark", "ignite", "initiate", "precipitate", "occasion"
        ],
        "effect_verbs": [
            "result", "follow", "ensue", "arise", "stem", "derive",
            "originate", "emerge", "develop", "proceed", "flow", "spring"
        ],
    },
    "Component-Whole": {
        "contain_verbs": [
            "contain", "include", "comprise", "consist", "hold",
            "incorporate", "encompass", "embrace", "enclose", "house"
        ],
        "compose_verbs": [
            "compose", "constitute", "form", "make", "build",
            "construct", "assemble", "combine", "integrate"
        ],
        "part_nouns": [
            "part", "component", "element", "piece", "portion",
            "section", "segment", "unit", "module", "fraction"
        ],
    },
    "Entity-Origin": {
        "origin_verbs": [
            "originate", "derive", "come", "emerge", "spring", "stem",
            "arise", "emanate", "flow", "issue", "proceed", "descend"
        ],
        "produce_verbs": [
            "produce", "create", "make", "generate", "manufacture",
            "develop", "build", "fabricate", "construct"
        ],
        "source_nouns": [
            "source", "origin", "root", "beginning", "start",
            "foundation", "basis", "birthplace", "cradle"
        ],
    },
    "Member-Collection": {
        "belong_verbs": [
            "belong", "join", "enter", "include", "incorporate"
        ],
        "group_nouns": [
            "group", "collection", "set", "class", "category",
            "family", "cluster", "array", "assembly", "gathering"
        ],
        "member_nouns": [
            "member", "element", "constituent", "participant",
            "affiliate", "associate", "component"
        ],
    },
    "Content-Container": {
        "contain_verbs": [
            "contain", "hold", "store", "keep", "house",
            "enclose", "embrace", "accommodate", "shelter"
        ],
        "fill_verbs": [
            "fill", "pack", "load", "stuff", "cram",
            "stock", "supply", "furnish", "equip"
        ],
        "container_nouns": [
            "container", "box", "vessel", "receptacle",
            "holder", "case", "package", "wrapper"
        ],
    },
    "Instrument-Agency": {
        "use_verbs": [
            "use", "employ", "utilize", "apply", "operate",
            "wield", "handle", "manipulate", "control"
        ],
        "tool_nouns": [
            "tool", "instrument", "device", "apparatus", "equipment",
            "implement", "utensil", "gadget", "mechanism"
        ],
    },
    "Entity-Destination": {
        "move_verbs": [
            "move", "go", "travel", "proceed", "head",
            "transfer", "transport", "send", "deliver", "ship"
        ],
        "destination_nouns": [
            "destination", "target", "goal", "endpoint",
            "terminus", "objective", "aim"
        ],
    },
    "Product-Producer": {
        "produce_verbs": [
            "produce", "create", "make", "manufacture", "generate",
            "develop", "build", "fabricate", "construct", "design"
        ],
        "producer_nouns": [
            "producer", "creator", "maker", "manufacturer",
            "developer", "builder", "designer", "author"
        ],
    },
    "Message-Topic": {
        "communicate_verbs": [
            "communicate", "convey", "express", "state", "declare",
            "announce", "proclaim", "report", "describe", "explain"
        ],
        "topic_nouns": [
            "topic", "subject", "theme", "matter", "issue",
            "point", "question", "content", "substance"
        ],
    },
}


@lru_cache(maxsize=10000)
def get_hypernyms(lemma: str, pos: str = 'v', depth: int = 2) -> Set[str]:
    """
    Get hypernyms up to specified depth in WordNet hierarchy.

    Args:
        lemma: Word lemma to look up
        pos: Part of speech ('v' for verb, 'n' for noun, 'a' for adjective)
        depth: How many levels up to traverse

    Returns:
        Set of hypernym lemmas
    """
    if not WORDNET_AVAILABLE:
        return set()

    synsets = wn.synsets(lemma, pos=pos)
    if not synsets:
        return set()

    hypernyms = set()

    # Only use first 2 senses to avoid noise
    for synset in synsets[:2]:
        current_hypers = synset.hypernyms()

        for d in range(depth):
            next_hypers = []
            for hyper in current_hypers:
                for lemma_obj in hyper.lemmas():
                    lemma_name = lemma_obj.name().lower().replace('_', ' ')
                    hypernyms.add(lemma_name)
                next_hypers.extend(hyper.hypernyms())
            current_hypers = next_hypers

    return hypernyms


@lru_cache(maxsize=10000)
def get_hyponyms(lemma: str, pos: str = 'v', depth: int = 1) -> Set[str]:
    """
    Get hyponyms (more specific terms) up to specified depth.

    Args:
        lemma: Word lemma to look up
        pos: Part of speech
        depth: How many levels down to traverse

    Returns:
        Set of hyponym lemmas
    """
    if not WORDNET_AVAILABLE:
        return set()

    synsets = wn.synsets(lemma, pos=pos)
    if not synsets:
        return set()

    hyponyms = set()

    for synset in synsets[:2]:
        current_hypos = synset.hyponyms()

        for d in range(depth):
            next_hypos = []
            for hypo in current_hypos:
                for lemma_obj in hypo.lemmas():
                    lemma_name = lemma_obj.name().lower().replace('_', ' ')
                    hyponyms.add(lemma_name)
                next_hypos.extend(hypo.hyponyms())
            current_hypos = next_hypos

    return hyponyms


def hypernym_matches(token_lemma: str, pattern_lemma: str, pos: str = 'v') -> bool:
    """
    Check if token matches pattern via hypernym chain.

    Returns True if:
    1. Direct match
    2. pattern_lemma is a hypernym of token_lemma
    3. token_lemma is a hypernym of pattern_lemma

    Args:
        token_lemma: Lemma from actual text
        pattern_lemma: Lemma from pattern
        pos: Part of speech for WordNet lookup

    Returns:
        True if match found
    """
    token_lemma = token_lemma.lower()
    pattern_lemma = pattern_lemma.lower()

    # Direct match
    if token_lemma == pattern_lemma:
        return True

    if not WORDNET_AVAILABLE:
        return False

    # Check if pattern_lemma is hypernym of token_lemma
    token_hypernyms = get_hypernyms(token_lemma, pos, depth=3)
    if pattern_lemma in token_hypernyms:
        return True

    # Check if token_lemma is hypernym of pattern_lemma (generalization)
    pattern_hypernyms = get_hypernyms(pattern_lemma, pos, depth=3)
    if token_lemma in pattern_hypernyms:
        return True

    return False


def relation_specific_match(
    token_lemma: str,
    relation: str,
    pos: str = 'v'
) -> tuple[bool, float, str]:
    """
    Check if token matches any relation-specific hypernym group.

    Args:
        token_lemma: Lemma from actual text
        relation: Relation type (e.g., "Cause-Effect")
        pos: Part of speech

    Returns:
        Tuple of (matched, confidence, group_name)
    """
    token_lemma = token_lemma.lower()

    # Extract base relation without direction
    base_relation = relation.split('(')[0] if '(' in relation else relation

    if base_relation not in RELATION_HYPERNYMS:
        return False, 0.0, ""

    groups = RELATION_HYPERNYMS[base_relation]

    for group_name, words in groups.items():
        # Direct match in curated list
        if token_lemma in [w.lower() for w in words]:
            return True, 0.9, group_name

        # Check if any curated word is a hypernym/hyponym of token
        if WORDNET_AVAILABLE:
            token_hypernyms = get_hypernyms(token_lemma, pos, depth=2)
            token_hyponyms = get_hyponyms(token_lemma, pos, depth=1)

            for word in words:
                if word.lower() in token_hypernyms:
                    return True, 0.7, group_name
                if word.lower() in token_hyponyms:
                    return True, 0.75, group_name

    return False, 0.0, ""


def expand_pattern_with_wordnet(pattern_lemma: str, relation: str, pos: str = 'v') -> List[str]:
    """
    Expand a pattern lemma with WordNet synonyms and hyponyms.

    Useful for generating additional pattern variations.

    Args:
        pattern_lemma: Original pattern lemma
        relation: Relation type for context
        pos: Part of speech

    Returns:
        List of expanded lemmas (includes original)
    """
    expanded = [pattern_lemma.lower()]

    if not WORDNET_AVAILABLE:
        return expanded

    # Add synonyms from same synset
    synsets = wn.synsets(pattern_lemma, pos=pos)
    for synset in synsets[:2]:
        for lemma in synset.lemmas():
            name = lemma.name().lower().replace('_', ' ')
            if name not in expanded:
                expanded.append(name)

    # Add hyponyms (more specific terms)
    hyponyms = get_hyponyms(pattern_lemma, pos, depth=1)
    for hypo in hyponyms:
        if hypo not in expanded:
            expanded.append(hypo)

    return expanded


def check_wordnet_available() -> bool:
    """Check if WordNet is available and loaded."""
    if not WORDNET_AVAILABLE:
        return False

    try:
        # Try a simple lookup to verify WordNet data is downloaded
        wn.synsets('test')
        return True
    except Exception:
        return False


if __name__ == "__main__":
    # Test WordNet availability and functionality
    print("WordNet Augmentor Module")
    print("=" * 50)

    if check_wordnet_available():
        print("WordNet is available and loaded.")

        # Test hypernym matching
        test_cases = [
            ("create", "cause", "v"),
            ("produce", "cause", "v"),
            ("box", "container", "n"),
            ("derive", "originate", "v"),
        ]

        print("\nHypernym matching tests:")
        for token, pattern, pos in test_cases:
            result = hypernym_matches(token, pattern, pos)
            print(f"  {token} matches {pattern}: {result}")

        # Test relation-specific matching
        print("\nRelation-specific matching tests:")
        for token in ["cause", "trigger", "produce", "generate"]:
            matched, conf, group = relation_specific_match(token, "Cause-Effect")
            print(f"  '{token}' in Cause-Effect: matched={matched}, conf={conf:.2f}, group={group}")

    else:
        print("WordNet is NOT available.")
        print("Install with: pip install nltk")
        print("Then run: python -c \"import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')\"")
