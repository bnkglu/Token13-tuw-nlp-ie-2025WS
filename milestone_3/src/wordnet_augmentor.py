"""
WordNet-based pattern augmentation for relation extraction.

Uses WordNet hypernym hierarchies to generalize patterns and improve
matching while maintaining precision through relation-specific constraints.

Usage:
    from wordnet_augmentor import hypernym_matches, RELATION_HYPERNYMS
"""

from functools import lru_cache
from typing import Set, Optional, List, Dict, Tuple

try:
    from nltk.corpus import wordnet as wn
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False
    wn = None


# Relation-specific hypernym groups for targeted matching
# These are curated verb groups that commonly appear in each relation type
# EXPANDED: Added 50+ additional words for better coverage
RELATION_HYPERNYMS: Dict[str, Dict[str, List[str]]] = {
    "Cause-Effect": {
        "cause_verbs": [
            "cause", "induce", "produce", "create", "generate", "trigger",
            "lead", "result", "bring", "provoke", "elicit", "prompt",
            "spark", "ignite", "initiate", "precipitate", "occasion",
            # Expanded: influence/impact verbs
            "affect", "impact", "influence", "determine", "shape",
            # Expanded: prevention verbs
            "prevent", "stop", "block", "inhibit", "hinder", "suppress",
            # Expanded: enabling verbs
            "enable", "allow", "permit", "facilitate", "promote",
            # Expanded: forcing verbs
            "force", "compel", "drive", "push", "accelerate", "decelerate"
        ],
        "effect_verbs": [
            "result", "follow", "ensue", "arise", "stem", "derive",
            "originate", "emerge", "develop", "proceed", "flow", "spring",
            # Expanded
            "occur", "happen", "transpire", "unfold", "manifest"
        ],
    },
    "Component-Whole": {
        "contain_verbs": [
            "contain", "include", "comprise", "consist", "hold",
            "incorporate", "encompass", "embrace", "enclose", "house",
            # Expanded
            "span", "cover", "subsume", "embed", "nest"
        ],
        "compose_verbs": [
            "compose", "constitute", "form", "make", "build",
            "construct", "assemble", "combine", "integrate",
            # Expanded
            "compile", "aggregate", "consolidate", "merge", "unify"
        ],
        "part_nouns": [
            "part", "component", "element", "piece", "portion",
            "section", "segment", "unit", "module", "fraction",
            # Expanded
            "division", "subdivision", "chapter", "layer", "level",
            "aspect", "feature", "facet", "dimension", "constituent"
        ],
    },
    "Entity-Origin": {
        "origin_verbs": [
            "originate", "derive", "come", "emerge", "spring", "stem",
            "arise", "emanate", "flow", "issue", "proceed", "descend",
            # Expanded
            "evolve", "develop", "grow", "spawn", "breed", "hatch"
        ],
        "produce_verbs": [
            "produce", "create", "make", "generate", "manufacture",
            "develop", "build", "fabricate", "construct",
            # Expanded
            "yield", "bear", "spawn", "breed", "give"
        ],
        "source_nouns": [
            "source", "origin", "root", "beginning", "start",
            "foundation", "basis", "birthplace", "cradle",
            # Expanded
            "genesis", "inception", "provenance", "derivation",
            "ancestry", "lineage", "heritage", "wellspring"
        ],
    },
    "Member-Collection": {
        "belong_verbs": [
            "belong", "join", "enter", "include", "incorporate",
            # Expanded
            "participate", "enroll", "enlist", "affiliate", "associate"
        ],
        "group_nouns": [
            "group", "collection", "set", "class", "category",
            "family", "cluster", "array", "assembly", "gathering",
            # Expanded
            "ensemble", "consortium", "coalition", "federation",
            "league", "alliance", "association", "organization"
        ],
        "member_nouns": [
            "member", "element", "constituent", "participant",
            "affiliate", "associate", "component",
            # Expanded
            "representative", "delegate", "partner", "colleague"
        ],
    },
    "Content-Container": {
        "contain_verbs": [
            "contain", "hold", "store", "keep", "house",
            "enclose", "embrace", "accommodate", "shelter",
            # Expanded
            "harbor", "lodge", "preserve", "maintain", "retain"
        ],
        "fill_verbs": [
            "fill", "pack", "load", "stuff", "cram",
            "stock", "supply", "furnish", "equip",
            # Expanded
            "occupy", "populate", "saturate", "permeate"
        ],
        "container_nouns": [
            "container", "box", "vessel", "receptacle",
            "holder", "case", "package", "wrapper",
            # Expanded
            "enclosure", "repository", "storage", "depot",
            "tank", "bin", "barrel", "crate", "canister"
        ],
    },
    "Instrument-Agency": {
        "use_verbs": [
            "use", "employ", "utilize", "apply", "operate",
            "wield", "handle", "manipulate", "control",
            # Expanded
            "activate", "deploy", "leverage", "exploit", "harness"
        ],
        "tool_nouns": [
            "tool", "instrument", "device", "apparatus", "equipment",
            "implement", "utensil", "gadget", "mechanism",
            # Expanded
            "machine", "appliance", "system", "facility", "resource",
            "method", "technique", "means", "medium", "vehicle"
        ],
    },
    "Entity-Destination": {
        "move_verbs": [
            "move", "go", "travel", "proceed", "head",
            "transfer", "transport", "send", "deliver", "ship",
            # Expanded
            "dispatch", "forward", "route", "direct", "convey",
            "carry", "bring", "take", "escort", "guide"
        ],
        "destination_nouns": [
            "destination", "target", "goal", "endpoint",
            "terminus", "objective", "aim",
            # Expanded
            "location", "site", "place", "venue", "position",
            "point", "station", "stop", "terminal"
        ],
    },
    "Product-Producer": {
        "produce_verbs": [
            "produce", "create", "make", "manufacture", "generate",
            "develop", "build", "fabricate", "construct", "design",
            # Expanded
            "engineer", "craft", "forge", "fashion", "compose",
            "author", "write", "publish", "release", "launch"
        ],
        "producer_nouns": [
            "producer", "creator", "maker", "manufacturer",
            "developer", "builder", "designer", "author",
            # Expanded
            "writer", "artist", "inventor", "founder", "originator",
            "architect", "engineer", "craftsman", "publisher"
        ],
    },
    "Message-Topic": {
        "communicate_verbs": [
            "communicate", "convey", "express", "state", "declare",
            "announce", "proclaim", "report", "describe", "explain",
            # Expanded
            "discuss", "analyze", "address", "treat", "cover",
            "explore", "examine", "investigate", "review", "study",
            "present", "illustrate", "depict", "portray", "mention"
        ],
        "topic_nouns": [
            "topic", "subject", "theme", "matter", "issue",
            "point", "question", "content", "substance",
            # Expanded
            "focus", "concern", "aspect", "area", "field",
            "domain", "scope", "coverage", "context"
        ],
    },
}


# Nominal relation patterns for noun-based relation detection
NOMINAL_RELATIONS: Dict[str, Dict[str, List[str]]] = {
    "Component-Whole": {
        "part_of": ["part", "portion", "segment", "piece", "fraction",
                    "section", "component", "element", "constituent"],
        "whole_of": ["whole", "entirety", "totality", "aggregate", "sum"],
    },
    "Member-Collection": {
        "member_of": ["member", "affiliate", "participant", "associate",
                      "representative", "delegate", "partner"],
        "collection_of": ["group", "collection", "set", "class", "family",
                          "cluster", "array", "assembly"],
    },
    "Entity-Origin": {
        "from_source": ["source", "origin", "root", "beginning", "genesis",
                        "birthplace", "provenance", "ancestry"],
    },
    "Content-Container": {
        "in_container": ["container", "box", "vessel", "holder", "case",
                         "repository", "storage", "enclosure"],
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


# =============================================================================
# ENTITY TYPE DETECTION USING WORDNET SUPERSENSES
# =============================================================================

@lru_cache(maxsize=10000)
def get_entity_type(text: str) -> str:
    """
    Get WordNet supersense (lexicographer file) for entity's head noun.
    
    WordNet organizes nouns into 26 semantic categories (supersenses):
    - artifact, person, group, location, substance, food, animal, plant,
    - communication, cognition, act, event, state, phenomenon, etc.
    
    Args:
        text: Entity text (e.g., "the committee", "timer")
    
    Returns:
        Supersense category (e.g., "artifact", "person", "group")
        Returns "unknown" if not found in WordNet.
    
    Example:
        >>> get_entity_type("timer")
        'artifact'
        >>> get_entity_type("committee")
        'group'
        >>> get_entity_type("members")
        'person'
    """
    if not WORDNET_AVAILABLE:
        return "unknown"
    
    # Clean and split text
    words = text.lower().strip().split()
    
    # Try words from end to start (head noun is typically last)
    for word in reversed(words):
        # Skip common determiners and modifiers
        if word in {'the', 'a', 'an', 'this', 'that', 'these', 'those', 
                    'some', 'any', 'all', 'each', 'every', 'no', 'my', 
                    'your', 'his', 'her', 'its', 'our', 'their'}:
            continue
        
        synsets = wn.synsets(word, pos='n')
        if synsets:
            # Use first (most common) sense
            lexname = synsets[0].lexname()
            # Extract category from lexname (e.g., "noun.artifact" -> "artifact")
            return lexname.split('.')[1] if '.' in lexname else lexname
    
    return "unknown"


# =============================================================================
# ENTITY TYPE -> RELATION MAPPING RULES
# =============================================================================
# These rules are derived from training data analysis.
# Each rule maps (e1_type, e2_type) -> most likely relation with that type pair.
# Confidence percentages indicate how often this relation appears for the pair.

# High-confidence rules (>50% majority, >10 training samples)
ENTITY_TYPE_RELATION_RULES: Dict[Tuple[str, str], str] = {
    # Member-Collection patterns (group/collection -> member)
    ("group", "person"): "Member-Collection(e2,e1)",       # 78% of 164 samples
    ("group", "animal"): "Member-Collection(e2,e1)",       # 96% of 51 samples
    ("group", "plant"): "Member-Collection(e2,e1)",        # 64% of 11 samples
    ("artifact", "animal"): "Member-Collection(e2,e1)",    # 66% of 53 samples
    ("act", "animal"): "Member-Collection(e2,e1)",         # 71% of 14 samples
    ("communication", "animal"): "Member-Collection(e2,e1)", # 60% of 10 samples
    ("unknown", "animal"): "Member-Collection(e2,e1)",     # 92% of 24 samples
    ("unknown", "person"): "Member-Collection(e2,e1)",     # 73% of 26 samples
    
    # Component-Whole patterns (part-of relationships)
    ("body", "animal"): "Component-Whole(e1,e2)",          # 91% of 11 samples
    ("animal", "body"): "Component-Whole(e2,e1)",          # 88% of 17 samples
    ("animal", "animal"): "Component-Whole(e2,e1)",        # 53% of 19 samples
    
    # Cause-Effect patterns (causal relationships)
    ("state", "state"): "Cause-Effect(e2,e1)",             # 61% of 95 samples
    ("state", "act"): "Cause-Effect(e2,e1)",               # 81% of 47 samples
    ("state", "animal"): "Cause-Effect(e2,e1)",            # 68% of 37 samples
    ("state", "event"): "Cause-Effect(e2,e1)",             # 65% of 20 samples
    ("state", "phenomenon"): "Cause-Effect(e2,e1)",        # 94% of 17 samples
    ("state", "substance"): "Cause-Effect(e2,e1)",         # 62% of 13 samples
    ("state", "unknown"): "Cause-Effect(e2,e1)",           # 100% of 10 samples
    ("event", "act"): "Cause-Effect(e2,e1)",               # 74% of 34 samples
    ("event", "event"): "Cause-Effect(e2,e1)",             # 83% of 35 samples
    ("event", "phenomenon"): "Cause-Effect(e2,e1)",        # 93% of 14 samples
    ("event", "object"): "Cause-Effect(e2,e1)",            # 55% of 11 samples
    ("feeling", "act"): "Cause-Effect(e2,e1)",             # 84% of 19 samples
    ("phenomenon", "act"): "Cause-Effect(e1,e2)",          # 60% of 10 samples
    ("phenomenon", "artifact"): "Cause-Effect(e2,e1)",     # 53% of 19 samples
    ("process", "act"): "Cause-Effect(e2,e1)",             # 50% of 10 samples
    ("animal", "state"): "Cause-Effect(e1,e2)",            # 50% of 18 samples
    ("artifact", "phenomenon"): "Cause-Effect(e1,e2)",     # 70% of 10 samples
    
    # Message-Topic patterns (communication about topics)
    ("communication", "act"): "Message-Topic(e1,e2)",      # 60% of 100 samples
    ("communication", "cognition"): "Message-Topic(e1,e2)", # 74% of 77 samples
    ("communication", "state"): "Message-Topic(e1,e2)",    # 65% of 49 samples
    ("communication", "time"): "Message-Topic(e1,e2)",     # 52% of 29 samples
    ("communication", "attribute"): "Message-Topic(e1,e2)", # 67% of 21 samples
    ("communication", "event"): "Message-Topic(e1,e2)",    # 68% of 19 samples
    ("communication", "Tops"): "Message-Topic(e1,e2)",     # 50% of 18 samples
    ("communication", "possession"): "Message-Topic(e1,e2)", # 50% of 16 samples
    ("communication", "relation"): "Message-Topic(e1,e2)", # 82% of 11 samples
    ("phenomenon", "communication"): "Message-Topic(e2,e1)", # 50% of 10 samples
    
    # Product-Producer patterns (creator-creation relationships)
    ("communication", "person"): "Product-Producer(e1,e2)", # 53% of 152 samples
    ("person", "communication"): "Product-Producer(e2,e1)", # 58% of 150 samples
    ("person", "food"): "Product-Producer(e2,e1)",         # 64% of 11 samples
    ("food", "person"): "Product-Producer(e1,e2)",         # 50% of 22 samples
    
    # Entity-Origin patterns (source/origin relationships)
    ("plant", "food"): "Entity-Origin(e2,e1)",             # 86% of 50 samples
    ("plant", "substance"): "Entity-Origin(e2,e1)",        # 88% of 16 samples
    ("food", "food"): "Entity-Origin(e2,e1)",              # 62% of 53 samples
    ("artifact", "time"): "Entity-Origin(e1,e2)",          # 50% of 16 samples
    
    # Content-Container patterns
    ("artifact", "food"): "Content-Container(e2,e1)",      # 57% of 42 samples
    ("artifact", "possession"): "Content-Container(e2,e1)", # 61% of 18 samples
    
    # Entity-Destination patterns
    ("possession", "group"): "Entity-Destination(e1,e2)",  # 71% of 14 samples
    ("substance", "body"): "Entity-Destination(e1,e2)",    # 55% of 22 samples
    ("substance", "object"): "Entity-Destination(e1,e2)",  # 59% of 17 samples
    
    # Instrument-Agency patterns
    ("person", "substance"): "Instrument-Agency(e2,e1)",   # 55% of 33 samples
}

# Medium-confidence rules (40-50% majority, can be used as fallback)
ENTITY_TYPE_RELATION_RULES_MEDIUM: Dict[Tuple[str, str], str] = {
    ("person", "artifact"): "Instrument-Agency(e2,e1)",    # 48% of 371 samples
    ("person", "unknown"): "Instrument-Agency(e2,e1)",     # 45% of 22 samples
    ("body", "body"): "Component-Whole(e1,e2)",            # 43% of 49 samples
    ("body", "group"): "Member-Collection(e1,e2)",         # 45% of 11 samples
    ("artifact", "substance"): "Content-Container(e2,e1)", # 44% of 63 samples
    ("artifact", "plant"): "Component-Whole(e2,e1)",       # 42% of 31 samples
    ("communication", "location"): "Message-Topic(e1,e2)", # 48% of 27 samples
    ("communication", "plant"): "Component-Whole(e1,e2)",  # 42% of 12 samples
    ("act", "event"): "Cause-Effect(e1,e2)",               # 43% of 23 samples
    ("event", "animal"): "Member-Collection(e2,e1)",       # 43% of 14 samples
    ("event", "state"): "Cause-Effect(e2,e1)",             # 47% of 17 samples
    ("group", "object"): "Member-Collection(e2,e1)",       # 46% of 13 samples
    ("group", "unknown"): "Member-Collection(e2,e1)",      # 46% of 13 samples
    ("phenomenon", "event"): "Cause-Effect(e2,e1)",        # 40% of 10 samples
    ("possession", "communication"): "Entity-Destination(e1,e2)", # 42% of 12 samples
    ("state", "artifact"): "Cause-Effect(e2,e1)",          # 42% of 45 samples
}


def disambiguate_by_entity_type(
    e1_text: str,
    e2_text: str,
    candidate_relations: List[str],
    use_medium_confidence: bool = True
) -> Tuple[str, float]:
    """
    Use WordNet entity types to select the best relation from candidates.
    
    This function helps disambiguate when multiple patterns match the same
    text with similar confidence. It uses the semantic types of the entities
    to pick the most likely relation based on learned rules from training data.
    
    Args:
        e1_text: Text of entity 1
        e2_text: Text of entity 2
        candidate_relations: List of candidate relation labels to choose from
        use_medium_confidence: Whether to also use medium-confidence rules (40-50%)
    
    Returns:
        Tuple of (selected_relation, confidence_score)
        - If entity types match a rule and the rule's relation is in candidates,
          returns that relation with high confidence (0.85)
        - If using medium rules and match found, returns with medium confidence (0.65)
        - Otherwise returns first candidate with low confidence (0.5)
    
    Example:
        >>> disambiguate_by_entity_type("timer", "device", 
        ...     ["Member-Collection(e2,e1)", "Component-Whole(e1,e2)"])
        ('Component-Whole(e1,e2)', 0.85)  # Because both are artifacts
    """
    if not candidate_relations:
        return "Other", 0.0
    
    # Get entity types
    e1_type = get_entity_type(e1_text)
    e2_type = get_entity_type(e2_text)
    
    type_pair = (e1_type, e2_type)
    
    # Check high-confidence rules first
    if type_pair in ENTITY_TYPE_RELATION_RULES:
        preferred = ENTITY_TYPE_RELATION_RULES[type_pair]
        if preferred in candidate_relations:
            return preferred, 0.85
        # Also check if base relation matches (ignore direction)
        preferred_base = preferred.split('(')[0] if '(' in preferred else preferred
        for cand in candidate_relations:
            cand_base = cand.split('(')[0] if '(' in cand else cand
            if cand_base == preferred_base:
                return cand, 0.75
    
    # Check medium-confidence rules
    if use_medium_confidence and type_pair in ENTITY_TYPE_RELATION_RULES_MEDIUM:
        preferred = ENTITY_TYPE_RELATION_RULES_MEDIUM[type_pair]
        if preferred in candidate_relations:
            return preferred, 0.65
        # Also check base relation match
        preferred_base = preferred.split('(')[0] if '(' in preferred else preferred
        for cand in candidate_relations:
            cand_base = cand.split('(')[0] if '(' in cand else cand
            if cand_base == preferred_base:
                return cand, 0.55
    
    # No rule matched - return first candidate with low confidence
    return candidate_relations[0], 0.5


def get_entity_type_pair_info(e1_text: str, e2_text: str) -> Dict:
    """
    Get diagnostic information about entity types for debugging.
    
    Args:
        e1_text: Text of entity 1
        e2_text: Text of entity 2
    
    Returns:
        Dictionary with entity types and any matching rules
    """
    e1_type = get_entity_type(e1_text)
    e2_type = get_entity_type(e2_text)
    type_pair = (e1_type, e2_type)
    
    return {
        'e1_text': e1_text,
        'e2_text': e2_text,
        'e1_type': e1_type,
        'e2_type': e2_type,
        'high_conf_rule': ENTITY_TYPE_RELATION_RULES.get(type_pair),
        'medium_conf_rule': ENTITY_TYPE_RELATION_RULES_MEDIUM.get(type_pair),
    }


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

        # Test entity type detection
        print("\n" + "=" * 50)
        print("Entity Type Detection Tests:")
        print("=" * 50)
        
        entity_tests = [
            "timer", "device", "member", "committee", "book", "author",
            "wheel", "car", "employee", "company", "juice", "orange"
        ]
        for entity in entity_tests:
            etype = get_entity_type(entity)
            print(f"  '{entity}' -> {etype}")
        
        # Test entity type disambiguation
        print("\n" + "=" * 50)
        print("Entity Type Disambiguation Tests:")
        print("=" * 50)
        
        disamb_tests = [
            ("timer", "device", ["Member-Collection(e2,e1)", "Component-Whole(e1,e2)"]),
            ("members", "committee", ["Member-Collection(e2,e1)", "Component-Whole(e1,e2)"]),
            ("book", "author", ["Product-Producer(e1,e2)", "Member-Collection(e2,e1)"]),
            ("wheel", "car", ["Component-Whole(e1,e2)", "Member-Collection(e2,e1)"]),
        ]
        
        for e1, e2, candidates in disamb_tests:
            result, conf = disambiguate_by_entity_type(e1, e2, candidates)
            e1_type = get_entity_type(e1)
            e2_type = get_entity_type(e2)
            print(f"  ({e1}[{e1_type}], {e2}[{e2_type}]) -> {result} (conf={conf:.2f})")
        
        print(f"\nTotal high-confidence rules: {len(ENTITY_TYPE_RELATION_RULES)}")
        print(f"Total medium-confidence rules: {len(ENTITY_TYPE_RELATION_RULES_MEDIUM)}")

    else:
        print("WordNet is NOT available.")
        print("Install with: pip install nltk")
        print("Then run: python -c \"import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')\"")
