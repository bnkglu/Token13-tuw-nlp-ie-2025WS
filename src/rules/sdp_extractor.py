"""SDP (Shortest Dependency Path) extraction and signature generation.

This module extracts SDPs between entity pairs from CoNLL-U formatted data
and generalizes them into abstract signatures for pattern mining.
"""

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import conllu


@dataclass
class SDPSignature:
    """Abstract signature representing a generalized SDP pattern.

    Attributes
    ----------
    pos_pattern : Tuple[str, ...]
        Sequence of POS tags along the path (e.g., ("NOUN", "ADP", "NOUN"))
    dep_pattern : Tuple[str, ...]
        Sequence of dependency relations (e.g., ("prep", "pobj"))
    trigger_words : Tuple[str, ...]
        Specific words kept as triggers (prepositions, verbs)
    direction : str
        Direction of the relation ("e1_to_e2" or "e2_to_e1")
    path_length : int
        Number of tokens in the SDP
    """

    pos_pattern: Tuple[str, ...]
    dep_pattern: Tuple[str, ...]
    trigger_words: Tuple[str, ...]
    direction: str
    path_length: int

    def __hash__(self) -> int:
        return hash((self.pos_pattern, self.dep_pattern, self.trigger_words))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SDPSignature):
            return False
        return (
            self.pos_pattern == other.pos_pattern
            and self.dep_pattern == other.dep_pattern
            and self.trigger_words == other.trigger_words
        )

    def to_string(self) -> str:
        """Return human-readable string representation."""
        parts = []
        for i, pos in enumerate(self.pos_pattern):
            if i < len(self.trigger_words) and self.trigger_words[i]:
                parts.append(f"{self.trigger_words[i]}:{pos}")
            else:
                parts.append(pos)
            if i < len(self.dep_pattern):
                parts.append(f"--({self.dep_pattern[i]})-->")
        return " ".join(parts)


@dataclass
class SDPExample:
    """A single SDP example with context.

    Attributes
    ----------
    sent_id : int
        Sentence ID from the dataset
    text : str
        Original sentence text
    relation : str
        Full relation label (e.g., "Component-Whole(e2,e1)")
    relation_type : str
        Relation type without direction (e.g., "Component-Whole")
    e1_text : str
        Text of entity 1
    e2_text : str
        Text of entity 2
    sdp_tokens : List[Dict[str, Any]]
        Tokens along the SDP
    sdp_token_ids : List[int]
        Token IDs along the SDP
    signature : Optional[SDPSignature]
        Generalized signature
    """

    sent_id: int
    text: str
    relation: str
    relation_type: str
    e1_text: str
    e2_text: str
    sdp_tokens: List[Dict[str, Any]]
    sdp_token_ids: List[int]
    signature: Optional[SDPSignature] = None


class SDPExtractor:
    """Extract shortest dependency paths from CoNLL-U data.

    Parameters
    ----------
    trigger_pos : Tuple[str, ...]
        POS tags to keep as trigger words (default: ADP, VERB, AUX, SCONJ)
    """

    # POS tags that should be kept as trigger words
    DEFAULT_TRIGGER_POS = ("ADP", "VERB", "AUX", "SCONJ", "CCONJ")

    def __init__(self, trigger_pos: Optional[Tuple[str, ...]] = None):
        self.trigger_pos = trigger_pos or self.DEFAULT_TRIGGER_POS

    def load_conllu(self, filepath: str) -> List[Dict[str, Any]]:
        """Load and parse a CoNLL-U file.

        Parameters
        ----------
        filepath : str
            Path to the CoNLL-U file

        Returns
        -------
        List[Dict[str, Any]]
            List of parsed examples with tokens and metadata
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = conllu.parse(f.read())

        examples = []
        for sent in data:
            example = self._parse_sentence(sent)
            if example:
                examples.append(example)

        return examples

    def _parse_sentence(self, sent: conllu.TokenList) -> Optional[Dict[str, Any]]:
        """Parse a single CoNLL-U sentence into a structured example."""
        metadata = sent.metadata.copy()

        # Parse required metadata
        if "relation" not in metadata:
            return None

        example = {
            "sent_id": int(metadata.get("sent_id", 0)),
            "text": metadata.get("text", ""),
            "relation": metadata.get("relation", ""),
            "comment": metadata.get("comment"),
        }

        # Parse entity spans from metadata
        for key in ["e1", "e2"]:
            if key in metadata:
                match = re.match(r"(.+?)\s+\[(\d+):(\d+)\]", metadata[key])
                if match:
                    word, start, end = match.groups()
                    example[key] = {
                        "text": word,
                        "start_token": int(start),
                        "end_token": int(end),
                    }

        # Parse tokens
        tokens = []
        e1_token_id = None
        e2_token_id = None

        for t in sent:
            # Skip multi-word tokens
            if isinstance(t["id"], tuple):
                continue

            token = {
                "id": int(t["id"]),
                "form": t["form"],
                "lemma": t["lemma"],
                "upos": t["upos"],
                "xpos": t["xpos"],
                "feats": t["feats"],
                "head": int(t["head"]) if t["head"] is not None else 0,
                "deprel": t["deprel"],
            }

            # Check for entity markers in MISC
            if t["misc"] and "Entity" in t["misc"]:
                entity_val = t["misc"]["Entity"]
                if entity_val == "e1":
                    e1_token_id = token["id"]
                    token["entity"] = "e1"
                elif entity_val == "e2":
                    e2_token_id = token["id"]
                    token["entity"] = "e2"

            tokens.append(token)

        example["tokens"] = tokens
        example["e1_token_id"] = e1_token_id
        example["e2_token_id"] = e2_token_id

        return example

    def extract_sdp(
        self, example: Dict[str, Any]
    ) -> Tuple[List[int], List[Dict[str, Any]]]:
        """Extract the shortest dependency path between e1 and e2.

        Parameters
        ----------
        example : Dict[str, Any]
            Parsed example with tokens and entity info

        Returns
        -------
        Tuple[List[int], List[Dict[str, Any]]]
            (path_token_ids, path_tokens)
        """
        tokens = example["tokens"]
        e1_id = example.get("e1_token_id")
        e2_id = example.get("e2_token_id")

        if e1_id is None or e2_id is None:
            return [], []

        # Build adjacency graph (undirected for BFS)
        graph = defaultdict(list)
        for tok in tokens:
            tid = tok["id"]
            head = tok["head"]
            if head != 0:
                graph[tid].append(head)
                graph[head].append(tid)

        # BFS to find shortest path
        queue = deque([(e1_id, [e1_id])])
        visited = {e1_id}

        while queue:
            node, path = queue.popleft()
            if node == e2_id:
                path_tokens = [self._get_token_by_id(tokens, tid) for tid in path]
                return path, path_tokens

            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return [], []

    def _get_token_by_id(
        self, tokens: List[Dict[str, Any]], token_id: int
    ) -> Optional[Dict[str, Any]]:
        """Get token by its ID."""
        for t in tokens:
            if t["id"] == token_id:
                return t
        return None

    def extract_dep_labels(
        self, tokens: List[Dict[str, Any]], path_ids: List[int]
    ) -> List[Tuple[str, str]]:
        """Extract dependency labels along the SDP with direction.

        Parameters
        ----------
        tokens : List[Dict[str, Any]]
            All tokens in the sentence
        path_ids : List[int]
            Token IDs along the SDP

        Returns
        -------
        List[Tuple[str, str]]
            List of (deprel, direction) tuples
        """
        token_map = {t["id"]: t for t in tokens}
        labels = []

        for i in range(len(path_ids) - 1):
            a_id, b_id = path_ids[i], path_ids[i + 1]
            tok_a = token_map.get(a_id)
            tok_b = token_map.get(b_id)

            if tok_a is None or tok_b is None:
                labels.append(("unknown", "unknown"))
                continue

            if tok_a["head"] == b_id:
                labels.append((tok_a["deprel"], "up"))  # child -> head
            elif tok_b["head"] == a_id:
                labels.append((tok_b["deprel"], "down"))  # head -> child
            else:
                labels.append(("unknown", "unknown"))

        return labels

    def generalize_sdp(
        self, path_tokens: List[Dict[str, Any]], dep_labels: List[Tuple[str, str]]
    ) -> SDPSignature:
        """Convert SDP to an abstract signature.

        Parameters
        ----------
        path_tokens : List[Dict[str, Any]]
            Tokens along the SDP
        dep_labels : List[Tuple[str, str]]
            Dependency labels with direction

        Returns
        -------
        SDPSignature
            Generalized signature
        """
        pos_pattern = []
        trigger_words = []

        for tok in path_tokens:
            pos = tok["upos"]
            pos_pattern.append(pos)

            # Keep lemma as trigger word if POS is in trigger list
            if pos in self.trigger_pos:
                trigger_words.append(tok["lemma"].lower())
            else:
                trigger_words.append("")

        # Extract just the deprels (without direction for now)
        dep_pattern = tuple(label for label, _ in dep_labels)

        # Determine direction based on first entity
        e1_in_path = any(tok.get("entity") == "e1" for tok in path_tokens)
        direction = "e1_to_e2" if e1_in_path else "e2_to_e1"

        return SDPSignature(
            pos_pattern=tuple(pos_pattern),
            dep_pattern=dep_pattern,
            trigger_words=tuple(trigger_words),
            direction=direction,
            path_length=len(path_tokens),
        )

    def extract_signature(self, example: Dict[str, Any]) -> Optional[SDPExample]:
        """Extract full SDP example with signature from a parsed example.

        Parameters
        ----------
        example : Dict[str, Any]
            Parsed CoNLL-U example

        Returns
        -------
        Optional[SDPExample]
            SDP example with signature, or None if extraction fails
        """
        path_ids, path_tokens = self.extract_sdp(example)

        if not path_tokens:
            return None

        dep_labels = self.extract_dep_labels(example["tokens"], path_ids)
        signature = self.generalize_sdp(path_tokens, dep_labels)

        # Parse relation type (without direction)
        relation = example.get("relation", "Other")
        relation_type = relation.split("(")[0] if "(" in relation else relation

        return SDPExample(
            sent_id=example["sent_id"],
            text=example["text"],
            relation=relation,
            relation_type=relation_type,
            e1_text=example.get("e1", {}).get("text", ""),
            e2_text=example.get("e2", {}).get("text", ""),
            sdp_tokens=path_tokens,
            sdp_token_ids=path_ids,
            signature=signature,
        )

    def extract_all_signatures(
        self, filepath: str, relation_filter: Optional[str] = None
    ) -> List[SDPExample]:
        """Extract signatures from all examples in a CoNLL-U file.

        Parameters
        ----------
        filepath : str
            Path to CoNLL-U file
        relation_filter : Optional[str]
            Filter by relation type (e.g., "Component-Whole")

        Returns
        -------
        List[SDPExample]
            List of SDP examples with signatures
        """
        examples = self.load_conllu(filepath)
        results = []

        for ex in examples:
            # Apply relation filter if specified
            if relation_filter:
                relation = ex.get("relation", "")
                if not relation.startswith(relation_filter):
                    continue

            sdp_example = self.extract_signature(ex)
            if sdp_example:
                results.append(sdp_example)

        return results
