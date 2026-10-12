"""Structural query helpers for CoNLL-U dependency trees."""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable

from .conllu import Token, TokenList


PHRASE_UPOS = {
    "noun": {"NOUN", "PROPN", "PRON", "NUM"},
    "np": {"NOUN", "PROPN", "PRON", "NUM"},
    "verb": {"VERB", "AUX"},
    "vp": {"VERB", "AUX"},
    "adjective": {"ADJ"},
    "adjp": {"ADJ"},
    "adverb": {"ADV"},
    "advp": {"ADV"},
    "clause": {"VERB", "AUX", "ADJ", "NOUN", "PROPN"},
}
CLAUSE_DEPRELS = {"root", "advcl", "acl", "acl:relcl", "ccomp", "csubj", "xcomp", "parataxis", "conj"}
CLAUSE_TYPE_BY_DEPREL = {
    "root": "main_clause",
    "acl:relcl": "relative_clause",
    "acl": "adnominal_clause",
    "advcl": "adverbial_clause",
    "ccomp": "complement_clause",
    "csubj": "subject_clause",
    "xcomp": "open_complement_clause",
    "parataxis": "paratactic_clause",
    "conj": "coordinate_clause",
}
CLAUSE_TYPE_ALIASES = {
    "main": "main_clause",
    "main_clause": "main_clause",
    "relative": "relative_clause",
    "relative_clause": "relative_clause",
    "relcl": "relative_clause",
    "adnominal": "adnominal_clause",
    "adnominal_clause": "adnominal_clause",
    "adverbial": "adverbial_clause",
    "adverbial_clause": "adverbial_clause",
    "advcl": "adverbial_clause",
    "complement": "complement_clause",
    "complement_clause": "complement_clause",
    "content": "complement_clause",
    "subject": "subject_clause",
    "subject_clause": "subject_clause",
    "open_complement": "open_complement_clause",
    "open_complement_clause": "open_complement_clause",
    "xcomp": "open_complement_clause",
    "paratactic": "paratactic_clause",
    "paratactic_clause": "paratactic_clause",
    "coordinate": "coordinate_clause",
    "coordinate_clause": "coordinate_clause",
}


def _word_tokens(sentence: TokenList, include_punct: bool = True) -> list[Token]:
    return [
        token for token in sentence
        if isinstance(token.get("id"), int) and (include_punct or token.get("upos") != "PUNCT")
    ]


def _id_map(sentence: TokenList, include_punct: bool = True) -> dict[int, Token]:
    return {token["id"]: token for token in _word_tokens(sentence, include_punct=include_punct)}


def resolve_token(sentence: TokenList, node: int | str | Token, occurrence: int = 0) -> Token:
    """Resolve a token by Token object, integer ID, FORM, or LEMMA."""
    if isinstance(node, Token):
        return node
    if isinstance(node, int):
        token = _id_map(sentence).get(node)
        if token is None:
            raise ValueError(f"No token with id={node}.")
        return token
    matches = [
        token for token in _word_tokens(sentence)
        if token.get("form") == node or token.get("lemma") == node
    ]
    if not matches:
        raise ValueError(f"No token with FORM or LEMMA matching {node!r}.")
    if occurrence >= len(matches):
        raise ValueError(f"Only {len(matches)} matching token(s), occurrence={occurrence} is out of range.")
    return matches[occurrence]


def get_root(sentence: TokenList) -> Token | None:
    """Return the unique root token, or None if the sentence does not have one."""
    roots = [token for token in _word_tokens(sentence) if token.get("head") == 0]
    return roots[0] if len(roots) == 1 else None


def get_head(sentence: TokenList, node: int | str | Token, occurrence: int = 0) -> Token | None:
    """Return the head token of a node, or None for the root."""
    token = resolve_token(sentence, node, occurrence=occurrence)
    head = token.get("head")
    if head == 0:
        return None
    return _id_map(sentence).get(head)


def get_dependents(
    sentence: TokenList,
    node: int | str | Token,
    occurrence: int = 0,
    deprel: str | Iterable[str] | None = None,
    include_punct: bool = False,
    base_deprel: bool = False,
) -> list[Token]:
    """Return direct dependents of a node."""
    token = resolve_token(sentence, node, occurrence=occurrence)
    wanted = None
    if deprel is not None:
        wanted = {deprel} if isinstance(deprel, str) else set(deprel)
    results = []
    for child in _word_tokens(sentence, include_punct=include_punct):
        if child.get("head") != token.get("id"):
            continue
        child_deprel = child.get("deprel")
        key = str(child_deprel).split(":", 1)[0] if base_deprel and child_deprel is not None else child_deprel
        if wanted is None or key in wanted:
            results.append(child)
    return sorted(results, key=lambda item: item.get("id"))


def get_siblings(
    sentence: TokenList,
    node: int | str | Token,
    occurrence: int = 0,
    include_self: bool = False,
    include_punct: bool = False,
) -> list[Token]:
    """Return tokens sharing the same head as the node."""
    token = resolve_token(sentence, node, occurrence=occurrence)
    head = token.get("head")
    siblings = [
        other for other in _word_tokens(sentence, include_punct=include_punct)
        if other.get("head") == head and (include_self or other is not token)
    ]
    return sorted(siblings, key=lambda item: item.get("id"))


def get_left_siblings(sentence: TokenList, node: int | str | Token, occurrence: int = 0, include_punct: bool = False) -> list[Token]:
    """Return same-head siblings to the left of the node."""
    token = resolve_token(sentence, node, occurrence=occurrence)
    return [sibling for sibling in get_siblings(sentence, token, include_punct=include_punct) if sibling.get("id") < token.get("id")]


def get_right_siblings(sentence: TokenList, node: int | str | Token, occurrence: int = 0, include_punct: bool = False) -> list[Token]:
    """Return same-head siblings to the right of the node."""
    token = resolve_token(sentence, node, occurrence=occurrence)
    return [sibling for sibling in get_siblings(sentence, token, include_punct=include_punct) if sibling.get("id") > token.get("id")]


def get_left_sibling(sentence: TokenList, node: int | str | Token, occurrence: int = 0, include_punct: bool = False) -> Token | None:
    """Return the nearest left sibling, if any."""
    siblings = get_left_siblings(sentence, node, occurrence=occurrence, include_punct=include_punct)
    return siblings[-1] if siblings else None


def get_right_sibling(sentence: TokenList, node: int | str | Token, occurrence: int = 0, include_punct: bool = False) -> Token | None:
    """Return the nearest right sibling, if any."""
    siblings = get_right_siblings(sentence, node, occurrence=occurrence, include_punct=include_punct)
    return siblings[0] if siblings else None


def get_token_relations(sentence: TokenList, node: int | str | Token, occurrence: int = 0, include_punct: bool = False) -> dict[str, Any]:
    """Return root, head, dependents, and siblings for one node."""
    token = resolve_token(sentence, node, occurrence=occurrence)
    return {
        "token": token,
        "root": get_root(sentence),
        "head": get_head(sentence, token),
        "dependents": get_dependents(sentence, token, include_punct=include_punct),
        "siblings": get_siblings(sentence, token, include_punct=include_punct),
        "left_siblings": get_left_siblings(sentence, token, include_punct=include_punct),
        "right_siblings": get_right_siblings(sentence, token, include_punct=include_punct),
        "left_sibling": get_left_sibling(sentence, token, include_punct=include_punct),
        "right_sibling": get_right_sibling(sentence, token, include_punct=include_punct),
    }


def is_connected_tree(sentence: TokenList, include_punct: bool = True) -> bool:
    """Return True if all tokens are connected to a single dependency root."""
    tokens = _word_tokens(sentence, include_punct=include_punct)
    if not tokens:
        return True
    id_to_token = {token["id"]: token for token in tokens}
    roots = [token for token in tokens if token.get("head") == 0]
    if len(roots) != 1:
        return False
    adjacency = {token_id: set() for token_id in id_to_token}
    for token in tokens:
        token_id = token.get("id")
        head = token.get("head")
        if head == 0:
            continue
        if head not in id_to_token:
            return False
        adjacency[token_id].add(head)
        adjacency[head].add(token_id)
    visited = set()
    queue = deque([roots[0]["id"]])
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        queue.extend(adjacency[current] - visited)
    return len(visited) == len(tokens)


def has_cycle(sentence: TokenList, include_punct: bool = True) -> bool:
    """Return True if following HEAD links finds a cycle."""
    id_to_token = _id_map(sentence, include_punct=include_punct)
    for token in id_to_token.values():
        seen = set()
        current = token
        while True:
            head = current.get("head")
            if not isinstance(head, int) or head == 0:
                break
            if head in seen:
                return True
            seen.add(head)
            current = id_to_token.get(head)
            if current is None:
                break
    return False


def is_dependency_tree(sentence: TokenList, include_punct: bool = True) -> bool:
    """Return True if the sentence is connected and acyclic with one root."""
    return is_connected_tree(sentence, include_punct=include_punct) and not has_cycle(sentence, include_punct=include_punct)


def dependency_arcs(sentence: TokenList, include_punct: bool = True) -> list[tuple[int, int]]:
    """Return directed dependency arcs as (head, dependent) ID pairs."""
    ids = set(_id_map(sentence, include_punct=include_punct))
    arcs = []
    for token in _word_tokens(sentence, include_punct=include_punct):
        head = token.get("head")
        dep = token.get("id")
        if isinstance(head, int) and head != 0 and head in ids:
            arcs.append((head, dep))
    return arcs


def crossing_dependencies(sentence: TokenList, include_punct: bool = True) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Return pairs of crossing dependency arcs."""
    crossings = []
    arcs = dependency_arcs(sentence, include_punct=include_punct)
    for i, first in enumerate(arcs):
        a, b = sorted(first)
        for second in arcs[i + 1:]:
            if set(first) & set(second):
                continue
            c, d = sorted(second)
            if (a < c < b < d) or (c < a < d < b):
                crossings.append((first, second))
    return crossings


def has_crossing_dependencies(sentence: TokenList, include_punct: bool = True) -> bool:
    """Return True if at least one pair of dependency arcs crosses."""
    return bool(crossing_dependencies(sentence, include_punct=include_punct))


def subtree_token_ids(sentence: TokenList, node: int | str | Token, occurrence: int = 0, include_punct: bool = True) -> set[int]:
    """Return token IDs in the dependency subtree rooted at node."""
    token = resolve_token(sentence, node, occurrence=occurrence)
    id_to_token = _id_map(sentence, include_punct=include_punct)
    root_id = token.get("id")
    if root_id not in id_to_token:
        return set()
    children = {token_id: [] for token_id in id_to_token}
    for item in id_to_token.values():
        head = item.get("head")
        if head in children:
            children[head].append(item.get("id"))
    found = set()
    queue = deque([root_id])
    while queue:
        current = queue.popleft()
        if current in found:
            continue
        found.add(current)
        queue.extend(children.get(current, []))
    return found


def is_contiguous_span(token_ids: Iterable[int]) -> bool:
    """Return True if token IDs form one continuous interval."""
    ids = sorted(token_ids)
    return not ids or ids == list(range(ids[0], ids[-1] + 1))


def _normalize_token_ids(sentence: TokenList, token_ids: Iterable[int | Token], include_punct: bool = True) -> set[int]:
    """Normalize token IDs or Token objects to an integer ID set."""
    valid_ids = set(_id_map(sentence, include_punct=include_punct))
    normalized = {
        item.get("id") if isinstance(item, Token) else item
        for item in token_ids
    }
    normalized = {item for item in normalized if isinstance(item, int)}
    unknown = normalized - valid_ids
    if unknown:
        raise ValueError(f"Unknown token IDs: {sorted(unknown)}.")
    return normalized


def is_string(sentence: TokenList, token_ids: Iterable[int | Token], include_punct: bool = True) -> bool:
    """Return True if tokens are continuous with respect to precedence."""
    ids = _normalize_token_ids(sentence, token_ids, include_punct=include_punct)
    return bool(ids) and is_contiguous_span(ids)


def catena_root(sentence: TokenList, token_ids: Iterable[int | Token], include_punct: bool = True) -> Token | None:
    """Return the unique root of a catena candidate."""
    ids = _normalize_token_ids(sentence, token_ids, include_punct=include_punct)
    if not ids:
        return None
    id_to_token = _id_map(sentence, include_punct=include_punct)
    roots = [
        token for token_id, token in id_to_token.items()
        if token_id in ids and token.get("head") not in ids
    ]
    return roots[0] if len(roots) == 1 else None


def is_catena(sentence: TokenList, token_ids: Iterable[int | Token], include_punct: bool = True) -> bool:
    """Return True if tokens are continuous with respect to dominance."""
    ids = _normalize_token_ids(sentence, token_ids, include_punct=include_punct)
    if not ids:
        return False
    id_to_token = _id_map(sentence, include_punct=include_punct)
    root = catena_root(sentence, ids, include_punct=include_punct)
    if root is None:
        return False
    adjacency = {token_id: set() for token_id in ids}
    for token_id in ids:
        head = id_to_token[token_id].get("head")
        if head in ids:
            adjacency[token_id].add(head)
            adjacency[head].add(token_id)
    visited = set()
    queue = deque([root.get("id")])
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        queue.extend(adjacency[current] - visited)
    return visited == ids


def is_component(sentence: TokenList, token_ids: Iterable[int | Token], include_punct: bool = True) -> bool:
    """Return True if tokens are both a string and a catena."""
    ids = _normalize_token_ids(sentence, token_ids, include_punct=include_punct)
    return is_string(sentence, ids, include_punct=include_punct) and is_catena(sentence, ids, include_punct=include_punct)


def is_constituent(sentence: TokenList, token_ids: Iterable[int | Token], include_punct: bool = True) -> bool:
    """Return True if tokens form a complete dependency subtree."""
    ids = _normalize_token_ids(sentence, token_ids, include_punct=include_punct)
    root = catena_root(sentence, ids, include_punct=include_punct)
    if root is None:
        return False
    return subtree_token_ids(sentence, root, include_punct=include_punct) == ids


def is_phrase(sentence: TokenList, token_ids: Iterable[int | Token], include_punct: bool = True) -> bool:
    """Return True if tokens form a multi-word complete dependency subtree."""
    ids = _normalize_token_ids(sentence, token_ids, include_punct=include_punct)
    return len(ids) >= 2 and is_constituent(sentence, ids, include_punct=include_punct)


def catena_head(sentence: TokenList, token_ids: Iterable[int | Token], include_punct: bool = True) -> Token | None:
    """Return the external head immediately dominating a catena."""
    ids = _normalize_token_ids(sentence, token_ids, include_punct=include_punct)
    root = catena_root(sentence, ids, include_punct=include_punct)
    if root is None:
        return None
    head = root.get("head")
    if not isinstance(head, int) or head == 0:
        return None
    return _id_map(sentence, include_punct=include_punct).get(head)


def get_parent(sentence: TokenList, node: int | str | Token, occurrence: int = 0) -> Token | None:
    """Return the parent word of a node."""
    return get_head(sentence, node, occurrence=occurrence)


def get_children(
    sentence: TokenList,
    node: int | str | Token,
    occurrence: int = 0,
    include_punct: bool = False,
) -> list[Token]:
    """Return child words immediately dominated by a node."""
    return get_dependents(sentence, node, occurrence=occurrence, include_punct=include_punct)


def get_pre_children(sentence: TokenList, node: int | str | Token, occurrence: int = 0, include_punct: bool = False) -> list[Token]:
    """Return child words preceding their parent."""
    token = resolve_token(sentence, node, occurrence=occurrence)
    return [child for child in get_children(sentence, token, include_punct=include_punct) if child.get("id") < token.get("id")]


def get_post_children(sentence: TokenList, node: int | str | Token, occurrence: int = 0, include_punct: bool = False) -> list[Token]:
    """Return child words following their parent."""
    token = resolve_token(sentence, node, occurrence=occurrence)
    return [child for child in get_children(sentence, token, include_punct=include_punct) if child.get("id") > token.get("id")]


def get_dependent_constituents(
    sentence: TokenList,
    node: int | str | Token,
    occurrence: int = 0,
    include_punct: bool = False,
) -> list[list[Token]]:
    """Return complete-subtree dependents immediately dominated by a node."""
    return [
        get_subtree_tokens(sentence, child, include_punct=include_punct, continuous_only=False)
        for child in get_children(sentence, node, occurrence=occurrence, include_punct=include_punct)
    ]


def get_pre_dependent_constituents(sentence: TokenList, node: int | str | Token, occurrence: int = 0, include_punct: bool = False) -> list[list[Token]]:
    """Return dependent constituents preceding their head."""
    token = resolve_token(sentence, node, occurrence=occurrence)
    return [
        get_subtree_tokens(sentence, child, include_punct=include_punct, continuous_only=False)
        for child in get_pre_children(sentence, token, include_punct=include_punct)
    ]


def get_post_dependent_constituents(sentence: TokenList, node: int | str | Token, occurrence: int = 0, include_punct: bool = False) -> list[list[Token]]:
    """Return dependent constituents following their head."""
    token = resolve_token(sentence, node, occurrence=occurrence)
    return [
        get_subtree_tokens(sentence, child, include_punct=include_punct, continuous_only=False)
        for child in get_post_children(sentence, token, include_punct=include_punct)
    ]


def is_projective_tree(sentence: TokenList, include_punct: bool = True) -> bool:
    """Return True if every dependency subtree is linearly contiguous."""
    if not is_dependency_tree(sentence, include_punct=include_punct):
        return False
    for token in _word_tokens(sentence, include_punct=include_punct):
        if not is_contiguous_span(subtree_token_ids(sentence, token, include_punct=include_punct)):
            return False
    return True


def dependency_distance(sentence: TokenList, node: int | str | Token, occurrence: int = 0, absolute: bool = True) -> int:
    """Return dependency distance from node to its head."""
    token = resolve_token(sentence, node, occurrence=occurrence)
    head = token.get("head")
    if not isinstance(head, int) or head == 0:
        return 0
    distance = token.get("id") - head
    return abs(distance) if absolute else distance


def directional_dependency_distance(sentence: TokenList, node: int | str | Token, occurrence: int = 0) -> int:
    """Return signed dependency distance from head to dependent."""
    return dependency_distance(sentence, node, occurrence=occurrence, absolute=False)


def node_level(sentence: TokenList, node: int | str | Token, occurrence: int = 0) -> int:
    """Return the number of HEAD steps from node to root."""
    token = resolve_token(sentence, node, occurrence=occurrence)
    id_to_token = _id_map(sentence)
    level = 0
    current = token
    seen = set()
    while True:
        head = current.get("head")
        if not isinstance(head, int) or head == 0:
            return level
        if head in seen:
            raise ValueError(f"Cycle detected while tracing token id {token.get('id')}.")
        seen.add(head)
        current = id_to_token.get(head)
        if current is None:
            return level
        level += 1


def hierarchical_distance(sentence: TokenList, first: int | str | Token, second: int | str | Token, occurrence: int = 0) -> int:
    """Return edge distance between two nodes in the dependency tree."""
    first_token = resolve_token(sentence, first, occurrence=occurrence)
    second_token = resolve_token(sentence, second, occurrence=occurrence)
    adjacency = {token.get("id"): set() for token in _word_tokens(sentence)}
    for token in _word_tokens(sentence):
        head = token.get("head")
        token_id = token.get("id")
        if head in adjacency:
            adjacency[token_id].add(head)
            adjacency[head].add(token_id)
    start = first_token.get("id")
    target = second_token.get("id")
    queue = deque([(start, 0)])
    seen = set()
    while queue:
        current, distance = queue.popleft()
        if current == target:
            return distance
        if current in seen:
            continue
        seen.add(current)
        for nxt in adjacency.get(current, set()):
            if nxt not in seen:
                queue.append((nxt, distance + 1))
    raise ValueError("Nodes are not connected.")


def node_metrics(sentence: TokenList, node: int | str | Token, occurrence: int = 0, include_punct: bool = False) -> dict[str, Any]:
    """Return simple node-level dependency metrics."""
    token = resolve_token(sentence, node, occurrence=occurrence)
    subtree_ids = subtree_token_ids(sentence, token, include_punct=include_punct)
    return {
        "id": token.get("id"),
        "form": token.get("form"),
        "head": token.get("head"),
        "deprel": token.get("deprel"),
        "dependency_distance": dependency_distance(sentence, token),
        "directional_dependency_distance": directional_dependency_distance(sentence, token),
        "level": node_level(sentence, token),
        "dependent_count": len(get_dependents(sentence, token, include_punct=include_punct)),
        "subtree_size": len(subtree_ids),
    }


def get_subtree_tokens(
    sentence: TokenList,
    node: int | str | Token,
    occurrence: int = 0,
    include_punct: bool = False,
    continuous_only: bool = False,
) -> list[Token]:
    """Return tokens in the subtree rooted at node."""
    ids = subtree_token_ids(sentence, node, occurrence=occurrence, include_punct=include_punct)
    if continuous_only and not is_contiguous_span(ids):
        raise ValueError("The subtree is not a continuous phrase span.")
    return [token for token in _word_tokens(sentence, include_punct=include_punct) if token.get("id") in ids]


def subtree_span(sentence: TokenList, node: int | str | Token, occurrence: int = 0, include_punct: bool = False) -> tuple[int, int]:
    """Return the linear token ID span of a subtree."""
    ids = sorted(subtree_token_ids(sentence, node, occurrence=occurrence, include_punct=include_punct))
    if not ids:
        raise ValueError("The subtree is empty.")
    return ids[0], ids[-1]


def phrase_type_for_node(token: Token) -> str:
    """Infer a coarse phrase type from UPOS and DEPREL."""
    upos = token.get("upos")
    deprel = token.get("deprel")
    if deprel in CLAUSE_DEPRELS and upos in PHRASE_UPOS["clause"]:
        return "clause"
    if upos in PHRASE_UPOS["np"]:
        return "np"
    if upos in PHRASE_UPOS["vp"]:
        return "vp"
    if upos in PHRASE_UPOS["adjp"]:
        return "adjp"
    if upos in PHRASE_UPOS["advp"]:
        return "advp"
    return "phrase"


def extract_headed_phrase(
    sentence: TokenList,
    node: int | str | Token,
    occurrence: int = 0,
    phrase_type: str = "auto",
    include_punct: bool = False,
    continuous_only: bool = True,
    as_text: bool = False,
) -> list[Token] | str:
    """Extract the phrase headed by a node as tokens or text."""
    token = resolve_token(sentence, node, occurrence=occurrence)
    inferred = phrase_type_for_node(token) if phrase_type == "auto" else phrase_type.lower()
    allowed = PHRASE_UPOS.get(inferred)
    if allowed is not None and token.get("upos") not in allowed:
        raise ValueError(f"Token UPOS={token.get('upos')} is not compatible with phrase_type={phrase_type!r}.")
    tokens = get_subtree_tokens(sentence, token, include_punct=include_punct, continuous_only=continuous_only)
    if as_text:
        return " ".join(str(item.get("form", "")) for item in tokens)
    return tokens


def extract_noun_phrase(sentence: TokenList, node: int | str | Token, **kwargs) -> list[Token] | str:
    """Extract a noun phrase headed by node."""
    return extract_headed_phrase(sentence, node, phrase_type="np", **kwargs)


def extract_verb_phrase(sentence: TokenList, node: int | str | Token, **kwargs) -> list[Token] | str:
    """Extract a verb phrase headed by node."""
    return extract_headed_phrase(sentence, node, phrase_type="vp", **kwargs)


def extract_adjective_phrase(sentence: TokenList, node: int | str | Token, **kwargs) -> list[Token] | str:
    """Extract an adjective phrase headed by node."""
    return extract_headed_phrase(sentence, node, phrase_type="adjp", **kwargs)


def extract_adverb_phrase(sentence: TokenList, node: int | str | Token, **kwargs) -> list[Token] | str:
    """Extract an adverb phrase headed by node."""
    return extract_headed_phrase(sentence, node, phrase_type="advp", **kwargs)


def extract_clause(sentence: TokenList, node: int | str | Token, **kwargs) -> list[Token] | str:
    """Extract a clause headed by node."""
    return extract_headed_phrase(sentence, node, phrase_type="clause", **kwargs)


def _normalize_clause_type(clause_type: str | None) -> str | None:
    """Normalize user-facing clause type names."""
    if clause_type is None:
        return None
    normalized = clause_type.lower().replace("-", "_").replace(" ", "_")
    return CLAUSE_TYPE_ALIASES.get(normalized, normalized)


def clause_type_for_node(sentence: TokenList, node: int | str | Token, occurrence: int = 0) -> str | None:
    """Return the clause type headed by a node, if the node is clause-like."""
    token = resolve_token(sentence, node, occurrence=occurrence)
    upos = token.get("upos")
    if upos not in PHRASE_UPOS["clause"]:
        return None
    deprel = token.get("deprel")
    if deprel in CLAUSE_TYPE_BY_DEPREL:
        return CLAUSE_TYPE_BY_DEPREL[deprel]
    base = str(deprel).split(":", 1)[0] if deprel else ""
    return CLAUSE_TYPE_BY_DEPREL.get(base)


def is_clause_node(
    sentence: TokenList,
    node: int | str | Token,
    occurrence: int = 0,
    clause_type: str | None = None,
) -> bool:
    """Return True if a node heads a clause, optionally of one clause type."""
    observed = clause_type_for_node(sentence, node, occurrence=occurrence)
    expected = _normalize_clause_type(clause_type)
    return observed is not None if expected is None else observed == expected


def extract_typed_clause(
    sentence: TokenList,
    node: int | str | Token,
    occurrence: int = 0,
    clause_type: str | None = None,
    include_punct: bool = False,
    continuous_only: bool = True,
    as_text: bool = False,
    strict: bool = False,
) -> list[Token] | str | None:
    """Extract a clause only when the node matches the requested clause type."""
    observed = clause_type_for_node(sentence, node, occurrence=occurrence)
    expected = _normalize_clause_type(clause_type)
    matches = observed is not None if expected is None else observed == expected
    if not matches:
        if strict:
            raise ValueError(f"Token does not head the requested clause type: expected={expected!r}, observed={observed!r}.")
        return None
    return extract_clause(
        sentence,
        node,
        occurrence=occurrence,
        include_punct=include_punct,
        continuous_only=continuous_only,
        as_text=as_text,
    )


def extract_main_clause(sentence: TokenList, node: int | str | Token, **kwargs) -> list[Token] | str | None:
    """Extract a main clause headed by node, if applicable."""
    return extract_typed_clause(sentence, node, clause_type="main_clause", **kwargs)


def extract_relative_clause(sentence: TokenList, node: int | str | Token, **kwargs) -> list[Token] | str | None:
    """Extract a relative clause headed by node, if applicable."""
    return extract_typed_clause(sentence, node, clause_type="relative_clause", **kwargs)


def extract_adnominal_clause(sentence: TokenList, node: int | str | Token, **kwargs) -> list[Token] | str | None:
    """Extract an adnominal clause headed by node, if applicable."""
    return extract_typed_clause(sentence, node, clause_type="adnominal_clause", **kwargs)


def extract_adverbial_clause(sentence: TokenList, node: int | str | Token, **kwargs) -> list[Token] | str | None:
    """Extract an adverbial clause headed by node, if applicable."""
    return extract_typed_clause(sentence, node, clause_type="adverbial_clause", **kwargs)


def extract_complement_clause(sentence: TokenList, node: int | str | Token, **kwargs) -> list[Token] | str | None:
    """Extract a complement clause headed by node, if applicable."""
    return extract_typed_clause(sentence, node, clause_type="complement_clause", **kwargs)


def extract_subject_clause(sentence: TokenList, node: int | str | Token, **kwargs) -> list[Token] | str | None:
    """Extract a subject clause headed by node, if applicable."""
    return extract_typed_clause(sentence, node, clause_type="subject_clause", **kwargs)


def extract_open_complement_clause(sentence: TokenList, node: int | str | Token, **kwargs) -> list[Token] | str | None:
    """Extract an open complement clause headed by node, if applicable."""
    return extract_typed_clause(sentence, node, clause_type="open_complement_clause", **kwargs)


def extract_paratactic_clause(sentence: TokenList, node: int | str | Token, **kwargs) -> list[Token] | str | None:
    """Extract a paratactic clause headed by node, if applicable."""
    return extract_typed_clause(sentence, node, clause_type="paratactic_clause", **kwargs)


def extract_coordinate_clause(sentence: TokenList, node: int | str | Token, **kwargs) -> list[Token] | str | None:
    """Extract a coordinate clause headed by node, if applicable."""
    return extract_typed_clause(sentence, node, clause_type="coordinate_clause", **kwargs)
