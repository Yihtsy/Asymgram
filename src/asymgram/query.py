"""Query and annotation helpers for CoNLL-U treebanks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from collections.abc import Iterable
import inspect
from pathlib import Path
import re
import shlex
from typing import Any, Callable, Iterator, Mapping

from .conllu import SentenceList, SentenceLists, Token, TokenList, parse
from .format_changer import read_conllu, read_conllu_sentence


@dataclass(frozen=True)
class TokenMatch:
    """A token match with sentence context."""

    sentence: TokenList
    token: Token
    sent_index: int
    token_index: int

    @property
    def sent_id(self) -> str:
        return str(getattr(self.sentence, "metadata", {}).get("sent_id", self.sent_index + 1))

    @property
    def token_id(self) -> Any:
        return self.token.get("id")

    def context(self, window: int = 5) -> str:
        """Return a tokenized context string around the match."""
        tokens = [tok for tok in self.sentence if isinstance(tok.get("id"), int)]
        idx = next((i for i, tok in enumerate(tokens) if tok is self.token), self.token_index)
        start = max(0, idx - window)
        end = min(len(tokens), idx + window + 1)
        forms = []
        for i, tok in enumerate(tokens[start:end], start=start):
            form = str(tok.get("form", ""))
            forms.append(f"[{form}]" if i == idx else form)
        return " ".join(forms)


def _load_if_path(data: Any) -> Any:
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.exists():
            return read_conllu(str(path))
    return data


def _parse_token_id_value(token_id: Any) -> Any:
    if isinstance(token_id, (int, tuple)):
        return token_id
    if isinstance(token_id, str):
        if "-" in token_id:
            start, end = token_id.split("-", 1)
            return (int(start), "-", int(end))
        if "." in token_id:
            base, index = token_id.split(".", 1)
            return (int(base), ".", int(index))
        return int(token_id)
    return token_id


def iter_sentences(data: Any) -> Iterator[TokenList]:
    """Yield TokenList objects from common asymgram data containers."""
    data = _load_if_path(data)
    if isinstance(data, TokenList):
        yield data
    elif isinstance(data, SentenceList):
        yield from data
    elif isinstance(data, SentenceLists):
        for sentence_list in data:
            yield from sentence_list
    elif isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
        for item in data:
            if isinstance(item, TokenList):
                yield item
            elif isinstance(item, SentenceList):
                yield from item
            else:
                raise TypeError(f"Unsupported iterable item type: {type(item)!r}")
    else:
        raise TypeError("data must be a path, TokenList, SentenceList, SentenceLists, or iterable of sentences")


def get_sentence_by_id(data: Any, sent_id: str, on_duplicate: str = "error") -> TokenList:
    """Return a sentence by sent_id."""
    if on_duplicate not in {"error", "first", "last", "all"}:
        raise ValueError("on_duplicate must be 'error', 'first', 'last', or 'all'.")
    if isinstance(data, (str, Path)) and Path(data).exists():
        result = read_conllu_sentence(str(data), sent_id, on_duplicate=on_duplicate)
        if on_duplicate == "all":
            return result
        return result

    matches = [
        sentence for sentence in iter_sentences(data)
        if str(sentence.metadata.get("sent_id")) == str(sent_id)
    ]
    if not matches:
        raise KeyError(f"sent_id not found: {sent_id}")
    if len(matches) > 1 and on_duplicate == "error":
        raise ValueError(f"Duplicate sent_id found: {sent_id}")
    if on_duplicate == "all":
        return matches
    if on_duplicate == "first":
        return matches[0]
    return matches[-1]


def get_token_by_position(
    data: Any,
    sent_id: str,
    token_id: Any,
    on_duplicate: str = "error",
) -> Token:
    """
    Return one token by sentence id and CoNLL-U token id.

    token_id may be an int, "7", "7.1", "2-3", or the corresponding tuple ID.
    """
    sentence = get_sentence_by_id(data, sent_id, on_duplicate=on_duplicate)
    if isinstance(sentence, list) and not isinstance(sentence, TokenList):
        raise ValueError("get_token_by_position() requires one sentence; use on_duplicate='first' or 'last'.")
    target_id = _parse_token_id_value(token_id)
    for token in sentence:
        if token.get("id") == target_id:
            return token
    raise KeyError(f"token_id not found in sent_id {sent_id}: {token_id}")


def iter_tokens(
    data: Any,
    include_punct: bool = True,
    include_empty: bool = False,
) -> Iterator[TokenMatch]:
    """Yield word-token matches with sentence context."""
    for sent_index, sentence in enumerate(iter_sentences(data)):
        token_index = 0
        for token in sentence:
            token_id = token.get("id")
            if not isinstance(token_id, int):
                if include_empty:
                    yield TokenMatch(sentence, token, sent_index, token_index)
                continue
            if not include_punct and token.get("upos") == "PUNCT":
                token_index += 1
                continue
            yield TokenMatch(sentence, token, sent_index, token_index)
            token_index += 1


def _base_deprel(deprel: Any) -> str:
    return str(deprel).split(":", 1)[0] if deprel is not None else ""


def _value_matches(value: Any, criterion: Any, regex: bool = False) -> bool:
    if criterion is None:
        return True
    if callable(criterion):
        return bool(criterion(value))
    if isinstance(criterion, re.Pattern):
        return bool(criterion.search(str(value)))
    if regex:
        return bool(re.search(str(criterion), str(value)))
    if isinstance(criterion, (set, list, tuple, frozenset)):
        return value in criterion or str(value) in {str(item) for item in criterion}
    return value == criterion or str(value) == str(criterion)


def token_matches(
    token: Token,
    regex: bool = False,
    base_deprel: bool = False,
    **criteria: Any,
) -> bool:
    """Return True if a token satisfies field criteria.

    Criteria values can be exact values, containers, compiled regex patterns, or
    callables that receive the token field value.
    """
    for field, criterion in criteria.items():
        if criterion is None:
            continue
        if field == "deprel" and base_deprel:
            value = _base_deprel(token.get("deprel"))
        elif field == "feats":
            feats = token.get("feats") or {}
            if callable(criterion):
                if not criterion(feats):
                    return False
                continue
            if isinstance(criterion, dict):
                if not all(feats.get(k) == v for k, v in criterion.items()):
                    return False
                continue
            value = feats
        elif field == "misc":
            misc = token.get("misc") or {}
            if callable(criterion):
                if not criterion(misc):
                    return False
                continue
            if isinstance(criterion, dict):
                if not all(misc.get(k) == v for k, v in criterion.items()):
                    return False
                continue
            value = misc
        else:
            value = token.get(field)
        if not _value_matches(value, criterion, regex=regex):
            return False
    return True


def find_tokens(
    data: Any,
    predicate: Callable[[Token], bool] | None = None,
    include_punct: bool = True,
    regex: bool = False,
    base_deprel: bool = False,
    **criteria: Any,
) -> list[TokenMatch]:
    """Find tokens by field criteria or a custom predicate."""
    matches = []
    for match in iter_tokens(data, include_punct=include_punct):
        token = match.token
        if predicate is not None and not predicate(token):
            continue
        if token_matches(token, regex=regex, base_deprel=base_deprel, **criteria):
            matches.append(match)
    return matches


def find_sentences(
    data: Any,
    predicate: Callable[[TokenList], bool] | None = None,
    token_predicate: Callable[[Token], bool] | None = None,
    **token_criteria: Any,
) -> list[TokenList]:
    """Find sentences by sentence predicate or by containing a matching token."""
    results = []
    for sentence in iter_sentences(data):
        if predicate is not None and predicate(sentence):
            results.append(sentence)
            continue
        if token_predicate is None and not token_criteria:
            continue
        for token in sentence:
            if not isinstance(token.get("id"), int):
                continue
            if token_predicate is not None and not token_predicate(token):
                continue
            if token_matches(token, **token_criteria):
                results.append(sentence)
                break
    return results


def concordance(
    data: Any,
    window: int = 5,
    predicate: Callable[[Token], bool] | None = None,
    **criteria: Any,
) -> list[dict[str, Any]]:
    """Return KWIC-style concordance rows for matching tokens."""
    rows = []
    for match in find_tokens(data, predicate=predicate, **criteria):
        tokens = [tok for tok in match.sentence if isinstance(tok.get("id"), int)]
        idx = next((i for i, tok in enumerate(tokens) if tok is match.token), match.token_index)
        left = " ".join(str(tok.get("form", "")) for tok in tokens[max(0, idx - window):idx])
        right = " ".join(str(tok.get("form", "")) for tok in tokens[idx + 1:idx + window + 1])
        rows.append({
            "sent_id": match.sent_id,
            "token_id": match.token_id,
            "left": left,
            "node": match.token.get("form"),
            "right": right,
            "upos": match.token.get("upos"),
            "deprel": match.token.get("deprel"),
        })
    return rows


def _copy_data(data: Any) -> Any:
    if isinstance(data, TokenList):
        return parse(data.serialize())[0]
    if isinstance(data, SentenceList):
        return parse(data.serialize())
    raise TypeError("copying is supported for TokenList and SentenceList inputs")


def update_tokens(
    data: Any,
    updates: dict[str, Any],
    predicate: Callable[[Token], bool] | None = None,
    in_place: bool = False,
    **criteria: Any,
) -> tuple[Any, int]:
    """Update matching tokens and return (data, number_changed)."""
    working = data if in_place else _copy_data(data)
    changed = 0
    for match in find_tokens(working, predicate=predicate, **criteria):
        for field, value in updates.items():
            match.token[field] = value(match.token) if callable(value) else value
        changed += 1
    return working, changed


# Structural dependency-tree queries

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


# Structured syntax-query language

QueryInput = str | Mapping[str, Any] | Callable[..., bool] | "SyntaxQuery"


@dataclass(frozen=True)
class SyntaxQuery:
    """A compiled token-and-sentence query.

    Token conditions are combined with logical AND. Sentence conditions, such
    as ``unit=clause``, are checked before the token quantifier is applied.
    """

    source: str
    token_predicates: tuple[Callable[[Token, TokenList], bool], ...] = field(
        default_factory=tuple, repr=False, compare=False
    )
    sentence_predicates: tuple[Callable[[TokenList], bool], ...] = field(
        default_factory=tuple, repr=False, compare=False
    )

    def matches_token(self, token: Token, sentence: TokenList) -> bool:
        """Return whether one token satisfies all token conditions."""
        return all(predicate(token, sentence) for predicate in self.token_predicates)

    def matches_sentence_conditions(self, sentence: TokenList) -> bool:
        """Return whether the sentence satisfies all sentence-only conditions."""
        return all(predicate(sentence) for predicate in self.sentence_predicates)


def _syntax_split_terms(expression: str) -> list[str]:
    terms: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None
    escaped = False
    for character in expression.strip():
        if escaped:
            current.append(character)
            escaped = False
            continue
        if character == "\\":
            current.append(character)
            escaped = True
            continue
        if quote:
            current.append(character)
            if character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
            current.append(character)
        elif character == "(":
            depth += 1
            current.append(character)
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced ')' in syntax query.")
            current.append(character)
        elif depth == 0 and (character.isspace() or character in {"&", ","}):
            if current:
                terms.append("".join(current))
                current = []
        else:
            current.append(character)
    if quote:
        raise ValueError("Unclosed quote in syntax query.")
    if depth:
        raise ValueError("Unbalanced parentheses in syntax query.")
    if current:
        terms.append("".join(current))
    return terms


def _syntax_decode_value(text: str, split_alternatives: bool = True) -> Any:
    text = text.strip()
    if not text:
        raise ValueError("Query values cannot be empty.")
    try:
        parts = shlex.split(text)
    except ValueError as error:
        raise ValueError(f"Invalid query value: {text!r}") from error
    value = parts[0] if len(parts) == 1 else text
    if split_alternatives and "|" in value:
        return tuple(_syntax_decode_value(part) for part in value.split("|"))
    lowered = value.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    return value


def _syntax_value_matches(observed: Any, expected: Any, operator: str) -> bool:
    if operator == "~=":
        try:
            return bool(re.search(str(expected), "" if observed is None else str(observed)))
        except re.error as error:
            raise ValueError(f"Invalid regular expression in query: {expected!r}") from error
    if isinstance(expected, (tuple, list, set, frozenset)):
        equal = any(observed == item or str(observed) == str(item) for item in expected)
    else:
        equal = observed == expected or str(observed) == str(expected)
    return not equal if operator == "!=" else equal


def _syntax_token_by_id(sentence: TokenList, token_id: Any) -> Token | None:
    return next(
        (token for token in sentence if isinstance(token.get("id"), int) and token.get("id") == token_id),
        None,
    )


def _syntax_field_value(token: Token, sentence: TokenList, field_name: str) -> Any:
    field_name = field_name.lower()
    if field_name.startswith("head."):
        head = _syntax_token_by_id(sentence, token.get("head"))
        return None if head is None else _syntax_field_value(head, sentence, field_name[5:])
    if field_name.startswith("feats."):
        feats = token.get("feats") or {}
        return feats.get(field_name.split(".", 1)[1]) if isinstance(feats, Mapping) else None
    if field_name.startswith("misc."):
        misc = token.get("misc") or {}
        return misc.get(field_name.split(".", 1)[1]) if isinstance(misc, Mapping) else None
    if field_name.startswith("metadata.") or field_name.startswith("sentence."):
        return sentence.metadata.get(field_name.split(".", 1)[1])
    if field_name == "clause_type":
        return clause_type_for_node(sentence, token)
    if field_name == "phrase_type":
        return phrase_type_for_node(token)
    return token.get(field_name)


def _syntax_word_tokens(sentence: TokenList, include_punct: bool = False) -> list[Token]:
    return [
        token for token in sentence
        if isinstance(token.get("id"), int) and (include_punct or token.get("upos") != "PUNCT")
    ]


def _syntax_children(sentence: TokenList, token: Token) -> list[Token]:
    return [child for child in _syntax_word_tokens(sentence, include_punct=True) if child.get("head") == token.get("id")]


def _syntax_descendants(sentence: TokenList, token: Token) -> list[Token]:
    found: list[Token] = []
    pending = list(_syntax_children(sentence, token))
    seen: set[int] = set()
    while pending:
        child = pending.pop()
        child_id = child.get("id")
        if child_id in seen:
            continue
        seen.add(child_id)
        found.append(child)
        pending.extend(_syntax_children(sentence, child))
    return found


def _syntax_sentence_has_unit(sentence: TokenList, unit_type: str) -> bool:
    unit_type = unit_type.lower().replace("-", "_").replace(" ", "_")
    aliases = {"catenae": "catena", "constituents": "constituent", "phrases": "phrase", "clauses": "clause"}
    unit_type = aliases.get(unit_type, unit_type)
    tokens = _syntax_word_tokens(sentence)
    if unit_type == "string":
        return len(tokens) >= 2
    if unit_type == "catena":
        ids = {token.get("id") for token in tokens}
        return any(token.get("head") in ids for token in tokens)
    if unit_type == "component":
        ids = {token.get("id") for token in tokens}
        return any(token.get("head") in ids and abs(token["id"] - token["head"]) == 1 for token in tokens)
    if unit_type in {"constituent", "phrase"}:
        return any(len(get_subtree_tokens(sentence, token)) >= 2 for token in tokens)
    if unit_type == "clause":
        return any(clause_type_for_node(sentence, token) is not None for token in tokens)
    raise ValueError(f"Unsupported unit type: {unit_type!r}")


def _syntax_callable_predicate(function: Callable[..., bool]) -> Callable[[Token, TokenList], bool]:
    try:
        parameters = inspect.signature(function).parameters.values()
        positional = [
            parameter for parameter in parameters
            if parameter.kind in {parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD}
        ]
        accepts_sentence = len(positional) >= 2 or any(
            parameter.kind == parameter.VAR_POSITIONAL for parameter in parameters
        )
    except (TypeError, ValueError):
        accepts_sentence = False
    if accepts_sentence:
        return lambda token, sentence: bool(function(token, sentence))
    return lambda token, sentence: bool(function(token))


def _syntax_comparison_predicate(field_name: str, expected: Any, operator: str = "="):
    if operator not in {"=", "!=", "~="}:
        raise ValueError(f"Unsupported query operator: {operator}")
    return lambda token, sentence: _syntax_value_matches(
        _syntax_field_value(token, sentence, field_name), expected, operator
    )


def _syntax_matches_related_token(nested: SyntaxQuery, candidate: Token, sentence: TokenList) -> bool:
    return nested.matches_sentence_conditions(sentence) and (
        not nested.token_predicates or nested.matches_token(candidate, sentence)
    )


def _syntax_compile_mapping(query: Mapping[str, Any]) -> SyntaxQuery:
    token_predicates = []
    sentence_predicates = []
    source_parts = []
    for field_name, expected in query.items():
        normalized = str(field_name).lower()
        if normalized == "unit":
            sentence_predicates.append(lambda sentence, value=str(expected): _syntax_sentence_has_unit(sentence, value))
            source_parts.append(f"unit={expected}")
            continue
        if normalized in {"has_child", "has_descendant"}:
            nested = compile_syntax_query(expected)
            relation = _syntax_children if normalized == "has_child" else _syntax_descendants
            token_predicates.append(
                lambda token, sentence, nested=nested, relation=relation: any(
                    _syntax_matches_related_token(nested, candidate, sentence)
                    for candidate in relation(sentence, token)
                )
            )
            source_parts.append(f"{normalized}({nested.source})")
            continue
        token_predicates.append(_syntax_comparison_predicate(normalized, expected))
        source_parts.append(f"{field_name}={expected}")
    return SyntaxQuery(" ".join(source_parts), tuple(token_predicates), tuple(sentence_predicates))


def _syntax_compile_string(expression: str) -> SyntaxQuery:
    token_predicates = []
    sentence_predicates = []
    for term in _syntax_split_terms(expression):
        relation_match = re.fullmatch(r"(has_child|has_descendant)\((.*)\)", term, flags=re.IGNORECASE)
        if relation_match:
            relation_name, nested_text = relation_match.groups()
            nested = compile_syntax_query(nested_text)
            relation = _syntax_children if relation_name.lower() == "has_child" else _syntax_descendants
            token_predicates.append(
                lambda token, sentence, nested=nested, relation=relation: any(
                    _syntax_matches_related_token(nested, candidate, sentence)
                    for candidate in relation(sentence, token)
                )
            )
            continue
        comparison = re.fullmatch(r"([A-Za-z_][\w.]*)\s*(!=|~=|=)\s*(.+)", term)
        if not comparison:
            raise ValueError(f"Invalid syntax-query term: {term!r}")
        field_name, operator, raw_value = comparison.groups()
        expected = _syntax_decode_value(raw_value, split_alternatives=operator != "~=")
        if field_name.lower() == "unit":
            if operator != "=":
                raise ValueError("unit only supports the '=' operator.")
            sentence_predicates.append(lambda sentence, value=str(expected): _syntax_sentence_has_unit(sentence, value))
        else:
            token_predicates.append(_syntax_comparison_predicate(field_name, expected, operator))
    return SyntaxQuery(expression.strip(), tuple(token_predicates), tuple(sentence_predicates))


def compile_syntax_query(query: QueryInput) -> SyntaxQuery:
    """Compile a string, mapping, callable, or existing ``SyntaxQuery``."""
    if isinstance(query, SyntaxQuery):
        return query
    if isinstance(query, str):
        if not query.strip():
            raise ValueError("Syntax query cannot be empty.")
        return _syntax_compile_string(query)
    if isinstance(query, Mapping):
        if not query:
            raise ValueError("Syntax query mapping cannot be empty.")
        return _syntax_compile_mapping(query)
    if callable(query):
        return SyntaxQuery(getattr(query, "__name__", "<callable>"), (_syntax_callable_predicate(query),))
    raise TypeError("query must be a string, mapping, callable, or SyntaxQuery")


def match_syntax_tokens(
    sentence: TokenList,
    query: QueryInput,
    include_punct: bool = False,
) -> list[Token]:
    """Return tokens matching a structured syntax query in one sentence."""
    compiled = compile_syntax_query(query)
    if not compiled.matches_sentence_conditions(sentence) or not compiled.token_predicates:
        return []
    return [
        token for token in _syntax_word_tokens(sentence, include_punct=include_punct)
        if compiled.matches_token(token, sentence)
    ]


def sentence_matches_syntax(
    sentence: TokenList,
    query: QueryInput,
    mode: str = "any",
    min_count: int = 1,
    include_punct: bool = False,
) -> bool:
    """Return whether a sentence matches, using any/all/at_least quantification."""
    compiled = compile_syntax_query(query)
    if not compiled.matches_sentence_conditions(sentence):
        return False
    if not compiled.token_predicates:
        return True
    tokens = _syntax_word_tokens(sentence, include_punct=include_punct)
    matched = sum(compiled.matches_token(token, sentence) for token in tokens)
    normalized = mode.lower().replace("-", "_")
    if normalized == "any":
        return matched >= 1
    if normalized == "all":
        return bool(tokens) and matched == len(tokens)
    if normalized in {"at_least", "count"}:
        if min_count < 1:
            raise ValueError("min_count must be at least 1.")
        return matched >= min_count
    raise ValueError("mode must be 'any', 'all', or 'at_least'.")


def find_syntax_matches(data: Any, query: QueryInput, include_punct: bool = False) -> list[TokenMatch]:
    """Return context-bearing token matches across common asymgram containers."""
    compiled = compile_syntax_query(query)
    results: list[TokenMatch] = []
    for sent_index, sentence in enumerate(iter_sentences(data)):
        if not compiled.matches_sentence_conditions(sentence):
            continue
        word_index = 0
        for token in sentence:
            if not isinstance(token.get("id"), int):
                continue
            if not include_punct and token.get("upos") == "PUNCT":
                continue
            if compiled.token_predicates and compiled.matches_token(token, sentence):
                results.append(TokenMatch(sentence, token, sent_index, word_index))
            word_index += 1
    return results
