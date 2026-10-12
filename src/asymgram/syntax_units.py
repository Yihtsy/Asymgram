"""Syntactic unit objects and counters for dependency trees."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Iterator

from .conllu import Token, TokenList
from .query import iter_sentences
from .tree_query import (
    catena_head,
    catena_root,
    clause_type_for_node,
    get_head,
    get_subtree_tokens,
    is_catena,
    is_component,
    is_constituent,
    is_phrase,
    is_string,
    phrase_type_for_node,
)


@dataclass(frozen=True)
class String:
    """A word combination continuous with respect to precedence."""

    sentence: TokenList
    token_ids: tuple[int, ...]
    sent_index: int | None = None

    @property
    def tokens(self) -> list[Token]:
        return _tokens_by_ids(self.sentence, self.token_ids)

    @property
    def text(self) -> str:
        return _unit_text(self.tokens)

    @property
    def span(self) -> tuple[int, int]:
        return self.token_ids[0], self.token_ids[-1]

    @property
    def size(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True)
class Catena:
    """A word combination continuous with respect to dominance."""

    sentence: TokenList
    token_ids: tuple[int, ...]
    sent_index: int | None = None

    @property
    def tokens(self) -> list[Token]:
        return _tokens_by_ids(self.sentence, self.token_ids)

    @property
    def text(self) -> str:
        return _unit_text(self.tokens)

    @property
    def root(self) -> Token | None:
        return catena_root(self.sentence, self.token_ids)

    @property
    def head(self) -> Token | None:
        return catena_head(self.sentence, self.token_ids)

    @property
    def size(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True)
class Component(Catena):
    """A word combination that is both a string and a catena."""

    @property
    def span(self) -> tuple[int, int]:
        return self.token_ids[0], self.token_ids[-1]


@dataclass(frozen=True)
class Constituent(Catena):
    """A complete dependency subtree."""

    @property
    def root(self) -> Token | None:
        return catena_root(self.sentence, self.token_ids)

    @property
    def head(self) -> Token | None:
        root = self.root
        return get_head(self.sentence, root) if root is not None else None


@dataclass(frozen=True)
class Phrase(Constituent):
    """A multi-word complete dependency subtree."""

    @property
    def phrase_type(self) -> str | None:
        root = self.root
        return phrase_type_for_node(root) if root is not None else None


@dataclass(frozen=True)
class Clause(Phrase):
    """A clause-like subtree headed by a clause node."""

    @property
    def clause_type(self) -> str | None:
        root = self.root
        return clause_type_for_node(self.sentence, root) if root is not None else None


UNIT_CLASSES = {
    "string": String,
    "catena": Catena,
    "component": Component,
    "constituent": Constituent,
    "phrase": Phrase,
    "clause": Clause,
}


def _word_tokens(sentence: TokenList, include_punct: bool = True) -> list[Token]:
    return [
        token for token in sentence
        if isinstance(token.get("id"), int) and (include_punct or token.get("upos") != "PUNCT")
    ]


def _tokens_by_ids(sentence: TokenList, token_ids: Iterable[int], include_punct: bool = True) -> list[Token]:
    wanted = set(token_ids)
    return [token for token in _word_tokens(sentence, include_punct=include_punct) if token.get("id") in wanted]


def _unit_text(tokens: Iterable[Token]) -> str:
    return " ".join(str(token.get("form", "")) for token in tokens)


def _all_id_combinations(sentence: TokenList, include_punct: bool = True) -> Iterator[tuple[int, ...]]:
    ids = [token["id"] for token in _word_tokens(sentence, include_punct=include_punct)]
    for size in range(1, len(ids) + 1):
        yield from combinations(ids, size)


def _normalize_unit_type(unit_type: str) -> str:
    normalized = unit_type.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "strings": "string",
        "catenae": "catena",
        "catenas": "catena",
        "components": "component",
        "constituents": "constituent",
        "phrases": "phrase",
        "clauses": "clause",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in UNIT_CLASSES:
        raise ValueError(f"Unsupported unit_type={unit_type!r}.")
    return normalized


def _unit_predicate(unit_type: str):
    return {
        "string": is_string,
        "catena": is_catena,
        "component": is_component,
        "constituent": is_constituent,
        "phrase": is_phrase,
    }[unit_type]


def iter_strings(sentence: TokenList, include_punct: bool = True, min_size: int = 1) -> Iterator[String]:
    """Yield all strings in a sentence."""
    tokens = _word_tokens(sentence, include_punct=include_punct)
    ids = [token["id"] for token in tokens]
    for start in range(len(ids)):
        for end in range(start + max(1, min_size), len(ids) + 1):
            yield String(sentence, tuple(ids[start:end]))


def iter_catenae(sentence: TokenList, include_punct: bool = True, min_size: int = 1) -> Iterator[Catena]:
    """Yield all catenae in a sentence."""
    for token_ids in _all_id_combinations(sentence, include_punct=include_punct):
        if len(token_ids) >= min_size and is_catena(sentence, token_ids, include_punct=include_punct):
            yield Catena(sentence, token_ids)


def iter_components(sentence: TokenList, include_punct: bool = True, min_size: int = 1) -> Iterator[Component]:
    """Yield all components in a sentence."""
    for token_ids in _all_id_combinations(sentence, include_punct=include_punct):
        if len(token_ids) >= min_size and is_component(sentence, token_ids, include_punct=include_punct):
            yield Component(sentence, token_ids)


def iter_constituents(sentence: TokenList, include_punct: bool = True, min_size: int = 1) -> Iterator[Constituent]:
    """Yield all complete dependency subtrees in a sentence."""
    for token in _word_tokens(sentence, include_punct=include_punct):
        token_ids = tuple(sorted(tok["id"] for tok in get_subtree_tokens(sentence, token, include_punct=include_punct)))
        if len(token_ids) >= min_size:
            yield Constituent(sentence, token_ids)


def iter_phrases(sentence: TokenList, include_punct: bool = True, min_size: int = 2) -> Iterator[Phrase]:
    """Yield all multi-word complete dependency subtrees in a sentence."""
    for constituent in iter_constituents(sentence, include_punct=include_punct, min_size=min_size):
        if is_phrase(sentence, constituent.token_ids, include_punct=include_punct):
            yield Phrase(sentence, constituent.token_ids)


def iter_clauses(sentence: TokenList, include_punct: bool = True, min_size: int = 1) -> Iterator[Clause]:
    """Yield all clause-like complete dependency subtrees in a sentence."""
    for token in _word_tokens(sentence, include_punct=include_punct):
        if clause_type_for_node(sentence, token) is None:
            continue
        token_ids = tuple(sorted(tok["id"] for tok in get_subtree_tokens(sentence, token, include_punct=include_punct)))
        if len(token_ids) >= min_size:
            yield Clause(sentence, token_ids)


def iter_units(data: Any, unit_type: str, include_punct: bool = True, min_size: int = 1):
    """Yield syntactic units from a sentence, sentence list, corpus, or path."""
    normalized = _normalize_unit_type(unit_type)
    iterator = {
        "string": iter_strings,
        "catena": iter_catenae,
        "component": iter_components,
        "constituent": iter_constituents,
        "phrase": iter_phrases,
        "clause": iter_clauses,
    }[normalized]
    for sent_index, sentence in enumerate(iter_sentences(data)):
        for unit in iterator(sentence, include_punct=include_punct, min_size=min_size):
            yield UNIT_CLASSES[normalized](unit.sentence, unit.token_ids, sent_index=sent_index)


def find_units(data: Any, unit_type: str, include_punct: bool = True, min_size: int = 1) -> list:
    """Return syntactic units as a list."""
    return list(iter_units(data, unit_type, include_punct=include_punct, min_size=min_size))


def count_units(data: Any, unit_type: str, include_punct: bool = True, min_size: int = 1) -> int:
    """Count syntactic units in a sentence, sentence list, corpus, or path."""
    return sum(1 for _ in iter_units(data, unit_type, include_punct=include_punct, min_size=min_size))


def count_strings(data: Any, include_punct: bool = True, min_size: int = 1) -> int:
    """Count strings."""
    return count_units(data, "string", include_punct=include_punct, min_size=min_size)


def count_catenae(data: Any, include_punct: bool = True, min_size: int = 1) -> int:
    """Count catenae."""
    return count_units(data, "catena", include_punct=include_punct, min_size=min_size)


def count_components(data: Any, include_punct: bool = True, min_size: int = 1) -> int:
    """Count components."""
    return count_units(data, "component", include_punct=include_punct, min_size=min_size)


def count_constituents(data: Any, include_punct: bool = True, min_size: int = 1) -> int:
    """Count constituents."""
    return count_units(data, "constituent", include_punct=include_punct, min_size=min_size)


def count_phrases(data: Any, include_punct: bool = True, min_size: int = 2) -> int:
    """Count phrases."""
    return count_units(data, "phrase", include_punct=include_punct, min_size=min_size)


def count_clauses(data: Any, include_punct: bool = True, min_size: int = 1) -> int:
    """Count clauses."""
    return count_units(data, "clause", include_punct=include_punct, min_size=min_size)


def unit_ratios(sentence: TokenList, include_punct: bool = True) -> dict[str, Any]:
    """Return Osborne-style ratios for catenae, components, and constituents."""
    n_tokens = len(_word_tokens(sentence, include_punct=include_punct))
    total = (2 ** n_tokens) - 1 if n_tokens else 0
    counts = {
        "string": count_strings(sentence, include_punct=include_punct),
        "catena": count_catenae(sentence, include_punct=include_punct),
        "component": count_components(sentence, include_punct=include_punct),
        "constituent": count_constituents(sentence, include_punct=include_punct),
        "phrase": count_phrases(sentence, include_punct=include_punct),
        "clause": count_clauses(sentence, include_punct=include_punct),
    }
    result = {"n_tokens": n_tokens, "all_combinations": total, **counts}
    for name in ("string", "catena", "component", "constituent", "phrase", "clause"):
        non_count = total - counts[name]
        result[f"non_{name}"] = non_count
        result[f"{name}_ratio"] = f"{counts[name]}:{non_count}"
    return result
