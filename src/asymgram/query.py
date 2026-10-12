"""Query and annotation helpers for CoNLL-U treebanks."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from pathlib import Path
import re
from typing import Any, Callable, Iterator

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
