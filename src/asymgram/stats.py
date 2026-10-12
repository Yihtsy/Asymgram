"""Descriptive statistics and dependency metrics for asymgram treebanks."""

from __future__ import annotations

from collections import Counter
import random
from statistics import mean
from typing import Any

from .conllu import TokenList
from .query import iter_sentences, iter_tokens


def _word_tokens(sentence: TokenList, include_punct: bool = False):
    return [
        tok for tok in sentence
        if isinstance(tok.get("id"), int) and (include_punct or tok.get("upos") != "PUNCT")
    ]


def frequency_table(data: Any, field: str = "form", lowercase: bool = False, include_punct: bool = False) -> Counter:
    """Count token values in one CoNLL-U field."""
    counter = Counter()
    for match in iter_tokens(data, include_punct=include_punct):
        value = match.token.get(field)
        if value is None:
            continue
        if lowercase:
            value = str(value).lower()
        counter[value] += 1
    return counter


def form_frequency(data: Any, lowercase: bool = False, include_punct: bool = False) -> Counter:
    """Count FORM values."""
    return frequency_table(data, "form", lowercase=lowercase, include_punct=include_punct)


def lemma_frequency(data: Any, lowercase: bool = False, include_punct: bool = False) -> Counter:
    """Count LEMMA values."""
    return frequency_table(data, "lemma", lowercase=lowercase, include_punct=include_punct)


def upos_distribution(data: Any, include_punct: bool = True) -> Counter:
    """Count UPOS tags."""
    return frequency_table(data, "upos", include_punct=include_punct)


def deprel_distribution(data: Any, include_punct: bool = True, base: bool = False) -> Counter:
    """Count DEPREL labels."""
    counter = Counter()
    for match in iter_tokens(data, include_punct=include_punct):
        deprel = match.token.get("deprel")
        if deprel is None:
            continue
        if base:
            deprel = str(deprel).split(":", 1)[0]
        counter[deprel] += 1
    return counter


def upos_deprel_table(data: Any, include_punct: bool = True, base_deprel: bool = False) -> Counter:
    """Count UPOS-DEPREL combinations."""
    counter = Counter()
    for match in iter_tokens(data, include_punct=include_punct):
        deprel = match.token.get("deprel")
        if base_deprel and deprel is not None:
            deprel = str(deprel).split(":", 1)[0]
        counter[(match.token.get("upos"), deprel)] += 1
    return counter


def sentence_length(sentence: TokenList, include_punct: bool = False) -> int:
    """Return sentence length in word tokens."""
    return len(_word_tokens(sentence, include_punct=include_punct))


def dependency_distances(sentence: TokenList, include_punct: bool = False, signed: bool = False) -> list[int]:
    """Return dependency distances for all non-root tokens."""
    token_ids = {tok.get("id") for tok in _word_tokens(sentence, include_punct=include_punct)}
    distances = []
    for tok in _word_tokens(sentence, include_punct=include_punct):
        head = tok.get("head")
        token_id = tok.get("id")
        if not isinstance(head, int) or head == 0:
            continue
        if not include_punct and head not in token_ids:
            continue
        distance = token_id - head
        distances.append(distance if signed else abs(distance))
    return distances


def mean_dependency_distance(sentence: TokenList, include_punct: bool = False) -> float:
    """Return mean dependency distance for one sentence."""
    distances = dependency_distances(sentence, include_punct=include_punct)
    return mean(distances) if distances else 0.0


def tree_depth(sentence: TokenList, include_punct: bool = False) -> int:
    """Return maximum dependency depth in a sentence."""
    tokens = _word_tokens(sentence, include_punct=include_punct)
    id_map = {tok.get("id"): tok for tok in tokens}
    max_depth = 0
    for tok in tokens:
        depth = 0
        current = tok
        seen = set()
        while True:
            head = current.get("head")
            if not isinstance(head, int) or head == 0:
                break
            if head in seen:
                raise ValueError(f"Cycle detected at token id {tok.get('id')}")
            seen.add(head)
            current = id_map.get(head)
            if current is None:
                break
            depth += 1
        max_depth = max(max_depth, depth)
    return max_depth


def sentence_metrics(sentence: TokenList, include_punct: bool = False) -> dict[str, Any]:
    """Return a compact metric row for one sentence."""
    distances = dependency_distances(sentence, include_punct=include_punct)
    return {
        "sent_id": getattr(sentence, "metadata", {}).get("sent_id"),
        "length": sentence_length(sentence, include_punct=include_punct),
        "dependency_edges": len(distances),
        "mean_dependency_distance": mean(distances) if distances else 0.0,
        "max_dependency_distance": max(distances) if distances else 0,
        "tree_depth": tree_depth(sentence, include_punct=include_punct),
    }


def corpus_summary(data: Any, include_punct: bool = False) -> dict[str, Any]:
    """Return corpus-level counts and averaged dependency metrics."""
    sentences = list(iter_sentences(data))
    metric_rows = [sentence_metrics(sent, include_punct=include_punct) for sent in sentences]
    token_count = sum(row["length"] for row in metric_rows)
    edge_count = sum(row["dependency_edges"] for row in metric_rows)
    all_distances = []
    for sent in sentences:
        all_distances.extend(dependency_distances(sent, include_punct=include_punct))
    return {
        "sentences": len(sentences),
        "tokens": token_count,
        "dependency_edges": edge_count,
        "mean_sentence_length": mean([row["length"] for row in metric_rows]) if metric_rows else 0.0,
        "mean_dependency_distance": mean(all_distances) if all_distances else 0.0,
        "max_dependency_distance": max(all_distances) if all_distances else 0,
        "mean_tree_depth": mean([row["tree_depth"] for row in metric_rows]) if metric_rows else 0.0,
    }


def sample_sentences(
    data: Any,
    n: int,
    seed: int | None = None,
    predicate=None,
    replace: bool = False,
) -> list[TokenList]:
    """Sample sentences, optionally after a sentence-level predicate filter."""
    sentences = [sent for sent in iter_sentences(data) if predicate is None or predicate(sent)]
    rng = random.Random(seed)
    if replace:
        return [rng.choice(sentences) for _ in range(n)] if sentences else []
    return rng.sample(sentences, min(n, len(sentences)))
