"""Sentence filtering and CoNLL-U subcorpus extraction."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .conllu import SentenceList, SentenceLists, Token, TokenList
from .format_changer import read_conllu
from .query import iter_sentences
from .query_syntax import QueryInput, compile_syntax_query, sentence_matches_syntax


def _copy_sentence(sentence: TokenList) -> TokenList:
    return TokenList(
        [Token(deepcopy(dict(token))) for token in sentence],
        metadata=deepcopy(sentence.metadata),
    )


def _query_source(query: QueryInput) -> str:
    return compile_syntax_query(query).source


def _document_metadata(data: Any, source_path: str | None, query_source: str, annotate: bool) -> dict:
    metadata = deepcopy(getattr(data, "metadata", {}))
    if source_path:
        metadata.setdefault("source_file", source_path)
    if annotate:
        metadata["subcorpus_query"] = query_source
    return metadata


def _matching_sentence_list(
    data: Any,
    query: QueryInput,
    mode: str,
    min_count: int,
    include_punct: bool,
    copy: bool,
    annotate: bool,
    source_path: str | None = None,
) -> SentenceList:
    compiled = compile_syntax_query(query)
    selected = []
    for sentence in iter_sentences(data):
        if not sentence_matches_syntax(
            sentence,
            compiled,
            mode=mode,
            min_count=min_count,
            include_punct=include_punct,
        ):
            continue
        result = _copy_sentence(sentence) if copy else sentence
        if annotate:
            result.metadata["subcorpus_query"] = compiled.source
            if source_path:
                result.metadata.setdefault("source_file", source_path)
        selected.append(result)
    return SentenceList(
        selected,
        metadata=_document_metadata(data, source_path, compiled.source, annotate),
    )


def filter_sentences(
    data: Any,
    query: QueryInput,
    mode: str = "any",
    min_count: int = 1,
    include_punct: bool = False,
    copy: bool = True,
    annotate: bool = True,
    flatten: bool = False,
    keep_empty: bool = False,
) -> SentenceList | SentenceLists:
    """Filter sentences while preserving asymgram container types and metadata.

    ``SentenceLists`` inputs preserve document grouping unless ``flatten=True``.
    Other supported inputs return a new ``SentenceList``.
    """
    source_path = None
    loaded = data
    if isinstance(data, (str, Path)):
        source_path = str(Path(data))
        loaded = read_conllu(source_path)

    if isinstance(loaded, SentenceLists):
        groups = []
        for sentence_list in loaded:
            group = _matching_sentence_list(
                sentence_list,
                query,
                mode,
                min_count,
                include_punct,
                copy,
                annotate,
                source_path=sentence_list.metadata.get("source_file"),
            )
            if group or keep_empty:
                groups.append(group)
        grouped = SentenceLists(groups)
        if not flatten:
            return grouped
        sentences = [sentence for group in grouped for sentence in group]
        metadata = {"subcorpus_query": _query_source(query)} if annotate else {}
        metadata["source_documents"] = len(loaded)
        return SentenceList(sentences, metadata=metadata)

    return _matching_sentence_list(
        loaded,
        query,
        mode,
        min_count,
        include_punct,
        copy,
        annotate,
        source_path=source_path,
    )


def extract_subcorpus(
    data: Any,
    query: QueryInput,
    **kwargs: Any,
) -> SentenceList | SentenceLists:
    """Extract a sentence subcorpus; this is the semantic alias of filter_sentences."""
    return filter_sentences(data, query, **kwargs)


def write_subcorpus(
    data: Any,
    output_path: str | Path,
    query: QueryInput | None = None,
    **kwargs: Any,
) -> SentenceList:
    """Extract, flatten if needed, and write one ``.conllu`` subcorpus."""
    output = Path(output_path)
    if output.suffix.lower() != ".conllu":
        raise ValueError(f"Output file must have a '.conllu' extension: {output}")

    options = dict(kwargs)
    if query is not None:
        options["flatten"] = True
        result = filter_sentences(data, query, **options)
    else:
        result = data
    if isinstance(result, SentenceLists):
        result = result.flatten()
    elif isinstance(result, TokenList):
        result = SentenceList([_copy_sentence(result)])
    elif not isinstance(result, SentenceList):
        result = SentenceList([_copy_sentence(sentence) for sentence in iter_sentences(result)])
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_conllu(output)
    return result
