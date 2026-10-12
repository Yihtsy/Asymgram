"""Query syntax for dependency-tree searches."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import re
import shlex
from typing import Any, Callable, Mapping

from .conllu import Token, TokenList
from .query import TokenMatch
from .tree_query import clause_type_for_node, get_subtree_tokens, phrase_type_for_node


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


def _split_terms(expression: str) -> list[str]:
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


def _decode_value(text: str, split_alternatives: bool = True) -> Any:
    text = text.strip()
    if not text:
        raise ValueError("Query values cannot be empty.")
    try:
        parts = shlex.split(text)
    except ValueError as error:
        raise ValueError(f"Invalid query value: {text!r}") from error
    value = parts[0] if len(parts) == 1 else text
    if split_alternatives and "|" in value:
        return tuple(_decode_value(part) for part in value.split("|"))
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


def _value_matches(observed: Any, expected: Any, operator: str) -> bool:
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


def _token_by_id(sentence: TokenList, token_id: Any) -> Token | None:
    return next(
        (token for token in sentence if isinstance(token.get("id"), int) and token.get("id") == token_id),
        None,
    )


def _field_value(token: Token, sentence: TokenList, field_name: str) -> Any:
    field_name = field_name.lower()
    if field_name.startswith("head."):
        head = _token_by_id(sentence, token.get("head"))
        return None if head is None else _field_value(head, sentence, field_name[5:])
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


def _word_tokens(sentence: TokenList, include_punct: bool = False) -> list[Token]:
    return [
        token for token in sentence
        if isinstance(token.get("id"), int) and (include_punct or token.get("upos") != "PUNCT")
    ]


def _children(sentence: TokenList, token: Token) -> list[Token]:
    return [child for child in _word_tokens(sentence, include_punct=True) if child.get("head") == token.get("id")]


def _descendants(sentence: TokenList, token: Token) -> list[Token]:
    found: list[Token] = []
    pending = list(_children(sentence, token))
    seen: set[int] = set()
    while pending:
        child = pending.pop()
        child_id = child.get("id")
        if child_id in seen:
            continue
        seen.add(child_id)
        found.append(child)
        pending.extend(_children(sentence, child))
    return found


def _sentence_has_unit(sentence: TokenList, unit_type: str) -> bool:
    unit_type = unit_type.lower().replace("-", "_").replace(" ", "_")
    aliases = {"catenae": "catena", "constituents": "constituent", "phrases": "phrase", "clauses": "clause"}
    unit_type = aliases.get(unit_type, unit_type)
    tokens = _word_tokens(sentence)
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


def _callable_predicate(function: Callable[..., bool]) -> Callable[[Token, TokenList], bool]:
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


def _comparison_predicate(field_name: str, expected: Any, operator: str = "="):
    if operator not in {"=", "!=", "~="}:
        raise ValueError(f"Unsupported query operator: {operator}")
    return lambda token, sentence: _value_matches(
        _field_value(token, sentence, field_name), expected, operator
    )


def _matches_related_token(nested: SyntaxQuery, candidate: Token, sentence: TokenList) -> bool:
    return nested.matches_sentence_conditions(sentence) and (
        not nested.token_predicates or nested.matches_token(candidate, sentence)
    )


def _compile_mapping(query: Mapping[str, Any]) -> SyntaxQuery:
    token_predicates = []
    sentence_predicates = []
    source_parts = []
    for field_name, expected in query.items():
        normalized = str(field_name).lower()
        if normalized == "unit":
            sentence_predicates.append(lambda sentence, value=str(expected): _sentence_has_unit(sentence, value))
            source_parts.append(f"unit={expected}")
            continue
        if normalized in {"has_child", "has_descendant"}:
            nested = compile_syntax_query(expected)
            relation = _children if normalized == "has_child" else _descendants
            token_predicates.append(
                lambda token, sentence, nested=nested, relation=relation: any(
                    _matches_related_token(nested, candidate, sentence)
                    for candidate in relation(sentence, token)
                )
            )
            source_parts.append(f"{normalized}({nested.source})")
            continue
        token_predicates.append(_comparison_predicate(normalized, expected))
        source_parts.append(f"{field_name}={expected}")
    return SyntaxQuery(" ".join(source_parts), tuple(token_predicates), tuple(sentence_predicates))


def _compile_string(expression: str) -> SyntaxQuery:
    token_predicates = []
    sentence_predicates = []
    for term in _split_terms(expression):
        relation_match = re.fullmatch(r"(has_child|has_descendant)\((.*)\)", term, flags=re.IGNORECASE)
        if relation_match:
            relation_name, nested_text = relation_match.groups()
            nested = compile_syntax_query(nested_text)
            relation = _children if relation_name.lower() == "has_child" else _descendants
            token_predicates.append(
                lambda token, sentence, nested=nested, relation=relation: any(
                    _matches_related_token(nested, candidate, sentence)
                    for candidate in relation(sentence, token)
                )
            )
            continue
        comparison = re.fullmatch(r"([A-Za-z_][\w.]*)\s*(!=|~=|=)\s*(.+)", term)
        if not comparison:
            raise ValueError(f"Invalid syntax-query term: {term!r}")
        field_name, operator, raw_value = comparison.groups()
        expected = _decode_value(raw_value, split_alternatives=operator != "~=")
        if field_name.lower() == "unit":
            if operator != "=":
                raise ValueError("unit only supports the '=' operator.")
            sentence_predicates.append(lambda sentence, value=str(expected): _sentence_has_unit(sentence, value))
        else:
            token_predicates.append(_comparison_predicate(field_name, expected, operator))
    return SyntaxQuery(expression.strip(), tuple(token_predicates), tuple(sentence_predicates))


def compile_syntax_query(query: QueryInput) -> SyntaxQuery:
    """Compile a string, mapping, callable, or existing ``SyntaxQuery``."""
    if isinstance(query, SyntaxQuery):
        return query
    if isinstance(query, str):
        if not query.strip():
            raise ValueError("Syntax query cannot be empty.")
        return _compile_string(query)
    if isinstance(query, Mapping):
        if not query:
            raise ValueError("Syntax query mapping cannot be empty.")
        return _compile_mapping(query)
    if callable(query):
        return SyntaxQuery(getattr(query, "__name__", "<callable>"), (_callable_predicate(query),))
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
        token for token in _word_tokens(sentence, include_punct=include_punct)
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
    tokens = _word_tokens(sentence, include_punct=include_punct)
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
    from .query import iter_sentences

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
