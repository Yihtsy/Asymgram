from .conllu import Token, TokenList, SentenceList
from .constants import CHINESE_COMMA
from copy import deepcopy
from collections import deque
from typing import Union
import re

UD_FUNCTION_HEAD_DEPRELS = ("case",)
UD_FUNCTION_HEAD_RESTORE_DEPRELS = ("case", "aux", "aux:pass", "mark", "cop")
UD_FUNCTION_HEAD_BASE_DEPRELS = tuple(rel.split(":", 1)[0] for rel in UD_FUNCTION_HEAD_DEPRELS)
UD_FUNCTION_HEAD_PROMOTED_DEPRELS = {
    "case": "adp",
}
UD_FUNCTION_HEAD_COMPLEMENT_DEPRELS = {
    "case": "pobj",
    "aux": "xcomp",
    "aux:pass": "xcomp",
    "cop": "xcomp",
}
UD_FUNCTION_HEAD_REVERSE_COMPLEMENTS = {
    "case": "pobj",
    "aux": "xcomp",
    "aux:pass": "xcomp",
    "mark": "mark",
    "cop": "xcomp",
}
UD_FUNCTION_HEAD_REVERSE_PROMOTED = {
    "case": "adp",
}
UD_FUNCTION_HEAD_REVERSE_UPOS = {
    "case": {"ADP"},
    "aux": {"AUX"},
    "aux:pass": {"AUX"},
    "mark": {"SCONJ", "CCONJ", "ADV", "PART", "ADP"},
    "cop": {"AUX"},
}
NO_SPACE_LANGUAGE_PREFIXES = ("zh", "ja", "ko", "th")

def _get_root_token_local(tokenlist: TokenList) -> Token:
    """Return the single root token without requiring SentenceList context."""
    roots = [tok for tok in tokenlist if isinstance(tok.get("id"), int) and tok.get("head") == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root token, found {len(roots)}.")
    return roots[0]

def _set_head_object_refs(tokenlist: TokenList) -> None:
    """Set transient Token.head object references from integer HEAD values."""
    id_map = {tok.get("id"): tok for tok in tokenlist if isinstance(tok.get("id"), int)}
    for tok in tokenlist:
        if not isinstance(tok.get("id"), int):
            continue
        head_id = tok.get("head")
        tok.head = None if head_id == 0 else id_map.get(head_id)

CJK_CHAR_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
CASE_COMPLEMENT_KEEP_DEPRELS = {
    "acl",
    "amod",
    "appos",
    "case",
    "clf",
    "compound",
    "det",
    "flat",
    "fixed",
    "nmod",
    "nummod",
}


def _base_deprel(deprel: str | None) -> str:
    return str(deprel).split(":", 1)[0] if deprel else ""


def _copy_tokenlist(tokenlist: TokenList) -> TokenList:
    return TokenList([Token(dict(tok)) for tok in tokenlist], deepcopy(tokenlist.metadata), build_refs=True)


def _copy_sentencelist(sentences: SentenceList) -> SentenceList:
    return SentenceList([_copy_tokenlist(sentence) for sentence in sentences], deepcopy(sentences.metadata))


def _replace_tokenlist_contents(target: TokenList, source: TokenList) -> TokenList:
    target[:] = [Token(dict(tok)) for tok in source]
    target.metadata = deepcopy(source.metadata)
    for tok in target:
        tok.tokenlist = target
    target._build_intertoken_refs()
    return target


def _get_merge_separator(tokens: list[Token], language: str = "auto", separator: str | None = None) -> str:
    if separator is not None:
        return separator

    normalized_language = (language or "auto").lower()
    if normalized_language != "auto":
        return "" if normalized_language.startswith(NO_SPACE_LANGUAGE_PREFIXES) else " "

    forms = [str(tok.get("form", "")) for tok in tokens if tok.get("form") not in (None, "_")]
    if any(re.search(r"[A-Za-z0-9]", form) for form in forms):
        return " "
    if any(CJK_CHAR_RE.search(form) for form in forms):
        return ""
    return " "


def _token_misc(token: Token) -> dict:
    misc = token.get("misc")
    return misc if isinstance(misc, dict) else {}


def _encoded_spacing_has_whitespace(value) -> bool:
    """Return whether an encoded SpacesBefore/SpacesAfter value contains whitespace."""
    if value in (None, "", "_"):
        return False
    return bool(str(value))


def _boundary_separator(left: Token, right: Token, fallback: str, force: bool = False) -> str:
    """Choose the separator for one original token boundary."""
    if force:
        return fallback

    left_misc = _token_misc(left)
    right_misc = _token_misc(right)
    if "SpacesAfter" in left_misc:
        return " " if _encoded_spacing_has_whitespace(left_misc["SpacesAfter"]) else ""
    if "SpacesBefore" in right_misc:
        return " " if _encoded_spacing_has_whitespace(right_misc["SpacesBefore"]) else ""
    if left_misc.get("SpaceAfter") == "No":
        return ""
    if right_misc.get("SpaceBefore") == "No":
        return ""
    return fallback


def _join_token_field(
    tokens: list[Token],
    field: str,
    fallback_separator: str,
    force_separator: bool = False,
) -> str | None:
    """Join one token field while respecting recorded inter-token spacing."""
    values = [token.get(field) for token in tokens]
    present = [value not in (None, "_") for value in values]
    if not any(present):
        return None

    result = ""
    previous_index = None
    for index, value in enumerate(values):
        if value in (None, "_"):
            continue
        if previous_index is not None:
            separators = [
                _boundary_separator(
                    tokens[boundary],
                    tokens[boundary + 1],
                    fallback_separator,
                    force=force_separator,
                )
                for boundary in range(previous_index, index)
            ]
            result += " " if any(separators) else ""
        result += str(value)
        previous_index = index
    return result


def _merge_token_misc(
    tokens: list[Token],
    root_misc,
    fallback_separator: str,
    force_separator: bool = False,
) -> dict:
    """Merge transliteration and boundary-spacing MISC attributes."""
    misc = _merge_misc_value(root_misc)

    for key in ("Translit", "LTranslit"):
        values = [_token_misc(token).get(key) for token in tokens]
        if any(value not in (None, "_") for value in values):
            if not all(value not in (None, "_") for value in values):
                misc.pop(key, None)
            else:
                merged = str(values[0])
                for index, value in enumerate(values[1:], start=1):
                    merged += _boundary_separator(
                        tokens[index - 1],
                        tokens[index],
                        fallback_separator,
                        force=force_separator,
                    )
                    merged += str(value)
                misc[key] = merged

    first_misc = _token_misc(tokens[0])
    last_misc = _token_misc(tokens[-1])
    for key in ("SpaceBefore", "SpacesBefore"):
        misc.pop(key, None)
        if key in first_misc:
            misc[key] = first_misc[key]
    for key in ("SpaceAfter", "SpacesAfter"):
        misc.pop(key, None)
        if key in last_misc:
            misc[key] = last_misc[key]
    return misc


def _ids_are_contiguous(token_ids: list[int]) -> bool:
    ordered = sorted(token_ids)
    return ordered == list(range(ordered[0], ordered[-1] + 1))


def _merge_misc_value(value):
    if isinstance(value, dict):
        return dict(value)
    return {}


def _encode_orig_forms(forms: list[str]) -> str:
    """Encode a custom comma-separated MISC value without using raw pipes."""
    escaped = []
    for form in forms:
        value = str(form)
        value = value.replace("%", "%25")
        value = value.replace(",", "%2C").replace("|", "%7C")
        value = value.replace("\t", "%09").replace("\r", "%0D").replace("\n", "%0A")
        escaped.append(value)
    return ",".join(escaped)


def _remap_token_id_value(value, old_to_new_id: dict[int, int]):
    if isinstance(value, int):
        return old_to_new_id.get(value, value)
    if isinstance(value, tuple) and len(value) == 3:
        first, sep, last = value
        if sep == "-":
            return (
                old_to_new_id.get(first, first),
                sep,
                old_to_new_id.get(last, last),
            )
        if sep == ".":
            return (
                old_to_new_id.get(first, first) if first != 0 else 0,
                sep,
                last,
            )
    return value


def _remap_deps_value(value, old_to_new_id: dict[int, int], removed_ids: set | None = None):
    if not isinstance(value, list):
        return value
    removed_ids = removed_ids or set()
    remapped = []
    for deprel, dep_head in value:
        if dep_head in removed_ids:
            continue
        remapped.append((deprel, _remap_token_id_value(dep_head, old_to_new_id)))
    return remapped or "_"


def _standard_token_row(values: dict) -> Token:
    row = {
        "id": values.get("id"),
        "form": values.get("form", "_"),
        "lemma": values.get("lemma", "_"),
        "upos": values.get("upos", "_"),
        "xpos": values.get("xpos", "_"),
        "feats": values.get("feats", "_"),
        "head": values.get("head", "_"),
        "deprel": values.get("deprel", "_"),
        "deps": values.get("deps", "_"),
        "misc": values.get("misc", "_"),
    }
    if row["deps"] == "_" and isinstance(row["head"], int) and row["deprel"] not in (None, "_"):
        row["deps"] = [(row["deprel"], row["head"])]
    return Token(row)


def _is_multiword_token_id(token_id) -> bool:
    return isinstance(token_id, tuple) and len(token_id) == 3 and token_id[1] == "-"


def _is_empty_node_id(token_id) -> bool:
    return isinstance(token_id, tuple) and len(token_id) == 3 and token_id[1] == "."


def _remove_nonword_token_lines(
    tokenlist: TokenList,
    id_predicate,
    in_place: bool = False,
    prune_deps: bool = False,
) -> TokenList:
    removed_ids = {
        token.get("id")
        for token in tokenlist
        if id_predicate(token.get("id"))
    }
    if not removed_ids:
        return tokenlist if in_place else _copy_tokenlist(tokenlist)

    new_tokens = []
    for token in tokenlist:
        token_id = token.get("id")
        if token_id in removed_ids:
            continue
        copied = Token(dict(deepcopy(token)))
        if prune_deps:
            copied["deps"] = _remap_deps_value(copied.get("deps"), {}, removed_ids=removed_ids)
        new_tokens.append(copied)

    result = TokenList(new_tokens, deepcopy(tokenlist.metadata), build_refs=True)
    if in_place:
        return _replace_tokenlist_contents(tokenlist, result)
    return result


def _apply_nonword_line_removal(data, id_predicate, in_place: bool = False, prune_deps: bool = False):
    if isinstance(data, TokenList):
        return _remove_nonword_token_lines(data, id_predicate, in_place=in_place, prune_deps=prune_deps)

    if isinstance(data, SentenceList):
        working_sentences = data if in_place else _copy_sentencelist(data)
        for index, sentence in enumerate(working_sentences):
            working_sentences[index] = _remove_nonword_token_lines(
                sentence,
                id_predicate,
                in_place=in_place,
                prune_deps=prune_deps,
            )
        if in_place:
            return data
        return SentenceList(list(working_sentences), deepcopy(working_sentences.metadata))

    working_items = data if in_place else [_copy_tokenlist(sentence) for sentence in data]
    return [
        _remove_nonword_token_lines(
            sentence,
            id_predicate,
            in_place=in_place,
            prune_deps=prune_deps,
        )
        for sentence in working_items
    ]


def _merge_token_group(
    tokenlist: TokenList,
    token_ids: set[int],
    root_id: int,
    separator: str,
    record_orig_forms: bool = True,
    force_separator: bool = False,
) -> TokenList:
    id_to_token = {tok["id"]: tok for tok in tokenlist if isinstance(tok.get("id"), int)}
    root_token = Token(dict(deepcopy(id_to_token[root_id])))
    ordered_tokens = [id_to_token[token_id] for token_id in sorted(token_ids)]

    form_parts = [str(tok.get("form")) for tok in ordered_tokens if tok.get("form") not in (None, "_")]
    root_token["form"] = _join_token_field(
        ordered_tokens, "form", separator, force_separator=force_separator
    ) or "_"
    root_token["lemma"] = _join_token_field(
        ordered_tokens, "lemma", separator, force_separator=force_separator
    ) or "_"
    misc = _merge_token_misc(
        ordered_tokens,
        root_token.get("misc"),
        separator,
        force_separator=force_separator,
    )
    if record_orig_forms:
        misc["OrigForms"] = _encode_orig_forms(form_parts)
    root_token["misc"] = misc or "_"

    new_tokens = []
    for token in tokenlist:
        token_id = token.get("id")
        if isinstance(token_id, int) and token_id in token_ids:
            if token_id == root_id:
                new_tokens.append(root_token)
            continue
        new_tokens.append(Token(dict(deepcopy(token))))

    old_to_new_id = {}
    next_id = 1
    for token in new_tokens:
        old_id = token.get("id")
        if isinstance(old_id, int):
            old_to_new_id[old_id] = next_id
            token["id"] = next_id
            next_id += 1

    root_new_id = old_to_new_id[root_id]
    for token_id in token_ids:
        old_to_new_id[token_id] = root_new_id

    for token in new_tokens:
        token_id = token.get("id")
        if not isinstance(token_id, int):
            token["id"] = _remap_token_id_value(token_id, old_to_new_id)
        head = token.get("head")
        if isinstance(head, int):
            token["head"] = old_to_new_id.get(head, head)
        token["deps"] = _remap_deps_value(token.get("deps"), old_to_new_id)

    return TokenList(new_tokens, deepcopy(tokenlist.metadata), build_refs=True)


def _collect_subtree_ids(tokenlist: TokenList, root_id: int) -> set[int]:
    id_to_token = {tok["id"]: tok for tok in tokenlist if isinstance(tok.get("id"), int)}
    if root_id not in id_to_token:
        raise ValueError(f"Token ID {root_id} not found.")

    subtree_ids = set()
    queue = deque([root_id])
    while queue:
        current_id = queue.popleft()
        if current_id in subtree_ids:
            continue
        subtree_ids.add(current_id)
        for token in id_to_token.values():
            if token.get("head") == current_id:
                queue.append(token["id"])
    return subtree_ids


def _remove_token_ids(
    tokenlist: TokenList,
    token_ids: set[int],
    in_place: bool = False,
    require_no_external_dependents: bool = True,
) -> TokenList:
    """
    Remove tokens and renumber IDs/HEADs without leaving dangling structures.
    """
    if not token_ids:
        return tokenlist if in_place else _copy_tokenlist(tokenlist)

    id_to_token = {tok["id"]: tok for tok in tokenlist if isinstance(tok.get("id"), int)}
    missing_ids = sorted(token_ids - set(id_to_token))
    if missing_ids:
        raise ValueError(f"Token IDs not found: {missing_ids}")

    kept_tokens = [
        tok for tok in tokenlist
        if isinstance(tok.get("id"), int) and tok["id"] not in token_ids
    ]

    if kept_tokens and any(id_to_token[token_id].get("head") == 0 for token_id in token_ids):
        raise ValueError("Cannot remove a root token while keeping other tokens.")

    if require_no_external_dependents:
        affected = [
            tok["id"] for tok in kept_tokens
            if tok.get("head") in token_ids
        ]
        if affected:
            raise ValueError(
                f"Cannot remove tokens {sorted(token_ids)} because kept tokens depend on them: {affected}"
            )

    new_tokens = []
    for token in tokenlist:
        token_id = token.get("id")
        if isinstance(token_id, int) and token_id in token_ids:
            continue
        new_tokens.append(Token(dict(deepcopy(token))))

    old_to_new_id = {}
    next_id = 1
    for token in new_tokens:
        old_id = token.get("id")
        if isinstance(old_id, int):
            old_to_new_id[old_id] = next_id
            token["id"] = next_id
            next_id += 1

    for token in new_tokens:
        token_id = token.get("id")
        if not isinstance(token_id, int):
            token["id"] = _remap_token_id_value(token_id, old_to_new_id)
        head = token.get("head")
        if isinstance(head, int) and head != 0:
            token["head"] = old_to_new_id.get(head, head)
        token["deps"] = _remap_deps_value(token.get("deps"), old_to_new_id, removed_ids=token_ids)

    result = TokenList(new_tokens, deepcopy(tokenlist.metadata), build_refs=True)
    if in_place:
        return _replace_tokenlist_contents(tokenlist, result)
    return result


def _apply_sentence_removal(data, remove_ids_getter, in_place: bool = False):
    if isinstance(data, TokenList):
        token_ids = remove_ids_getter(data)
        return _remove_token_ids(data, token_ids, in_place=in_place)

    if isinstance(data, SentenceList):
        working_sentences = data if in_place else _copy_sentencelist(data)
        for index, sentence in enumerate(working_sentences):
            token_ids = remove_ids_getter(sentence)
            working_sentences[index] = _remove_token_ids(sentence, token_ids, in_place=in_place)
        if in_place:
            return data
        return SentenceList(list(working_sentences), deepcopy(working_sentences.metadata))

    working_items = data if in_place else [_copy_tokenlist(sentence) for sentence in data]
    return [
        _remove_token_ids(sentence, remove_ids_getter(sentence), in_place=in_place)
        for sentence in working_items
    ]


def remove_multiword_tokens(data, in_place: bool = False):
    """
    Remove CoNLL-U multiword token range lines such as ID 2-3.

    Multiword token lines are surface-range rows, not syntactic word nodes, so
    integer word IDs, HEAD, and DEPS are left unchanged.
    """
    return _apply_nonword_line_removal(data, _is_multiword_token_id, in_place=in_place)


def remove_empty_nodes(data, in_place: bool = False):
    """
    Remove CoNLL-U empty node lines such as ID 7.1.

    Empty nodes are enhanced-dependency nodes. When they are removed, enhanced
    DEPS edges pointing to the removed empty node IDs are removed as well.
    """
    return _apply_nonword_line_removal(data, _is_empty_node_id, in_place=in_place, prune_deps=True)


def convert_ud_function_dependents(
    data,
    target_deprels: tuple[str, ...] = UD_FUNCTION_HEAD_DEPRELS,
    in_place: bool = False,
):
    """
    Convert selected UD function-word dependents into local syntactic heads.

    By default, the conversion only targets UD `case` dependents. This turns a
    UD adpositional structure such as `V -> O -> P(case)` into a
    function-headed structure `V -> P(adp) -> O(pobj)`. Additional relations can
    be passed explicitly through `target_deprels`, but they are not converted by
    default.

    For `case`, nominal-internal dependents are kept under the nominal
    complement, the promoted adposition receives `deprel=adp`, and the nominal
    complement receives `deprel=pobj`.

    Parameters:
        data: A TokenList, SentenceList, or iterable of TokenList objects.
        target_deprels: Dependency relations to reverse. Subtypes are matched by
            exact label or by base relation.
        in_place: If True, mutate the input object. If False, return a converted
            copy.

    Returns:
        The converted object, matching the input shape.
    """
    target_set = set(target_deprels)
    target_bases = {_base_deprel(rel) for rel in target_deprels if ":" not in rel}

    def is_target_relation(deprel: str | None) -> bool:
        if deprel == "aux:pass" and "aux:pass" not in target_set:
            return False
        return deprel in target_set or _base_deprel(deprel) in target_bases

    def convert_sentence(sentence: TokenList) -> TokenList:
        working = sentence if in_place else _copy_tokenlist(sentence)
        id2token = {tok["id"]: tok for tok in working if isinstance(tok.get("id"), int)}
        if not id2token:
            return working

        original_children = {token_id: [] for token_id in id2token}
        for tok in id2token.values():
            head = tok.get("head")
            if head in original_children:
                original_children[head].append(tok)

        processed_heads = set()
        for lexical_id in sorted(id2token):
            if lexical_id in processed_heads:
                continue
            lexical = id2token[lexical_id]
            function_children = [
                child for child in original_children.get(lexical_id, [])
                if is_target_relation(child.get("deprel"))
            ]
            if not function_children:
                continue

            processed_heads.add(lexical_id)
            function_children.sort(key=lambda tok: tok["id"])
            top_function = function_children[0]
            last_function = function_children[-1]
            old_head = lexical.get("head")
            old_deprel = lexical.get("deprel")
            top_original_deprel = top_function.get("deprel")
            top_original_base = _base_deprel(top_original_deprel)
            last_original_deprel = last_function.get("deprel")
            promoted_deprel = UD_FUNCTION_HEAD_PROMOTED_DEPRELS.get(
                top_original_deprel,
                UD_FUNCTION_HEAD_PROMOTED_DEPRELS.get(top_original_base, old_deprel)
            )
            complement_deprel = UD_FUNCTION_HEAD_COMPLEMENT_DEPRELS.get(
                top_original_deprel,
                UD_FUNCTION_HEAD_COMPLEMENT_DEPRELS.get(top_original_base, last_original_deprel)
            )
            if old_head == 0:
                promoted_deprel = "root"

            # The promoted function word inherits the lexical head's external role.
            top_function["head"] = old_head
            top_function["deprel"] = promoted_deprel

            # Multiple function words form a local chain in surface order.
            for previous, current in zip(function_children, function_children[1:]):
                current["head"] = previous["id"]

            # The original lexical head becomes the complement of the innermost function word.
            lexical["head"] = last_function["id"]
            lexical["deprel"] = complement_deprel

            original_function_ids = {child["id"] for child in function_children}
            for child in original_children.get(lexical_id, []):
                if child["id"] in original_function_ids:
                    continue
                if top_original_base == "case" and _base_deprel(child.get("deprel")) in CASE_COMPLEMENT_KEEP_DEPRELS:
                    continue
                child["head"] = top_function["id"]

        return TokenList([Token(dict(tok)) for tok in working], deepcopy(working.metadata), build_refs=True)

    if isinstance(data, TokenList):
        return convert_sentence(data)

    if isinstance(data, SentenceList):
        working_sentences = data if in_place else _copy_sentencelist(data)
        for index, sentence in enumerate(working_sentences):
            working_sentences[index] = convert_sentence(sentence)
        return SentenceList(list(working_sentences), deepcopy(working_sentences.metadata))

    working_items = data if in_place else [_copy_tokenlist(sentence) for sentence in data]
    return [convert_sentence(sentence) for sentence in working_items]


def restore_ud_function_dependents(
    data,
    target_deprels: tuple[str, ...] = UD_FUNCTION_HEAD_RESTORE_DEPRELS,
    in_place: bool = False,
):
    """
    Restore local function-headed structures back to UD-style dependencies.

    For example, `V -> P(adp) -> N(pobj)` can be restored to
    `V -> N -> P(case)`. For `aux`, `aux:pass`, and `cop`, an `xcomp`
    complement is restored as the lexical head and the function word becomes
    the selected UD dependent. For `mark`, the complement relation currently
    follows the forward conversion convention and is restored from `mark`.
    """
    target_set = set(target_deprels)
    target_bases = {_base_deprel(rel) for rel in target_deprels if ":" not in rel}

    def target_requested(target: str) -> bool:
        if target == "aux:pass" and "aux:pass" not in target_set:
            return False
        return target in target_set or _base_deprel(target) in target_bases

    def infer_xcomp_function_relation(function: Token, lexical: Token, children: dict[int, list[Token]]) -> str:
        lexical_children = children.get(lexical["id"], [])
        function_children = children.get(function["id"], [])
        if any(child.get("deprel") == "nsubj:pass" for child in lexical_children + function_children):
            return "aux:pass"
        lexical_upos = lexical.get("upos")
        function_lemma = str(function.get("lemma", "")).lower()
        if lexical_upos in {"ADJ", "NOUN", "PROPN", "PRON", "NUM"}:
            return "cop"
        if function_lemma == "be" and lexical_upos not in {"VERB"}:
            return "cop"
        return "aux"

    def find_candidates(sentence: TokenList, id2token: dict[int, Token], children: dict[int, list[Token]]):
        candidates = []
        for function_id in sorted(id2token):
            function = id2token[function_id]
            function_deprel = function.get("deprel")
            for target in target_deprels:
                complement_deprel = UD_FUNCTION_HEAD_REVERSE_COMPLEMENTS.get(target)
                if complement_deprel is None:
                    continue
                allowed_upos = UD_FUNCTION_HEAD_REVERSE_UPOS.get(target)
                if allowed_upos and function.get("upos") not in allowed_upos:
                    continue
                promoted_deprel = UD_FUNCTION_HEAD_REVERSE_PROMOTED.get(target)
                if promoted_deprel and function_deprel not in {promoted_deprel, "root"}:
                    continue
                complements = [
                    child for child in children.get(function_id, [])
                    if child.get("deprel") == complement_deprel
                ]
                if complements:
                    complements.sort(key=lambda tok: tok["id"])
                    lexical = complements[0]
                    restored_target = target
                    if complement_deprel == "xcomp":
                        restored_target = infer_xcomp_function_relation(function, lexical, children)
                        if not target_requested(restored_target):
                            continue
                    elif not target_requested(target):
                        continue
                    candidates.append((function_id, restored_target, lexical))
                    break
        return candidates

    def restore_sentence(sentence: TokenList) -> TokenList:
        working = sentence if in_place else _copy_tokenlist(sentence)

        while True:
            id2token = {tok["id"]: tok for tok in working if isinstance(tok.get("id"), int)}
            children = {token_id: [] for token_id in id2token}
            for tok in id2token.values():
                head = tok.get("head")
                if head in children:
                    children[head].append(tok)

            candidates = find_candidates(working, id2token, children)
            if not candidates:
                break

            function_id, restored_deprel, lexical = candidates[0]
            function = id2token[function_id]
            old_head = function.get("head")
            old_deprel = function.get("deprel")

            lexical["head"] = old_head
            lexical["deprel"] = "root" if old_head == 0 else old_deprel
            function["head"] = lexical["id"]
            function["deprel"] = restored_deprel

            for child in children.get(function_id, []):
                if child["id"] in {lexical["id"], function["id"]}:
                    continue
                child["head"] = lexical["id"]

        result = TokenList([Token(dict(tok)) for tok in working], deepcopy(working.metadata), build_refs=True)
        if in_place:
            return _replace_tokenlist_contents(sentence, result)
        return result

    if isinstance(data, TokenList):
        return restore_sentence(data)

    if isinstance(data, SentenceList):
        working_sentences = data if in_place else _copy_sentencelist(data)
        for index, sentence in enumerate(working_sentences):
            working_sentences[index] = restore_sentence(sentence)
        return SentenceList(list(working_sentences), deepcopy(working_sentences.metadata))

    working_items = data if in_place else [_copy_tokenlist(sentence) for sentence in data]
    return [restore_sentence(sentence) for sentence in working_items]

def is_subtree(tokenlist: TokenList, start_id: int, end_id: int) -> bool:
    """
    Check if tokens from start_id to end_id (inclusive) form a connected dependency subtree.

    Parameters:
        tokenlist: TokenList parsed from CoNLL-U
        start_id: Start token ID (1-based, inclusive)
        end_id: End token ID (1-based, inclusive)

    Returns:
        bool: True if the specified range forms a subtree; False otherwise.
    """
    # Collect all token ids in the range
    id_set = set(range(start_id, end_id + 1))
    id_to_token = {t["id"]: t for t in tokenlist if isinstance(t["id"], int)}

    # Ensure all specified ids are present in tokenlist
    if not id_set.issubset(id_to_token):
        return False

    local_roots = []
    children = {tid: [] for tid in id_set}

    for tid in id_set:
        head = id_to_token[tid]["head"]
        if head not in id_set:
            local_roots.append(tid)
        else:
            children[head].append(tid)

    if len(local_roots) != 1:
        return False

    # Traverse subtree from the root
    visited = set()
    queue = deque([local_roots[0]])
    while queue:
        current = queue.popleft()
        visited.add(current)
        for child in children.get(current, []):
            if child not in visited:
                queue.append(child)

    return visited == id_set

def is_marginal_subtree(token: "Token", exclude_punct: bool = True) -> bool:
    """
    Determine whether the subtree rooted at `token` lies at the left or right edge of the sentence.

    Parameters:
        token (Token): The root of the subtree to be tested.
        exclude_punct (bool): Whether to ignore punctuation tokens at the sentence edges.

    Returns:
        bool: True if the subtree lies at the edge (left or right), possibly after ignoring edge punctuation.
    """
    if not hasattr(token, "tokenlist") or token.tokenlist is None:
        raise ValueError("Token must belong to a TokenList.")

    tokenlist = token.tokenlist

    # Get subtree span
    span_start, span_end = token.get_subtree_span(
        exclude_left_punct=exclude_punct,
        exclude_right_punct=exclude_punct
    )

    # Get all token IDs in the sentence (exclude comments/multiword tokens)
    sentence_ids = [t["id"] for t in tokenlist if isinstance(t["id"], int)]

    if not sentence_ids:
        return False  # Empty sentence?

    sentence_start = sentence_ids[0]
    sentence_end = sentence_ids[-1]

    # Optional: skip edge punctuation tokens in the sentence itself
    if exclude_punct:
        tokens = [t for t in tokenlist if isinstance(t["id"], int)]
        while tokens and tokens[0].get("upos") == "PUNCT":
            tokens = tokens[1:]
        while tokens and tokens[-1].get("upos") == "PUNCT":
            tokens = tokens[:-1]
        if not tokens:
            return False  # sentence is only punctuation
        sentence_start = tokens[0]["id"]
        sentence_end = tokens[-1]["id"]

    return span_start == sentence_start or span_end == sentence_end

def get_local_root(tokenlist: TokenList, start_id: int, end_id: int) -> int:
    """
    Return the local root ID of a subtree defined by token IDs from start_id to end_id.
    Raises an error if the specified span does not form a subtree.

    Parameters:
        tokenlist: TokenList parsed from CoNLL-U
        start_id: Start token ID (1-based, inclusive)
        end_id: End token ID (1-based, inclusive)

    Returns:
        int: ID of the local root

    Raises:
        ValueError: If the range does not form a valid subtree
    """
    if not is_subtree(tokenlist, start_id, end_id):
        raise ValueError(f"The token span [{start_id}, {end_id}] does not form a valid subtree.")

    id_set = set(range(start_id, end_id + 1))
    id_to_token = {t["id"]: t for t in tokenlist if isinstance(t["id"], int)}

    # Find the local root
    for tid in id_set:
        head = id_to_token[tid]["head"]
        if head not in id_set:
            return tid

    # In theory should never reach here
    raise RuntimeError("Unexpected error: subtree verified but root not found.")

def merge_tokens(
    tokenlist: TokenList,
    start_id: int,
    end_id: int,
    record_orig_forms: bool = False,
    language: str = "auto",
    separator: str | None = None,
) -> TokenList:
    """
    Merge tokens with IDs in [start_id, end_id] into one node (local root), adjusting IDs, HEAD, and DEPS.

    Parameters:
        tokenlist (TokenList): Input sentence.
        start_id (int): Start ID (inclusive, 1-based).
        end_id (int): End ID (inclusive, 1-based).
        record_orig_forms (bool): If True, store merged source forms in MISC as OrigForms=form1,form2.
        language (str): Language hint used when source spacing is not explicit.
        separator (str | None): Forced separator for FORM, LEMMA, Translit, and LTranslit.
            If None, SpaceAfter/SpacesAfter/SpacesBefore are respected before
            falling back to the language/script heuristic.

    Returns:
        TokenList: New token list with subtree merged and IDs, HEAD, and DEPS adjusted.
    """
    if not is_subtree(tokenlist, start_id, end_id):
        raise ValueError(f"Span [{start_id}, {end_id}] is not a valid subtree.")

    id_set = set(range(start_id, end_id + 1))
    id_to_token = {t["id"]: t for t in tokenlist if isinstance(t["id"], int)}

    # Find local root
    root_id = next(tid for tid in id_set if id_to_token[tid]["head"] not in id_set)
    root_token = deepcopy(id_to_token[root_id])

    # Merge surface, lemma, transliteration, and outer spacing metadata.
    sorted_tokens = [id_to_token[i] for i in sorted(id_set)]
    fallback_separator = _get_merge_separator(
        sorted_tokens, language=language, separator=separator
    )
    force_separator = separator is not None
    root_token["form"] = _join_token_field(
        sorted_tokens,
        "form",
        fallback_separator,
        force_separator=force_separator,
    ) or "_"
    root_token["lemma"] = _join_token_field(
        sorted_tokens,
        "lemma",
        fallback_separator,
        force_separator=force_separator,
    ) or "_"
    misc = _merge_token_misc(
        sorted_tokens,
        root_token.get("misc"),
        fallback_separator,
        force_separator=force_separator,
    )
    if record_orig_forms:
        misc["OrigForms"] = _encode_orig_forms(
            [str(t.get("form")) for t in sorted_tokens if t.get("form") not in (None, "_")]
        )
    root_token["misc"] = misc or "_"

    # Build new token list
    new_tokens = []
    for token in tokenlist:
        tid = token.get("id")
        if isinstance(tid, int):
            if tid == root_id:
                new_tokens.append(root_token)
            elif tid in id_set:
                continue  # skip other nodes in subtree
            else:
                t = deepcopy(token)
                # Redirect heads pointing into the subtree to root.
                if t["head"] in id_set:
                    t["head"] = root_id
                new_tokens.append(t)
        else:
            new_tokens.append(deepcopy(token))  # comments etc.

    # Reassign new sequential IDs and update HEAD/DEPS references.
    old_to_new_id = {}
    next_id = 1
    for token in new_tokens:
        old_id = token.get("id")
        if isinstance(old_id, int):
            old_to_new_id[old_id] = next_id
            token["id"] = next_id
            next_id += 1

    for token in new_tokens:
        token_id = token.get("id")
        if not isinstance(token_id, int):
            token["id"] = _remap_token_id_value(token_id, old_to_new_id)
        head = token.get("head")
        if isinstance(head, int):
            token["head"] = old_to_new_id.get(head, head)
        token["deps"] = _remap_deps_value(token.get("deps"), old_to_new_id)

    return TokenList(new_tokens, metadata=deepcopy(tokenlist.metadata), build_refs=True)


def _merge_dependents_with_heads(
    data,
    deprel: str,
    language: str = "auto",
    separator: str | None = None,
    in_place: bool = False,
    raise_on_non_contiguous: bool = False,
    relation_name: str | None = None,
    record_orig_forms: bool = True,
):
    """
    Merge dependents with their heads for one dependency relation.

    A head is merged only when all of its direct dependents with the selected
    dependency relation and the head itself form one continuous surface span.
    If several matching dependents attach to the same head and the whole group
    is continuous, they are merged together in one step. The merged token keeps
    all columns from the original head except FORM and LEMMA, which are rebuilt
    from the merged span.

    If a removed dependent token has its own dependents, those dependents are
    reattached to the merged head so that the dependency tree stays connected.

    Parameters:
        data: A TokenList, SentenceList, or iterable of TokenList objects.
        deprel: The dependency relation to merge. Subtypes are matched by base
            relation, so "compound:nn" matches "compound".
        language: Language hint used when token boundaries do not explicitly
            record spacing. Codes starting with zh/ja/ko/th use no spaces;
            other languages use spaces. "auto" uses a script heuristic.
        separator: Explicit separator for FORM, LEMMA, Translit, and LTranslit.
            If omitted, SpaceAfter/SpacesAfter/SpacesBefore are respected.
        in_place: If True, mutate the input object. If False, return a converted
            copy.
        raise_on_non_contiguous: If True, raise an error when a head has
            matching dependents that are not continuous with it. If False, skip
            such cases.
        relation_name: Optional name for error messages.
        record_orig_forms: If True, store the merged source forms in MISC as
            OrigForms=form1,form2.

    Returns:
        The transformed object, matching the input shape.
    """
    target_base = _base_deprel(deprel)
    label = relation_name or target_base

    def is_target_dependent(token: Token) -> bool:
        return _base_deprel(token.get("deprel")) == target_base

    def merge_sentence(sentence: TokenList) -> TokenList:
        working = sentence if in_place else _copy_tokenlist(sentence)

        while True:
            id_to_token = {tok["id"]: tok for tok in working if isinstance(tok.get("id"), int)}
            merge_candidates = []
            skipped = []

            for head_id in sorted(id_to_token):
                matching_children = [
                    tok for tok in id_to_token.values()
                    if tok.get("head") == head_id and is_target_dependent(tok)
                ]
                if not matching_children:
                    continue

                group_ids = {head_id, *(tok["id"] for tok in matching_children)}
                if not _ids_are_contiguous(list(group_ids)):
                    skipped.append((head_id, sorted(group_ids)))
                    continue

                merge_candidates.append((min(group_ids), head_id, group_ids))

            if skipped and raise_on_non_contiguous:
                head_id, group_ids = skipped[0]
                raise ValueError(
                    f"{label} group headed by token {head_id} is not continuous: {group_ids}"
                )

            if not merge_candidates:
                break

            _, head_id, group_ids = sorted(merge_candidates)[0]
            ordered_tokens = [id_to_token[token_id] for token_id in sorted(group_ids)]
            joiner = _get_merge_separator(ordered_tokens, language=language, separator=separator)
            working = _merge_token_group(
                working,
                group_ids,
                head_id,
                joiner,
                record_orig_forms=record_orig_forms,
                force_separator=separator is not None,
            )

        if in_place:
            return _replace_tokenlist_contents(sentence, working)
        return working

    if isinstance(data, TokenList):
        return merge_sentence(data)

    if isinstance(data, SentenceList):
        working_sentences = data if in_place else _copy_sentencelist(data)
        for index, sentence in enumerate(working_sentences):
            working_sentences[index] = merge_sentence(sentence)
        return SentenceList(list(working_sentences), deepcopy(working_sentences.metadata))

    working_items = data if in_place else [_copy_tokenlist(sentence) for sentence in data]
    return [merge_sentence(sentence) for sentence in working_items]


def merge_compounds_with_heads(
    data,
    deprel: str = "compound",
    language: str = "auto",
    separator: str | None = None,
    in_place: bool = False,
    raise_on_non_contiguous: bool = False,
    record_orig_forms: bool = True,
):
    """
    Merge compound dependents with their lexical heads.

    Direct and nested compound groups are merged when each group is continuous.
    The merged token keeps all columns from the original head except FORM and
    LEMMA, which are rebuilt from the merged span.
    """
    return _merge_dependents_with_heads(
        data,
        deprel=deprel,
        language=language,
        separator=separator,
        in_place=in_place,
        raise_on_non_contiguous=raise_on_non_contiguous,
        relation_name="compound",
        record_orig_forms=record_orig_forms,
    )


def merge_fixed_expressions(
    data,
    deprel: str = "fixed",
    language: str = "auto",
    separator: str | None = None,
    in_place: bool = False,
    raise_on_non_contiguous: bool = False,
    record_orig_forms: bool = True,
):
    """
    Merge fixed multiword expressions with their fixed head.

    This targets UD `fixed` groups such as "in terms of", where the following
    function words are attached to the first word with `fixed`. The merged
    token keeps all columns from the original fixed head except FORM and LEMMA.
    """
    return _merge_dependents_with_heads(
        data,
        deprel=deprel,
        language=language,
        separator=separator,
        in_place=in_place,
        raise_on_non_contiguous=raise_on_non_contiguous,
        relation_name="fixed",
        record_orig_forms=record_orig_forms,
    )


def merge_flat_expressions(
    data,
    deprel: str = "flat",
    language: str = "auto",
    separator: str | None = None,
    in_place: bool = False,
    raise_on_non_contiguous: bool = False,
    record_orig_forms: bool = True,
):
    """
    Merge flat multiword expressions with their head.

    This targets UD `flat` groups, often used for names and other flat
    multiword expressions. The merged token keeps all columns from the original
    head except FORM, LEMMA, and optional OrigForms in MISC.
    """
    return _merge_dependents_with_heads(
        data,
        deprel=deprel,
        language=language,
        separator=separator,
        in_place=in_place,
        raise_on_non_contiguous=raise_on_non_contiguous,
        relation_name="flat",
        record_orig_forms=record_orig_forms,
    )


def merge_goeswith_tokens(
    data,
    deprel: str = "goeswith",
    language: str = "auto",
    separator: str | None = "",
    in_place: bool = False,
    raise_on_non_contiguous: bool = False,
    record_orig_forms: bool = True,
):
    """
    Merge tokens linked by UD `goeswith`.

    `goeswith` usually represents parts of a word that were split incorrectly,
    so the default separator is the empty string.
    """
    return _merge_dependents_with_heads(
        data,
        deprel=deprel,
        language=language,
        separator=separator,
        in_place=in_place,
        raise_on_non_contiguous=raise_on_non_contiguous,
        relation_name="goeswith",
        record_orig_forms=record_orig_forms,
    )


def _apply_sentence_transform(data, transform_sentence, in_place: bool = False):
    if isinstance(data, TokenList):
        return transform_sentence(data)

    if isinstance(data, SentenceList):
        working_sentences = data if in_place else _copy_sentencelist(data)
        for index, sentence in enumerate(working_sentences):
            working_sentences[index] = transform_sentence(sentence)
        if in_place:
            return data
        return SentenceList(list(working_sentences), deepcopy(working_sentences.metadata))

    working_items = data if in_place else [_copy_tokenlist(sentence) for sentence in data]
    return [transform_sentence(sentence) for sentence in working_items]


def insert_token(
    tokenlist: TokenList,
    token_attrs: dict,
    reference_id: int | None = None,
    position: str = "after",
    in_place: bool = False,
) -> TokenList:
    """
    Insert one ordinary word token and renumber integer IDs, HEAD, and DEPS.

    Parameters:
        tokenlist: Sentence to edit.
        token_attrs: CoNLL-U fields for the new token. FORM is required.
        reference_id: Existing integer token ID used as insertion anchor. If
            None, append the token after the last ordinary word.
        position: "before" or "after" the reference token.
        in_place: If True, mutate the input TokenList.
    """
    if position not in {"before", "after"}:
        raise ValueError("position must be 'before' or 'after'.")
    if "form" not in token_attrs:
        raise ValueError("token_attrs must include 'form'.")

    working = tokenlist if in_place else _copy_tokenlist(tokenlist)
    word_ids = [tok["id"] for tok in working if isinstance(tok.get("id"), int)]
    if reference_id is None:
        reference_id = max(word_ids) if word_ids else None
        position = "after"
    elif reference_id not in word_ids:
        raise ValueError(f"Reference token ID {reference_id} not found.")

    temp_id = -10**12
    new_token = _standard_token_row({"id": temp_id, **token_attrs})

    new_tokens = []
    inserted = False
    for token in working:
        token_id = token.get("id")
        if reference_id is not None and token_id == reference_id and position == "before":
            new_tokens.append(new_token)
            inserted = True
        new_tokens.append(Token(dict(deepcopy(token))))
        if reference_id is not None and token_id == reference_id and position == "after":
            new_tokens.append(new_token)
            inserted = True
    if reference_id is None or not inserted:
        new_tokens.append(new_token)

    old_to_new_id = {}
    next_id = 1
    for token in new_tokens:
        old_id = token.get("id")
        if isinstance(old_id, int):
            old_to_new_id[old_id] = next_id
            token["id"] = next_id
            next_id += 1

    for token in new_tokens:
        token_id = token.get("id")
        if not isinstance(token_id, int):
            token["id"] = _remap_token_id_value(token_id, old_to_new_id)
        head = token.get("head")
        if isinstance(head, int):
            token["head"] = _remap_token_id_value(head, old_to_new_id)
        token["deps"] = _remap_deps_value(token.get("deps"), old_to_new_id)

    result = TokenList(new_tokens, deepcopy(working.metadata), build_refs=True)
    if in_place:
        return _replace_tokenlist_contents(tokenlist, result)
    return result


def insert_empty_node(
    tokenlist: TokenList,
    base_id: int,
    token_attrs: dict,
    empty_index: int | None = None,
    in_place: bool = False,
) -> TokenList:
    """
    Insert one CoNLL-U empty node such as 7.1 or 0.1.

    Empty nodes do not change ordinary integer IDs or basic HEAD values. By UD
    convention, i.1 is placed after token i, while 0.1 is placed before token 1.
    Basic HEAD and DEPREL are always "_"; enhanced relations belong in DEPS.
    """
    if not isinstance(base_id, int) or base_id < 0:
        raise ValueError("base_id must be a non-negative integer.")
    if "form" not in token_attrs:
        raise ValueError("token_attrs must include 'form'.")

    working = tokenlist if in_place else _copy_tokenlist(tokenlist)
    word_ids = {tok["id"] for tok in working if isinstance(tok.get("id"), int)}
    if base_id != 0 and base_id not in word_ids:
        raise ValueError(f"Base token ID {base_id} not found.")

    existing_suffixes = [
        token.get("id")[2]
        for token in working
        if _is_empty_node_id(token.get("id")) and token.get("id")[0] == base_id
    ]
    if empty_index is None:
        empty_index = max(existing_suffixes, default=0) + 1
    elif empty_index in existing_suffixes:
        raise ValueError(f"Empty node ID {base_id}.{empty_index} already exists.")
    if empty_index <= 0:
        raise ValueError("empty_index must be positive.")

    values = {
        "id": (base_id, ".", empty_index),
        "head": "_",
        "deprel": "_",
        **token_attrs,
    }
    values["head"] = "_"
    values["deprel"] = "_"
    new_token = _standard_token_row(values)

    insert_at = 0
    if base_id != 0:
        for index, token in enumerate(working):
            token_id = token.get("id")
            if token_id == base_id or (_is_empty_node_id(token_id) and token_id[0] == base_id):
                insert_at = index + 1
    else:
        for index, token in enumerate(working):
            token_id = token.get("id")
            if _is_empty_node_id(token_id) and token_id[0] == 0:
                insert_at = index + 1

    new_tokens = [Token(dict(deepcopy(tok))) for tok in working]
    new_tokens.insert(insert_at, new_token)
    result = TokenList(new_tokens, deepcopy(working.metadata), build_refs=True)
    if in_place:
        return _replace_tokenlist_contents(tokenlist, result)
    return result


def merge_hyphenated_compounds(
    data,
    hyphen_forms: tuple[str, ...] = ("-", "‐", "‑", "‒", "–", "—"),
    left_deprels: tuple[str, ...] = ("compound", "amod", "flat", "fixed"),
    in_place: bool = False,
    record_orig_forms: bool = True,
):
    """
    Merge continuous word-hyphen-word compounds such as long - term.

    The default pattern requires the left word and the hyphen token to depend on
    the right word, and the left word's dependency relation to be one of
    compound/amod/flat/fixed. The three-token span must form a valid subtree.
    """
    allowed_left_bases = {_base_deprel(label) for label in left_deprels}

    def is_hyphen(token: Token) -> bool:
        return token.get("form") in hyphen_forms and token.get("upos") == "PUNCT"

    def transform_sentence(sentence: TokenList) -> TokenList:
        working = sentence if in_place else _copy_tokenlist(sentence)
        while True:
            id_to_token = {tok["id"]: tok for tok in working if isinstance(tok.get("id"), int)}
            candidate = None
            for left_id in sorted(id_to_token):
                hyphen_id = left_id + 1
                right_id = left_id + 2
                if hyphen_id not in id_to_token or right_id not in id_to_token:
                    continue
                left = id_to_token[left_id]
                hyphen = id_to_token[hyphen_id]
                right = id_to_token[right_id]
                if not is_hyphen(hyphen):
                    continue
                if left.get("head") != right_id or hyphen.get("head") != right_id:
                    continue
                if _base_deprel(left.get("deprel")) not in allowed_left_bases:
                    continue
                if not is_subtree(working, left_id, right_id):
                    continue
                candidate = (left_id, right_id)
                break
            if candidate is None:
                break
            working = merge_tokens(
                working,
                candidate[0],
                candidate[1],
                record_orig_forms=record_orig_forms,
                separator="",
            )
        if in_place:
            return _replace_tokenlist_contents(sentence, working)
        return working

    return _apply_sentence_transform(data, transform_sentence, in_place=in_place)


def split_token(tokenlist: TokenList, target_id: int, direction: str, new_token_attrs: dict) -> TokenList:
    """
    Split a token at the character level and insert a new dependent token either before or after.

    Parameters:
        tokenlist (TokenList): CoNLL-U sentence.
        target_id (int): ID of the token to split (1-based).
        direction (str): 'forward' or 'backward' to indicate insertion before or after the target character.
        new_token_attrs (dict): dict like {"upos": "PART", "deprel": "aux", ...}

    Returns:
        TokenList: Updated token list with the new token inserted and IDs/heads adjusted.
    """
    id_to_token = {t["id"]: t for t in tokenlist if isinstance(t["id"], int)}
    if target_id not in id_to_token:
        raise ValueError(f"Token ID {target_id} not found.")

    target_token = deepcopy(id_to_token[target_id])
    original_form = target_token["form"]

    if len(original_form) < 2:
        raise ValueError("Cannot split token with less than 2 characters.")

    # Split the form
    if direction == "leftward":
        new_char = original_form[0]
        remaining = original_form[1:]
    elif direction == "rightward":
        new_char = original_form[-1]
        remaining = original_form[:-1]
    else:
        raise ValueError("Direction must be 'leftward' or 'rightward'.")

    # Update target token form
    target_token["form"] = remaining
    has_lemma = "lemma" in target_token and target_token["lemma"] != "_"
    if has_lemma:
        target_token["lemma"] = remaining
    target_token = Token(target_token)

    # Create new token
    new_token = Token({
        "id": -1,  # temporary ID
        "form": new_char,
        "lemma": new_char if has_lemma else "_",
        "upostag": new_token_attrs.get("upos", "_"),
        "xpostag": "_",
        "feats": "_",
        "head": target_id,  # point to the original token
        "deprel": new_token_attrs.get("deprel", "dep"),
        "deps": "_",
        "misc": "_"
    })

    # Rebuild the token list
    new_tokens = []
    for token in tokenlist:
        tid = token.get("id")
        if isinstance(tid, int):
            if tid == target_id:
                if direction == "forward":
                    new_tokens.append(new_token)
                    new_tokens.append(target_token)
                else:
                    new_tokens.append(target_token)
                    new_tokens.append(new_token)
            else:
                new_tokens.append(deepcopy(token))
        else:
            new_tokens.append(deepcopy(token))  # metadata lines

    # Reassign IDs and heads
    old_to_new_id = {}
    for i, token in enumerate(new_tokens):
        old_id = token.get("id")
        if isinstance(old_id, int) or old_id == -1:
            new_id = i + 1
            old_to_new_id[old_id] = new_id
            token["id"] = new_id

    for token in new_tokens:
        head = token.get("head")
        if isinstance(head, int):
            token["head"] = old_to_new_id.get(head, head)

    return TokenList(new_tokens, metadata=tokenlist.metadata)

def remove_punct(data, in_place: bool = False):
    """
    Remove punctuation tokens from either a TokenList or a SentenceList.
    If input is a TokenList, returns a new TokenList without punctuation.
    If input is a SentenceList, returns a new SentenceList with punctuation removed from each sentence.
    """
    def get_punct_ids(sentence: TokenList) -> set[int]:
        token_ids = set()
        for token in sentence:
            upos = token.get("upos")
            deprel = token.get("deprel")
            if upos == "PUNCT" and deprel == "punct":
                token_ids.add(token["id"])
            elif upos == "PUNCT" or deprel == "punct":
                print(f"Warning: Token ID={token['id']} meets only one of UPOS or DEPREL conditions. Not removed.")
        return token_ids

    return _apply_sentence_removal(data, get_punct_ids, in_place=in_place)


def remove_edge_punct(data, side: str = "both", in_place: bool = False):
    """
    Remove punctuation tokens only at sentence edges.

    Parameters:
        data: A TokenList, SentenceList, or iterable of TokenList objects.
        side: "left", "right", or "both".
        in_place: If True, mutate input where supported.
    """
    if side not in {"left", "right", "both"}:
        raise ValueError("side must be 'left', 'right', or 'both'.")

    def get_edge_punct_ids(sentence: TokenList) -> set[int]:
        word_tokens = [tok for tok in sentence if isinstance(tok.get("id"), int)]
        token_ids = set()
        if side in {"left", "both"}:
            for token in word_tokens:
                if token.get("upos") == "PUNCT" and token.get("deprel") == "punct":
                    token_ids.add(token["id"])
                    continue
                break
        if side in {"right", "both"}:
            for token in reversed(word_tokens):
                if token.get("upos") == "PUNCT" and token.get("deprel") == "punct":
                    token_ids.add(token["id"])
                    continue
                break
        return token_ids

    return _apply_sentence_removal(data, get_edge_punct_ids, in_place=in_place)


DEFAULT_PUNCT_NORMALIZATION = {
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "；": ";",
    "：": ":",
    "（": "(",
    "）": ")",
    "“": "\"",
    "”": "\"",
    "‘": "'",
    "’": "'",
    "—": "-",
    "–": "-",
}


def normalize_punctuation(
    data,
    mapping: dict[str, str] | None = None,
    set_upos: bool = True,
    set_deprel: bool = False,
    in_place: bool = False,
):
    """
    Normalize punctuation forms and lemmas.

    By default this maps common fullwidth or curly punctuation to ASCII forms
    and sets UPOS to PUNCT for normalized punctuation tokens. It does not change
    HEAD unless paired with another operation.
    """
    punct_map = mapping or DEFAULT_PUNCT_NORMALIZATION
    punct_values = set(punct_map.values()) | set(punct_map.keys())

    def transform_sentence(sentence: TokenList) -> TokenList:
        working = sentence if in_place else _copy_tokenlist(sentence)
        for token in working:
            if not isinstance(token.get("id"), int):
                continue
            form = token.get("form")
            normalized = punct_map.get(form)
            if normalized is None and form not in punct_values:
                continue
            if normalized is not None:
                token["form"] = normalized
                if token.get("lemma") == form or token.get("lemma") in (None, "_"):
                    token["lemma"] = normalized
            if set_upos:
                token["upos"] = "PUNCT"
            if set_deprel and token.get("deprel") in (None, "_", "dep"):
                token["deprel"] = "punct"
        result = TokenList([Token(dict(tok)) for tok in working], deepcopy(working.metadata), build_refs=True)
        if in_place:
            return _replace_tokenlist_contents(sentence, result)
        return result

    return _apply_sentence_transform(data, transform_sentence, in_place=in_place)


def remove_reparandum(data, in_place: bool = False):
    """
    Remove reparandum tokens when doing so does not leave external dependents.

    The function removes tokens whose base dependency relation is `reparandum`,
    then renumbers token IDs and HEADs. If a kept token depends on a removed
    token, a ValueError is raised.
    """
    def get_reparandum_ids(sentence: TokenList) -> set[int]:
        return {
            token["id"] for token in sentence
            if isinstance(token.get("id"), int) and _base_deprel(token.get("deprel")) == "reparandum"
        }

    return _apply_sentence_removal(data, get_reparandum_ids, in_place=in_place)


def remove_discourse_interjections(data, in_place: bool = False):
    """
    Remove discourse interjections when both UPOS and DEPREL match.

    Only tokens with `UPOS=INTJ` and base dependency relation `discourse` are
    removed. The deletion is rejected if it would leave kept tokens depending
    on a removed token.
    """
    def get_discourse_intj_ids(sentence: TokenList) -> set[int]:
        return {
            token["id"] for token in sentence
            if (
                isinstance(token.get("id"), int)
                and token.get("upos") == "INTJ"
                and _base_deprel(token.get("deprel")) == "discourse"
            )
        }

    return _apply_sentence_removal(data, get_discourse_intj_ids, in_place=in_place)


def remove_subtree(tokenlist: TokenList, root: int | Token, in_place: bool = False) -> TokenList:
    """
    Remove the complete subtree rooted at one token.

    The root may be a token ID or a Token object. The complete descendant set is
    removed together, then token IDs and HEADs are renumbered.
    """
    root_id = root.get("id") if isinstance(root, Token) else root
    if not isinstance(root_id, int):
        raise TypeError("root must be a token ID or a Token object with an integer ID.")
    subtree_ids = _collect_subtree_ids(tokenlist, root_id)
    return _remove_token_ids(tokenlist, subtree_ids, in_place=in_place)


def graft_subtree(
    tokenlist: TokenList,
    root: int | Token,
    new_head: int | Token,
    deprel: str | None = None,
    in_place: bool = False,
) -> TokenList:
    """
    Reattach a complete subtree to a new head.

    The operation changes only the HEAD of the subtree root, plus DEPREL when
    provided. The new head cannot be inside the moved subtree, because that
    would create a cycle.
    """
    root_id = root.get("id") if isinstance(root, Token) else root
    new_head_id = new_head.get("id") if isinstance(new_head, Token) else new_head
    if not isinstance(root_id, int) or not isinstance(new_head_id, int):
        raise TypeError("root and new_head must be token IDs or Token objects with integer IDs.")

    working = tokenlist if in_place else _copy_tokenlist(tokenlist)
    id_to_token = {tok["id"]: tok for tok in working if isinstance(tok.get("id"), int)}
    if root_id not in id_to_token:
        raise ValueError(f"Token ID {root_id} not found.")
    if new_head_id not in id_to_token:
        raise ValueError(f"New head token ID {new_head_id} not found.")

    subtree_ids = _collect_subtree_ids(working, root_id)
    if new_head_id in subtree_ids:
        raise ValueError("Cannot graft a subtree under one of its own descendants.")

    id_to_token[root_id]["head"] = new_head_id
    if deprel is not None:
        id_to_token[root_id]["deprel"] = deprel

    result = TokenList([Token(dict(tok)) for tok in working], deepcopy(working.metadata), build_refs=True)
    if in_place:
        return _replace_tokenlist_contents(tokenlist, result)
    return result

def remove_leaf_node(tokenlist: TokenList, token: Token, in_place: bool = True) -> Union[None, TokenList]:
    """
    Remove a leaf node from a TokenList.

    Parameters:
        tokenlist (TokenList): The sentence to operate on.
        token (Token): The token to remove (must be a leaf node).
        in_place (bool): If True, modifies the TokenList in place.
                         If False, returns a new TokenList with the token removed.

    Returns:
        TokenList or None: A new TokenList if in_place is False; otherwise None.
    """
    if not token.is_leaf_node():
        raise ValueError("The token is not a leaf node and cannot be removed.")

    if not in_place:
        # Clone the TokenList and rebind internal references
        tokenlist = TokenList([Token(dict(tok)) for tok in tokenlist], tokenlist.metadata.copy())
        for tok in tokenlist:
            tok.tokenlist = tokenlist
        token = tokenlist[token["id"] - 1]  # Locate the corresponding token in the clone

    # Remove the token and re-index
    tokenlist.remove(token)
    tokenlist.refresh_ids()
    tokenlist.refresh_heads()
    tokenlist.refresh_metadata_text()

    if not in_place:
        return tokenlist

def merge_sentences(
    sentence1: TokenList,
    sentence2: TokenList,
    deprel: str = "parataxis",
    main_sent_id: int = 1,
    insert_punct_if_missing: bool = True,
    inserted_punct: str = "，",
    insert_comma_if_missing: bool | None = None,
    in_place: bool = True
) -> TokenList:
    """
    Merge two TokenList sentences into one.

    Parameters:
        sentence1 (TokenList): The first sentence (always on the left).
        sentence2 (TokenList): The second sentence (always on the right).
        deprel (str): The deprel label to assign when linking the root of the sub sentence to the main sentence.
        main_sent_id (int): Which sentence is the main clause (1 or 2).
        insert_punct_if_missing (bool): Whether to insert punctuation if the left sentence does not end in punctuation.
        inserted_punct (str): The punctuation FORM/LEMMA to insert when insert_punct_if_missing=True.
        insert_comma_if_missing (bool | None): Deprecated alias for insert_punct_if_missing.
        in_place (bool): If True, requires both sentences to come from the same SentenceList and be adjacent.

    Returns:
        TokenList: A new merged TokenList.
    """
    if insert_comma_if_missing is not None:
        insert_punct_if_missing = insert_comma_if_missing

    if main_sent_id not in (1, 2):
        raise ValueError("main_sent_id must be either 1 or 2.")
    if insert_punct_if_missing and not inserted_punct:
        raise ValueError("inserted_punct must be a non-empty string when insert_punct_if_missing=True.")

    # ==== Check continuity if in_place ====
    if in_place:
        s1_list = getattr(sentence1, "sentencelist", None)
        s2_list = getattr(sentence2, "sentencelist", None)
        if s1_list is None or s2_list is None or s1_list is not s2_list:
            raise ValueError("Sentences must belong to the same SentenceList when in_place=True.")
        i1 = s1_list.index(sentence1)
        i2 = s2_list.index(sentence2)
        if i2 != i1 + 1:
            raise ValueError("Sentences must be adjacent in the SentenceList.")

    # ==== Clone sentence1 and sentence2 ====
    clone_sent1 = TokenList([Token(dict(tok)) for tok in sentence1 if isinstance(tok.get("id"), int)],
                             metadata=sentence1.metadata.copy())
    clone_sent2 = TokenList([Token(dict(tok)) for tok in sentence2 if isinstance(tok.get("id"), int)],
                             metadata=sentence2.metadata.copy())
    _set_head_object_refs(clone_sent1)
    _set_head_object_refs(clone_sent2)

    # ==== Find roots ====
    root1 = _get_root_token_local(clone_sent1)
    root2 = _get_root_token_local(clone_sent2)
    main_root, sub_root = (root1, root2) if main_sent_id == 1 else (root2, root1)

    added_punct = False
    if insert_punct_if_missing and clone_sent1[-1].get("upos") != "PUNCT":
        punct = deepcopy(CHINESE_COMMA)
        punct["form"] = inserted_punct
        punct["lemma"] = inserted_punct
        clone_sent1.append(punct)
        punct.head = root1
        added_punct = True

    # ==== Link sub_root to main_root ====
    sub_root.head = main_root
    sub_root["deprel"] = deprel

    # ==== Merge metadata  ====
    connector = inserted_punct if added_punct else ""
    merged_metadata = {
        "sent_id": clone_sent1.metadata.get("sent_id", ""),
        #"text": clone_sent1.metadata.get("text", "") + connector + clone_sent2.metadata.get("text", "")
    }

    # ==== Combine tokens and build new TokenList ====
    combined_tokens = clone_sent1 + clone_sent2
    merged = TokenList(combined_tokens, merged_metadata, build_refs=False)
    merged.refresh_ids()
    merged.refresh_heads()
    deps_value = sub_root.get("deps")
    new_sub_edge = (deprel, main_root.get("id"))
    if isinstance(deps_value, list):
        kept_deps = [
            edge for edge in deps_value
            if not (len(edge) == 2 and edge[0] == "root" and edge[1] == 0)
        ]
        if new_sub_edge not in kept_deps:
            kept_deps.append(new_sub_edge)
        sub_root["deps"] = kept_deps
    else:
        sub_root["deps"] = [new_sub_edge]
    _set_head_object_refs(merged)
    merged.refresh_metadata_text()

    # ==== Replace in SentenceList if in_place ====
    if in_place:
        slist = sentence1.sentencelist
        i1 = slist.index(sentence1)
        slist[i1] = merged
        del slist[i1 + 1]
        merged.sentencelist = slist

    return merged


def merge_sentence_sequence(
    sentences,
    start_index: int = 0,
    end_index: int | None = None,
    deprel: str = "parataxis",
    main_sent_id: int = 1,
    insert_punct_if_missing: bool = True,
    inserted_punct: str = "，",
) -> TokenList:
    """
    Merge a consecutive sentence span into one TokenList.

    Parameters:
        sentences: A SentenceList or a list-like object of TokenList items.
        start_index: Inclusive start index of the span to merge.
        end_index: Exclusive end index. If None, merge through the end.
        deprel: Relation used when attaching each later sentence root.
        main_sent_id: Passed to merge_sentences for every pairwise merge.
        insert_punct_if_missing: Whether to insert punctuation between parts.
        inserted_punct: Punctuation FORM/LEMMA to insert when needed.
    """
    if end_index is None:
        end_index = len(sentences)
    if start_index < 0 or end_index > len(sentences) or start_index >= end_index:
        raise ValueError("Invalid sentence span.")

    selected = list(sentences[start_index:end_index])
    if len(selected) < 2:
        raise ValueError("At least two sentences are required for merge_sentence_sequence().")
    if any(not isinstance(sentence, TokenList) for sentence in selected):
        raise TypeError("merge_sentence_sequence() expects TokenList items.")

    source_sent_ids = [
        sentence.metadata.get("sent_id", "")
        for sentence in selected
        if sentence.metadata.get("sent_id")
    ]

    merged = _copy_tokenlist(selected[0])
    for next_sentence in selected[1:]:
        merged = merge_sentences(
            merged,
            _copy_tokenlist(next_sentence),
            deprel=deprel,
            main_sent_id=main_sent_id,
            insert_punct_if_missing=insert_punct_if_missing,
            inserted_punct=inserted_punct,
            in_place=False,
        )

    if source_sent_ids:
        merged.metadata["sent_id"] = "+".join(source_sent_ids)
        merged.metadata["source_sent_ids"] = "|".join(source_sent_ids)
    merged.refresh_metadata_text()
    return merged


def split_sentence_at_node(
    sentence: TokenList,
    target_node: Token,
    insert_punct_if_missing: bool = True,
    inserted_punct: str = "，",
    insert_comma_if_missing: bool | None = None,
    in_place: bool = True
):
    """
    Split a sentence at the subtree rooted at target_node.
    """

    # ==== Safety checks ====
    if insert_comma_if_missing is not None:
        insert_punct_if_missing = insert_comma_if_missing
    if insert_punct_if_missing and not inserted_punct:
        raise ValueError("inserted_punct must be a non-empty string when insert_punct_if_missing=True.")

    if target_node.tokenlist is not sentence:
        raise ValueError("The target node does not belong to the given sentence.")

    root = _get_root_token_local(sentence)
    if target_node is root:
        raise ValueError("Cannot split at the root node.")

    if not is_marginal_subtree(target_node, exclude_punct=True):
        raise ValueError("The subtree must be located at one margin of the sentence.")

    # ==== Get subtree span ====
    min_id, max_id = target_node.get_subtree_span(
        exclude_left_punct=True, exclude_right_punct=True
    )

    total_ids = sorted(t["id"] for t in sentence if isinstance(t["id"], int))
    sentence_min_id = total_ids[0]
    sentence_max_id = total_ids[-1]

    # ==== Determine split direction ====
    is_left_split = (min_id == sentence_min_id)

    if is_left_split:
        left_ids = [i for i in total_ids if i <= max_id]
        right_ids = [i for i in total_ids if i > max_id]
    else:
        left_ids = [i for i in total_ids if i < min_id]
        right_ids = [i for i in total_ids if i >= min_id]

    # ==== Clone tokens ====
    left_tokens = [Token(dict(t)) for t in sentence if t["id"] in left_ids]
    right_tokens = [Token(dict(t)) for t in sentence if t["id"] in right_ids]

    # ==== Build TokenLists without refs ====
    left_part = TokenList(left_tokens, deepcopy(sentence.metadata), build_refs=True)
    right_part = TokenList(right_tokens, deepcopy(sentence.metadata), build_refs=True)

    for tok in left_part:
        tok.set_tokenlist(left_part)
    for tok in right_part:
        tok.set_tokenlist(right_part)

    # ==== Refresh IDs ====
    left_part.refresh_ids()
    right_part.refresh_ids()

    # ==== Insert comma if necessary ====
    if insert_comma_if_missing and is_left_split:
        last_tok = left_part[-1]
        if last_tok.get("upos") != "PUNCT":
            comma = Token(deepcopy(CHINESE_COMMA))
            comma.set_tokenlist(left_part)
            comma.head = last_tok  # temporarily attach to last token
            left_part.append(comma)
            left_part.refresh_ids()  # Refresh IDs after insertion

    # ==== Sync back head IDs from logical refs ====
    left_part.refresh_heads()
    right_part.refresh_heads()

    # ==== Refresh metadata text ====
    left_part.refresh_metadata_text()
    right_part.refresh_metadata_text()

    # ==== In-place update ====
    if in_place:
        sentencelist = sentence.sentencelist
        if sentencelist is None:
            raise ValueError("Sentence is not part of any SentenceList.")
        idx = sentencelist.index(sentence)
        sentencelist[idx] = left_part
        sentencelist.insert(idx + 1, right_part)
        left_part.sentencelist = sentencelist
        right_part.sentencelist = sentencelist
        return

    return (left_part, right_part)

def postprocess_stanza_depparse(sentences):
    """
    Process a single .conllu file: modify tokens as needed.
    """

    # ==== 词表定义 ====
    ADVERBS = {"还", "就", "也", "又", "都", "先", "还是", "但是", "但", "所以"}
    DIRECTIONAL_VERBS = {"上", "上去", "下去", "下来", "起来", "出去", "出来", "回来", "过去", "过来", "进去", "进来"}
    LOCALIZERS = {"上", "上面", "之间", "前", "前面", "里", "里面", "下", "下面", "旁边", "边上", "边", "底下"}
    NEGATIVES = {"不", "没", "没有"}
    EMPHASIZERS = {"正", "就是", "也"}
    TENSE_ASPECT_MARKERS = {"了", "着", "过", "一下", "在", "掉", "好", "完", "没有", "没"}
    MODALITY_MAKERS = {"应该", "要", "可能", "可以"}
    DEGREE_MODIFIERS = {"很", "非常", "挺", "比较", "蛮", "有点", "好", "死", "更", "一点", "太", "稍微"}
    DISCOURSE_MARKERS = {"结果", "因为", "然后", "而且", "所以"}
    MIRATIVE = {"没想到", "想不到", "哪知道"}

    VERB = ("弄", "搞", "搅", "绞", "烧", "烤", "烘", "挤", "拧", "晾", "晒", "淋")
    RESULTATIVE = ("干", "焦", "坏", "湿", "脏")
    VR = tuple(v + r for v in VERB for r in RESULTATIVE)

    for i, sentence in enumerate(sentences):
        id2token = {tok["id"]: tok for tok in sentence if isinstance(tok["id"], int)}

        for token in sentence:
            form = token.get("form")
            upos = token.get("upos")
            deprel = token.get("deprel")
            id = token.get("id")
            head = token.get("head")

            ##### 共性问题 #####
            # 规则1：“了2”和句末“的”之前处理为discourse
            # 修改为discourse:sp
            if (form == "了" or form == "的") and deprel == "discourse":
                token["deprel"] = "discourse:sp"

            # 规则2：焦点敏感算子和一些副词性成分，之前处理为mark
            # 建议处理为advmod
            elif form in ADVERBS and upos == "ADV" and deprel == "mark":
                token["deprel"] = "advmod"

            # 规则3：“后来”“最后”等副词性时间成分处理为nmod:tmod的，改为obl:tmod
            elif deprel == "nmod:tmod":
                token["deprel"] = "obl:tmod"

            # 时体成分
            elif form == "一下" and deprel == "mark":
                token["upos"] = "AUX"
                token["deprel"] = "aux"

            elif form == "在" and deprel == "advmod" and upos == "ADV":
                token["upos"] = "AUX"
                token["deprel"] = "aux:ta"

            # 规则3：趋向动词建议分析为aux
            # 只有原先为mark的才改，原先为主要动词的不改
            elif form in DIRECTIONAL_VERBS and deprel == "mark":
                token["deprel"] = "compound:dir"

            ##### 细分标签 #####
            # aux细分aux:ta时体标记
            elif form in TENSE_ASPECT_MARKERS and deprel == "aux":
                token["deprel"] = "aux:ta"

            # aux细分aux:modal情态标记
            elif form in MODALITY_MAKERS and deprel == "aux":
                token["deprel"] = "aux:modal"

            # 方位词
            elif form in LOCALIZERS and deprel == "acl":
                token["deprel"] = "case:loc"

            # 否定词
            elif form in NEGATIVES and upos == "ADV" and deprel == "advmod":
                token["deprel"] = "advmod:neg"

            # advmod细分advmod:deg程度修饰语
            elif form in DEGREE_MODIFIERS and deprel == "advmod":
                token["deprel"] = "advmod:deg"

            # 句首话语标记
            elif form in DISCOURSE_MARKERS and (deprel == "advmod" or deprel == "mark"):
                token["deprel"] = "discourse"

            # 意外标记
            elif form in MIRATIVE:
                token["deprel"] = "advmod"
            
            # “VP以后/的时候”
            elif form in ("以后", "的时候") and deprel == "mark":
                token["deprel"] = "mark:tmod"
            
            elif form in ("一") and deprel == "mark":
                token["deprel"] = "mark:ta"
                
            ##### 任务特有的词汇Task-specific #####
            elif form in ("三毛", "流浪记") and upos != "PROPN":
                token["upos"] = "PROPN"

            # “干”的词性
            elif form == "干" and upos != "ADJ":
                token["upos"] = "ADJ"
            
            # “小”的词性
            elif form in ("小", "大", "新") and upos != "ADJ":
                token["upos"] = "ADJ"
                token["deprel"] = "amod"

            # “结果”的词性
            #elif form == "结果":
            #    token["upos"] = "ADV"
            #    token["deprel"] = "advmod"

            # “滴水”的词性
            elif form == "滴水" and upos != "VERB":
                token["upos"] = "VERB"

        ##### 新增三条结构处理规则 #####

        # 1. 修复数量名结构（nummod → clf → noun）
        for token in sentence:
            if token.get("deprel") not in {"nummod", "det"}:
                continue
            modifier = token
            clf_id = modifier.get("head")
            clf = id2token.get(clf_id, {})
            noun_id = clf.get("head")
            if clf.get("deprel") != "clf" or noun_id not in id2token:
                continue
            modifier["head"] = noun_id
            clf["head"] = modifier["id"]

        '''
        # 2. 合并“这 个”/“那 个”结构
        j = 0
        while j < len(sentence) - 1:
            t1, t2 = sentence[j], sentence[j + 1]
            if (
                t1.get("form") in {"这", "那"}
                and t2.get("form") == "个"
                and isinstance(t1.get("id"), int)
                and isinstance(t2.get("id"), int)
                and t1["id"] + 1 == t2["id"]
            ):
                try:
                    sentence = merge_tokens(sentence, t1["id"], t2["id"])
                    sentences[i] = sentence
                    id2token = {tok["id"]: tok for tok in sentence if isinstance(tok["id"], int)}
                    j = 0
                    continue
                except Exception as e:
                    print(f"❌ 合并失败：第{j+1}对 → {e}")
            j += 1

        
        # 3. 拆分动补结构
        for token in sentence:
            if isinstance(token["id"], int) and token["form"] in VR:
                sentence = split_token(
                    tokenlist=sentence,
                    target_id=token["id"],
                    direction="rightward",
                    new_token_attrs={"upos": "ADJ", "deprel": "compound:vv"}
                )
                sentences[i] = sentence
                break  # 每句只处理一次动补结构
        '''
    return sentences
