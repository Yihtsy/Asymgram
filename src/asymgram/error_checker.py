from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable, Iterator

from .conllu import SentenceList, TokenList, parse, parse_incr


UD_UPOS = {
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
}

UD_DEPRELS = {
    "acl",
    "advcl",
    "advmod",
    "amod",
    "appos",
    "aux",
    "case",
    "cc",
    "ccomp",
    "clf",
    "compound",
    "conj",
    "cop",
    "csubj",
    "dep",
    "det",
    "discourse",
    "dislocated",
    "expl",
    "fixed",
    "flat",
    "goeswith",
    "iobj",
    "list",
    "mark",
    "nmod",
    "nsubj",
    "nummod",
    "obj",
    "obl",
    "orphan",
    "parataxis",
    "punct",
    "reparandum",
    "ref",
    "root",
    "vocative",
    "xcomp",
}

UD_FEATURES = {
    "Abbr",
    "Animacy",
    "Aspect",
    "Case",
    "Clusivity",
    "Definite",
    "Degree",
    "Deixis",
    "DeixisRef",
    "Evident",
    "ExtPos",
    "Foreign",
    "Gender",
    "Mood",
    "NounClass",
    "NumForm",
    "NumType",
    "Number",
    "Other",
    "Person",
    "Polarity",
    "Polite",
    "Poss",
    "PronType",
    "Reflex",
    "Style",
    "Tense",
    "Typo",
    "VerbForm",
    "Voice",
}

STRONG_UPOS_DEPREL_WARNINGS = {
    ("ADP", "nsubj"),
    ("ADP", "obj"),
    ("ADP", "iobj"),
    ("DET", "nsubj"),
    ("DET", "obj"),
    ("DET", "iobj"),
    ("DET", "advmod"),
    ("CCONJ", "case"),
    ("CCONJ", "det"),
    ("CCONJ", "amod"),
    ("SCONJ", "det"),
    ("SCONJ", "amod"),
    ("SCONJ", "obj"),
    ("PUNCT", "nsubj"),
    ("PUNCT", "obj"),
    ("PUNCT", "amod"),
    ("NUM", "amod"),
    ("NUM", "det"),
    ("ADV", "amod"),
    ("ADJ", "advmod"),
    ("AUX", "amod"),
    ("AUX", "nmod"),
    ("AUX", "det"),
}

EXPECTED_DEPRELS_BY_UPOS = {
    "ADP": {"case", "fixed", "mark", "adp"},
    "DET": {"det", "fixed"},
    "CCONJ": {"cc", "fixed"},
    "SCONJ": {"mark", "fixed"},
    "PUNCT": {"punct"},
    "NUM": {"nummod", "compound", "flat", "root", "conj", "obl", "nmod"},
    "ADV": {"advmod", "fixed", "discourse", "mark", "root"},
    "AUX": {"aux", "aux:pass", "cop"},
    "INTJ": {"discourse", "root"},
}

ID_RE = re.compile(r"^[1-9][0-9]*$")
MWT_ID_RE = re.compile(r"^([1-9][0-9]*)-([1-9][0-9]*)$")
EMPTY_ID_RE = re.compile(r"^([1-9][0-9]*)\.([1-9][0-9]*)$")
DEPS_HEAD_RE = re.compile(r"^(0|[1-9][0-9]*(?:\.[1-9][0-9]*)?):(.+)$")
FEATURE_NAME_RE = re.compile(r"^[A-Z][A-Za-z0-9]*$")
FEATURE_VALUE_RE = re.compile(r"^[A-Za-z0-9_,.-]+$")


@dataclass(frozen=True)
class Diagnostic:
    code: str
    level: str
    message: str
    sent_id: str = "NA"
    token_id: Any = None
    line: int | None = None
    field: str | None = None

    def as_tuple(self) -> tuple[str, str, str, str, Any, int | None, str | None]:
        return (self.level, self.code, self.message, self.sent_id, self.token_id, self.line, self.field)


def _tok_get(token: Any, key: str, default: Any = None) -> Any:
    if key == "upos":
        return token.get("upos", token.get("upostag", default))
    return token.get(key, default)


def _is_word_id(token_id: Any) -> bool:
    return isinstance(token_id, int)


def _is_mwt_id(token_id: Any) -> bool:
    return isinstance(token_id, tuple) and len(token_id) == 3 and token_id[1] == "-"


def _is_empty_id(token_id: Any) -> bool:
    return isinstance(token_id, tuple) and len(token_id) == 3 and token_id[1] == "."


def _base_deprel(deprel: Any) -> str:
    return str(deprel).split(":", 1)[0] if deprel is not None else ""


def _format_dep_head(head: Any) -> str:
    if isinstance(head, tuple) and len(head) == 3 and head[1] == ".":
        return f"{head[0]}.{head[2]}"
    return str(head)


def _same_base_deprel(first: Any, second: Any) -> bool:
    return _base_deprel(first) == _base_deprel(second)


def _position(sent: Any, token: Any | None = None) -> tuple[str, Any]:
    sent_id = getattr(sent, "metadata", {}).get("sent_id", "NA")
    token_id = _tok_get(token, "id") if token is not None else None
    return sent_id, token_id


def _diag(
    code: str,
    level: str,
    message: str,
    sent: Any | None = None,
    token: Any | None = None,
    line: int | None = None,
    field: str | None = None,
) -> Diagnostic:
    sent_id, token_id = _position(sent, token) if sent is not None else ("NA", None)
    return Diagnostic(code, level, message, sent_id, token_id, line, field)


def is_valid_deprel(deprel: Any) -> bool:
    if not isinstance(deprel, str) or not deprel or deprel == "_":
        return False
    parts = deprel.split(":")
    if not parts[0] in UD_DEPRELS:
        return False
    return all(part for part in parts[1:])


def parse_feats(feats: Any) -> tuple[dict[str, set[str]], list[str]]:
    if feats in (None, "_"):
        return {}, []
    if isinstance(feats, dict):
        return {str(k): set(v if isinstance(v, list) else str(v).split(",")) for k, v in feats.items()}, []
    errors = []
    parsed: dict[str, set[str]] = {}
    for feature in str(feats).split("|"):
        if "=" not in feature:
            errors.append(f"feature lacks '=': {feature}")
            continue
        name, value = feature.split("=", 1)
        if not FEATURE_NAME_RE.match(name):
            errors.append(f"malformed feature name: {name}")
        values = set(value.split(","))
        if not value or any(not FEATURE_VALUE_RE.match(v) for v in values):
            errors.append(f"malformed value for {name}: {value}")
        if name in parsed:
            errors.append(f"duplicate feature: {name}")
            parsed[name].update(values)
        else:
            parsed[name] = values
    return parsed, errors


def parse_deps(deps: Any) -> tuple[list[tuple[str, str]], list[str]]:
    if deps in (None, "_"):
        return [], []
    parsed = []
    errors = []
    if isinstance(deps, list):
        for item in deps:
            if isinstance(item, tuple) and len(item) == 2:
                rel, head = item
                parsed.append((_format_dep_head(head), str(rel)))
            else:
                errors.append(f"malformed DEPS item: {item}")
        return parsed, errors
    for item in str(deps).split("|"):
        match = DEPS_HEAD_RE.match(item)
        if not match:
            errors.append(f"malformed DEPS item: {item}")
            continue
        parsed.append((match.group(1), match.group(2)))
    return parsed, errors


def text_from_forms(sent: Any, consider_spaceafter: bool = True) -> str:
    parts = []
    covered_word_ids = set()
    for token in sent:
        token_id = _tok_get(token, "id")
        if _is_mwt_id(token_id):
            start, _, end = token_id
            covered_word_ids.update(range(start, end + 1))
            parts.append(str(_tok_get(token, "form", "")))
        elif _is_word_id(token_id):
            if token_id in covered_word_ids:
                continue
            parts.append(str(_tok_get(token, "form", "")))
        else:
            continue
        misc = _tok_get(token, "misc")
        no_space = False
        if isinstance(misc, dict):
            no_space = misc.get("SpaceAfter") == "No"
        elif isinstance(misc, str):
            no_space = "SpaceAfter=No" in misc
        if consider_spaceafter and not no_space:
            parts.append(" ")
    return "".join(parts).rstrip()


def raw_format_diagnostics(text: str) -> list[Diagnostic]:
    diagnostics = []
    sentence_has_token = False
    previous_line_was_blank = True

    for line_no, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.rstrip("\n")
        if not line.strip():
            if not previous_line_was_blank:
                sentence_has_token = False
            previous_line_was_blank = True
            continue

        if line.startswith("#"):
            if sentence_has_token and not previous_line_was_blank:
                diagnostics.append(
                    Diagnostic("MISSING_SENTENCE_SEPARATOR", "ERROR", "metadata starts before a blank sentence separator", line=line_no)
                )
            if line.startswith("# sent_id") or line.startswith("# text"):
                if " = " not in line:
                    diagnostics.append(
                        Diagnostic("MALFORMED_METADATA", "ERROR", "metadata line lacks ' = '", line=line_no)
                    )
            elif "=" in line and " = " not in line:
                diagnostics.append(
                    Diagnostic("MALFORMED_METADATA", "WARNING", "metadata assignment should use ' = '", line=line_no)
                )
            previous_line_was_blank = False
            continue

        cols = line.split("\t")
        if len(cols) != 10:
            level = "ERROR"
            message = f"token line has {len(cols)} columns, expected 10"
            if "\t" not in line and len(line.split()) == 10:
                message += "; columns appear to be separated by spaces"
            diagnostics.append(Diagnostic("COLUMN_COUNT", level, message, line=line_no))
            previous_line_was_blank = False
            sentence_has_token = True
            continue

        raw_id, _, _, raw_upos, _, raw_feats, raw_head, raw_deprel, raw_deps, _ = cols
        if not (ID_RE.match(raw_id) or MWT_ID_RE.match(raw_id) or EMPTY_ID_RE.match(raw_id)):
            diagnostics.append(Diagnostic("MALFORMED_ID", "ERROR", f"malformed ID value: {raw_id}", line=line_no, field="id"))
        mwt_match = MWT_ID_RE.match(raw_id)
        if mwt_match and int(mwt_match.group(1)) >= int(mwt_match.group(2)):
            diagnostics.append(Diagnostic("MWT_RANGE_ORDER", "ERROR", f"multiword token range is reversed: {raw_id}", line=line_no, field="id"))
        if ID_RE.match(raw_id):
            if not re.match(r"^(0|[1-9][0-9]*)$", raw_head):
                diagnostics.append(Diagnostic("MALFORMED_HEAD", "ERROR", f"HEAD must be 0 or a positive integer: {raw_head}", line=line_no, field="head"))
        elif raw_head != "_":
            diagnostics.append(Diagnostic("MWT_EMPTY_HEAD", "ERROR", "multiword/empty node HEAD should be '_'", line=line_no, field="head"))
        if raw_upos != "_" and raw_upos not in UD_UPOS:
            diagnostics.append(Diagnostic("INVALID_UPOS", "ERROR", f"invalid UPOS tag: {raw_upos}", line=line_no, field="upos"))
        if raw_deprel != "_" and not is_valid_deprel(raw_deprel):
            diagnostics.append(Diagnostic("INVALID_DEPREL", "ERROR", f"invalid dependency relation: {raw_deprel}", line=line_no, field="deprel"))

        _, feat_errors = parse_feats(raw_feats)
        for error in feat_errors:
            diagnostics.append(Diagnostic("MALFORMED_FEATS", "ERROR", error, line=line_no, field="feats"))
        _, deps_errors = parse_deps(raw_deps)
        for error in deps_errors:
            diagnostics.append(Diagnostic("MALFORMED_DEPS", "ERROR", error, line=line_no, field="deps"))

        previous_line_was_blank = False
        sentence_has_token = True

    return diagnostics


def check_sentence(sent: Any) -> list[Diagnostic]:
    diagnostics = []
    word_tokens = [tok for tok in sent if _is_word_id(_tok_get(tok, "id"))]
    word_ids = [_tok_get(tok, "id") for tok in word_tokens]
    word_id_set = set(word_ids)
    word_by_id = {_tok_get(tok, "id"): tok for tok in word_tokens}
    mwt_ranges = [_tok_get(tok, "id") for tok in sent if _is_mwt_id(_tok_get(tok, "id"))]
    empty_ids = [_tok_get(tok, "id") for tok in sent if _is_empty_id(_tok_get(tok, "id"))]

    if word_ids != list(range(1, len(word_ids) + 1)):
        diagnostics.append(
            _diag(
                "WORD_ID_SEQUENCE",
                "ERROR",
                f"word IDs must be continuous and ordered from 1 to n; found {word_ids}",
                sent,
            )
        )
    for token_id, count in Counter(word_ids).items():
        if count > 1:
            diagnostics.append(_diag("DUPLICATE_ID", "ERROR", f"duplicate word ID: {token_id}", sent))

    for token in sent:
        token_id = _tok_get(token, "id")
        if _is_mwt_id(token_id):
            start, _, end = token_id
            if start >= end:
                diagnostics.append(_diag("MWT_RANGE_ORDER", "ERROR", "multiword token range is reversed", sent, token))
            missing = [i for i in range(start, end + 1) if i not in word_id_set]
            if missing:
                diagnostics.append(
                    _diag("MWT_MISSING_CONSTITUENT", "ERROR", f"multiword token range points to missing IDs {missing}", sent, token)
                )
            continue
        if _is_empty_id(token_id):
            base, _, suffix = token_id
            if base not in word_id_set:
                diagnostics.append(_diag("EMPTY_NODE_BASE_MISSING", "ERROR", "empty node base ID is absent", sent, token))
            if suffix < 1:
                diagnostics.append(_diag("EMPTY_NODE_ID", "ERROR", "empty node suffix must be positive", sent, token))

    for first in mwt_ranges:
        for second in mwt_ranges:
            if first is second:
                continue
            first_span = set(range(first[0], first[2] + 1))
            second_span = set(range(second[0], second[2] + 1))
            if first_span & second_span and first != second:
                diagnostics.append(
                    _diag("MWT_OVERLAP", "ERROR", f"overlapping multiword token ranges: {first} and {second}", sent)
                )
                break

    empty_by_base = defaultdict(list)
    for base, _, suffix in empty_ids:
        empty_by_base[base].append(suffix)
    for base, suffixes in empty_by_base.items():
        expected = list(range(1, len(suffixes) + 1))
        if sorted(suffixes) != expected:
            diagnostics.append(
                _diag("EMPTY_NODE_SEQUENCE", "ERROR", f"empty node IDs for base {base} are not consecutive: {suffixes}", sent)
            )

    head0_tokens = []
    root_deprel_tokens = []
    for token in word_tokens:
        token_id = _tok_get(token, "id")
        head = _tok_get(token, "head")
        deprel = _tok_get(token, "deprel")
        upos = _tok_get(token, "upos")

        if not isinstance(head, int):
            diagnostics.append(_diag("HEAD_TYPE", "ERROR", "HEAD must be an integer in the basic tree", sent, token, field="head"))
        elif head == 0:
            head0_tokens.append(token)
        elif head == token_id:
            diagnostics.append(_diag("SELF_LOOP", "ERROR", "token cannot take itself as HEAD", sent, token, field="head"))
        elif head not in word_id_set:
            diagnostics.append(_diag("HEAD_NOT_FOUND", "ERROR", f"HEAD points to absent word ID {head}", sent, token, field="head"))

        if deprel == "root":
            root_deprel_tokens.append(token)
        if head == 0 and deprel != "root":
            diagnostics.append(_diag("HEAD_ROOT_MISMATCH", "ERROR", "HEAD=0 requires DEPREL=root", sent, token))
        if head != 0 and deprel == "root":
            diagnostics.append(_diag("DEPREL_ROOT_MISMATCH", "ERROR", "DEPREL=root requires HEAD=0", sent, token))

        if upos not in UD_UPOS:
            diagnostics.append(_diag("INVALID_UPOS", "ERROR", f"invalid UPOS tag: {upos}", sent, token, field="upos"))
        if not is_valid_deprel(deprel):
            diagnostics.append(_diag("INVALID_DEPREL", "ERROR", f"invalid dependency relation: {deprel}", sent, token, field="deprel"))

        if upos == "PUNCT" and deprel != "punct":
            diagnostics.append(_diag("PUNCT_NON_PUNCT", "WARNING", "UPOS=PUNCT normally requires DEPREL=punct", sent, token))
        if upos != "PUNCT" and deprel == "punct":
            diagnostics.append(_diag("NON_PUNCT_PUNCT", "WARNING", "DEPREL=punct normally requires UPOS=PUNCT", sent, token))
        if upos == "PUNCT" and head == 0:
            diagnostics.append(_diag("PUNCT_AS_ROOT", "WARNING", "punctuation should not normally be root", sent, token))
        if head in word_id_set:
            head_token = word_by_id.get(head)
            if head_token is not None and _tok_get(head_token, "upos") == "PUNCT" and upos != "PUNCT":
                diagnostics.append(_diag("PUNCT_AS_HEAD", "WARNING", "punctuation is used as a lexical head", sent, token))
            if head_token is not None and _base_deprel(deprel) == "conj" and _base_deprel(_tok_get(head_token, "deprel")) == "conj":
                diagnostics.append(
                    _diag(
                        "CONJ_CHAIN_HEAD",
                        "WARNING",
                        "conj token depends on another conj token; UD normally attaches conjuncts to the first conjunct",
                        sent,
                        token,
                    )
                )

        base_deprel = _base_deprel(deprel)
        if (upos, deprel) in STRONG_UPOS_DEPREL_WARNINGS or (upos, base_deprel) in STRONG_UPOS_DEPREL_WARNINGS:
            diagnostics.append(_diag("SUSPICIOUS_UPOS_DEPREL", "WARNING", f"suspicious UPOS-DEPREL combination: {upos} & {deprel}", sent, token))
        expected = EXPECTED_DEPRELS_BY_UPOS.get(upos)
        if expected and deprel not in expected and base_deprel not in {rel.split(':', 1)[0] for rel in expected}:
            diagnostics.append(_diag("ATYPICAL_UPOS_DEPREL", "WARNING", f"atypical UPOS-DEPREL combination: {upos} & {deprel}", sent, token))

        feats, feat_errors = parse_feats(_tok_get(token, "feats"))
        for error in feat_errors:
            diagnostics.append(_diag("MALFORMED_FEATS", "ERROR", error, sent, token, field="feats"))
        for feature_name in feats:
            if feature_name not in UD_FEATURES:
                diagnostics.append(
                    _diag("UPOS_FEATS_MISMATCH", "WARNING", f"{feature_name} is not a UD universal feature", sent, token, field="feats")
                )
        for feature_name, values in feats.items():
            if len(values) > 1 and feature_name in {"Number", "Gender", "Case", "Person", "Tense", "Degree", "VerbForm"}:
                diagnostics.append(
                    _diag("MUTUALLY_SUSPICIOUS_FEATS", "WARNING", f"{feature_name} has multiple values: {sorted(values)}", sent, token, field="feats")
                )

        deps, deps_errors = parse_deps(_tok_get(token, "deps"))
        for error in deps_errors:
            diagnostics.append(_diag("MALFORMED_DEPS", "ERROR", error, sent, token, field="deps"))
        seen_deps = Counter(deps)
        for dep, count in seen_deps.items():
            if count > 1:
                diagnostics.append(_diag("DUPLICATE_DEPS_EDGE", "ERROR", f"duplicate enhanced dependency: {dep}", sent, token, field="deps"))
        deps_by_head = defaultdict(list)
        for dep_head, dep_rel in deps:
            deps_by_head[dep_head].append(dep_rel)
        for dep_head, dep_rels in deps_by_head.items():
            if len(set(dep_rels)) > 1:
                diagnostics.append(
                    _diag(
                        "MULTIPLE_DEPS_RELS_SAME_HEAD",
                        "WARNING",
                        f"DEPS contains multiple relations from the same head {dep_head}: {sorted(set(dep_rels))}",
                        sent,
                        token,
                        field="deps",
                    )
                )
        for dep_head, dep_rel in deps:
            if dep_head == str(token_id):
                diagnostics.append(_diag("ENHANCED_SELF_LOOP", "ERROR", "DEPS contains an enhanced self-loop", sent, token, field="deps"))
            if isinstance(dep_rel, str) and dep_rel.count(":") > 1:
                diagnostics.append(
                    _diag(
                        "OVER_SPECIFIED_DEPS_DEPREL",
                        "WARNING",
                        f"enhanced dependency relation has multiple subtype levels: {dep_rel}",
                        sent,
                        token,
                        field="deps",
                    )
                )
            if dep_head != "0":
                base = int(dep_head.split(".", 1)[0])
                if base not in word_id_set:
                    diagnostics.append(_diag("DEPS_HEAD_NOT_FOUND", "ERROR", f"DEPS references absent node {dep_head}", sent, token, field="deps"))
                else:
                    dep_head_token = word_by_id.get(base)
                    if dep_head_token is not None and _base_deprel(dep_rel) == "conj" and _base_deprel(_tok_get(dep_head_token, "deprel")) == "conj":
                        diagnostics.append(
                            _diag(
                                "ENHANCED_CONJ_CHAIN_HEAD",
                                "WARNING",
                                "enhanced conj edge points to another conj token; check whether conjuncts should attach to the first conjunct",
                                sent,
                                token,
                                field="deps",
                            )
                        )
            if not is_valid_deprel(dep_rel):
                diagnostics.append(_diag("INVALID_DEPS_DEPREL", "ERROR", f"invalid DEPS relation: {dep_rel}", sent, token, field="deps"))

    enhanced_conj_edges = []
    for token in word_tokens:
        token_id = _tok_get(token, "id")
        deps, deps_errors = parse_deps(_tok_get(token, "deps"))
        if deps_errors:
            continue
        for dep_head, dep_rel in deps:
            if _base_deprel(dep_rel) == "conj" and dep_head != "0":
                enhanced_conj_edges.append((dep_head, str(token_id), token))
            if dep_head == "0" and dep_rel == "root" and _tok_get(token, "head") != 0:
                diagnostics.append(
                    _diag(
                        "ENHANCED_ROOT_NONROOT_TOKEN",
                        "WARNING",
                        "DEPS contains 0:root for a token that is not the basic root",
                        sent,
                        token,
                        field="deps",
                    )
                )
    enhanced_conj_pairs = {(head, dependent) for head, dependent, _ in enhanced_conj_edges}
    emitted_reciprocal_conj_pairs = set()
    for head, dependent, token in enhanced_conj_edges:
        if (dependent, head) in enhanced_conj_pairs:
            pair_key = tuple(sorted((head, dependent)))
            if pair_key in emitted_reciprocal_conj_pairs:
                continue
            emitted_reciprocal_conj_pairs.add(pair_key)
            diagnostics.append(
                _diag(
                    "ENHANCED_CONJ_RECIPROCAL",
                    "WARNING",
                    f"enhanced conj edges appear in both directions between {head} and {dependent}",
                    sent,
                    token,
                    field="deps",
                )
            )

    if len(head0_tokens) == 0:
        diagnostics.append(_diag("NO_ROOT", "ERROR", "sentence has no token with HEAD=0", sent))
    if len(head0_tokens) > 1:
        diagnostics.append(_diag("MULTIPLE_ROOTS", "ERROR", f"sentence has multiple HEAD=0 tokens: {[_tok_get(t, 'id') for t in head0_tokens]}", sent))
    if len(root_deprel_tokens) == 0:
        diagnostics.append(_diag("NO_ROOT_DEPREL", "ERROR", "sentence has no token with DEPREL=root", sent))
    if len(root_deprel_tokens) > 1:
        diagnostics.append(
            _diag("MULTIPLE_ROOT_DEPRELS", "ERROR", f"sentence has multiple DEPREL=root tokens: {[_tok_get(t, 'id') for t in root_deprel_tokens]}", sent)
        )
    diagnostics.extend(_tree_structure_diagnostics(sent, word_tokens, word_id_set))

    metadata_text = getattr(sent, "metadata", {}).get("text")
    if metadata_text:
        form_text = text_from_forms(sent)
        compact_form_text = text_from_forms(sent, consider_spaceafter=False)
        if metadata_text != form_text and metadata_text != compact_form_text:
            diagnostics.append(
                _diag("TEXT_FORM_MISMATCH", "WARNING", f"# text differs from token FORM reconstruction: {metadata_text!r} != {form_text!r}", sent)
            )

    return diagnostics


def _tree_structure_diagnostics(sent: Any, word_tokens: list[Any], word_id_set: set[int]) -> list[Diagnostic]:
    diagnostics = []
    by_id = {_tok_get(tok, "id"): tok for tok in word_tokens}

    for token in word_tokens:
        start_id = _tok_get(token, "id")
        current_id = start_id
        visited = []
        while current_id != 0:
            if current_id in visited:
                cycle = visited[visited.index(current_id):] + [current_id]
                diagnostics.append(_diag("CYCLE", "ERROR", f"HEAD chain contains a cycle: {cycle}", sent, token))
                break
            visited.append(current_id)
            current = by_id.get(current_id)
            if current is None:
                diagnostics.append(_diag("DISCONNECTED_TREE", "ERROR", "HEAD chain leaves the sentence", sent, token))
                break
            head = _tok_get(current, "head")
            if not isinstance(head, int):
                break
            if head != 0 and head not in word_id_set:
                break
            current_id = head
        else:
            continue
    return diagnostics


def corpus_consistency_diagnostics(
    sentences: Iterable[Any],
    min_count: int = 5,
    minority_ratio: float = 0.05,
    rare_upos_deprel_max_count: int = 1,
    deps_basic_edge_majority_ratio: float = 0.8,
) -> list[Diagnostic]:
    rows = []
    for sent in sentences:
        for token in sent:
            if not _is_word_id(_tok_get(token, "id")):
                continue
            rows.append((sent, token))

    diagnostics = []
    upos_deprel = Counter((_tok_get(tok, "upos"), _tok_get(tok, "deprel")) for _, tok in rows)

    rare_pairs = {
        pair
        for pair, count in upos_deprel.items()
        if count <= rare_upos_deprel_max_count and pair[0] in EXPECTED_DEPRELS_BY_UPOS
    }
    for sent, token in rows:
        pair = (_tok_get(token, "upos"), _tok_get(token, "deprel"))
        if pair in rare_pairs:
            diagnostics.append(
                _diag("RARE_UPOS_DEPREL", "WARNING", f"rare UPOS-DEPREL combination in corpus: {pair[0]} & {pair[1]}", sent, token)
            )

    deps_rows = []
    for sent, token in rows:
        head = _tok_get(token, "head")
        deprel = _tok_get(token, "deprel")
        deps, deps_errors = parse_deps(_tok_get(token, "deps"))
        if deps_errors or not deps or not isinstance(head, int):
            continue
        basic_edge_seen = any(dep_head == str(head) and _same_base_deprel(dep_rel, deprel) for dep_head, dep_rel in deps)
        deps_rows.append((sent, token, head, deprel, basic_edge_seen))

    if deps_rows:
        preserved_count = sum(1 for _, _, _, _, basic_edge_seen in deps_rows if basic_edge_seen)
        if preserved_count / len(deps_rows) >= deps_basic_edge_majority_ratio:
            for sent, token, head, deprel, basic_edge_seen in deps_rows:
                if not basic_edge_seen:
                    diagnostics.append(
                        _diag(
                            "BASIC_EDGE_MISSING_FROM_DEPS",
                            "WARNING",
                            f"DEPS does not preserve the basic edge {head}:{deprel}",
                            sent,
                            token,
                            field="deps",
                        )
                    )

    return diagnostics


def check_sentence_list(sentences: Iterable[Any], include_corpus_checks: bool = True) -> list[Diagnostic]:
    sentence_list = list(sentences)
    diagnostics = []
    sent_id_counts = Counter(getattr(sent, "metadata", {}).get("sent_id") for sent in sentence_list)
    for sent_id, count in sent_id_counts.items():
        if sent_id and count > 1:
            diagnostics.append(
                Diagnostic("DUPLICATE_SENT_ID", "ERROR", f"sent_id appears {count} times in checked corpus: {sent_id}", sent_id=sent_id)
            )
    for sent in sentence_list:
        diagnostics.extend(check_sentence(sent))
    if include_corpus_checks:
        diagnostics.extend(corpus_consistency_diagnostics(sentence_list))
    return diagnostics


def check_conllu_text(text: str, include_corpus_checks: bool = True) -> list[Diagnostic]:
    diagnostics = raw_format_diagnostics(text)
    try:
        sentences = parse(text)
    except Exception as exc:
        diagnostics.append(Diagnostic("PARSE_ERROR", "ERROR", f"CoNLL-U parser failed: {exc}"))
        return diagnostics
    diagnostics.extend(check_sentence_list(sentences, include_corpus_checks=include_corpus_checks))
    return diagnostics


def check_file(path: str, include_corpus_checks: bool = True) -> list[Diagnostic]:
    with open(path, "r", encoding="utf-8") as f:
        return check_conllu_text(f.read(), include_corpus_checks=include_corpus_checks)


def _markdown_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _diagnostic_location(diagnostic: Diagnostic) -> str:
    parts = []
    if diagnostic.sent_id != "NA":
        parts.append(f"sent={diagnostic.sent_id}")
    if diagnostic.token_id is not None:
        parts.append(f"tok={diagnostic.token_id}")
    if diagnostic.line is not None:
        parts.append(f"line={diagnostic.line}")
    return ", ".join(parts) or "NA"


def _display_source_path(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path.absolute())


def _load_report_input(data: Any, include_corpus_checks: bool = True) -> tuple[list[Any], list[Diagnostic], str, list[str]]:
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.exists() and path.is_dir():
            sentences = []
            diagnostics = []
            files = sorted(path.rglob("*.conllu"))
            source_files = [_display_source_path(file_path) for file_path in files]
            for file_path in files:
                text = file_path.read_text(encoding="utf-8")
                diagnostics.extend(raw_format_diagnostics(text))
                try:
                    parsed = list(parse(text))
                except Exception as exc:
                    diagnostics.append(Diagnostic("PARSE_ERROR", "ERROR", f"{file_path}: CoNLL-U parser failed: {exc}"))
                    continue
                sentences.extend(parsed)
            diagnostics.extend(check_sentence_list(sentences, include_corpus_checks=include_corpus_checks))
            return sentences, diagnostics, _display_source_path(path), source_files
        if path.exists() and path.is_file():
            text = path.read_text(encoding="utf-8")
            source_file = _display_source_path(path)
            diagnostics = raw_format_diagnostics(text)
            try:
                sentences = list(parse(text))
            except Exception as exc:
                diagnostics.append(Diagnostic("PARSE_ERROR", "ERROR", f"CoNLL-U parser failed: {exc}"))
                return [], diagnostics, source_file, [source_file]
            diagnostics.extend(check_sentence_list(sentences, include_corpus_checks=include_corpus_checks))
            return sentences, diagnostics, source_file, [source_file]
        text = str(data)
        diagnostics = raw_format_diagnostics(text)
        try:
            sentences = list(parse(text))
        except Exception as exc:
            diagnostics.append(Diagnostic("PARSE_ERROR", "ERROR", f"CoNLL-U parser failed: {exc}"))
            return [], diagnostics, "raw text", []
        diagnostics.extend(check_sentence_list(sentences, include_corpus_checks=include_corpus_checks))
        return sentences, diagnostics, "raw text", []

    if isinstance(data, TokenList):
        sentences = [data]
    elif isinstance(data, SentenceList):
        sentences = list(data)
    else:
        sentences = list(data)
    diagnostics = check_sentence_list(sentences, include_corpus_checks=include_corpus_checks)
    return sentences, diagnostics, type(data).__name__, []


def _collect_token_rows(sentences: Iterable[Any]) -> list[tuple[Any, Any]]:
    rows = []
    for sent in sentences:
        for token in sent:
            if _is_word_id(_tok_get(token, "id")):
                rows.append((sent, token))
    return rows


def _combination_status(upos: str, deprel: str, count: int, rare_max_count: int) -> str:
    base_deprel = _base_deprel(deprel)
    if not is_valid_deprel(deprel):
        return "non_ud"
    if (upos, deprel) in STRONG_UPOS_DEPREL_WARNINGS or (upos, base_deprel) in STRONG_UPOS_DEPREL_WARNINGS:
        return "suspicious"
    expected = EXPECTED_DEPRELS_BY_UPOS.get(upos)
    if expected and deprel not in expected and base_deprel not in {rel.split(":", 1)[0] for rel in expected}:
        return "atypical"
    if count <= rare_max_count:
        return "rare"
    return "common"


def _format_counter_table(title: str, counter: Counter, limit: int) -> list[str]:
    lines = [f"## {title}", "", "| Item | Count |", "|---|---:|"]
    for item, count in counter.most_common(limit):
        lines.append(f"| {_markdown_escape(item)} | {count} |")
    if not counter:
        lines.append("| NA | 0 |")
    lines.append("")
    return lines


DIAGNOSTIC_DESCRIPTIONS = {
    "ATYPICAL_UPOS_DEPREL": "UPOS-DEPREL combination falls outside the built-in expected mapping.",
    "BASIC_EDGE_MISSING_FROM_DEPS": "Enhanced dependencies do not preserve the corresponding basic dependency edge.",
    "COLUMN_COUNT": "Token line does not have the standard 10 CoNLL-U columns.",
    "CONJ_CHAIN_HEAD": "A conjunct depends on another conjunct instead of the first conjunct.",
    "CYCLE": "Basic HEAD chain contains a cycle.",
    "DEPREL_ROOT_MISMATCH": "DEPREL=root is used on a token whose HEAD is not 0.",
    "DEPS_HEAD_NOT_FOUND": "Enhanced dependency head references an absent node.",
    "DISCONNECTED_TREE": "Basic HEAD chain leaves the sentence instead of reaching the root.",
    "DUPLICATE_DEPS_EDGE": "DEPS contains duplicate enhanced dependency edges.",
    "DUPLICATE_ID": "Sentence contains duplicate word IDs.",
    "DUPLICATE_SENT_ID": "The same sent_id appears more than once.",
    "EMPTY_NODE_BASE_MISSING": "Empty node ID refers to a missing base token.",
    "EMPTY_NODE_ID": "Empty node suffix is malformed.",
    "EMPTY_NODE_SEQUENCE": "Empty node suffixes for the same base are not consecutive.",
    "ENHANCED_CONJ_CHAIN_HEAD": "Enhanced conj edge points to another conjunct instead of the first conjunct.",
    "ENHANCED_CONJ_RECIPROCAL": "Enhanced conj edges appear in both directions between the same two nodes.",
    "ENHANCED_ROOT_NONROOT_TOKEN": "DEPS contains 0:root for a token that is not the basic root.",
    "ENHANCED_SELF_LOOP": "DEPS contains an enhanced self-loop.",
    "HEAD_NOT_FOUND": "HEAD points to an absent word ID.",
    "HEAD_ROOT_MISMATCH": "HEAD=0 is used without DEPREL=root.",
    "HEAD_TYPE": "HEAD is not an integer in the basic tree.",
    "INVALID_DEPREL": "DEPREL is outside the UD standard relation inventory or has malformed subtype syntax.",
    "INVALID_DEPS_DEPREL": "DEPS relation is outside the UD standard relation inventory or has malformed subtype syntax.",
    "INVALID_UPOS": "UPOS is outside the UD universal POS inventory.",
    "MALFORMED_DEPS": "DEPS field is malformed.",
    "MALFORMED_FEATS": "FEATS field is malformed.",
    "MALFORMED_HEAD": "HEAD field is malformed in raw CoNLL-U.",
    "MALFORMED_ID": "ID field is malformed in raw CoNLL-U.",
    "MALFORMED_METADATA": "Metadata line is malformed.",
    "MISSING_SENTENCE_SEPARATOR": "Metadata starts before a blank sentence separator.",
    "MULTIPLE_DEPS_RELS_SAME_HEAD": "DEPS contains multiple relation labels from the same head to the same dependent.",
    "MULTIPLE_ROOT_DEPRELS": "Sentence has multiple tokens with DEPREL=root.",
    "MULTIPLE_ROOTS": "Sentence has multiple HEAD=0 tokens.",
    "MUTUALLY_SUSPICIOUS_FEATS": "A morphological feature has multiple values that are often mutually exclusive.",
    "MWT_EMPTY_HEAD": "Multiword or empty-node line has a basic HEAD where '_' is expected.",
    "MWT_MISSING_CONSTITUENT": "Multiword token range points to missing word IDs.",
    "MWT_OVERLAP": "Multiword token ranges overlap.",
    "MWT_RANGE_ORDER": "Multiword token range is reversed.",
    "NON_PUNCT_PUNCT": "DEPREL=punct is used on a non-PUNCT token.",
    "NO_ROOT": "Sentence has no HEAD=0 token.",
    "NO_ROOT_DEPREL": "Sentence has no token with DEPREL=root.",
    "OVER_SPECIFIED_DEPS_DEPREL": "Enhanced dependency relation has multiple subtype levels.",
    "PARSE_ERROR": "CoNLL-U parser failed.",
    "PUNCT_AS_HEAD": "Punctuation is used as a lexical head.",
    "PUNCT_AS_ROOT": "Punctuation is used as root.",
    "PUNCT_NON_PUNCT": "UPOS=PUNCT is used without DEPREL=punct.",
    "RARE_UPOS_DEPREL": "UPOS-DEPREL combination is valid but rare in this corpus.",
    "SELF_LOOP": "Token takes itself as HEAD.",
    "SUSPICIOUS_UPOS_DEPREL": "UPOS-DEPREL combination is in the strong-warning list.",
    "TEXT_FORM_MISMATCH": "# text differs from reconstruction from token FORM values.",
    "UPOS_FEATS_MISMATCH": "Morphological feature name is not in the UD universal feature inventory.",
    "WORD_ID_SEQUENCE": "Word IDs are not continuous and ordered from 1 to n.",
}


def _diagnostic_description(diagnostic: Diagnostic) -> str:
    return DIAGNOSTIC_DESCRIPTIONS.get(diagnostic.code, diagnostic.message)


def _format_locations(diagnostics: list[Diagnostic]) -> str:
    locations = [_diagnostic_location(diagnostic) for diagnostic in diagnostics]
    unique_locations = list(dict.fromkeys(locations))
    return "<br>".join(_markdown_escape(location) for location in unique_locations)


def _group_diagnostics(diagnostics: Iterable[Diagnostic]) -> list[tuple[str, str, str, str, list[Diagnostic]]]:
    groups: dict[tuple[str, str, str, str], list[Diagnostic]] = defaultdict(list)
    for diagnostic in diagnostics:
        key = (
            diagnostic.level,
            diagnostic.code,
            diagnostic.field or "",
            _diagnostic_description(diagnostic),
        )
        groups[key].append(diagnostic)
    level_order = {"ERROR": 0, "WARNING": 1}
    return [
        (level, code, field, message, items)
        for (level, code, field, message), items in sorted(
            groups.items(),
            key=lambda item: (
                level_order.get(item[0][0], 9),
                item[0][1],
                item[0][2],
                item[0][3],
            ),
        )
    ]


def _write_report_output(report_text: str, output_path: str | Path | None, output_format: str) -> None:
    if output_path is None:
        return
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format.lower() == "pdf":
        _write_pdf_report(report_text, output)
    elif output_format.lower() in {"md", "markdown"}:
        output.write_text(report_text, encoding="utf-8")
    else:
        raise ValueError("output_format must be 'md' or 'pdf'.")


def _write_pdf_report(markdown_text: str, output_path: Path) -> None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    except Exception as exc:
        raise RuntimeError("PDF output requires reportlab. Use output_format='md' or install reportlab.") from exc

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(output_path), pagesize=A4)
    story = []
    for line in markdown_text.splitlines():
        if not line.strip():
            story.append(Spacer(1, 6))
            continue
        if line.startswith("# "):
            story.append(Paragraph(line[2:], styles["Title"]))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:], styles["Heading2"]))
        elif line.startswith("### "):
            story.append(Paragraph(line[4:], styles["Heading3"]))
        elif line.startswith("|"):
            story.append(Paragraph(_markdown_escape(line), styles["Code"]))
        else:
            story.append(Paragraph(_markdown_escape(line), styles["BodyText"]))
    doc.build(story)


def generate_error_report(
    data: Any,
    output_path: str | Path | None = None,
    output_format: str = "md",
    include_corpus_checks: bool = True,
    top_n: int = 30,
    rare_upos_deprel_max_count: int = 2,
    examples_per_combo: int = 5,
) -> str:
    """
    Generate a Markdown or PDF error report for CoNLL-U diagnostics.

    The report includes error diagnostics plus a combination-level aggregation
    of non-UD, suspicious, rare, or atypical UPOS-DEPREL combinations. These
    combinations are also counted above as token-level diagnostics when their
    status is diagnostic; the review table groups them for manual inspection.

    Parameters:
        data: A CoNLL-U file path, folder path, raw CoNLL-U text, TokenList,
            SentenceList, or iterable of TokenList objects.
        output_path: Optional path for saving the report.
        output_format: "md" or "pdf". If output_path ends with ".pdf", PDF is
            used automatically.
        include_corpus_checks: Whether to include existing corpus consistency
            diagnostics.
        top_n: Maximum number of grouped diagnostic rows and review rows.
        rare_upos_deprel_max_count: Combination frequency threshold for marking
            UPOS-DEPREL pairs as rare.
        examples_per_combo: Number of token examples shown for each highlighted
            UPOS-DEPREL pair.

    Returns:
        The Markdown report text. PDF output also returns the source Markdown.
    """
    if output_path is not None:
        suffix = Path(output_path).suffix.lower()
        if suffix == ".pdf":
            output_format = "pdf"
        elif suffix in {".md", ".markdown"}:
            output_format = "md"

    sentences, diagnostics, source_name, source_files = _load_report_input(data, include_corpus_checks=include_corpus_checks)
    token_rows = _collect_token_rows(sentences)
    level_counts = Counter(d.level for d in diagnostics)
    code_counts = Counter(d.code for d in diagnostics)
    deprel_counts = Counter(_tok_get(tok, "deprel") for _, tok in token_rows)
    non_ud_deprel_counts = Counter(
        {deprel: count for deprel, count in deprel_counts.items() if not is_valid_deprel(deprel)}
    )
    combo_counts = Counter((_tok_get(tok, "upos"), _tok_get(tok, "deprel")) for _, tok in token_rows)

    combo_examples: dict[tuple[str, str], list[tuple[str, Any, str, int]]] = defaultdict(list)
    for sent, token in token_rows:
        pair = (_tok_get(token, "upos"), _tok_get(token, "deprel"))
        if len(combo_examples[pair]) >= examples_per_combo:
            continue
        combo_examples[pair].append(
            (
                getattr(sent, "metadata", {}).get("sent_id", "NA"),
                _tok_get(token, "id"),
                _tok_get(token, "form", ""),
                _tok_get(token, "head"),
            )
        )

    highlighted_combos = []
    for (upos, deprel), count in combo_counts.items():
        status = _combination_status(upos, deprel, count, rare_upos_deprel_max_count)
        if status != "common":
            highlighted_combos.append((status, upos, deprel, count))
    status_order = {"non_ud": 0, "suspicious": 1, "atypical": 2, "rare": 3}
    highlighted_combos.sort(key=lambda row: (status_order.get(row[0], 9), row[3], row[1], row[2]))
    selected_combos = []
    selected_keys = set()
    for status in status_order:
        first_for_status = next((row for row in highlighted_combos if row[0] == status), None)
        if first_for_status is not None and len(selected_combos) < top_n:
            selected_combos.append(first_for_status)
            selected_keys.add(first_for_status)
    for row in highlighted_combos:
        if len(selected_combos) >= top_n:
            break
        if row in selected_keys:
            continue
        selected_combos.append(row)
        selected_keys.add(row)

    lines = [
        "# CoNLL-U Error Report",
        "",
        f"- Source: `{_markdown_escape(source_name)}`",
        f"- Checked CoNLL-U path: `{_markdown_escape(source_name)}`",
        f"- Checked CoNLL-U files: {len(source_files) if source_files else 'NA'}",
        f"- Sentences: {len(sentences)}",
        f"- Tokens: {len(token_rows)}",
        f"- Diagnostics: {len(diagnostics)}",
        f"- Errors: {level_counts.get('ERROR', 0)}",
        f"- Warnings: {level_counts.get('WARNING', 0)}",
        "",
        "## Diagnostic Summary",
        "",
        "| Level | Count |",
        "|---|---:|",
    ]
    for level in ("ERROR", "WARNING"):
        lines.append(f"| {level} | {level_counts.get(level, 0)} |")
    lines.append("")
    lines.append("## Checked CoNLL-U Files")
    lines.append("")
    lines.append("| File |")
    lines.append("|---|")
    for source_file in source_files[:top_n]:
        lines.append(f"| `{_markdown_escape(source_file)}` |")
    if len(source_files) > top_n:
        lines.append(f"| ... {len(source_files) - top_n} more files omitted |")
    if not source_files:
        lines.append("| NA |")
    lines.append("")
    lines.extend(_format_counter_table("Diagnostic Codes", code_counts, top_n))

    lines.append("## Diagnostic Details")
    lines.append("")
    lines.append("Diagnostics are grouped by level, code, field, and issue type. The Locations column lists all matching locations in the report scope.")
    lines.append("")
    lines.append("| Level | Code | Count | Field | Issue | Locations |")
    lines.append("|---|---|---:|---|---|---|")
    grouped_diagnostics = _group_diagnostics(diagnostics)
    for level, code, field, message, items in grouped_diagnostics[:top_n]:
        lines.append(
            f"| {level} | {code} | {len(items)} | {_markdown_escape(field)} | "
            f"{_markdown_escape(message)} | {_format_locations(items)} |"
        )
    if len(grouped_diagnostics) > top_n:
        lines.append(f"| ... | ... | {len(grouped_diagnostics) - top_n} more groups omitted | ... | ... | ... |")
    if not diagnostics:
        lines.append("| NA | NA | 0 | NA | No diagnostics found | NA |")
    lines.append("")

    lines.append("## UPOS-DEPREL Combinations for Review")
    lines.append("")
    lines.append("These rows aggregate token-level diagnostics already counted above, plus any highlighted dependency-label combinations that need manual inspection.")
    lines.append("`non_ud` means the DEPREL label is outside the UD standard inventory; `suspicious` means the pair is in the strong-warning list; `atypical` means the pair falls outside the built-in expected UPOS-DEPREL mapping; `rare` means the pair is valid but appears at or below the corpus frequency threshold. When multiple labels apply, the report prioritizes non-UD, suspicious, and atypical statuses before rare.")
    lines.append("")
    lines.append("| Status | UPOS | DEPREL | Count | Examples |")
    lines.append("|---|---|---|---:|---|")
    for status, upos, deprel, count in selected_combos:
        examples = "; ".join(
            f"{sent_id}:{token_id}:{form}->h{head}"
            for sent_id, token_id, form, head in combo_examples[(upos, deprel)]
        )
        lines.append(f"| {status} | {upos} | {_markdown_escape(deprel)} | {count} | {_markdown_escape(examples)} |")
    if not highlighted_combos:
        lines.append("| NA | NA | NA | 0 | No highlighted combinations |")
    lines.append("")

    lines.append("## Non-UD DEPREL Labels")
    lines.append("")
    lines.append("These dependency labels are not in the UD standard relation inventory, though they may occur in converted or legacy corpora.")
    lines.append("")
    lines.append("| DEPREL | Count |")
    lines.append("|---|---:|")
    for deprel, count in non_ud_deprel_counts.most_common(top_n):
        lines.append(f"| {_markdown_escape(deprel)} | {count} |")
    if not non_ud_deprel_counts:
        lines.append("| NA | 0 |")
    lines.append("")

    report_text = "\n".join(lines)
    _write_report_output(report_text, output_path, output_format)

    return report_text


def generate_summary_report(
    data: Any,
    output_path: str | Path | None = None,
    output_format: str = "md",
    top_n: int = 50,
    rare_upos_deprel_max_count: int = 2,
) -> str:
    """
    Generate a Markdown or PDF corpus summary report for CoNLL-U data.

    This report focuses on descriptive corpus statistics rather than diagnostic
    error checking. Use generate_error_report for error and warning reports.
    """
    if output_path is not None:
        suffix = Path(output_path).suffix.lower()
        if suffix == ".pdf":
            output_format = "pdf"
        elif suffix in {".md", ".markdown"}:
            output_format = "md"

    sentences, diagnostics, source_name, source_files = _load_report_input(data, include_corpus_checks=False)
    token_rows = _collect_token_rows(sentences)
    upos_counts = Counter(_tok_get(tok, "upos") for _, tok in token_rows)
    deprel_counts = Counter(_tok_get(tok, "deprel") for _, tok in token_rows)
    root_upos_counts = Counter(_tok_get(tok, "upos") for _, tok in token_rows if _tok_get(tok, "head") == 0)
    combo_counts = Counter((_tok_get(tok, "upos"), _tok_get(tok, "deprel")) for _, tok in token_rows)

    lines = [
        "# CoNLL-U Corpus Summary",
        "",
        f"- Source: `{_markdown_escape(source_name)}`",
        f"- Checked CoNLL-U path: `{_markdown_escape(source_name)}`",
        f"- Checked CoNLL-U files: {len(source_files) if source_files else 'NA'}",
        f"- Sentences: {len(sentences)}",
        f"- Tokens: {len(token_rows)}",
        "",
        "## Checked CoNLL-U Files",
        "",
        "| File |",
        "|---|",
    ]
    for source_file in source_files[:top_n]:
        lines.append(f"| `{_markdown_escape(source_file)}` |")
    if len(source_files) > top_n:
        lines.append(f"| ... {len(source_files) - top_n} more files omitted |")
    if not source_files:
        lines.append("| NA |")
    lines.append("")

    lines.extend(_format_counter_table("UPOS Distribution", upos_counts, top_n))
    lines.extend(_format_counter_table("DEPREL Distribution", deprel_counts, top_n))
    lines.extend(_format_counter_table("Root UPOS Distribution", root_upos_counts, top_n))

    lines.append("## Full UPOS-DEPREL Frequency Table")
    lines.append("")
    lines.append("| UPOS | DEPREL | Count | Status |")
    lines.append("|---|---|---:|---|")
    for (upos, deprel), count in sorted(combo_counts.items(), key=lambda item: (-item[1], str(item[0][0]), str(item[0][1])))[:top_n * 3]:
        status = _combination_status(upos, deprel, count, rare_upos_deprel_max_count)
        lines.append(f"| {upos} | {_markdown_escape(deprel)} | {count} | {status} |")
    lines.append("")

    if diagnostics:
        lines.append("## Parse Diagnostics")
        lines.append("")
        lines.append("Parsing produced diagnostics while loading the corpus. Use generate_error_report for full error-checking output.")
        lines.append("")
        lines.extend(_format_counter_table("Diagnostic Codes", Counter(d.code for d in diagnostics), top_n))

    report_text = "\n".join(lines)
    _write_report_output(report_text, output_path, output_format)

    return report_text


def generate_analysis_report(*args, **kwargs) -> str:
    """Backward-compatible alias for the diagnostic analysis report."""
    return generate_error_report(*args, **kwargs)


def iter_error_tuples(diagnostics: Iterable[Diagnostic]) -> Iterator[tuple[str, str, str, str, Any, int | None, str | None]]:
    for diagnostic in diagnostics:
        yield diagnostic.as_tuple()


def two_unidirectional_depr(token, sent):
    deps, errors = parse_deps(_tok_get(token, "deps"))
    if errors:
        return False
    return any(dep1[1] == dep2[1] and dep1 != dep2 for dep1 in deps for dep2 in deps)


def punct_as_root(token, sent, enh=False):
    if enh is False:
        return _tok_get(token, "head") == 0 and _tok_get(token, "upos") == "PUNCT"
    return False


def head_deprel_mismatch(token, sent, enh=False):
    if enh is False:
        head = _tok_get(token, "head")
        deprel = _tok_get(token, "deprel")
        return (head == 0 and deprel != "root") or (head != 0 and deprel == "root")
    return False


def upos_deprel_mismatch(token, sent):
    upos = _tok_get(token, "upos")
    deprel = _tok_get(token, "deprel")
    return (upos == "PUNCT" and deprel != "punct") or (upos != "PUNCT" and deprel == "punct")


def find_id_equals_head_errors(sentences):
    """
    Given a SentenceList, return all tokens where 'id' == 'head'.
    Returns a list of tuples: (newdoc_id, sent_id, token_id)
    """
    doc_id = getattr(sentences, "metadata", {}).get("newdoc id", "NA")
    errors = []
    for sent in sentences:
        sent_id = getattr(sent, "metadata", {}).get("sent_id", "NA")
        for token in sent:
            if _is_word_id(_tok_get(token, "id")) and _tok_get(token, "head") == _tok_get(token, "id"):
                errors.append((doc_id, sent_id, _tok_get(token, "id")))
    return errors


def exists_error(sent):
    return any(diagnostic.level == "ERROR" for diagnostic in check_sentence(sent))
