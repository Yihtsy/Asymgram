from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re
from typing import Any, Iterable, Iterator

from .conllu import parse, parse_incr


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

UPOS_FEATS = {
    "PUNCT": set(),
    "SYM": set(),
    "ADP": {"AdpType", "Case", "Gender", "Number", "Person", "Polarity", "PronType"},
    "AUX": {
        "Aspect",
        "Clusivity",
        "Evident",
        "Mood",
        "Number",
        "Person",
        "Polarity",
        "Tense",
        "VerbForm",
        "Voice",
    },
    "VERB": {
        "Aspect",
        "Clusivity",
        "Evident",
        "Mood",
        "Number",
        "Person",
        "Polarity",
        "Tense",
        "VerbForm",
        "Voice",
    },
    "ADJ": {"Case", "Degree", "Gender", "Number", "NumType", "Polarity", "VerbForm"},
    "ADV": {"Degree", "Polarity", "PronType"},
    "DET": {"Case", "Definite", "Gender", "Number", "NumType", "Person", "Poss", "PronType"},
    "NUM": {"Case", "Gender", "Number", "NumForm", "NumType"},
    "PRON": {
        "Case",
        "Clusivity",
        "Definite",
        "Gender",
        "Number",
        "Person",
        "Poss",
        "PronType",
        "Reflex",
    },
    "NOUN": {"Animacy", "Case", "Definite", "Gender", "Number"},
    "PROPN": {"Animacy", "Case", "Definite", "Gender", "NameType", "Number"},
    "PART": {"PartType", "Polarity", "PronType"},
    "CCONJ": {"ConjType"},
    "SCONJ": {"PronType"},
    "INTJ": {"Polarity"},
    "X": set(),
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
    ("NUM", "nmod"),
    ("ADV", "amod"),
    ("ADJ", "advmod"),
    ("AUX", "amod"),
    ("AUX", "nmod"),
    ("AUX", "det"),
}

EXPECTED_DEPRELS_BY_UPOS = {
    "ADP": {"case", "fixed", "mark"},
    "DET": {"det", "fixed"},
    "CCONJ": {"cc", "fixed"},
    "SCONJ": {"mark", "fixed"},
    "PUNCT": {"punct"},
    "NUM": {"nummod", "compound", "flat", "root", "conj"},
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
                parsed.append((str(head), str(rel)))
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
    for token in sent:
        if not _is_word_id(_tok_get(token, "id")):
            continue
        parts.append(str(_tok_get(token, "form", "")))
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
    sent_ids = Counter()
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
                elif line.startswith("# sent_id"):
                    sent_id = line.split("=", 1)[1].strip()
                    sent_ids[sent_id] += 1
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

    for sent_id, count in sent_ids.items():
        if count > 1:
            diagnostics.append(
                Diagnostic("DUPLICATE_SENT_ID", "ERROR", f"sent_id appears {count} times: {sent_id}")
            )

    return diagnostics


def check_sentence(sent: Any) -> list[Diagnostic]:
    diagnostics = []
    word_tokens = [tok for tok in sent if _is_word_id(_tok_get(tok, "id"))]
    word_ids = [_tok_get(tok, "id") for tok in word_tokens]
    word_id_set = set(word_ids)
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
            head_token = next((tok for tok in word_tokens if _tok_get(tok, "id") == head), None)
            if head_token is not None and _tok_get(head_token, "upos") == "PUNCT" and upos != "PUNCT":
                diagnostics.append(_diag("PUNCT_AS_HEAD", "WARNING", "punctuation is used as a lexical head", sent, token))

        base_deprel = _base_deprel(deprel)
        if (upos, deprel) in STRONG_UPOS_DEPREL_WARNINGS or (upos, base_deprel) in STRONG_UPOS_DEPREL_WARNINGS:
            diagnostics.append(_diag("SUSPICIOUS_UPOS_DEPREL", "WARNING", f"suspicious UPOS-DEPREL combination: {upos} & {deprel}", sent, token))
        expected = EXPECTED_DEPRELS_BY_UPOS.get(upos)
        if expected and deprel not in expected and base_deprel not in {rel.split(':', 1)[0] for rel in expected}:
            diagnostics.append(_diag("ATYPICAL_UPOS_DEPREL", "INFO", f"atypical UPOS-DEPREL combination: {upos} & {deprel}", sent, token))

        feats, feat_errors = parse_feats(_tok_get(token, "feats"))
        for error in feat_errors:
            diagnostics.append(_diag("MALFORMED_FEATS", "ERROR", error, sent, token, field="feats"))
        allowed_feats = UPOS_FEATS.get(upos)
        if allowed_feats is not None:
            for feature_name in feats:
                if feature_name not in allowed_feats and allowed_feats:
                    diagnostics.append(
                        _diag("UPOS_FEATS_MISMATCH", "WARNING", f"{feature_name} is unusual for UPOS={upos}", sent, token, field="feats")
                    )
                if not allowed_feats and feature_name:
                    diagnostics.append(
                        _diag("UPOS_FEATS_MISMATCH", "WARNING", f"morphological feature is unusual for UPOS={upos}", sent, token, field="feats")
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
        for dep_head, dep_rel in deps:
            if dep_head == str(token_id):
                diagnostics.append(_diag("ENHANCED_SELF_LOOP", "ERROR", "DEPS contains an enhanced self-loop", sent, token, field="deps"))
            if dep_head != "0":
                base = int(dep_head.split(".", 1)[0])
                if base not in word_id_set:
                    diagnostics.append(_diag("DEPS_HEAD_NOT_FOUND", "ERROR", f"DEPS references absent node {dep_head}", sent, token, field="deps"))
            if not is_valid_deprel(dep_rel):
                diagnostics.append(_diag("INVALID_DEPS_DEPREL", "ERROR", f"invalid DEPS relation: {dep_rel}", sent, token, field="deps"))

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
    for token in head0_tokens:
        upos = _tok_get(token, "upos")
        if upos not in {"VERB", "AUX"}:
            diagnostics.append(_diag("ATYPICAL_ROOT_UPOS", "INFO", f"non-verbal root requires manual inspection: UPOS={upos}", sent, token))

    diagnostics.extend(_tree_structure_diagnostics(sent, word_tokens, word_id_set))

    metadata_text = getattr(sent, "metadata", {}).get("text")
    if metadata_text:
        form_text = text_from_forms(sent)
        compact_form_text = "".join(str(_tok_get(tok, "form", "")) for tok in word_tokens)
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
) -> list[Diagnostic]:
    rows = []
    for sent in sentences:
        for token in sent:
            if not _is_word_id(_tok_get(token, "id")):
                continue
            rows.append((sent, token))

    diagnostics = []
    upos_deprel = Counter((_tok_get(tok, "upos"), _tok_get(tok, "deprel")) for _, tok in rows)
    form_upos = defaultdict(Counter)
    lemma_upos = defaultdict(Counter)
    lemma_deprel = defaultdict(Counter)

    for _, token in rows:
        form_upos[str(_tok_get(token, "form", "")).lower()][_tok_get(token, "upos")] += 1
        lemma_upos[str(_tok_get(token, "lemma", "")).lower()][_tok_get(token, "upos")] += 1
        lemma_deprel[str(_tok_get(token, "lemma", "")).lower()][_tok_get(token, "deprel")] += 1

    rare_pairs = {
        pair
        for pair, count in upos_deprel.items()
        if count <= rare_upos_deprel_max_count and pair[0] in EXPECTED_DEPRELS_BY_UPOS
    }
    for sent, token in rows:
        pair = (_tok_get(token, "upos"), _tok_get(token, "deprel"))
        if pair in rare_pairs:
            diagnostics.append(
                _diag("RARE_UPOS_DEPREL", "INFO", f"rare UPOS-DEPREL combination in corpus: {pair[0]} & {pair[1]}", sent, token)
            )

    def add_minority_warnings(table: dict[str, Counter], code: str, label: str) -> None:
        for key, counts in table.items():
            total = sum(counts.values())
            if key in {"", "_"} or total < min_count or len(counts) < 2:
                continue
            majority = counts.most_common(1)[0][0]
            for value, count in counts.items():
                if value == majority:
                    continue
                if count / total <= minority_ratio:
                    for sent, token in rows:
                        token_key = str(_tok_get(token, label, "")).lower()
                        observed = _tok_get(token, "deprel" if code.endswith("DEPREL") else "upos")
                        if token_key == key and observed == value:
                            diagnostics.append(
                                _diag(code, "INFO", f"{label}={key!r} has minority annotation {value!r} ({count}/{total})", sent, token)
                            )

    add_minority_warnings(form_upos, "FORM_UPOS_INCONSISTENCY", "form")
    add_minority_warnings(lemma_upos, "LEMMA_UPOS_INCONSISTENCY", "lemma")
    add_minority_warnings(lemma_deprel, "LEMMA_DEPREL_INCONSISTENCY", "lemma")
    return diagnostics


def check_sentence_list(sentences: Iterable[Any], include_corpus_checks: bool = True) -> list[Diagnostic]:
    sentence_list = list(sentences)
    diagnostics = []
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
