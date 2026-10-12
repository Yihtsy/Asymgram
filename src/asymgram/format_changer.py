import os
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from .conllu import (
    CONLLUB_COLUMNS,
    CONLLUC_COLUMNS,
    ConlluBasicSentenceList,
    ConlluBasicTokenList,
    ConlluCoreSentenceList,
    ConlluCoreTokenList,
    SentenceLists,
    SentenceList,
    Token,
    TokenList,
    parse,
    parse_incr,
)
from typing import Generator

XML_CORPUS_PROFILES = {
    "generic": {
        "sentence_tags": ("s", "sentence", "seg"),
        "token_tags": ("w", "word", "tok", "token", "c", "pc"),
        "form_attrs": ("form", "word", "orth"),
        "lemma_attrs": ("lemma", "lem", "hw"),
        "xpos_attrs": ("pos", "tag", "c5", "msd"),
        "upos_attrs": ("upos",),
        "punct_tags": ("c", "pc"),
    },
    "bnc": {
        "sentence_tags": ("s",),
        "token_tags": ("w", "c"),
        "form_attrs": ("form",),
        "lemma_attrs": ("lemma", "hw"),
        "xpos_attrs": ("c5", "pos"),
        "upos_attrs": ("upos", "class"),
        "punct_tags": ("c",),
    },
    "xces": {
        "sentence_tags": ("s",),
        "token_tags": ("w", "c"),
        "form_attrs": ("form", "orth"),
        "lemma_attrs": ("lemma", "lem"),
        "xpos_attrs": ("ana", "pos", "msd"),
        "upos_attrs": ("upos",),
        "punct_tags": ("c",),
    },
    "tei": {
        "sentence_tags": ("s",),
        "token_tags": ("w", "pc"),
        "form_attrs": ("form",),
        "lemma_attrs": ("lemma",),
        "xpos_attrs": ("pos", "ana"),
        "upos_attrs": ("upos",),
        "punct_tags": ("pc",),
    },
}

# CHAT notation reference. Entries are descriptive and do not by themselves
# control conversion or cleaning behavior. Symbols marked as corpus-specific
# were observed in local DementiaBank Chou or Delaware CHAT files.
# Reference: MacWhinney, B. (2000). The CHILDES Project: Tools for Analyzing
# Talk (3rd ed., Vol. 1: Transcription Format and Programs). Mahwah, NJ:
# Lawrence Erlbaum Associates.
CHAT_SPECIAL_SYMBOLS = {
    # Special-form suffixes documented by MacWhinney (2000), chapter 5.
    "@b": "babbling or sound play",
    "@c": "child-invented form",
    "@d": "dialect form",
    "@f": "family-specific form",
    "@fp": "filled pause",
    "@i": "interjection or interactional form",
    "@l": "letter name or letter form",
    "@n": "neologism",
    "@o": "onomatopoeic form",
    "@p": "phonologically consistent form with unclear meaning",
    "@pr": "phrasal repetition",
    "@s": "second-language form",
    "@sc": "form reduced to schwa",
    "@sl": "signed-language form",
    "@sas": "simultaneous sign and speech",
    "@t": "test word or investigator-created nonce form",
    "@u": "UNIBET or other phonological transcription on the main tier",
    "@x": "word excluded from selected analyses",
    "@wp": "word play",
    "@g": "general special form",
    "@q": "corpus-specific special-form suffix; consult corpus documentation",
    "@k": "corpus-specific special-form suffix; consult corpus documentation",
    "@si": "corpus-specific special-form suffix; consult corpus documentation",

    # Unclear, untranscribed, nonverbal, and fragmentary material.
    "xxx": "unintelligible speech excluded from word counts",
    "xx": "unintelligible speech counted as one word",
    "yyy": "unintelligible speech represented on a %pho tier and excluded from word counts",
    "yy": "unintelligible speech represented on a %pho tier and counted as one word",
    "www": "material deliberately left untranscribed, normally explained on %exp",
    "0": "action without speech or an attachment point for a dependent tier",
    "&text": "phonological fragment or incomplete/nonlexical form",
    "&-text": "modern CHAT filled-pause or fragment notation observed in corpora",
    "&=event": "non-speech event such as &=laughs or &=coughs observed in corpora",

    # Pauses and prosodic marking. Hash forms are used in the 2000 manual;
    # parenthesized-dot forms are common in the local modern CHAT corpora.
    "#": "unfilled pause",
    "##": "longer unfilled pause",
    "###": "very long unfilled pause",
    "#N": "pause with an estimated or measured duration",
    "#d": "disfluent pause",
    "(.)": "short pause in modern CHAT notation",
    "(..)": "medium pause in modern CHAT notation",
    "(...)": "long pause in modern CHAT notation",
    "-:": "lengthening or drawling of the preceding word",
    "[!]": "stress on the preceding word or scoped material",
    "[!!]": "contrastive stress on the preceding word or scoped material",

    # Basic and special utterance terminators or linkers.
    ".": "declarative utterance terminator",
    "?": "question utterance terminator",
    "!": "imperative or emphatic utterance terminator",
    "+...": "trailing off or incomplete utterance",
    "+..?": "trailing off of a question",
    "+!?": "question with exclamatory force",
    "+/.": "interruption",
    "+/?": "interruption of a question",
    "+//.": "self-interruption",
    "+//?": "self-interruption of a question",
    "+^": "quick uptake after the previous speaker",
    "+<": "unscoped or lazy overlap with the previous turn",
    "+,": "self-completion after interruption",
    "++": "completion of another speaker's utterance",
    "[c]": "clause delimiter",

    # Scoping, replacement, uncertainty, errors, retracing, and overlap.
    "<text>": "explicit scope for a following bracketed CHAT code",
    "[= text]": "explanation of the preceding word or scoped material",
    "[=! text]": "paralinguistic or prosodic description",
    "[: text]": "replacement or normalized target text",
    "[:=x text]": "translation from language x",
    "[=? text]": "alternative transcription",
    "[%xxx: text]": "dependent-tier material embedded on the main tier",
    "[% text]": "comment embedded on the main tier",
    "[$text]": "project code embedded on the main tier",
    "[?]": "best-guess or uncertain transcription",
    "[*]": "error marking; an optional error code may follow the asterisk",
    "[/]": "retracing or repetition without correction",
    "[//]": "retracing with correction",
    "[///]": "retracing with full reformulation",
    "[/-]": "false start without retracing",
    "[/?]": "unclear retracing type",
    "[>]": "scoped overlap followed by the next speaker's overlap",
    "[<]": "scoped overlap preceded by the previous speaker's overlap",
    "[<>]": "scope overlaps both preceding and following speakers",
    "[>N] / [<N]": "numbered overlap pairing",
    "[+ code]": "utterance-level postcode; the code is project-defined",

    # Media alignment marks are delimited by control character U+0015.
    "\x15START_END\x15": "media time alignment in milliseconds",
}

_CLITIC_TO_FORM = {
    "be&3S": "'s",
    "not": "n't",
    "to": "to",
    "be&PRES": "'re",
    "be&1S": "'m",
    "have": "'ve",
    "us": "'s",
    "genmod": "'d",
    "will": "'ll",
}

_EMPTY_WORD_MOR_TO_FORM = {
    "not": "n't",
    "be": "'s",
    "have": "'ve",
    "will": "'ll",
}

_CHA_POS_TO_CONLLU_UPOS = {
    "n": "NOUN",
    "noun": "NOUN",
    "v": "VERB",
    "verb": "VERB",
    "part": "PART",
    "det:art": "DET",
    "det:dem": "DET",
    "det:poss": "DET",
    "det": "DET",
    "cop": "AUX",
    "mod": "AUX",
    "aux": "AUX",
    "adj": "ADJ",
    "adv": "ADV",
    "adv:tem": "ADV",
    "det:num": "NUM",
    "qn": "NUM",
    "num": "NUM",
    "intj": "INTJ",
    "meta": "X",
    "prep": "ADP",
    "adp": "ADP",
    "pro:sub": "PRON",
    "pro:obj": "PRON",
    "pro:indef": "PRON",
    "pro:rel": "PRON",
    "pro:per": "PRON",
    "pro:exist": "PRON",
    "pron": "PRON",
    "propn": "PROPN",
    "co": "PART",
    "inf": "PART",
    "neg": "PART",
    "coord": "CCONJ",
    "cconj": "CCONJ",
    "conj": "SCONJ",
    "sconj": "SCONJ",
    "punct": "PUNCT",
    "cm": "PUNCT",
    "over#part": "VERB",
    "n:let": "SYM",
    "neo": "X",
    "on": "INTJ",
    "sing": "X",
    "x": "X",
    "": "PUNCT",
}

_CHAT_TEXT_NO_SPACE_PATTERNS = (
    r"\s*\[\+ exc\]",
    r"\s*\[/\]",
    r"\s*\[//\]",
    r"\s*\+\.\.\.",
    r"\s*\+//\.?",
    r"\s*\+(?:\"|/|\.|\?)+",
)

_CHAT_TEXT_INLINE_PATTERNS = (
    r"@q",
)

_CHAT_UNKNOWN_WORDS = frozenset({"xx", "xxx", "yy", "yyy", "www"})
_CHAT_CONTROL_TOKENS = frozenset({"‡", "„"})
_CHAT_SPECIAL_FORM_SUFFIX_RE = re.compile(
    r"(?<=\S)@[A-Za-z][A-Za-z0-9:_-]*(?=\s|[.!?,;:]|$)",
    flags=re.IGNORECASE,
)

_MOR_FEATURES = {
    "Acc": ("Case", "Acc"),
    "Art": ("PronType", "Art"),
    "Cmp": ("Degree", "Cmp"),
    "Def": ("Definite", "Def"),
    "Dem": ("PronType", "Dem"),
    "Emp": ("Emph", "Yes"),
    "Fem": ("Gender", "Fem"),
    "Gen": ("Case", "Gen"),
    "Ger": ("VerbForm", "Ger"),
    "Indef": ("PronType", "Ind"),
    "Inf": ("VerbForm", "Inf"),
    "Int": ("PronType", "Int"),
    "Masc": ("Gender", "Masc"),
    "Nom": ("Case", "Nom"),
    "Past": ("Tense", "Past"),
    "Plur": ("Number", "Plur"),
    "Pres": ("Tense", "Pres"),
    "Prs": ("PronType", "Prs"),
    "Ptan": ("Number", "Ptan"),
    "Rcp": ("PronType", "Rcp"),
    "Rel": ("PronType", "Rel"),
    "Sing": ("Number", "Sing"),
    "Sup": ("Degree", "Sup"),
    "Tot": ("PronType", "Tot"),
    "reflx": ("Reflex", "Yes"),
}


def remove_timer_marks(text: str) -> str:
    timer_marks_regex = re.compile(r"\x15-?(\d+)_(\d+)-?\x15")
    return timer_marks_regex.sub("", text).rstrip()


def _replace_chat_replacements(text: str) -> str:
    scoped_replacement = re.compile(r"<[^<>]*>\s*\[:\s*([^\]]+)\]")
    word_replacement = re.compile(r"(?<!\S)\S+\s*\[:\s*([^\]]+)\]")
    previous = None
    while previous != text:
        previous = text
        text = scoped_replacement.sub(lambda match: match.group(1), text)
        text = word_replacement.sub(lambda match: match.group(1), text)
    return text


def _remove_repetition_marks(text: str) -> str:
    previous_item_repetition_regex = re.compile(
        r"(^|\s)(?:<[^<>]*>|\S+)(?:\s+\[(?!/{1,3}\])[^\]]*\])*"
        r"\s+\[(?:/|//|///)\]"
    )
    previous = None
    while previous != text:
        previous = text
        text = previous_item_repetition_regex.sub(r"\1", text)
    return text


def clean_chat_text(text: str) -> str:
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = remove_timer_marks(text)
    text = _replace_chat_replacements(text)
    text = _remove_repetition_marks(text)
    text = re.sub(r"<([^<>]*)>", r" \1 ", text)
    text = re.sub(r"(?<!\S)&\S*", " ", text)
    text = re.sub(r"(?<!\S)(?:xx|xxx|yy|yyy|www)(?!\S)", " ", text, flags=re.IGNORECASE)
    text = _CHAT_SPECIAL_FORM_SUFFIX_RE.sub("", text)
    text = re.sub(r"(?<!\S)@[A-Za-z][A-Za-z0-9]*(?!\S)", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\(\.{1,3}\)", " ", text)
    text = text.replace("‡", " ").replace("„", " ")
    for pattern in _CHAT_TEXT_NO_SPACE_PATTERNS:
        text = re.sub(pattern, "", text)
    for pattern in _CHAT_TEXT_INLINE_PATTERNS:
        text = re.sub(pattern, "", text)
    text = text.replace("<", " ").replace(">", " ")
    return re.sub(r"\s+", " ", text).strip()


def _split_cha_mor(mor: str) -> tuple[str, list[str]]:
    if not mor:
        return "_", []
    if "-" not in mor:
        return mor, []
    lemma, *tags = mor.split("-")
    return lemma or "_", tags


def _cha_mor_tags_to_feats(tags: list[str], upos: str, lemma: str = "") -> dict[str, str] | str:
    feats = {}
    for tag in tags:
        if tag == "Fin":
            feats["VerbForm"] = "Fin"
            continue
        if tag == "Imp":
            feats["Mood"] = "Imp"
            continue
        if tag == "Part":
            feats["VerbForm"] = "Part"
            continue
        if tag == "S":
            feats["Number"] = "Sing"
            continue
        if tag == "P":
            feats["Number"] = "Plur"
            continue
        if tag in {"P1", "P2", "P3"}:
            feats["Number"] = "Plur"
            if upos in {"PRON", "VERB", "AUX"}:
                feats["Person"] = tag[-1]
            continue
        if tag == "Neg":
            feats["Polarity"] = "Neg"
            continue
        if tag == "Ind":
            if upos in {"VERB", "AUX"}:
                feats["Mood"] = "Ind"
            else:
                feats["Definite"] = "Ind"
            continue
        if tag in {"S1", "S2", "S3"}:
            if upos in {"VERB", "AUX"}:
                feats["Number"] = "Sing"
                feats["Person"] = tag[-1]
            elif upos == "PRON":
                feats["Number"] = "Sing"
                lemma_person = {
                    "i": "1",
                    "me": "1",
                    "my": "1",
                    "mine": "1",
                    "we": "1",
                    "us": "1",
                    "our": "1",
                    "you": "2",
                    "your": "2",
                    "he": "3",
                    "him": "3",
                    "his": "3",
                    "she": "3",
                    "her": "3",
                    "it": "3",
                    "its": "3",
                    "they": "3",
                    "them": "3",
                    "their": "3",
                }.get(lemma.lower())
                if lemma_person is not None:
                    feats["Person"] = lemma_person
            continue
        name_value = _MOR_FEATURES.get(tag)
        if name_value is not None:
            name, value = name_value
            feats[name] = value
    return feats or "_"


def _cha_mor_tags_to_misc(tags: list[str]) -> dict[str, str] | str:
    if "irr" in tags:
        return {"IrregularVerb": "Yes"}
    return "_"


def _merge_misc(*values) -> dict[str, str] | str:
    merged = {}
    for value in values:
        if isinstance(value, dict):
            merged.update(value)
    return merged or "_"


def _cha_surface_form(token, lemma: str) -> str:
    form = token.word
    if form == "POSTCLITIC":
        return _CLITIC_TO_FORM.get(token.mor, token.word)
    if form:
        if form.endswith("n't") and lemma and lemma != "_":
            return lemma
        if form.endswith("'s") and lemma and lemma != "_":
            return lemma
        return form
    return _EMPTY_WORD_MOR_TO_FORM.get(lemma, lemma)


def _strip_chat_special_form_suffix(value: str) -> str:
    return _CHAT_SPECIAL_FORM_SUFFIX_RE.sub("", value or "")


def _is_chat_skip_token(
    form: str,
    lemma: str,
    xpos: str = "",
    deprel: str = "",
) -> bool:
    values = {str(form or "").strip(), str(lemma or "").strip()}
    lowered = {value.lower() for value in values}
    if any(value.startswith(("+", "&")) for value in values):
        return True
    if lowered & _CHAT_UNKNOWN_WORDS or values & _CHAT_CONTROL_TOKENS:
        return True
    if str(xpos or "").lower() == "beg" or str(deprel or "").lower() == "begp":
        return True
    return any(re.fullmatch(r"\[[^\]]*\]", value) is not None for value in values)


def _parse_wor_tier(wor_text: str) -> list[dict[str, str | None]]:
    if not wor_text:
        return []
    item_regex = re.compile(r"(\S+)(?:\s+\x15-?(\d+)_(\d+)-?\x15)?")
    items = []
    for match in item_regex.finditer(wor_text):
        word = match.group(1)
        if word.startswith("\x15"):
            continue
        items.append({"word": word, "start": match.group(2), "end": match.group(3)})
    for index, item in enumerate(items):
        if not item.get("start") or not item.get("end"):
            continue
        previous_index = index - 1
        while previous_index >= 0:
            previous_item = items[previous_index]
            if previous_item.get("start") or previous_item.get("end") or _is_chat_punctuation_word(previous_item["word"]):
                break
            previous_item["start"] = item["start"]
            previous_item["end"] = item["end"]
            previous_index -= 1
    previous_timing = None
    for item in items:
        if item.get("start") and item.get("end"):
            previous_timing = item
            continue
        if previous_timing is not None and not _is_chat_punctuation_word(item["word"]):
            item["start"] = previous_timing["start"]
            item["end"] = previous_timing["end"]
    return items


def _is_chat_punctuation_word(word: str | None) -> bool:
    return bool(word) and re.fullmatch(r"[^\w]+", word) is not None


def _normalize_cha_alignment_word(word: str | None) -> str:
    word = (word or "").strip().lower()
    return word.replace(":", "").replace("$co", "")


def _align_cha_words(token_words: list[str], wor_words: list[str]) -> dict[int, int]:
    token_count = len(token_words)
    wor_count = len(wor_words)
    scores = [[0] * (wor_count + 1) for _ in range(token_count + 1)]

    for token_index in range(token_count - 1, -1, -1):
        for wor_index in range(wor_count - 1, -1, -1):
            best = max(scores[token_index + 1][wor_index], scores[token_index][wor_index + 1])
            if token_words[token_index] == wor_words[wor_index]:
                best = max(best, 1 + scores[token_index + 1][wor_index + 1])
            scores[token_index][wor_index] = best

    mapping = {}
    token_index = 0
    wor_index = 0
    while token_index < token_count and wor_index < wor_count:
        if scores[token_index][wor_index + 1] == scores[token_index][wor_index]:
            wor_index += 1
            continue
        if (
            token_words[token_index] == wor_words[wor_index]
            and 1 + scores[token_index + 1][wor_index + 1] == scores[token_index][wor_index]
        ):
            mapping[token_index] = wor_index
            token_index += 1
            wor_index += 1
            continue
        token_index += 1

    return mapping


def _fill_cha_alignment_gaps(
    token_to_wor: dict[int, int],
    token_words: list[str],
    wor_items: list[dict[str, str | None]],
) -> dict[int, int]:
    used_wor_indexes = set(token_to_wor.values())

    for token_index in range(len(token_words)):
        if token_index in token_to_wor:
            continue

        previous_token_indexes = [index for index in token_to_wor if index < token_index]
        next_token_indexes = [index for index in token_to_wor if index > token_index]
        lower_bound = token_to_wor[max(previous_token_indexes)] + 1 if previous_token_indexes else 0
        upper_bound = token_to_wor[min(next_token_indexes)] if next_token_indexes else len(wor_items)
        candidates = [
            wor_index
            for wor_index in range(lower_bound, upper_bound)
            if wor_index not in used_wor_indexes
            and not _is_chat_punctuation_word(wor_items[wor_index]["word"])
            and wor_items[wor_index].get("start")
            and wor_items[wor_index].get("end")
        ]
        if len(candidates) == 1:
            token_to_wor[token_index] = candidates[0]
            used_wor_indexes.add(candidates[0])

    return token_to_wor


def _cha_utterance_timing(utterance) -> dict[str, str | None] | None:
    timer_marks_regex = re.compile(r"\x15-?(\d+)_(\d+)-?\x15")
    for tier_name, tier_text in utterance.tiers.items():
        if tier_name.startswith("%"):
            continue
        matches = timer_marks_regex.findall(tier_text)
        if matches:
            start, end = matches[-1]
            return {"word": None, "start": start, "end": end, "source": "Utterance"}
    return None


def _cha_token_timings(utterance) -> list[dict[str, str | None] | None]:
    wor_items = _parse_wor_tier(utterance.tiers.get("%wor", ""))
    timings = [None] * len(utterance.tokens)
    previous_timing = None
    utterance_timing = _cha_utterance_timing(utterance)

    token_positions = []
    token_words = []
    for index, token in enumerate(utterance.tokens):
        if token.word:
            token_positions.append(index)
            token_words.append(_normalize_cha_alignment_word(token.word))

    wor_words = [_normalize_cha_alignment_word(item["word"]) for item in wor_items]
    token_to_wor = _align_cha_words(token_words, wor_words)
    token_to_wor = _fill_cha_alignment_gaps(token_to_wor, token_words, wor_items)

    for token_word_index, token_index in enumerate(token_positions):
        wor_index = token_to_wor.get(token_word_index)
        if wor_index is not None and wor_index < len(wor_items):
            previous_timing = wor_items[wor_index]
            timings[token_index] = previous_timing

        next_index = token_index + 1
        while next_index < len(utterance.tokens) and not utterance.tokens[next_index].word:
            timings[next_index] = previous_timing
            next_index += 1

    for token_index, token in enumerate(utterance.tokens):
        timing = timings[token_index]
        if timing is not None and timing.get("start") and timing.get("end"):
            previous_timing = timings[token_index]
            continue
        if not token.word:
            timings[token_index] = previous_timing or utterance_timing
        elif utterance_timing is not None and not _is_chat_punctuation_word(token.word):
            timings[token_index] = utterance_timing
            previous_timing = utterance_timing

    return timings


def _cha_timing_to_misc(timing: dict[str, str | None] | None) -> dict[str, str] | str:
    if not timing or not timing.get("start") or not timing.get("end"):
        return "_"
    misc = {"TimeStart": str(timing["start"]), "TimeEnd": str(timing["end"])}
    if timing.get("source"):
        misc["TimeSource"] = str(timing["source"])
    return misc


def _cha_contraction_misc(tokens, index: int, timing: dict[str, str | None] | None) -> dict[str, str] | str:
    if not timing or not timing.get("word"):
        return "_"
    token = tokens[index]
    original_form = str(timing["word"])

    if not token.word and token.mor in _EMPTY_WORD_MOR_TO_FORM and index > 0:
        previous_word = tokens[index - 1].word
        if previous_word:
            original_form = str(previous_word)
    if _is_chat_skip_token(original_form, original_form):
        return "_"

    if token.word and index + 1 < len(tokens) and not tokens[index + 1].word:
        return {"OrigForm": original_form, "ContractionPart": "1"}
    if not token.word:
        return {"OrigForm": original_form, "ContractionPart": "2"}
    return "_"


def _is_possessive_s(token, deprel: str) -> bool:
    return token.word == "s" and token.pos == "part" and deprel == "case"


def _normalize_cha_deprel(deprel: str) -> str:
    deprel = deprel.lower()
    if deprel == "incroot":
        return "root"
    return deprel.replace("-", ":")


def _renumber_cha_token_dicts(
    token_dicts: list[dict],
    original_heads: dict[int, int] | None = None,
) -> list[dict]:
    old_to_new = {token["id"]: idx for idx, token in enumerate(token_dicts, 1)}
    original_heads = original_heads or {}

    def retained_head(old_head: int) -> int:
        seen = set()
        while old_head not in old_to_new and old_head != 0 and old_head not in seen:
            seen.add(old_head)
            old_head = original_heads.get(old_head, 0)
        return old_head

    for token in token_dicts:
        old_id = token["id"]
        old_head = retained_head(token["head"])
        token["id"] = old_to_new[old_id]
        if old_head == 0:
            token["head"] = 0
        else:
            token["head"] = old_to_new.get(old_head, 0)
    return token_dicts


def _cha_utterance_has_gra(utterance) -> bool:
    for token in utterance.tokens:
        if not hasattr(token, "gra") or not hasattr(token.gra, "dep"):
            return False
    return True


def _cha_reader_utterances(reader, participant: str):
    try:
        return reader.utterances(participants=participant)
    except TypeError:
        return [
            utterance
            for utterance in reader.utterances()
            if getattr(utterance, "participant", None) == participant
        ]


def _cha_headers_to_metadata(headers) -> dict:
    if not headers:
        return {}
    first_header = headers[0]
    if isinstance(first_header, dict):
        return dict(first_header)

    metadata = {}
    for key in (
        "languages",
        "participants",
        "date",
        "media",
        "location",
        "situation",
        "transcriber",
        "comment",
    ):
        value = getattr(first_header, key, None)
        if value not in (None, "", [], {}):
            metadata[f"CHAT_{key}"] = str(value)
    if not metadata:
        metadata["CHAT_header"] = str(first_header)
    return metadata


def cha_reader_to_sentencelist(reader, participant: str = "PAR") -> SentenceList:
    """
    Convert a pylangacq Reader into asymgram's enhanced SentenceList.

    Parameters:
        reader: A pylangacq Reader object.
        participant: CHAT participant code to extract, defaulting to "PAR".

    Returns:
        SentenceList: An asymgram SentenceList whose sentences are asymgram TokenList
        objects and whose tokens are asymgram Token objects.
    """
    utterances = [utterance for utterance in _cha_reader_utterances(reader, participant) if _cha_utterance_has_gra(utterance)]
    headers = reader.headers()
    metadata = _cha_headers_to_metadata(headers)
    sentences = []

    for idx, utterance in enumerate(utterances, 1):
        token_dicts = []
        token_timings = _cha_token_timings(utterance)
        original_heads = {token.gra.dep: token.gra.head for token in utterance.tokens}
        for token_index, token in enumerate(utterance.tokens):
            xpos = token.pos or ("punct" if token.gra.rel == "PUNCT" else "")
            deprel = _normalize_cha_deprel(token.gra.rel)
            upos = _CHA_POS_TO_CONLLU_UPOS.get(xpos, "_")
            lemma, mor_tags = _split_cha_mor(token.mor)
            form = _strip_chat_special_form_suffix(_cha_surface_form(token, lemma))
            lemma = _strip_chat_special_form_suffix(lemma)
            if _is_chat_skip_token(form, lemma, xpos=xpos, deprel=deprel):
                continue

            if token.gra.rel == "PUNCT":
                lemma = token.word
                xpos = "punct"
                upos = "PUNCT"
            if not token.word and lemma == "be":
                upos = "AUX"
            timing = token_timings[token_index] if token_index < len(token_timings) else None
            misc = _merge_misc(
                _cha_mor_tags_to_misc(mor_tags),
                _cha_timing_to_misc(timing),
                _cha_contraction_misc(utterance.tokens, token_index, timing),
            )
            if _is_possessive_s(token, deprel):
                form = "'s"
                lemma = "'s"
                misc = _merge_misc(misc, {"SpaceBefore": "No"})

            token_dicts.append(
                {
                    "id": token.gra.dep,
                    "form": form,
                    "lemma": lemma,
                    "upos": upos,
                    "xpos": xpos,
                    "feats": _cha_mor_tags_to_feats(mor_tags, upos, lemma),
                    "head": token.gra.head,
                    "deprel": deprel,
                    "deps": "_",
                    "misc": misc,
                }
            )

        text = clean_chat_text(utterance.tiers.get(participant, ""))
        token_dicts = _renumber_cha_token_dicts(token_dicts, original_heads=original_heads)
        sentences.append(TokenList([Token(token_dict) for token_dict in token_dicts], metadata={"sent_id": idx, "text": text}))

    return SentenceList(sentences, metadata=metadata)


def cha_file_to_sentencelist(file_path: str, participant: str = "PAR") -> SentenceList:
    """
    Parse a .cha file and return asymgram's enhanced SentenceList.
    """
    import pylangacq

    file_path = os.fspath(file_path)
    if hasattr(pylangacq, "Reader"):
        reader = pylangacq.Reader.from_files([file_path])
    else:
        try:
            reader = pylangacq.read_chat(file_path)
        except TypeError:
            reader = pylangacq.read_chat(file_path, strict=False)
        except ValueError:
            reader = pylangacq.read_chat(file_path, strict=False)
    return cha_reader_to_sentencelist(reader, participant=participant)


def cha_folder_to_sentencelists(folder_path: str, participant: str = "PAR", recursive: bool = False) -> SentenceLists:
    """
    Parse .cha files in a folder and return an asymgram SentenceLists object.
    """
    folder_path = os.fspath(folder_path)
    pattern = "**/*.cha" if recursive else "*.cha"
    from pathlib import Path

    sentence_lists = [
        cha_file_to_sentencelist(str(path), participant=participant)
        for path in sorted(Path(folder_path).glob(pattern))
    ]
    return SentenceLists(sentence_lists)


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _xml_attr(element, attr_names: tuple[str, ...], default: str = "_") -> str:
    for name in attr_names:
        if name in element.attrib and element.attrib[name] not in ("", None):
            return str(element.attrib[name])
        namespaced = next(
            (value for key, value in element.attrib.items() if _xml_local_name(key) == name and value not in ("", None)),
            None,
        )
        if namespaced is not None:
            return str(namespaced)
    return default


def _xml_text(element) -> str:
    text = "".join(element.itertext()).strip()
    return re.sub(r"\s+", " ", text)


def _xml_metadata_id(element, fallback: str) -> str:
    for key in ("xml:id", "id", "n", "sid"):
        if key in element.attrib:
            return str(element.attrib[key])
    for key, value in element.attrib.items():
        if _xml_local_name(key) == "id":
            return str(value)
    return fallback


def _xml_attrs_to_misc(element) -> dict[str, str] | str:
    misc = {}
    for key, value in element.attrib.items():
        local_key = _xml_local_name(key)
        if local_key in {"id", "n", "sid", "form", "word", "orth", "lemma", "lem", "hw", "pos", "tag", "c5", "msd", "ana", "upos", "class"}:
            continue
        safe_key = re.sub(r"[^A-Za-z0-9_.-]", "_", local_key)
        if safe_key and value not in ("", None):
            misc[f"XML_{safe_key}"] = str(value)
    return misc or "_"


def _xml_token_to_dict(element, token_id: int, profile: dict) -> dict:
    tag_name = _xml_local_name(element.tag)
    form = _xml_attr(element, profile["form_attrs"], default="") or _xml_text(element)
    lemma = _xml_attr(element, profile["lemma_attrs"])
    xpos = _xml_attr(element, profile["xpos_attrs"])
    upos = _xml_attr(element, profile["upos_attrs"])
    if upos == "_" and tag_name in profile.get("punct_tags", ()):
        upos = "PUNCT"
    return {
        "id": token_id,
        "form": form or "_",
        "lemma": lemma,
        "upos": upos,
        "xpos": xpos,
        "feats": "_",
        "head": "_",
        "deprel": "_",
        "deps": "_",
        "misc": _xml_attrs_to_misc(element),
    }


def corpus_xml_file_to_sentencelist(
    file_path: str,
    profile: str = "auto",
    sentence_tags: tuple[str, ...] | None = None,
    token_tags: tuple[str, ...] | None = None,
) -> SentenceList:
    """
    Convert a word-level XML corpus file to SentenceList without adding syntax.

    Built-in profiles include "generic", "bnc", "xces", and "tei". The BNC
    profile handles the common <s>/<w>/<c> layout used by BNC-style XML.
    """
    file_path = os.fspath(file_path)
    profile_name = "bnc" if profile == "auto" else profile
    if profile_name not in XML_CORPUS_PROFILES:
        raise ValueError(f"Unknown XML corpus profile: {profile}")
    profile_config = dict(XML_CORPUS_PROFILES[profile_name])
    if sentence_tags is not None:
        profile_config["sentence_tags"] = sentence_tags
    if token_tags is not None:
        profile_config["token_tags"] = token_tags

    tree = ET.parse(file_path)
    root = tree.getroot()
    sentence_tag_set = set(profile_config["sentence_tags"])
    token_tag_set = set(profile_config["token_tags"])
    sentences = []

    for sent_index, sent_element in enumerate(root.iter(), 1):
        if _xml_local_name(sent_element.tag) not in sentence_tag_set:
            continue
        token_dicts = []
        for token_element in sent_element.iter():
            if token_element is sent_element:
                continue
            if _xml_local_name(token_element.tag) not in token_tag_set:
                continue
            token_dicts.append(_xml_token_to_dict(token_element, len(token_dicts) + 1, profile_config))
        if not token_dicts:
            continue
        sent_id = _xml_metadata_id(sent_element, str(sent_index))
        text = "".join(
            token["form"] if token.get("upos") == "PUNCT" else (" " + token["form"])
            for token in token_dicts
        ).strip()
        metadata = {"sent_id": sent_id, "text": text, "xml_profile": profile_name}
        sentences.append(TokenList([Token(token_dict) for token_dict in token_dicts], metadata=metadata))

    return SentenceList(sentences, metadata={"source_file": file_path, "xml_profile": profile_name})


def corpus_xml_folder_to_sentencelists(
    folder_path: str,
    profile: str = "auto",
    recursive: bool = False,
) -> SentenceLists:
    """Convert XML corpus files in a folder to SentenceLists."""
    pattern = "**/*.xml" if recursive else "*.xml"
    sentence_lists = [
        corpus_xml_file_to_sentencelist(str(path), profile=profile)
        for path in sorted(Path(os.fspath(folder_path)).glob(pattern))
    ]
    return SentenceLists(sentence_lists)


def add_stanza_syntax(
    sentences: SentenceList,
    lang: str = "en",
    processors: str = "tokenize,pos,lemma,depparse",
    package: str | None = None,
    pipeline=None,
    in_place: bool = False,
) -> SentenceList:
    """
    Add POS, lemma, HEAD, and DEPREL with Stanza using existing tokenization.

    This function is intentionally separate from XML conversion. It requires
    stanza and the relevant language model to be installed or supplied through
    an existing pipeline.
    """
    if not in_place:
        sentences = parse(sentences.serialize())
    if pipeline is None:
        try:
            import stanza
        except ImportError as exc:
            raise ImportError("add_stanza_syntax() requires stanza to be installed.") from exc
        kwargs = {
            "lang": lang,
            "processors": processors,
            "tokenize_pretokenized": True,
            "verbose": False,
        }
        if package is not None:
            kwargs["package"] = package
        pipeline = stanza.Pipeline(**kwargs)

    tokenized = [
        [token["form"] for token in sentence if isinstance(token.get("id"), int)]
        for sentence in sentences
    ]
    doc = pipeline(tokenized)
    for sentence, stanza_sentence in zip(sentences, doc.sentences):
        word_tokens = [token for token in sentence if isinstance(token.get("id"), int)]
        if len(word_tokens) != len(stanza_sentence.words):
            sent_id = sentence.metadata.get("sent_id", "?")
            raise ValueError(f"Stanza token count mismatch for sent_id={sent_id}.")
        for token, word in zip(word_tokens, stanza_sentence.words):
            token["lemma"] = word.lemma or token.get("lemma", "_")
            token["upos"] = word.upos or token.get("upos", "_")
            token["xpos"] = word.xpos or token.get("xpos", "_")
            token["feats"] = word.feats or "_"
            token["head"] = int(word.head)
            token["deprel"] = word.deprel or "_"
            token["deps"] = [(token["deprel"], token["head"])] if token["deprel"] != "_" else "_"
    return sentences


def conllu_to_text(conllu_file_path, text_file_path, remove_punct=False, style="cont"):
    # Read CoNLL-U files
    with open(conllu_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    # Parsing CoNLL-U data
    #sentences = conllu.parse(data)
    sentences = parse(data)
    
    # Extract plain text
    with open(text_file_path, 'w', encoding='utf-8') as f:
        if style == "vert": # one token per line
            for sentence in sentences:
                for token in sentence:
                    if remove_punct == True: # skip PUNCT
                        if token['upostag'] == 'PUNCT' or token['deprel'] == 'punct':
                            continue
                    f.write(token['form'] + '\n')
                f.write('\n')  # A blank line after each sentence
        elif style == "hori": # horizontal style
            pass
        elif style == "cont": # continuous style
            pass

def _parse_conllu_subset_id(value: str):
    if "-" in value:
        start, end = value.split("-", 1)
        return (int(start), "-", int(end))
    if "." in value:
        base, index = value.split(".", 1)
        return (int(base), ".", int(index))
    return int(value)


def _parse_conllu_subset_value(column: str, value: str):
    if value == "_":
        return "_"
    if column == "id":
        return _parse_conllu_subset_id(value)
    if column == "head":
        return int(value)
    return value


def _read_conllu_subset(file_path: str, expected_columns: tuple[str, ...]):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    blocks = [block for block in re.split(r"\n\s*\n", text.strip()) if block.strip()]
    columns = None
    sentences = []
    doc_metadata = {}
    tokenlist_class = ConlluBasicTokenList if expected_columns == CONLLUB_COLUMNS else ConlluCoreTokenList
    sentencelist_class = ConlluBasicSentenceList if expected_columns == CONLLUB_COLUMNS else ConlluCoreSentenceList

    for block in blocks:
        metadata = {}
        rows = []
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                comment = line[1:].strip()
                if comment.startswith("global.columns"):
                    _, raw_columns = comment.split("=", 1)
                    columns = tuple(column.lower() for column in raw_columns.split())
                    continue
                if "=" in comment:
                    key, value = comment.split("=", 1)
                    metadata[key.strip()] = value.strip()
                continue
            if columns is None:
                raise ValueError("CoNLL-U Plus subset file must declare '# global.columns = ...'.")
            parts = line.split("\t")
            if len(parts) != len(columns):
                raise ValueError(f"Expected {len(columns)} columns, got {len(parts)}: {line}")
            row = {
                "id": "_",
                "form": "_",
                "lemma": "_",
                "upos": "_",
                "xpos": "_",
                "feats": "_",
                "head": "_",
                "deprel": "_",
                "deps": "_",
                "misc": "_",
            }
            for column, value in zip(columns, parts):
                row[column] = _parse_conllu_subset_value(column, value)
            rows.append(Token(row))

        if rows:
            if tuple(columns) != expected_columns:
                expected = " ".join(column.upper() for column in expected_columns)
                actual = " ".join(column.upper() for column in columns)
                raise ValueError(f"Expected global.columns '{expected}', got '{actual}'.")
            for key in ("newdoc id", "newpar id"):
                if key in metadata:
                    doc_metadata[key] = metadata.pop(key)
            sentences.append(tokenlist_class(rows, metadata=metadata))
        elif metadata:
            doc_metadata.update(metadata)

    return sentencelist_class(sentences, metadata=doc_metadata)


def read_conllu(file_path: str) -> SentenceList:
    """
    Parse a single .conllu file into a SentenceList.

    Parameters:
        file_path (str): Path to the .conllu file

    Returns:
        SentenceList: Parsed list of sentences
    """
    file_path = os.fspath(file_path)
    if file_path.endswith(".conllub"):
        return _read_conllu_subset(file_path, CONLLUB_COLUMNS)
    if file_path.endswith(".conlluc"):
        return _read_conllu_subset(file_path, CONLLUC_COLUMNS)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    #return SentenceList(parse(content))
    return parse(content)

def write_conllu(sentences, output_path):
    """
    Write SentenceList to a .conllu file.
    """
    output_path = os.fspath(output_path)
    if not output_path.endswith(".conllu"):
        raise ValueError(f"Output file must have a '.conllu' extension: got '{output_path}'")
    with open(output_path, "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(sentence.serialize())


def write_conllub(sentences, output_path):
    """Write data to CoNLL-U Basic (.conllub)."""
    output_path = os.fspath(output_path)
    if not output_path.endswith(".conllub"):
        raise ValueError(f"Output file must have a '.conllub' extension: got '{output_path}'")
    data = sentences if isinstance(sentences, ConlluBasicSentenceList) else sentences.to_conllub()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(data.serialize())


def write_conlluc(sentences, output_path):
    """Write data to CoNLL-U Core (.conlluc)."""
    output_path = os.fspath(output_path)
    if not output_path.endswith(".conlluc"):
        raise ValueError(f"Output file must have a '.conlluc' extension: got '{output_path}'")
    data = sentences if isinstance(sentences, ConlluCoreSentenceList) else sentences.to_conlluc()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(data.serialize())


def read_conllu_sentence(file_path: str, sent_id: str, on_duplicate: str = "error"):
    """
    Read one sentence by sent_id without loading unrelated CoNLL-U sentences.

    on_duplicate may be "error", "first", "last", or "all".
    """
    if on_duplicate not in {"error", "first", "last", "all"}:
        raise ValueError("on_duplicate must be 'error', 'first', 'last', or 'all'.")

    file_path = os.fspath(file_path)
    matches = []

    if file_path.endswith((".conllub", ".conlluc")):
        sentences = read_conllu(file_path)
        iterator = iter(sentences)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            iterator = parse_incr(f)
            for sentence in iterator:
                if sentence.metadata.get("sent_id") == sent_id:
                    if on_duplicate == "first":
                        return sentence
                    matches.append(sentence)
                    if len(matches) > 1 and on_duplicate == "error":
                        raise ValueError(f"Duplicate sent_id found: {sent_id}")
        iterator = None

    if iterator is not None:
        for sentence in iterator:
            if sentence.metadata.get("sent_id") == sent_id:
                if on_duplicate == "first":
                    return sentence
                matches.append(sentence)
                if len(matches) > 1 and on_duplicate == "error":
                    raise ValueError(f"Duplicate sent_id found: {sent_id}")

    if not matches:
        raise KeyError(f"sent_id not found: {sent_id}")
    if on_duplicate == "all":
        return matches
    return matches[-1]

def conllu_folder2sentencelists(folder_path: str) -> Generator[SentenceList, None, None]:
    """
    Yield SentenceList objects from all .conllu files in a folder, in sorted order by filename.

    Parameters:
        folder_path (str): Path to the folder containing .conllu files

    Yields:
        SentenceList: Parsed list of sentences from each .conllu file
    """
    folder_path = os.fspath(folder_path)
    conllu_files = sorted(
        f for f in os.listdir(folder_path) if f.endswith(".conllu")
    )

    for filename in conllu_files:
        full_path = os.path.join(folder_path, filename)
        with open(full_path, "r", encoding="utf-8") as f:
            yield from parse_incr(f)

def load_sentence_lists(data) -> SentenceLists:
    """
    Load data and convert to a SentenceLists object.
    
    Parameters:
        data (str or SentenceList): directory, file path, or already parsed SentenceList
    
    Returns:
        SentenceLists: A wrapper around a list of SentenceList objects
    """
    sentence_lists = []

    if isinstance(data, str) and os.path.isdir(data):
        conllu_files = [f for f in os.listdir(data) if f.endswith(".conllu")]
        if not conllu_files:
            raise ValueError(f"No .conllu files found in directory: {data}")
        for fname in conllu_files:
            with open(os.path.join(data, fname), "r", encoding="utf-8") as f:
                sentence_lists.append(SentenceList(parse(f.read())))

    elif isinstance(data, str) and data.endswith(".conllu"):
        if not os.path.exists(data):
            raise FileNotFoundError(f"File not found: {data}")
        with open(data, "r", encoding="utf-8") as f:
            sentence_lists.append(SentenceList(parse(f.read())))

    elif isinstance(data, SentenceList):
        sentence_lists.append(data)

    else:
        raise TypeError("Input must be a SentenceList, a .conllu file path, or a directory path.")

    return SentenceLists(sentence_lists)

def load_sentence_list(file_path: str) -> SentenceList:
    """
    Load a single .conllu file and convert it to a SentenceList object.

    Parameters:
        file_path (str): Path to the .conllu file.

    Returns:
        SentenceList: Parsed sentence list from the given file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the input is not a .conllu file.
    """
    if not isinstance(file_path, str):
        raise TypeError("Input must be a file path string.")
    
    if not file_path.endswith(".conllu"):
        raise ValueError("Only .conllu files are supported.")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
        sentences = parse(data)
        return SentenceList(sentences)

def conllu_to_txt(folderpath: str):
    """
    Convert all .conllu files in the specified folder to .txt files
    and save them in a new folder with the suffix '_txt'.

    Parameters:
        folderpath (str): The path to the folder containing .conllu files.
    """
    if not folderpath or not os.path.isdir(folderpath):
        print("Invalid folder path.")
        return

    # Determine the parent folder and the new target folder path
    parent_folder = os.path.dirname(folderpath)
    new_folder_name = os.path.basename(folderpath) + "_txt"
    target_folder = os.path.join(parent_folder, new_folder_name)

    # Create the target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)

    # Process each .conllu file in the source folder
    for filename in os.listdir(folderpath):
        if filename.endswith(".conllu"):
            source_path = os.path.join(folderpath, filename)
            new_filename = os.path.splitext(filename)[0] + ".txt"
            target_path = os.path.join(target_folder, new_filename)

            # Copy the file content to a .txt file
            shutil.copyfile(source_path, target_path)
            print(f"Converted: {filename} -> {new_filename}")

    print(f"\nAll .conllu files have been converted and saved in: {target_folder}")

def spacy_doc2conllu_style_text(doc, filename: str = "", sent_index: int = 0, sent_id_length: int = 3):
    """
    Convert spaCy documentation to CoNLL format
    - The HEAD column of the ROOT node is 0
    - Add meta information: # newdoc id (only added in the first sentence), # sent_id, # text
    - The default sent_id is three digits
    """
    conllu_lines = []
    for token in doc:
        head = token.head.i + 1 if token.head != token else 0
        if token.dep_ == "punct" and token.head == token:
            head = 0
        conllu_lines.append(
            f"{token.i+1}\t{token.text}\t_\t{token.pos_}\t_\t_\t{head}\t{token.dep_}\t_\t_"
        )
    
    lines = []
    if sent_index == 0:
        lines.append(f"# newdoc id = {filename}")
    sent_id = f"{filename}_{str(sent_index+1).zfill(sent_id_length)}"
    lines.append(f"# sent_id = {sent_id}")
    lines.append(f"# text = {doc.text}")
    lines.extend(conllu_lines)

    return "\n".join(lines) + "\n"


def stanza_doc2conllu_style_text(doc, filename: str = None, sent_start_index: int = 0, sent_id_length: int = 3) -> str:
    """
    Convert a Stanza Document into CoNLL-U formatted string.
    - Add "# newdoc id" to the first sentence
    - Add "# sent_id" and "# text" for each sentence; text is the original sentence with spaces removed
    - Support start index offset (for continuous sentence numbering)
    """
    lines = []
    for i, sentence in enumerate(doc.sentences):
        global_sent_index = sent_start_index + i
        
        if global_sent_index == 0 and filename:
            lines.append(f"# newdoc id = {filename}")
        
        if filename:
            sent_id = f"{filename}_{str(global_sent_index + 1).zfill(sent_id_length)}"
        else:
            sent_id = f"{str(global_sent_index + 1).zfill(sent_id_length)}"
        clean_text = sentence.text.replace(" ", "")  # Remove spaces to restore original sentence
        lines.append(f"# sent_id = {sent_id}")
        lines.append(f"# text = {clean_text}")

        for word in sentence.words:
            head = word.head if word.head is not None else 0
            line = f"{word.id}\t{word.text}\t_\t{word.upos}\t_\t_\t{head}\t{word.deprel}\t_\t_"
            lines.append(line)
        
        lines.append("")  # Blank line between sentences

    return "\n".join(lines)


def hanlp_doc2conllu_style_text(doc: dict, filename: str = None, sent_start_index: int = 0, sent_id_length: int = 3) -> str:
    """
    Convert HanLP dependency parsing output into CoNLL-U formatted text.
    - Uses "tok/fine" for tokenization, "pos/ctb" for POS tags, and "dep" for dependencies.
    - POS tags are written into the XPOS column (5th), leaving UPOS (4th) as underscore.
    - Adds "# newdoc id", "# sent_id", and "# text" for each sentence.
    - Supports offset and customizable sent_id formatting.
    """
    lines = []

    tokens = doc['tok/fine']
    pos_tags = doc['pos/ctb']
    dependencies = doc['dep']

    for i, (words, pos, deps) in enumerate(zip(tokens, pos_tags, dependencies)):
        global_sent_index = sent_start_index + i

        if global_sent_index == 0 and filename:
            lines.append(f"# newdoc id = {filename}")

        sent_id = f"{filename + '_' if filename else ''}{str(global_sent_index + 1).zfill(sent_id_length)}"
        lines.append(f"# sent_id = {sent_id}")
        lines.append(f"# text = {''.join(words)}")

        for idx, (word, xpos, (head, rel)) in enumerate(zip(words, pos, deps)):
            head = int(head)
            line = f"{idx + 1}\t{word}\t_\t_\t{xpos}\t_\t{head}\t{rel}\t_\t_"
            lines.append(line)

        lines.append("")

    return "\n".join(lines)

def extract_all_deprels(
    conllu_folder: str,
    include_punct: bool = False,
    include_root: bool = False
) -> list[str]:
    """
    Extract all unique dependency relation types (deprel) from .conllu files in a folder.

    Parameters:
        conllu_folder (str): Path to the folder containing .conllu files.
        include_punct (bool): Whether to include 'punct' deprel.
        include_root (bool): Whether to include 'root' deprel.

    Returns:
        List[str]: A sorted list of unique deprel labels.
    """
    all_deprels = set()

    for filename in os.listdir(conllu_folder):
        if filename.endswith(".conllu"):
            filepath = os.path.join(conllu_folder, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    sentences = parse(f.read())
                    for sentence in sentences:
                        for token in sentence:
                            deprel = token.get("deprel")
                            if deprel is None:
                                continue
                            if not include_punct and deprel == "punct":
                                continue
                            if not include_root and deprel == "root":
                                continue
                            all_deprels.add(deprel)
            except Exception as e:
                print(f"Failed to read {filename}: {e}")

    return sorted(all_deprels)

def conllu_to_latex_standalone(conllu_data):
    latex_code = "\\documentclass{standalone}\n"
    latex_code += "\\usepackage{tikz-dependency}\n"
    latex_code += "\\usepackage{fontspec}\n"
    latex_code += "\\setmainfont{Times New Roman} % Replace with the font to be used\n"
    latex_code += "\\begin{document}\n"
    latex_code += "    \\begin{dependency}\n"
    latex_code += "        \\begin{deptext}[column sep=3em]\n"

    lines = conllu_data.strip().split("\n")
    forms_line = ""
    upos_line = ""
    for line in lines:
        if line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 10:
            continue
        form, upos = parts[1], parts[3]
        forms_line += f"{form} \\& "
        upos_line += f"{upos} \\& "
    
    forms_line = forms_line.rstrip(" \\&")  # Remove the last column separator
    upos_line = upos_line.rstrip(" \\&")    # Remove the last column separator
    latex_code += f"            {forms_line} \\\\\n"
    latex_code += f"            {upos_line} \\\\\n"
    latex_code += "        \\end{deptext}\n"

    for line in lines:
        if line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 10:
            continue
        head, deprel = parts[6], parts[7]
        if head == '0':
            continue
        latex_code += f"        \\depedge{{{head}}}{{{parts[0]}}}{{\\textbf{{{deprel}}}}}\n"
    
    latex_code += "    \\end{dependency}\n"
    latex_code += "\\end{document}"

    return latex_code

'''
# Example ConLL-U data
conllu_data = """
# newpar
# sent_id = art-sjadziba-17841
# text = На лекцыі Аляксандр распавядзе хітрыкі працы на канцэртах, колькі трэба рабіць здымкаў, як на наступны дзень пасля канцэрта выпусціць якасны рэпартаж, колькі часу сыходзіць на адзін канцэрт і шмат чаго яшчэ.
# genre = social-media
1   На  на  ADP IN  _   2   case    2:case  _
2   лекцыі  лекцыя  NOUN    NN  Animacy=Inan|Case=Loc|Gender=Fem|Number=Sing    4   obl 4:obl:на:loc    _
3   Аляксандр   Аляксандр   PROPN   NNP Animacy=Anim|Case=Nom|Gender=Masc|NameType=Giv|Number=Sing  4   nsubj   4:nsubj _
4   распавядзе  распавесці  VERB    VBC Aspect=Perf|Mood=Ind|Number=Sing|Person=3|Tense=Fut|VerbForm=Fin|Voice=Act  0   root    0:root  _
5   хітрыкі хітрыка NOUN    NN  Animacy=Inan|Case=Acc|Gender=Fem|Number=Plur    4   obj 4:obj|11:conj   _
6   працы   праца   NOUN    NN  Animacy=Inan|Case=Gen|Gender=Fem|Number=Sing    5   nmod    5:nmod:gen  _
7   на  на  ADP IN  _   8   case    8:case  _
8   канцэртах   канцэрт NOUN    NN  Animacy=Inan|Case=Loc|Gender=Masc|Number=Plur   6   nmod    6:nmod:на:loc   SpaceAfter=No
9   ,   ,   PUNCT   PUNCT   _   10  punct   10:punct    _
10  колькі  колькі  NUM CD  Case=Acc    13  nummod:gov  13:nummod:gov   _
11  трэба   трэба   VERB    PRED    _   5   acl 5:acl   _
12  рабіць  рабіць  VERB    VB  Aspect=Imp|VerbForm=Inf|Voice=Act   11  csubj   11:csubj    _
13  здымкаў здымак  NOUN    NN  Animacy=Inan|Case=Gen|Gender=Masc|Number=Plur   12  obj 12:obj  SpaceAfter=No
14  ,   ,   PUNCT   PUNCT   _   21  punct   21:punct    _
15  як  як  ADV WRB Degree=Pos  21  advmod  21:advmod   _
16  на  на  ADP IN  _   18  case    18:case _
17  наступны    наступны    ADJ JJL Animacy=Inan|Case=Acc|Degree=Pos|Gender=Masc|Number=Sing    18  amod    18:amod _
18  дзень   дзень   NOUN    NN  Animacy=Inan|Case=Acc|Gender=Masc|Number=Sing   21  obl 21:obl:на:acc   _
19  пасля   пасля   ADP IN  _   20  case    20:case _
20  канцэрта    канцэрт NOUN    NN  Animacy=Inan|Case=Gen|Gender=Masc|Number=Sing   21  obl 21:obl:пасля:gen    _
21  выпусціць   выпусціць   VERB    VB  Aspect=Perf|VerbForm=Inf|Voice=Act  11  conj    5:acl|11:conj   _
22  якасны  якасны  ADJ JJL Case=Nom|Degree=Pos|Gender=Masc|Number=Sing 23  amod    23:amod _
23  рэпартаж    рэпартаж    NOUN    NN  Animacy=Inan|Case=Nom|Gender=Masc|Number=Sing   21  nsubj:pass  21:nsubj:pass   SpaceAfter=No
24  ,   ,   PUNCT   PUNCT   _   27  punct   27:punct    _
25  колькі  колькі  NUM CD  Animacy=Inan|Case=Acc   26  nummod:gov  26:nummod:gov   _
26  часу    час NOUN    NN  Animacy=Inan|Case=Gen|Gender=Masc|Number=Sing   27  nsubj   27:nsubj    _
27  сыходзіць   сыходзіць   VERB    VBC Aspect=Imp|Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act  23  acl 23:acl  _
28  на  на  ADP IN  _   30  case    30:case _
29  адзін   адзін   NUM CD  Animacy=Inan|Case=Acc|Gender=Masc|Number=Sing|NumType=Card  30  nummod  30:nummod   _
30  канцэрт канцэрт NOUN    NN  Animacy=Inan|Case=Acc|Gender=Masc|Number=Sing   27  obl 27:obl:на:acc   _
31  і   і   CCONJ   CC  _   33  cc  33:cc   _
32  шмат    шмат    ADV RB  Degree=Pos  33  advmod  33:advmod   _
33  чаго    што PRON    WP  Animacy=Inan|Case=Gen|Gender=Neut|Number=Sing|PronType=Int  11  conj    5:ref   _
34  яшчэ    яшчэ    ADV RB  Degree=Pos  33  advmod  33:advmod   SpaceAfter=No
35  .   .   PUNCT   PUNCT   _   4   punct   4:punct _
"""

latex_code = conllu_to_latex_standalone(conllu_data)
print(latex_code)
'''
