"""Language- and parser-specific postprocessing profiles."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import re
from typing import Any, Iterable, Mapping

from ..conllu import SentenceList, Token, TokenList, parse
from ._core import merge_tokens


ZH_SUFFIX_CHARS = (
    "队", "馆", "所", "局",
    "化", "性",
    "者", "师", "员",
    "党", "区", "路", "市", "室", "处", "科", "厂",
    "机", "品", "力",
)

ZH_DEGREE_PREFIXES = ("很", "蛮", "挺")
ZH_NEGATIVE_PREFIXES = ("不",)
ZH_EXCLUDED_NEGATIVE_WORDS = ("不行",)
ZH_TAI_EXCLUDED_WORDS = ("太阳",)

ZH_STANZA_ADVERBS = {"还", "就", "也", "又", "都", "先", "还是", "但是", "但", "所以"}
ZH_STANZA_DIRECTIONAL_VERBS = {"上", "上去", "下去", "下来", "起来", "出去", "出来", "回来", "过去", "过来", "进去", "进来"}
ZH_STANZA_LOCALIZERS = {"上", "上面", "之间", "前", "前面", "里", "里面", "下", "下面", "旁边", "边上", "边", "底下"}
ZH_STANZA_NEGATIVES = {"不", "没", "没有"}
ZH_STANZA_TENSE_ASPECT_MARKERS = {"了", "着", "过", "一下", "在", "掉", "好", "完", "没有", "没"}
ZH_STANZA_MODALITY_MARKERS = {"应该", "要", "可能", "可以"}
ZH_STANZA_DEGREE_MODIFIERS = {"很", "非常", "挺", "比较", "蛮", "有点", "好", "死", "更", "一点", "太", "稍微"}
ZH_STANZA_DISCOURSE_MARKERS = {"结果", "因为", "然后", "而且", "所以"}
ZH_STANZA_MIRATIVES = {"没想到", "想不到", "哪知道"}

# Lexicons below only trigger together with syntactic and/or POS constraints.
# They are deliberately centralized so corpus-specific profiles can override them.
ZH_PASSIVE_MARKERS = {"被", "給", "给", "叫", "讓", "让", "為", "为", "受", "遭", "遭到"}
ZH_BA_MARKERS = {"把", "將", "将"}
ZH_RELATIVE_MARKERS = {"的"}
ZH_MANNER_MARKERS = {"地"}
ZH_SENTENCE_PARTICLES = {
    "啊", "呀", "吧", "呢", "嗎", "吗", "嘛", "啦", "囉", "啰", "喔", "哦", "噢",
    "罷了", "罢了", "而已", "來著", "来着", "唄", "呗", "嘍", "喽", "哇", "哪",
    "了", "的",
}
ZH_LOCALIZERS = {
    "上", "上面", "以上", "下", "下面", "以下", "中", "之中", "當中", "当中", "內", "内",
    "裡", "里", "裏", "裡面", "里面", "內部", "内部", "外", "外面", "之外", "前", "前面",
    "之前", "後", "后", "後面", "后面", "之後", "之后", "左", "左邊", "左边", "右", "右邊",
    "右边", "旁", "旁邊", "旁边", "邊", "边", "附近", "周圍", "周围", "四周", "間", "间",
    "之間", "之间", "對面", "对面", "底下", "頂上", "顶上", "以內", "以内", "以外",
}
ZH_DIRECTIONAL_COMPLEMENTS = {
    "上", "下", "出", "進", "进", "回", "過", "过", "開", "开", "起", "來", "来", "去",
    "上來", "上来", "上去", "下來", "下来", "下去", "出來", "出来", "出去", "進來", "进来",
    "進去", "进去", "回來", "回来", "回去", "過來", "过来", "過去", "过去", "開來", "开来",
    "開去", "开去", "起來", "起来", "起去",
}
ZH_RESULTATIVE_COMPLEMENTS = {
    "完", "好", "成", "到", "見", "见", "懂", "清", "清楚", "明白", "對", "对", "錯", "错",
    "住", "走", "掉", "破", "開", "开", "死", "醒", "醉", "倒", "滿", "满", "光", "著", "着",
    "夠", "够", "通", "透", "牢", "緊", "紧", "平", "乾淨", "干净", "壞", "坏", "累", "哭",
}
ZH_DURATION_FREQUENCY_NOUNS = {
    "次", "遍", "回", "趟", "下", "番", "陣", "阵", "會兒", "会儿", "分鐘", "分钟", "小時",
    "小时", "秒", "天", "日", "週", "周", "星期", "月", "年", "夜", "上午", "下午", "晚上",
}
ZH_TEMPORAL_NOUNS = {
    "今天", "今日", "昨天", "昨日", "明天", "明日", "前天", "後天", "后天", "現在", "现在",
    "當時", "当时", "目前", "從前", "从前", "以前", "以後", "以后", "最近", "剛才", "刚才",
    "早上", "上午", "中午", "下午", "晚上", "夜間", "夜间", "凌晨", "週末", "周末", "平時",
    "平时", "每天", "每日", "每週", "每周", "每月", "每年", "春天", "夏天", "秋天", "冬天",
    "時候", "时候", "期間", "期间", "年代", "世紀", "世纪",
}
ZH_VERB_OBJECT_COMPOUNDS = {
    ("吃", "飯"), ("吃", "饭"), ("睡", "覺"), ("睡", "觉"), ("讀", "書"), ("读", "书"),
    ("上", "班"), ("下", "班"), ("上", "課"), ("上", "课"), ("畢", "業"), ("毕", "业"),
    ("結", "婚"), ("结", "婚"), ("離", "婚"), ("离", "婚"), ("生", "氣"), ("生", "气"),
    ("幫", "忙"), ("帮", "忙"), ("見", "面"), ("见", "面"), ("聊", "天"), ("散", "步"),
    ("洗", "澡"), ("游", "泳"), ("跳", "舞"), ("唱", "歌"), ("開", "會"), ("开", "会"),
    ("打", "電話"), ("打", "电话"), ("打", "針"), ("打", "针"), ("發", "言"), ("发", "言"),
    ("道", "歉"), ("請", "假"), ("请", "假"), ("報", "名"), ("报", "名"), ("考", "試"),
    ("考", "试"), ("畢", "業"), ("毕", "业"), ("鼓", "掌"), ("握", "手"), ("點", "頭"),
    ("点", "头"), ("搖", "頭"), ("摇", "头"), ("操", "心"), ("擔", "心"), ("担", "心"),
}

EN_PASSIVE_AUXILIARIES = {"be", "get"}
EN_PHRASAL_PARTICLES = {
    "about", "across", "ahead", "along", "apart", "around", "aside", "away", "back", "by", "down",
    "forth", "forward", "in", "off", "on", "out", "over", "past", "round", "through", "together", "up",
}
EN_PREDETERMINERS = {"all", "both", "half"}
EN_PRECONJUNCTS = {"both", "either", "neither"}
EN_RELATIVE_PRONOUNS = {"that", "which", "who", "whom", "whose", "what", "whatever", "whoever", "whichever"}
EN_RELATIVE_ADVERBS = {"when", "where", "why", "whereby", "wherein"}
EN_NAME_DESCRIPTORS = {
    "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "miss", "dr", "dr.", "prof", "prof.", "professor",
    "president", "senator", "governor", "judge", "justice", "captain", "general", "sir", "dame", "saint",
    "st.", "father", "mother", "brother", "sister", "king", "queen", "prince", "princess", "pope",
    "jr", "jr.", "sr", "sr.", "ph.d.", "inc", "inc.", "corp", "corp.", "ltd", "ltd.", "llc", "plc",
}

DEPREL_LEXICONS = {
    "zh": {
        "passive_markers": ZH_PASSIVE_MARKERS,
        "ba_markers": ZH_BA_MARKERS,
        "relative_markers": ZH_RELATIVE_MARKERS,
        "manner_markers": ZH_MANNER_MARKERS,
        "sentence_particles": ZH_SENTENCE_PARTICLES,
        "localizers": ZH_LOCALIZERS,
        "directional_complements": ZH_DIRECTIONAL_COMPLEMENTS,
        "resultative_complements": ZH_RESULTATIVE_COMPLEMENTS,
        "duration_frequency_nouns": ZH_DURATION_FREQUENCY_NOUNS,
        "temporal_nouns": ZH_TEMPORAL_NOUNS,
        "verb_object_compounds": ZH_VERB_OBJECT_COMPOUNDS,
    },
    "en": {
        "passive_auxiliaries": EN_PASSIVE_AUXILIARIES,
        "phrasal_particles": EN_PHRASAL_PARTICLES,
        "predeterminers": EN_PREDETERMINERS,
        "preconjuncts": EN_PRECONJUNCTS,
        "relative_pronouns": EN_RELATIVE_PRONOUNS,
        "relative_adverbs": EN_RELATIVE_ADVERBS,
        "name_descriptors": EN_NAME_DESCRIPTORS,
    },
}

POSTPROCESS_PROFILES = {
    "zh_stanza_default": {
        "language": "zh",
        "parser": "stanza",
        "version": "1.15.0 / gsdsimp / UD 2.18",
        "mode": "fidelity",
        "description": "Alias for the measured zero-damage GSDSimp fidelity profile.",
    },
    "zh_stanza_1": {
        "language": "zh",
        "parser": "stanza",
        "version": "1.x compatibility alias",
        "mode": "fidelity",
        "description": "Compatibility alias for zh_stanza_default.",
    },
    "zh_stanza_1_15_gsdsimp_fidelity": {
        "language": "zh",
        "parser": "stanza",
        "version": "1.15.0 / gsdsimp / UD 2.18",
        "mode": "fidelity",
        "description": "Preserve raw analysis; no tested automatic rule had positive net LAS gain.",
    },
    "zh_stanza_1_15_gsdsimp_candidates": {
        "language": "zh",
        "parser": "stanza",
        "version": "1.15.0 / gsdsimp / UD 2.18",
        "mode": "candidates",
        "description": "Preserve the tree and record the number of review candidates.",
    },
    "zh_stanza_1_15_gsdsimp_research": {
        "language": "zh",
        "parser": "stanza",
        "version": "1.15.0 / gsdsimp / UD 2.18",
        "mode": "research",
        "description": "Convert output to asymgram research subtypes; not an accuracy profile.",
    },
    "zh_stanza_legacy_relabel": {
        "language": "zh",
        "parser": "stanza",
        "version": "legacy",
        "mode": "legacy",
        "description": "Preserve the pre-profile Stanza relabeling and classifier rotation behavior.",
    },
    "zh_stanza_ud": {
        "language": "zh",
        "parser": "stanza",
        "version": "1.x / UD v2",
        "mode": "research",
        "description": "Compatibility research conversion; not an accuracy profile.",
    },
    "zh_ud_refine": {
        "language": "zh",
        "parser": "any",
        "version": "UD v2",
        "mode": "research",
        "description": "Refine basic relations to official Chinese UD relation subtypes.",
    },
    "en_ewt_refine": {
        "language": "en",
        "parser": "any",
        "version": "UD English/EWT v2",
        "mode": "research",
        "description": "Refine basic relations to English UD/EWT relation subtypes.",
    },
}


def available_postprocess_profiles() -> dict[str, dict[str, str]]:
    """Return registered postprocessing profile metadata."""
    return deepcopy(POSTPROCESS_PROFILES)


def get_deprel_lexicon(language: str) -> dict[str, set]:
    """Return a copy of the built-in lexical cues for a language."""
    language = language.lower().split("-")[0].split("_")[0]
    if language not in DEPREL_LEXICONS:
        raise ValueError(f"No dependency-relation lexicon for language: {language}")
    return deepcopy(DEPREL_LEXICONS[language])


def _copy_tokenlist(sentence: TokenList) -> TokenList:
    return TokenList([Token(deepcopy(dict(token))) for token in sentence], metadata=deepcopy(sentence.metadata))


def _copy_data(data: Any) -> Any:
    if isinstance(data, TokenList):
        return _copy_tokenlist(data)
    if isinstance(data, SentenceList):
        return SentenceList([_copy_tokenlist(sentence) for sentence in data], metadata=deepcopy(data.metadata))
    return [_copy_tokenlist(sentence) for sentence in data]


def _iter_sentences(data: Any):
    if isinstance(data, TokenList):
        yield data
    elif isinstance(data, SentenceList):
        yield from data
    else:
        yield from data


def _replace_basic_edge_in_deps(token: Token, old_head: Any, old_deprel: Any) -> None:
    """Update a matching basic edge without dropping or reordering enhanced edges."""
    deps = token.get("deps")
    if not isinstance(deps, list):
        return
    new_head = token.get("head")
    new_deprel = token.get("deprel")
    for index, edge in enumerate(deps):
        if not isinstance(edge, tuple) or len(edge) != 2:
            continue
        relation, head = edge
        if head == old_head and relation == old_deprel:
            deps[index] = (new_deprel, new_head)
            break


def _set_deprel(token: Token, deprel: str, report: Counter | None = None) -> bool:
    old_deprel = token.get("deprel")
    if old_deprel == deprel:
        return False
    old_head = token.get("head")
    token["deprel"] = deprel
    _replace_basic_edge_in_deps(token, old_head, old_deprel)
    if report is not None:
        report[deprel] += 1
    return True


def _id_map(sentence: TokenList) -> dict[int, Token]:
    return {token["id"]: token for token in sentence if isinstance(token.get("id"), int)}


def _children_map(sentence: TokenList) -> dict[int, list[Token]]:
    children: dict[int, list[Token]] = {}
    for token in sentence:
        if isinstance(token.get("id"), int) and isinstance(token.get("head"), int):
            children.setdefault(token["head"], []).append(token)
    return children


def _form(token: Token | None) -> str:
    if not token:
        return ""
    lemma = token.get("lemma")
    value = token.get("form") if lemma in (None, "_") else lemma
    return str(value or "").lower()


def _feature(token: Token, name: str) -> Any:
    feats = token.get("feats")
    return feats.get(name) if isinstance(feats, Mapping) else None


def _has_case_child(children: Mapping[int, list[Token]], token_id: int, forms: set[str] | None = None) -> bool:
    for child in children.get(token_id, []):
        if str(child.get("deprel", "")).split(":", 1)[0] != "case":
            continue
        if forms is None or _form(child) in forms:
            return True
    return False


def _is_near_sentence_end(sentence: TokenList, token_id: int) -> bool:
    trailing = [
        token for token in sentence
        if isinstance(token.get("id"), int) and token["id"] > token_id
    ]
    return all(token.get("upos") in {"PUNCT", "SYM"} for token in trailing)


def _merged_lexicon(language: str, overrides: Mapping[str, Iterable] | None) -> dict[str, set]:
    lexicon = get_deprel_lexicon(language)
    if overrides:
        for name, values in overrides.items():
            if name not in lexicon:
                raise ValueError(f"Unknown {language} lexicon category: {name}")
            lexicon[name].update(values)
    return lexicon


def _apply_zh_stanza_token_relabels(sentence: TokenList) -> int:
    changed = 0
    for token in sentence:
        if not isinstance(token.get("id"), int):
            continue
        before = dict(token)
        old_head = token.get("head")
        old_deprel = token.get("deprel")
        form = token.get("form")
        upos = token.get("upos")
        deprel = token.get("deprel")

        if form in {"了", "的"} and deprel == "discourse":
            token["deprel"] = "discourse:sp"
        elif form in ZH_STANZA_ADVERBS and upos == "ADV" and deprel == "mark":
            token["deprel"] = "advmod"
        elif deprel == "nmod:tmod":
            token["deprel"] = "obl:tmod"
        elif form == "一下" and deprel == "mark":
            token["upos"] = "AUX"
            token["deprel"] = "aux"
        elif form == "在" and deprel == "advmod" and upos == "ADV":
            token["upos"] = "AUX"
            token["deprel"] = "aux:ta"
        elif form in ZH_STANZA_DIRECTIONAL_VERBS and deprel == "mark":
            token["deprel"] = "compound:dir"
        elif form in ZH_STANZA_TENSE_ASPECT_MARKERS and deprel == "aux":
            token["deprel"] = "aux:ta"
        elif form in ZH_STANZA_MODALITY_MARKERS and deprel == "aux":
            token["deprel"] = "aux:modal"
        elif form in ZH_STANZA_LOCALIZERS and deprel == "acl":
            token["deprel"] = "case:loc"
        elif form in ZH_STANZA_NEGATIVES and upos == "ADV" and deprel == "advmod":
            token["deprel"] = "advmod:neg"
        elif form in ZH_STANZA_DEGREE_MODIFIERS and deprel == "advmod":
            token["deprel"] = "advmod:deg"
        elif form in ZH_STANZA_DISCOURSE_MARKERS and deprel in {"advmod", "mark"}:
            token["deprel"] = "discourse"
        elif form in ZH_STANZA_MIRATIVES:
            token["deprel"] = "advmod"
        elif form in {"以后", "的时候"} and deprel == "mark":
            token["deprel"] = "mark:tmod"
        elif form == "一" and deprel == "mark":
            token["deprel"] = "mark:ta"

        if dict(token) != before:
            _replace_basic_edge_in_deps(token, old_head, old_deprel)
            changed += 1
    return changed


def _repair_zh_stanza_classifier_chain(sentence: TokenList) -> int:
    changed = 0
    id2token = _id_map(sentence)
    for modifier in sentence:
        if modifier.get("deprel") not in {"nummod", "det"}:
            continue
        clf = id2token.get(modifier.get("head"))
        if not clf or clf.get("deprel") != "clf":
            continue
        noun_id = clf.get("head")
        if noun_id not in id2token:
            continue
        old = (modifier.get("head"), clf.get("head"))
        modifier_old_deprel = modifier.get("deprel")
        clf_old_deprel = clf.get("deprel")
        modifier["head"] = noun_id
        clf["head"] = modifier["id"]
        _replace_basic_edge_in_deps(modifier, old[0], modifier_old_deprel)
        _replace_basic_edge_in_deps(clf, old[1], clf_old_deprel)
        if old != (modifier.get("head"), clf.get("head")):
            changed += 1
    return changed


def _refine_zh_sentence(sentence: TokenList, lexicon: Mapping[str, set], report: Counter) -> None:
    id2token = _id_map(sentence)
    children = _children_map(sentence)

    # Function words provide the structural cues needed by later nominal rules.
    for token in sentence:
        if not isinstance(token.get("id"), int):
            continue
        form = _form(token)
        deprel = str(token.get("deprel") or "")
        head = id2token.get(token.get("head"))

        if (
            form in lexicon["passive_markers"]
            and deprel in {"aux", "mark"}
            and head
            and head.get("upos") in {"VERB", "ADJ"}
        ):
            _set_deprel(token, "aux:pass", report)
        elif (
            form in lexicon["relative_markers"]
            and deprel == "mark"
            and head
            and token["id"] > head["id"]
        ):
            _set_deprel(token, "mark:rel", report)
        elif (
            form in lexicon["manner_markers"]
            and deprel in {"mark", "case"}
            and head
            and token["id"] > head["id"]
        ):
            _set_deprel(token, "mark:adv", report)
        elif (
            form in lexicon["localizers"]
            and deprel == "case"
            and head
            and head.get("upos") in {"NOUN", "PROPN", "PRON", "NUM"}
            and token["id"] > head["id"]
        ):
            _set_deprel(token, "case:loc", report)
        elif (
            form in lexicon["sentence_particles"]
            and deprel == "discourse"
            and _is_near_sentence_end(sentence, token["id"])
        ):
            _set_deprel(token, "discourse:sp", report)

    children = _children_map(sentence)
    passive_heads = {
        token.get("head") for token in sentence if token.get("deprel") == "aux:pass"
    }
    passive_heads.update(
        token.get("head")
        for token in sentence
        if token.get("deprel") == "obl"
        and isinstance(token.get("id"), int)
        and _has_case_child(children, token["id"], lexicon["passive_markers"])
    )
    for token in sentence:
        if not isinstance(token.get("id"), int):
            continue
        token_id = token["id"]
        deprel = str(token.get("deprel") or "")
        head = id2token.get(token.get("head"))
        form = _form(token)

        if token.get("head") in passive_heads and deprel == "nsubj":
            _set_deprel(token, "nsubj:pass", report)
        elif token.get("head") in passive_heads and deprel == "csubj":
            _set_deprel(token, "csubj:pass", report)
        elif deprel == "obl" and _has_case_child(children, token_id, lexicon["passive_markers"]):
            _set_deprel(token, "obl:agent", report)
        elif deprel in {"obl", "obj"} and _has_case_child(children, token_id, lexicon["ba_markers"]):
            _set_deprel(token, "obl:patient", report)
        elif deprel in {"obl", "nmod"} and form in lexicon["temporal_nouns"]:
            _set_deprel(token, "obl:tmod" if deprel == "obl" else "nmod:tmod", report)
        elif (
            deprel in {"advmod", "obl"}
            and form in lexicon["duration_frequency_nouns"]
            and any(child.get("deprel") in {"nummod", "det", "clf"} for child in children.get(token_id, []))
        ):
            _set_deprel(token, "advmod:df", report)
        elif (
            form in lexicon["directional_complements"]
            and deprel in {"compound", "xcomp", "mark", "advmod"}
            and head
            and head.get("upos") == "VERB"
            and token_id > head["id"]
        ):
            _set_deprel(token, "compound:dir", report)
        elif (
            form == "得"
            and deprel in {"compound", "mark", "aux"}
            and head
            and head.get("upos") in {"VERB", "ADJ"}
            and any(
                child.get("deprel") == "xcomp" and child["id"] > token_id
                for child in children.get(head["id"], [])
            )
        ):
            _set_deprel(token, "compound:ext", report)
        elif (
            form in lexicon["resultative_complements"]
            and deprel in {"compound", "xcomp", "advcl"}
            and head
            and head.get("upos") == "VERB"
            and token.get("upos") in {"VERB", "ADJ"}
            and 0 < token_id - head["id"] <= 2
            and not any(child.get("deprel") == "compound:ext" for child in children.get(head["id"], []))
        ):
            _set_deprel(token, "compound:vv", report)
        elif (
            deprel in {"obj", "compound"}
            and head
            and (_form(head), form) in lexicon["verb_object_compounds"]
        ):
            _set_deprel(token, "compound:vo", report)
        elif (
            deprel == "dislocated"
            and head
            and (_form(head), form) in lexicon["verb_object_compounds"]
        ):
            _set_deprel(token, "dislocated:vo", report)


def _is_english_passive_head(token: Token, children: Mapping[int, list[Token]]) -> bool:
    if _feature(token, "Voice") == "Pass":
        return True
    return any(child.get("deprel") == "aux:pass" for child in children.get(token["id"], []))


def _refine_en_sentence(sentence: TokenList, lexicon: Mapping[str, set], report: Counter) -> None:
    id2token = _id_map(sentence)
    children = _children_map(sentence)

    for token in sentence:
        if not isinstance(token.get("id"), int):
            continue
        deprel = str(token.get("deprel") or "")
        head = id2token.get(token.get("head"))
        if (
            deprel == "aux"
            and _form(token) in lexicon["passive_auxiliaries"]
            and head
            and (
                _feature(head, "VerbForm") == "Part"
                or head.get("xpos") == "VBN"
                or _feature(head, "Voice") == "Pass"
            )
        ):
            _set_deprel(token, "aux:pass", report)

    children = _children_map(sentence)
    for token in sentence:
        if not isinstance(token.get("id"), int):
            continue
        token_id = token["id"]
        deprel = str(token.get("deprel") or "")
        form = _form(token)
        head = id2token.get(token.get("head"))
        head_passive = bool(head and _is_english_passive_head(head, children))

        if deprel == "nsubj" and head_passive:
            _set_deprel(token, "nsubj:pass", report)
        elif deprel == "csubj" and head_passive:
            _set_deprel(token, "csubj:pass", report)
        elif deprel == "obl" and head_passive and _has_case_child(children, token_id, {"by"}):
            _set_deprel(token, "obl:agent", report)
        elif deprel == "nmod" and (
            _feature(token, "Poss") == "Yes"
            or _has_case_child(children, token_id, {"'s", "’s", "'", "’"})
        ):
            _set_deprel(token, "nmod:poss", report)
        elif (
            deprel == "nmod"
            and head
            and head.get("upos") == "PROPN"
            and form in lexicon["name_descriptors"]
        ):
            _set_deprel(token, "nmod:desc", report)
        elif deprel == "nmod" and not _has_case_child(children, token_id):
            _set_deprel(token, "nmod:unmarked", report)
        elif deprel == "obl" and not _has_case_child(children, token_id):
            _set_deprel(token, "obl:unmarked", report)
        elif (
            deprel in {"compound", "advmod"}
            and form in lexicon["phrasal_particles"]
            and head
            and head.get("upos") == "VERB"
            and token.get("upos") in {"PART", "ADV", "ADP"}
        ):
            _set_deprel(token, "compound:prt", report)
        elif (
            deprel == "det"
            and form in lexicon["predeterminers"]
            and any(child.get("deprel") == "det" and child["id"] != token_id for child in children.get(token.get("head"), []))
        ):
            _set_deprel(token, "det:predet", report)
        elif deprel == "cc" and form in lexicon["preconjuncts"]:
            _set_deprel(token, "cc:preconj", report)
        elif deprel in {"acl", "advcl"}:
            dependent_forms = {_form(child) for child in children.get(token_id, [])}
            if deprel == "acl" and dependent_forms & (lexicon["relative_pronouns"] | lexicon["relative_adverbs"]):
                _set_deprel(token, "acl:relcl", report)
            elif deprel == "advcl" and dependent_forms & lexicon["relative_adverbs"]:
                _set_deprel(token, "advcl:relcl", report)
        elif deprel == "flat" and (_feature(token, "Foreign") == "Yes" or token.get("upos") == "X"):
            _set_deprel(token, "flat:foreign", report)


def refine_language_deprels(
    data: Any,
    language: str,
    in_place: bool = False,
    lexicon_overrides: Mapping[str, Iterable] | None = None,
    return_report: bool = False,
):
    """Refine basic DEPREL values using official language-specific UD subtypes.

    Rules combine lexical cues with existing POS and dependency structure. They
    do not attempt to repair arbitrary parse structures. Enhanced DEPS edges are
    preserved in their original order; only a matching basic edge is relabeled.
    """
    language = language.lower().split("-")[0].split("_")[0]
    if language not in {"zh", "en"}:
        raise ValueError("language must be 'zh' or 'en'")
    working = data if in_place else _copy_data(data)
    lexicon = _merged_lexicon(language, lexicon_overrides)
    report: Counter = Counter()
    refiner = _refine_zh_sentence if language == "zh" else _refine_en_sentence
    profile = "zh_ud_refine" if language == "zh" else "en_ewt_refine"
    for sentence in _iter_sentences(working):
        refiner(sentence, lexicon, report)
        sentence.metadata["deprel_refine_profile"] = profile
    if isinstance(working, SentenceList):
        working.metadata["deprel_refine_profile"] = profile
    if return_report:
        return working, dict(sorted(report.items()))
    return working


def refine_zh_deprels(
    data: Any,
    in_place: bool = False,
    lexicon_overrides: Mapping[str, Iterable] | None = None,
    return_report: bool = False,
):
    """Refine relations with official Chinese UD v2 subtypes."""
    return refine_language_deprels(
        data, "zh", in_place=in_place,
        lexicon_overrides=lexicon_overrides, return_report=return_report,
    )


def refine_en_deprels(
    data: Any,
    in_place: bool = False,
    lexicon_overrides: Mapping[str, Iterable] | None = None,
    return_report: bool = False,
):
    """Refine relations with English UD/EWT v2 subtypes."""
    return refine_language_deprels(
        data, "en", in_place=in_place,
        lexicon_overrides=lexicon_overrides, return_report=return_report,
    )


def diagnose_zh_stanza_candidates(data: Any) -> list[dict[str, Any]]:
    """Return review candidates for the measured Stanza Chinese profile.

    The detector is deliberately non-mutating.  Its categories come from a
    Stanza 1.15.0 ``gsdsimp`` evaluation against UD Chinese GSDSimp 2.18;
    they identify weak regions, not guaranteed errors.
    """
    candidates: list[dict[str, Any]] = []
    weak_clause_relations = {"advcl", "ccomp", "csubj", "parataxis", "xcomp", "acl:relcl"}

    for sentence_index, sentence in enumerate(_iter_sentences(data)):
        sent_id = sentence.metadata.get("sent_id")
        id2token = _id_map(sentence)
        children = _children_map(sentence)

        def add(token: Token, category: str, reason: str, alternatives: Iterable[str]) -> None:
            candidates.append({
                "sentence_index": sentence_index,
                "sent_id": sent_id,
                "token_id": token.get("id"),
                "form": token.get("form"),
                "category": category,
                "observed_head": token.get("head"),
                "observed_deprel": token.get("deprel"),
                "alternatives": list(alternatives),
                "reason": reason,
                "action": "review",
            })

        for token in sentence:
            if not isinstance(token.get("id"), int):
                continue
            form = str(token.get("form") or "")
            deprel = str(token.get("deprel") or "")
            head = id2token.get(token.get("head"))

            if form in {"的", "之"} and deprel in {"case", "mark", "mark:rel"}:
                add(
                    token,
                    "de_function",
                    "Stanza frequently confuses nominal 的/之 case with relative-clause marking.",
                    ("case", "mark:rel"),
                )
            if deprel in weak_clause_relations:
                add(
                    token,
                    "clause_attachment",
                    "This relation belongs to a low-LAS clause/attachment group in the measured profile.",
                    sorted(weak_clause_relations | {"root"}),
                )
            if (
                (deprel == "nmod" and head and head.get("upos") in {"VERB", "ADJ"})
                or (deprel in {"nsubj", "obj", "obl"} and head and head.get("upos") in {"NOUN", "PROPN"})
            ):
                add(
                    token,
                    "core_oblique_attachment",
                    "The relation crosses the usual nominal/predicative category expectation and falls in a recurrent swap group.",
                    ("nmod", "nsubj", "obj", "obl"),
                )

            if deprel in {"nummod", "det"}:
                classifier = id2token.get(token.get("head"))
                if classifier and classifier.get("deprel") in {"clf", "nmod"}:
                    noun = id2token.get(classifier.get("head"))
                    if noun and noun.get("upos") in {"NOUN", "PROPN", "PRON"}:
                        add(
                            token,
                            "classifier_topology",
                            "Classifier-chain rotation was observed but had negative net LAS and requires review.",
                            ("NUM/DET->NOUN, CLF->NUM/DET", "NUM/DET->CLF->NOUN"),
                        )

            if token.get("upos") in {"NOUN", "VERB", "ADP", "SCONJ", "PART", "PROPN"}:
                if (
                    form in {"的", "地", "得", "在", "为", "因", "后", "上", "下", "中"}
                    or any(child.get("deprel") in weak_clause_relations for child in children.get(token["id"], []))
                ):
                    add(
                        token,
                        "upos_context",
                        "The form or its clause context falls in a recurrent Stanza UPOS confusion group.",
                        ("NOUN/VERB", "VERB/ADP", "SCONJ/PART", "NOUN/PROPN"),
                    )
    return candidates


def apply_postprocess_profile(data: Any, profile: str = "zh_stanza_default", in_place: bool = False):
    """Apply a language/parser-specific postprocessing profile."""
    if profile == "zh_stanza_1":
        profile = "zh_stanza_default"
    if profile not in POSTPROCESS_PROFILES:
        raise ValueError(f"Unknown postprocessing profile: {profile}")

    working = data if in_place else _copy_data(data)
    fidelity_profiles = {"zh_stanza_default", "zh_stanza_1_15_gsdsimp_fidelity"}
    if profile in fidelity_profiles:
        for sentence in _iter_sentences(working):
            sentence.metadata["postprocess_profile"] = profile
        if isinstance(working, SentenceList):
            working.metadata["postprocess_profile"] = profile
    elif profile == "zh_stanza_1_15_gsdsimp_candidates":
        candidates = diagnose_zh_stanza_candidates(working)
        counts = Counter(candidate["sentence_index"] for candidate in candidates)
        for sentence_index, sentence in enumerate(_iter_sentences(working)):
            sentence.metadata["postprocess_profile"] = profile
            sentence.metadata["postprocess_candidate_count"] = counts[sentence_index]
        if isinstance(working, SentenceList):
            working.metadata["postprocess_profile"] = profile
            working.metadata["postprocess_candidate_count"] = len(candidates)
    elif profile in {"zh_stanza_legacy_relabel", "zh_stanza_1_15_gsdsimp_research", "zh_stanza_ud"}:
        for sentence in _iter_sentences(working):
            _apply_zh_stanza_token_relabels(sentence)
            _repair_zh_stanza_classifier_chain(sentence)
            sentence.metadata["postprocess_profile"] = profile
            if profile in {"zh_stanza_1_15_gsdsimp_research", "zh_stanza_ud"}:
                _refine_zh_sentence(sentence, get_deprel_lexicon("zh"), Counter())
        if isinstance(working, SentenceList):
            working.metadata["postprocess_profile"] = profile
    elif profile in {"zh_ud_refine", "en_ewt_refine"}:
        language = "zh" if profile == "zh_ud_refine" else "en"
        lexicon = get_deprel_lexicon(language)
        report: Counter = Counter()
        refiner = _refine_zh_sentence if language == "zh" else _refine_en_sentence
        for sentence in _iter_sentences(working):
            refiner(sentence, lexicon, report)
            sentence.metadata["postprocess_profile"] = profile
        if isinstance(working, SentenceList):
            working.metadata["postprocess_profile"] = profile
    return working


def postprocess_zh_stanza(data: Any, in_place: bool = False):
    """Apply the measured zero-damage Chinese Stanza fidelity profile."""
    return apply_postprocess_profile(data, profile="zh_stanza_default", in_place=in_place)


def _merge_adjacent_ids(sentence: TokenList, first_id: int, second_id: int) -> TokenList:
    start, end = sorted((first_id, second_id))
    if end != start + 1:
        raise ValueError("Only adjacent token IDs can be merged.")
    return merge_tokens(sentence, start, end, record_orig_forms=True)


def merge_zh_suffix_compounds(
    data: Any,
    suffix_chars: Iterable[str] = ZH_SUFFIX_CHARS,
    min_stem_len: int = 2,
    in_place: bool = False,
):
    """
    Merge Chinese stem + bound suffix splits such as AB + 者/性/化.

    This is a word-token postprocessing rule. It only merges adjacent tokens
    where the left token has at least min_stem_len Chinese characters and the
    right token is in suffix_chars.
    """
    suffix_set = set(suffix_chars)
    working = data if in_place else _copy_data(data)

    if isinstance(working, TokenList):
        sentence_slots = [(None, working)]
    elif isinstance(working, SentenceList):
        sentence_slots = list(enumerate(working))
    else:
        sentence_slots = list(enumerate(working))

    for sent_index, sentence in sentence_slots:
        changed = True
        while changed:
            changed = False
            tokens = [token for token in sentence if isinstance(token.get("id"), int)]
            for left, right in zip(tokens, tokens[1:]):
                if not isinstance(left.get("id"), int) or not isinstance(right.get("id"), int):
                    continue
                if right.get("form") not in suffix_set:
                    continue
                left_form = str(left.get("form", ""))
                if len(left_form) < min_stem_len:
                    continue
                sentence = _merge_adjacent_ids(sentence, left["id"], right["id"])
                if sent_index is not None:
                    working[sent_index] = sentence
                else:
                    working = sentence
                changed = True
                break
    return working


def postprocess_zh_tokenized_text(
    text: str,
    merge_suffix_chars: Iterable[str] = ZH_SUFFIX_CHARS,
    split_degree_prefixes: Iterable[str] = ZH_DEGREE_PREFIXES,
    split_negative_prefixes: Iterable[str] = ZH_NEGATIVE_PREFIXES,
) -> str:
    """
    Light Chinese tokenized-text cleanup before parsing.

    The function keeps the rules systematic: bound suffix merging and common
    degree/negative prefix splitting. Corpus- or task-specific lexical lists
    should be supplied by callers instead of being hard-coded here.
    """
    for prefix in split_degree_prefixes:
        text = re.sub(rf"(?<!\S){re.escape(prefix)}([^\s])(?!\S)", rf"{prefix} \1", text)
    for prefix in split_negative_prefixes:
        excluded = "|".join(re.escape(word) for word in ZH_EXCLUDED_NEGATIVE_WORDS)
        text = re.sub(rf"(?<!\S){re.escape(prefix)}(?!({excluded})(?!\S))([^\s])(?!\S)", rf"{prefix} \2", text)
    text = re.sub(r"(?<!\S)太(?!阳)([^\s])(?!\S)", r"太 \1", text)
    suffix_pattern = "|".join(re.escape(char) for char in merge_suffix_chars)
    text = re.sub(rf"(?<!\S)([^\s]{{2,}})\s+({suffix_pattern})(?!\S)", r"\1\2", text)
    return text
