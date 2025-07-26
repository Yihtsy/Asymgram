from .conllu import Token, TokenList, SentenceList
from .constants import CHINESE_COMMA
from copy import deepcopy
from collections import deque
from typing import Union

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

def merge_tokens(tokenlist: TokenList, start_id: int, end_id: int) -> TokenList:
    """
    Merge tokens with IDs in [start_id, end_id] into one node (local root), adjusting IDs and heads.

    Parameters:
        tokenlist (TokenList): Input sentence.
        start_id (int): Start ID (inclusive, 1-based).
        end_id (int): End ID (inclusive, 1-based).

    Returns:
        TokenList: New token list with subtree merged and IDs/heads adjusted.
    """
    if not is_subtree(tokenlist, start_id, end_id):
        raise ValueError(f"Span [{start_id}, {end_id}] is not a valid subtree.")

    id_set = set(range(start_id, end_id + 1))
    id_to_token = {t["id"]: t for t in tokenlist if isinstance(t["id"], int)}

    # Find local root
    root_id = next(tid for tid in id_set if id_to_token[tid]["head"] not in id_set)
    root_token = deepcopy(id_to_token[root_id])

    # Merge form and lemma
    sorted_tokens = [id_to_token[i] for i in sorted(id_set)]
    merged_form = "".join(t["form"] for t in sorted_tokens if t["form"] != "_")
    merged_lemma = "".join(t.get("lemma", "") for t in sorted_tokens if t.get("lemma", "_") != "_")

    root_token["form"] = merged_form
    root_token["lemma"] = merged_lemma if merged_lemma else "_"

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
                # Redirect heads pointing into the subtree → to root
                if t["head"] in id_set:
                    t["head"] = root_id
                new_tokens.append(t)
        else:
            new_tokens.append(deepcopy(token))  # comments etc.

    # Reassign new sequential IDs and update heads
    old_to_new_id = {}
    for i, token in enumerate(new_tokens):
        old_id = token.get("id")
        if isinstance(old_id, int):
            new_id = i + 1
            old_to_new_id[old_id] = new_id
            token["id"] = new_id

    for token in new_tokens:
        head = token.get("head")
        if isinstance(head, int):
            token["head"] = old_to_new_id.get(head, head)

    return TokenList(new_tokens, metadata=tokenlist.metadata)

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

def remove_punct(data):
    """
    Remove punctuation tokens from either a TokenList or a SentenceList.
    If input is a TokenList, returns a new TokenList without punctuation.
    If input is a SentenceList, returns a new SentenceList with punctuation removed from each sentence.
    """
    if isinstance(data, TokenList):
        tokens_to_keep = []
        tokens_to_delete = []

        for token in data:
            upos = token.get("upos")
            deprel = token.get("deprel")
            if upos == "PUNCT" and deprel == "punct":
                tokens_to_delete.append(token)
            elif upos == "PUNCT" or deprel == "punct":
                print(f"Warning: Token ID={token['id']} meets only one of UPOS or DEPREL conditions. Not removed.")
                tokens_to_keep.append(token)
            else:
                tokens_to_keep.append(token)

        # Create ID mapping
        old_to_new_ids = {}
        new_id = 1
        for token in tokens_to_keep:
            old_to_new_ids[token["id"]] = new_id
            new_id += 1

        # Update IDs and HEADs
        updated_tokens = []
        for token in tokens_to_keep:
            new_token = token.copy()
            new_token["id"] = old_to_new_ids[token["id"]]
            if new_token["head"] != 0:
                new_token["head"] = old_to_new_ids.get(new_token["head"], 0)
            updated_tokens.append(new_token)

        return TokenList(updated_tokens, metadata=data.metadata)

    elif isinstance(data, SentenceList):
        new_sentences = SentenceList()
        for sentence in data:
            new_sentence = remove_punct(sentence)
            new_sentences.append(new_sentence)
        return new_sentences

    else:
        raise TypeError("Input must be either a TokenList or a SentenceList.")

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
    insert_comma_if_missing: bool = True,
    in_place: bool = True
) -> TokenList:
    """
    Merge two TokenList sentences into one.

    Parameters:
        sentence1 (TokenList): The first sentence (always on the left).
        sentence2 (TokenList): The second sentence (always on the right).
        deprel (str): The deprel label to assign when linking the root of the sub sentence to the main sentence.
        main_sent_id (int): Which sentence is the main clause (1 or 2).
        insert_comma_if_missing (bool): Whether to insert a comma if the left sentence does not end in punctuation.
        in_place (bool): If True, requires both sentences to come from the same SentenceList and be adjacent.

    Returns:
        TokenList: A new merged TokenList.
    """
    if main_sent_id not in (1, 2):
        raise ValueError("main_sent_id must be either 1 or 2.")

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
    for tok in clone_sent1:
        tok.tokenlist = clone_sent1
    for tok in clone_sent2:
        tok.tokenlist = clone_sent2

    # ==== Find roots ====
    root1 = clone_sent1.get_root()
    root2 = clone_sent2.get_root()
    main_root, sub_root = (root1, root2) if main_sent_id == 1 else (root2, root1)

    added_comma = False
    if insert_comma_if_missing and clone_sent1[-1].get("upos") != "PUNCT":
        comma = deepcopy(CHINESE_COMMA)
        clone_sent1.append(comma)
        comma.head = root1
        added_comma = True

    # ==== Link sub_root to main_root ====
    sub_root.head = main_root
    sub_root["deprel"] = deprel

    # ==== Merge metadata  ====
    connector = "，" if added_comma else ""
    merged_metadata = {
        "sent_id": clone_sent1.metadata.get("sent_id", ""),
        #"text": clone_sent1.metadata.get("text", "") + connector + clone_sent2.metadata.get("text", "")
    }

    # ==== Combine tokens and build new TokenList ====
    combined_tokens = clone_sent1 + clone_sent2
    merged = TokenList(combined_tokens, merged_metadata, build_refs=False)
    merged.refresh_ids()
    merged.refresh_heads()
    merged.refresh_metadata_text()

    # ==== Replace in SentenceList if in_place ====
    if in_place:
        slist = sentence1.sentencelist
        i1 = slist.index(sentence1)
        slist[i1] = merged
        del slist[i1 + 1]
        merged.sentencelist = slist

    return merged

def split_sentence_at_node(
    sentence: TokenList,
    target_node: Token,
    insert_comma_if_missing: bool = True,
    in_place: bool = True
):
    """
    Split a sentence at the subtree rooted at target_node.
    """

    # ==== Safety checks ====
    if target_node.tokenlist is not sentence:
        raise ValueError("The target node does not belong to the given sentence.")

    root = sentence.get_root()
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