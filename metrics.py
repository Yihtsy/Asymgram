from .conllu import Token, TokenList, SentenceList, to_sentencelist

Treebank = SentenceList | TokenList

'''
def uas(gold: Treebank, pred: Treebank) -> float:
    gold = to_sentencelist(gold)
    pred = to_sentencelist(pred)

    correct = 0
    total = 0

    for g_sent, p_sent in zip(gold, pred):
        for g_tok, p_tok in zip(g_sent, p_sent):
            token_id = str(g_tok.get("id", ""))
            if "-" in token_id or "." in token_id or token_id == "":
                continue  # skip multi-word/empty nodes
            total += 1
            if g_tok.get("head") == p_tok.get("head"):
                correct += 1

    return correct / total if total > 0 else 0.0
'''

def uas(gold: Treebank, pred: Treebank) -> float:
    gold = to_sentencelist(gold)
    pred = to_sentencelist(pred)

    correct = 0
    total = 0

    for g_sent, p_sent in zip(gold, pred):
        for g_tok, p_tok in zip(g_sent, p_sent):
            token_id = str(g_tok.id)
            if "-" in token_id or "." in token_id or token_id == "":
                continue # skip multi-word/empty nodes
            total += 1
            if g_tok.head == p_tok.head:
                correct += 1

    return correct / total if total > 0 else 0.0

'''
def las(gold: Treebank, pred: Treebank) -> float:
    gold = to_sentencelist(gold)
    pred = to_sentencelist(pred)

    correct = 0
    total = 0

    for g_sent, p_sent in zip(gold, pred):
        for g_tok, p_tok in zip(g_sent, p_sent):
            token_id = str(g_tok.get("id", ""))
            if "-" in token_id or "." in token_id or token_id == "":
                continue
            total += 1
            if (g_tok.get("head") == p_tok.get("head") and
                g_tok.get("deprel") == p_tok.get("deprel")):
                correct += 1

    return correct / total if total > 0 else 0.0
'''
def las(gold: Treebank, pred: Treebank) -> float:
    gold = to_sentencelist(gold)
    pred = to_sentencelist(pred)

    correct = 0
    total = 0

    for g_sent, p_sent in zip(gold, pred):
        for g_tok, p_tok in zip(g_sent, p_sent):
            token_id = str(g_tok.id)
            if "-" in token_id or "." in token_id or token_id == "":
                continue # skip multi-word/empty nodes
            total += 1
            if g_tok.head == p_tok.head and g_tok.deprel == p_tok.deprel:
                correct += 1

    return correct / total if total > 0 else 0.0