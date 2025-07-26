from .conllu import parse_incr

def two_unidirectional_depr(token, sent):
    if token['deps']:
        if any(depr1[1] == depr2[1] and id(depr1) != id(depr2) for depr1 in token['deps'] for depr2 in token['deps']):
            return True
    return False

def punct_as_root(token, sent, enh = False):
    if enh == False:
        if token['head'] == 0 and token['upostag'] == 'PUNCT':
            return True

def head_deprel_mismatch(token, sent, enh = False):
    if enh == False:
        if token['head'] == 0:
            if not token['deprel'] == 'root':
                return True

def upos_deprel_mismatch(token, sent):
    if token['upostag'] == 'PUNCT':
        if not token['deprel'] == 'punct':
            return True

def find_id_equals_head_errors(sentences):
    """
    Given a SentenceList, return all tokens where 'id' == 'head'.
    Returns a list of tuples: (newdoc_id, sent_id, token_id)
    """
    doc_id = sentences.metadata.get("newdoc id", "NA")  # Get document ID from SentenceList-level metadata
    errors = []
    for sent in sentences:
        sent_id = sent.metadata.get("sent_id", "NA")  # Get sentence ID from sentence-level metadata
        for token in sent:
            # Check if the token is a regular word (not multiword) and has head == id
            if isinstance(token["id"], int) and token["head"] == token["id"]:
                errors.append((doc_id, sent_id, token["id"]))
    return errors

def exists_error(sent):
    for token in sent:
        if any([
            head_deprel_mismatch(token, sent),
            two_unidirectional_depr(token, sent),
            punct_as_root(token, sent)
        ]):
            return True