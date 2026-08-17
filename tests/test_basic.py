import asymgram as ag
from asymgram.error_checker import check_conllu_text


SAMPLE = """
# sent_id = demo-1
# text = I saw her.
1	I	_	PRON	_	_	2	nsubj	2:nsubj	_
2	saw	_	VERB	_	_	0	root	0:root	_
3	her	_	PRON	_	_	2	obj	2:obj	SpaceAfter=No
4	.	_	PUNCT	_	_	2	punct	2:punct	_
"""


def test_parse_and_root():
    sentences = ag.parse(SAMPLE)
    assert sentences[0].get_roottoken().form == "saw"
    assert sentences.to_tokenized_text() == "I saw her ."


def test_error_checker_clean_sample_has_no_errors():
    diagnostics = check_conllu_text(SAMPLE, include_corpus_checks=False)
    assert [d for d in diagnostics if d.level == "ERROR"] == []
