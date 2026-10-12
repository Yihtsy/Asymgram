import asymgram as ag


SAMPLE = """
# sent_id = demo-1
# text = I saw her .
1	I	I	PRON	_	_	2	nsubj	_	_
2	saw	see	VERB	_	_	0	root	_	_
3	her	she	PRON	_	_	2	obj	_	_
4	.	.	PUNCT	_	_	2	punct	_	_

# sent_id = demo-2
# text = she smiled .
1	she	she	PRON	_	_	2	nsubj	_	_
2	smiled	smile	VERB	_	_	0	root	_	_
3	.	.	PUNCT	_	_	2	punct	_	_
"""


def test_find_tokens_and_concordance():
    sentences = ag.parse(SAMPLE)
    pronouns = ag.find_tokens(sentences, upos="PRON", include_punct=False)
    assert [m.token["form"] for m in pronouns] == ["I", "her", "she"]
    rows = ag.concordance(sentences, form="saw", window=1)
    assert rows[0]["left"] == "I"
    assert rows[0]["right"] == "her"


def test_update_tokens_returns_copy():
    sentences = ag.parse(SAMPLE)
    updated, changed = ag.update_tokens(sentences, {"upos": "NOUN"}, form="her")
    assert changed == 1
    assert ag.find_tokens(updated, form="her")[0].token["upos"] == "NOUN"
    assert ag.find_tokens(sentences, form="her")[0].token["upos"] == "PRON"


def test_stats_and_sampling():
    sentences = ag.parse(SAMPLE)
    assert ag.form_frequency(sentences)["she"] == 1
    assert ag.upos_deprel_table(sentences)[("PRON", "nsubj")] == 2
    assert ag.dependency_distances(sentences[0]) == [1, 1]
    summary = ag.corpus_summary(sentences)
    assert summary["sentences"] == 2
    assert summary["tokens"] == 5
    assert len(ag.sample_sentences(sentences, 1, seed=1)) == 1
