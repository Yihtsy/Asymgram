import asymgram as ag


def test_osborne_style_unit_counts():
    text = """# sent_id = 1
# text = Catenae are everywhere .
1	Catenae	catena	NOUN	noun	_	2	nsubj	_	_
2	are	be	VERB	verb	_	0	root	_	_
3	everywhere	everywhere	ADV	adv	_	2	advmod	_	_
4	.	.	PUNCT	punct	_	2	punct	_	_
"""
    sentence = ag.parse(text)[0]

    assert ag.count_strings(sentence, include_punct=False) == 6
    assert ag.count_catenae(sentence, include_punct=False) == 6
    assert ag.count_components(sentence, include_punct=False) == 6
    assert ag.count_constituents(sentence, include_punct=False) == 3
    assert ag.count_phrases(sentence, include_punct=False) == 1
    assert ag.count_clauses(sentence, include_punct=False) == 1

    ratios = ag.unit_ratios(sentence, include_punct=False)
    assert ratios["all_combinations"] == 7
    assert ratios["catena_ratio"] == "6:1"


def test_clause_unit_properties():
    text = """# sent_id = 1
# text = I know he left .
1	I	I	PRON	pron	_	2	nsubj	_	_
2	know	know	VERB	verb	_	0	root	_	_
3	he	he	PRON	pron	_	4	nsubj	_	_
4	left	leave	VERB	verb	_	2	ccomp	_	_
5	.	.	PUNCT	punct	_	2	punct	_	_
"""
    sentence = ag.parse(text)[0]
    clauses = list(ag.iter_clauses(sentence, include_punct=False))

    assert [clause.clause_type for clause in clauses] == ["main_clause", "complement_clause"]
    assert [clause.text for clause in clauses] == ["I know he left", "he left"]
