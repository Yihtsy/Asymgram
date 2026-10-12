import asymgram as ag


SAMPLE = """
# sent_id = clause-1
# text = I know the man who left because he wanted to rest .
1	I	I	PRON	_	_	2	nsubj	_	_
2	know	know	VERB	_	_	0	root	_	_
3	the	the	DET	_	_	4	det	_	_
4	man	man	NOUN	_	_	2	obj	_	_
5	who	who	PRON	_	_	6	nsubj	_	_
6	left	leave	VERB	_	_	4	acl:relcl	_	_
7	because	because	SCONJ	_	_	9	mark	_	_
8	he	he	PRON	_	_	9	nsubj	_	_
9	wanted	want	VERB	_	_	2	advcl	_	_
10	to	to	PART	_	_	11	mark	_	_
11	rest	rest	VERB	_	_	9	xcomp	_	_
12	.	.	PUNCT	_	_	2	punct	_	_

# sent_id = clause-2
# text = I think he left .
1	I	I	PRON	_	_	2	nsubj	_	_
2	think	think	VERB	_	_	0	root	_	_
3	he	he	PRON	_	_	4	nsubj	_	_
4	left	leave	VERB	_	_	2	ccomp	_	_
5	.	.	PUNCT	_	_	2	punct	_	_
"""


def test_clause_type_recognition_and_extraction():
    s1, s2 = ag.parse(SAMPLE)
    assert ag.clause_type_for_node(s1, "know") == "main_clause"
    assert ag.clause_type_for_node(s1, "left") == "relative_clause"
    assert ag.clause_type_for_node(s1, "wanted") == "adverbial_clause"
    assert ag.clause_type_for_node(s1, "rest") == "open_complement_clause"
    assert ag.clause_type_for_node(s2, "left") == "complement_clause"
    assert ag.is_clause_node(s1, "left", clause_type="relative")
    assert ag.extract_relative_clause(s1, "left", as_text=True) == "who left"
    assert ag.extract_adverbial_clause(s1, "wanted", as_text=True) == "because he wanted to rest"
    assert ag.extract_open_complement_clause(s1, "rest", as_text=True) == "to rest"
    assert ag.extract_complement_clause(s2, "left", as_text=True) == "he left"
    assert ag.extract_relative_clause(s1, "wanted", as_text=True) is None
