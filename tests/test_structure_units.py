import asymgram as ag


SAMPLE = """
# sent_id = units-1
# text = The students understood the surprised instructor .
1	The	the	DET	_	_	2	det	_	_
2	students	student	NOUN	_	_	3	nsubj	_	_
3	understood	understand	VERB	_	_	0	root	_	_
4	the	the	DET	_	_	6	det	_	_
5	surprised	surprised	ADJ	_	_	6	amod	_	_
6	instructor	instructor	NOUN	_	_	3	obj	_	_
7	.	.	PUNCT	_	_	3	punct	_	_
"""


def test_generic_units():
    sent = ag.parse(SAMPLE)[0]
    assert ag.is_string(sent, [1, 2])
    assert not ag.is_string(sent, [2, 4])
    assert ag.is_catena(sent, [1, 2])
    assert ag.is_catena(sent, [3, 6])
    assert not ag.is_catena(sent, [2, 4])
    assert ag.is_component(sent, [1, 2])
    assert not ag.is_component(sent, [3, 6])
    assert ag.is_constituent(sent, [1, 2])
    assert not ag.is_constituent(sent, [3, 6])
    assert ag.is_phrase(sent, [4, 5, 6])
    assert ag.catena_root(sent, [4, 5, 6])["form"] == "instructor"
    assert ag.catena_head(sent, [4, 5, 6])["form"] == "understood"


def test_relational_units():
    sent = ag.parse(SAMPLE)[0]
    assert ag.get_parent(sent, "students")["form"] == "understood"
    assert [tok["form"] for tok in ag.get_children(sent, "understood")] == ["students", "instructor"]
    assert [tok["form"] for tok in ag.get_pre_children(sent, "understood")] == ["students"]
    assert [tok["form"] for tok in ag.get_post_children(sent, "understood")] == ["instructor"]
    dependents = ag.get_dependent_constituents(sent, "understood")
    assert [[tok["form"] for tok in dep] for dep in dependents] == [["The", "students"], ["the", "surprised", "instructor"]]
