import asymgram as ag


PROJECTIVE = """
# sent_id = p1
# text = The very small dog chased the cat .
1	The	the	DET	_	_	4	det	_	_
2	very	very	ADV	_	_	3	advmod	_	_
3	small	small	ADJ	_	_	4	amod	_	_
4	dog	dog	NOUN	_	_	5	nsubj	_	_
5	chased	chase	VERB	_	_	0	root	_	_
6	the	the	DET	_	_	7	det	_	_
7	cat	cat	NOUN	_	_	5	obj	_	_
8	.	.	PUNCT	_	_	5	punct	_	_
"""


CROSSING = """
# sent_id = c1
# text = a b c d
1	a	a	NOUN	_	_	3	dep	_	_
2	b	b	NOUN	_	_	4	dep	_	_
3	c	c	NOUN	_	_	0	root	_	_
4	d	d	NOUN	_	_	3	dep	_	_
"""


def test_tree_judgements():
    sent = ag.parse(PROJECTIVE)[0]
    assert ag.is_connected_tree(sent)
    assert ag.is_dependency_tree(sent)
    assert ag.is_projective_tree(sent)
    assert not ag.has_crossing_dependencies(sent)

    crossing = ag.parse(CROSSING)[0]
    assert ag.has_crossing_dependencies(crossing)
    assert not ag.is_projective_tree(crossing)


def test_token_relations_and_metrics():
    sent = ag.parse(PROJECTIVE)[0]
    dog = ag.resolve_token(sent, "dog")
    assert ag.get_head(sent, dog)["form"] == "chased"
    assert ag.get_root(sent)["form"] == "chased"
    assert [tok["form"] for tok in ag.get_dependents(sent, dog)] == ["The", "small"]
    assert ag.get_right_sibling(sent, dog)["form"] == "cat"
    assert ag.get_left_sibling(sent, "cat")["form"] == "dog"

    metrics = ag.node_metrics(sent, "dog")
    assert metrics["dependency_distance"] == 1
    assert metrics["directional_dependency_distance"] == -1
    assert metrics["level"] == 1
    assert metrics["subtree_size"] == 4


def test_phrase_extraction():
    sent = ag.parse(PROJECTIVE)[0]
    assert ag.extract_noun_phrase(sent, "dog", as_text=True) == "The very small dog"
    assert ag.extract_verb_phrase(sent, "chased", as_text=True) == "The very small dog chased the cat"
    assert ag.subtree_span(sent, "dog") == (1, 4)
