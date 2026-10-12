"""Embedded GRS conversion contracts and structural regression examples."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import pytest
import asymgram as ag

TEXT = '# sent_id = simple\n1\tShe\tshe\tPRON\t_\t_\t2\tnsubj\t2:nsubj\tSpaceAfter=No\n2\truns\trun\tVERB\t_\t_\t0\troot\t0:root\t_\n\n'


def test_memory_types_copy_and_in_place():
    sentence = ag.parse(TEXT)[0]
    original = deepcopy(sentence)
    result = ag.ud_to_sud(sentence)
    assert result[0]['deprel'] == 'subj'
    assert result[0]['deps'] is None
    result[0]['misc']['SpaceAfter'] = 'Yes'
    assert sentence == original
    document = ag.SentenceList([sentence], metadata={'source': ['original']})
    converted = ag.ud_to_sud(document)
    converted.metadata['source'].append('changed')
    assert document.metadata['source'] == ['original']
    corpus = ag.ud_to_sud(ag.SentenceLists([document]))
    assert isinstance(corpus, ag.SentenceLists)
    assert ag.sud_to_ud(corpus)[0][0][0]['deprel'] == 'nsubj'
    assert ag.ud_to_sud(sentence, in_place=True) is sentence
    assert sentence[0]['deprel'] == 'subj'


def test_file_directory_and_comments(tmp_path):
    source = tmp_path / 'in.conllu'
    source.write_text('# free comment\n' + TEXT, encoding='utf-8')
    output = ag.ud_to_sud(source)
    assert output == source.with_name('in.sud.conllu')
    assert '# free comment' in output.read_text(encoding='utf-8')
    with pytest.raises(FileExistsError):
        ag.ud_to_sud(source)
    ag.ud_to_sud(source, output_path='', overwrite=True)
    assert ag.sud_to_ud(output).name == 'in.sud.ud.conllu'
    directory = tmp_path / 'corpus'
    (directory / 'sub').mkdir(parents=True)
    (directory / 'sub' / 'one.CONLLU').write_text(TEXT, encoding='utf-8')
    assert ag.ud_to_sud(directory, recursive=True) == [tmp_path / 'corpus_sud' / 'sub' / 'one.CONLLU']
    with pytest.raises(ValueError):
        ag.ud_to_sud(directory, output_dir=directory / 'out')
    with pytest.raises(ValueError):
        ag.ud_to_sud(source, output_path=source, overwrite=True)
    output.write_text('existing', encoding='utf-8')
    with pytest.raises(RuntimeError, match='simple'):
        ag.convert_ud_to_sud(source, output, max_steps=0)
    assert output.read_text(encoding='utf-8') == 'existing'


def test_preposition_full_scheme_and_reverse():
    sentence = ag.parse('1\tgo\tgo\tVERB\t_\t_\t0\troot\t_\t_\n'
                        '2\tto\tto\tADP\t_\t_\t3\tcase\t_\t_\n'
                        '3\tschool\tschool\tNOUN\t_\t_\t1\tobl\t_\t_\n\n')[0]
    result = ag.ud_to_sud(sentence)
    assert (result[1]['head'], result[1]['deprel']) == (1, 'udep')
    assert (result[2]['head'], result[2]['deprel']) == (2, 'comp:obj')
    reverse = ag.sud_to_ud(result)
    assert (reverse[1]['head'], reverse[1]['deprel']) == (3, 'case')
    assert (reverse[2]['head'], reverse[2]['deprel']) == (1, 'obl')


def test_forward_uses_bundled_official_grs_without_external_executable():
    sentence = ag.parse(TEXT)[0]
    result = ag.ud_to_sud(sentence)
    assert ag.sud_to_ud(result)[0]['deprel'] == 'nsubj'
    package = Path(ag.__file__).parent
    assert (package / 'resources' / 'sud' / 'grs' / 'UD_to_SUD.grs').is_file()
    assert (package / 'resources' / 'sud' / 'grs' / 'SUD_to_UD.grs').is_file()


def test_bracketed_misc_and_empty_nodes():
    sentence = ag.parse('1\traciste\traciste\tADJ\t_\tNumber=Sing\t0\troot\t_\tGender[ctxt]=Masc\n\n')[0]
    result = ag.sud_to_ud(sentence)
    assert result[0]['feats']['Gender'] == 'Masc'
    assert result[0]['misc'] == {'Exponence[Gender]': 'Absent'}
    sentence = ag.parse('1-2\tcan\'t\t_\t_\t_\t_\t_\t_\t_\t_\n'
                        '1\tca\tcan\tAUX\t_\t_\t2\taux\t2:aux\t_\n'
                        '2\tnot\tnot\tVERB\t_\t_\t0\troot\t0:root\t_\n'
                        '2.1\tgo\tgo\tVERB\t_\t_\t_\t_\t2:conj\t_\n\n')[0]
    result = ag.ud_to_sud(sentence)
    assert any(t['id'] == (1, '-', 2) for t in result)
    assert all(t['id'] != (2, '.', 1) for t in result)


def test_explicit_legacy_targets_and_argument_validation():
    sentence = ag.parse(TEXT)[0]
    assert ag.ud_to_sud(sentence, target_deprels=('case',))[0]['deprel'] == 'nsubj'
    with pytest.raises(TypeError):
        ag.ud_to_sud([sentence])
    with pytest.raises(ValueError):
        ag.ud_to_sud(sentence, output_path='out.conllu')


def test_coordinated_adpositions_keep_coordination_on_function_word():
    sentence = ag.parse('1\torders\torder\tNOUN\t_\t_\t0\troot\t_\t_\n'
                        '2\tat\tat\tADP\t_\t_\t1\tudep\t_\t_\n'
                        '3\tor\tor\tCCONJ\t_\t_\t4\tcc\t_\t_\n'
                        '4\tover\tover\tADP\t_\t_\t2\tconj:coord\t_\t_\n'
                        '5\tamount\tamount\tNOUN\t_\tShared=Yes\t4\tcomp:obj\t_\t_\n\n')[0]
    converted = ag.sud_to_ud(sentence)
    assert (converted[3]['head'], converted[3]['deprel']) == (2, 'conj')
    assert (converted[1]['head'], converted[1]['deprel']) == (5, 'case')
