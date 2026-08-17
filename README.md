# Asymgram

**Asymgram** stands for **Asymmetrency Grammar**, a syntactic framework under development that is closely related to dependency grammar. The framework is designed to support syntactic annotation, manipulation, and structural analysis of dependency-based linguistic data.

The name "AG" is also chosen in honor of the author's favorite esports club: Chengdu AG (All Gamers).

The Python package `asymgram` is a small toolkit for working with CoNLL-U dependency treebanks. It wraps parsed CoNLL-U tokens and sentences with convenient object references, provides tree manipulation helpers, evaluates dependency parses, and includes diagnostics for common annotation and format errors.

The package is designed for corpus linguistics, dependency grammar research, Universal Dependencies style treebank work, and quick inspection of parser output.

## Features

- Enhanced CoNLL-U objects: `Token`, `TokenList`, `SentenceList`, and `SentenceLists`.
- Context-aware token navigation, including head lookup, dependent lookup, subtree span extraction, and node depth.
- Sentence and token manipulation helpers for merging tokens, splitting tokens, removing leaf nodes, merging sentences, and splitting sentences at marginal subtrees.
- Basic dependency evaluation metrics: UAS and LAS.
- CoNLL-U readers, writers, text reconstruction, and simple conversion helpers.
- Error checking for common CoNLL-U and dependency treebank issues, including invalid IDs, malformed HEAD values, self-loops, cycles, missing roots, multiple roots, invalid UPOS/DEPREL labels, malformed FEATS/DEPS, punctuation mismatches, and suspicious UPOS-DEPREL combinations.

## Installation

```bash
pip install asymgram
```

For local development:

```bash
git clone https://github.com/Yihtsy/asymgram.git
cd asymgram
pip install -e .
```

Optional NLP-related dependencies can be installed with:

```bash
pip install "asymgram[nlp]"
```

## Quick Start

```python
import asymgram as ag

text = """
# sent_id = demo-1
# text = I saw her.
1	I	_	PRON	_	_	2	nsubj	2:nsubj	_
2	saw	_	VERB	_	_	0	root	0:root	_
3	her	_	PRON	_	_	2	obj	2:obj	SpaceAfter=No
4	.	_	PUNCT	_	_	2	punct	2:punct	_
"""

sentences = ag.parse(text)
sentence = sentences[0]
root = sentence.get_roottoken()

print(root.form)
print([tok.form for tok in root.get_deptokens(include_punct=True)])
print(sentences.to_tokenized_text())
```

## Error Checking

`asymgram.error_checker` can inspect raw CoNLL-U text, files, parsed sentences, or sentence lists.

```python
from asymgram.error_checker import check_conllu_text, check_file

diagnostics = check_conllu_text(text)

for diagnostic in diagnostics:
    print(diagnostic.level, diagnostic.code, diagnostic.sent_id, diagnostic.token_id, diagnostic.message)

file_diagnostics = check_file("treebank.conllu")
```

Each diagnostic is represented by a `Diagnostic` object:

```python
Diagnostic(
    code="SELF_LOOP",
    level="ERROR",
    message="token cannot take itself as HEAD",
    sent_id="demo-1",
    token_id=3,
)
```

Diagnostic levels are:

- `ERROR`: structural or format problems that are usually automatically decidable.
- `WARNING`: strong annotation warnings that should be manually checked.
- `INFO`: low-frequency or linguistically atypical patterns that may still be valid.

## Common APIs

```python
import asymgram as ag

sentences = ag.read_conllu("sample.conllu")
sentences.to_conllu("copy.conllu")

root = sentences[0].get_roottoken()
children = root.get_deptokens()
span = root.get_subtree_span()

uas = ag.uas(gold_sentences, predicted_sentences)
las = ag.las(gold_sentences, predicted_sentences)
```

## Notes

`asymgram` follows the CoNLL-U conventions used by Universal Dependencies, but it is not limited to official UD treebanks. Some diagnostics intentionally return warnings or info-level messages because many unusual UPOS-DEPREL combinations can be valid in ellipsis, multiword expressions, headless constructions, dates, measurements, names, and language-specific analyses.

## License

License information has not yet been specified.
