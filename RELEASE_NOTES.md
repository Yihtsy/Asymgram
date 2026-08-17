# asymgram 0.1.0

Initial public release of `asymgram`.

## Highlights

- Enhanced CoNLL-U data classes: `Token`, `TokenList`, `SentenceList`, and `SentenceLists`.
- CoNLL-U parsing, incremental parsing, serialization, text reconstruction, and file helpers.
- Dependency tree navigation helpers for heads, dependents, subtree spans, leaf nodes, and depth.
- Tree manipulation utilities for token merging/splitting, sentence merging/splitting, punctuation removal, and Chinese dependency-parse postprocessing helpers.
- UAS and LAS evaluation metrics.
- Expanded CoNLL-U error diagnostics for format, ID, HEAD/root, cycle, UPOS/DEPREL, FEATS, DEPS, punctuation, and corpus-level consistency checks.

## Installation

```bash
pip install asymgram
```
