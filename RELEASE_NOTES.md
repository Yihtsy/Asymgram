# asymgram 0.2.5

- Converted `tree_manipulation` from a single module into a package while preserving top-level `asymgram` function imports.
- Moved parser and language postprocessing into `tree_manipulation.postprocessing`.
- Removed the experimental analysis-sequence API from the public package.
- Consolidated general token/sentence queries, structural tree queries, and the syntax-query DSL into `query.py`.
- Removed the empty or compatibility-only query modules `tree_query.py`, `query_syntax.py`, and `syntax_query.py`.
- Removed the compatibility-only private modules `grs_execution.py`, `grs_matcher.py`, and `sud_python.py`, plus the duplicated `sud_conversion.py` file wrapper; their implementations now live in `scheme_changer.py`.
- Kept the bundled official UD/SUD `.grs` rules and `.lp` lexicon resources in distribution metadata.
- Declared the existing `numpy` runtime dependency in package metadata.

# asymgram 0.2.2

- Restored complete universal UD/SUD conversion as direct Python functions, replacing the restricted default with 128 forward and 150 reverse rules.
- Removed runtime GRS parsing and external Grew requirements; predicates, command functions, and execution order are checked-in Python.
- Preserved sentence/document/corpus and file/directory APIs, original file comments, atomic file output, traces, and rewrite bounds.
- Retained explicit `target_deprels` as experimental local-alternation compatibility, not as an official SUD implementation or a lossless converter.
- Kept the existing `0.2.1` tree-manipulation fixes. Complete-conversion verification and packaging comparison are recorded in the development manual.
- Python 3.14 source suite: 20 tests passed. Full forward PUD and EWT test outputs match official SUD2.18 files byte-for-byte; full reverse regression covers PUD, EWT test, and Chinese-GSDSimp test.

# asymgram 0.2.1

Feature release with expanded treebank manipulation, scheme-conversion, syntax-query, subcorpus, and reporting support.

## Highlights

- Added compact syntax-query APIs: `SyntaxQuery`, `compile_syntax_query()`, `match_syntax_tokens()`, `sentence_matches_syntax()`, and `find_syntax_matches()`.
- Added subcorpus extraction APIs: `filter_sentences()`, `extract_subcorpus()`, and `write_subcorpus()`.
- Added Python rule-based scheme-conversion APIs: `ud_to_sud()`, `sud_to_ud()`, `convert_ud_to_sud()`, and `convert_sud_to_ud()` for core function-dependent/function-head alternations.
- Expanded treebank manipulation helpers for non-word line removal, token insertion, empty-node insertion, punctuation normalization, edge-punctuation removal, hyphenated-compound merging, and sentence-sequence merging.
- Restored and integrated newer local conversion helpers for CHAT, XML corpus input, ConLLUB/ConLLUC output, sentence lookup, and token-position lookup.
- Added language-specific postprocessing APIs: `apply_postprocess_profile()`, `postprocess_zh_stanza()`, and `postprocess_zh_tokenized_text()`.
- Preserved newer CoNLL-U diagnostic reporting with `generate_error_report()` and `generate_summary_report()`, while adding `generate_analysis_report()` as a compatibility alias.
- Added structural-unit and tree-query APIs developed after `0.1.0`, including catena/component/constituent/phrase/clause iteration and counting.

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
