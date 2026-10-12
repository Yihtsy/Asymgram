# asymgram.error_checker

This document summarizes the current CoNLL-U error-checking behavior in `asymgram.error_checker`.

## Main APIs

- `check_conllu_text(text, include_corpus_checks=True)`: check raw CoNLL-U text.
- `check_file(path, include_corpus_checks=True)`: check one CoNLL-U file.
- `check_sentence(sent)`: check one parsed sentence.
- `check_sentence_list(sentences, include_corpus_checks=True)`: check parsed sentences.
- `generate_error_report(data, output_path=None, output_format="md")`: generate an error-focused report.
- `generate_summary_report(data, output_path=None, output_format="md")`: generate descriptive corpus statistics.

## Diagnostic Levels

- `ERROR`: structural or format problems that are usually automatically decidable.
- `WARNING`: annotation patterns or corpus patterns that should be manually checked.

The checker no longer emits `INFO`.

## Error Report

`generate_error_report()` focuses on diagnostics. It includes:

- source path and checked CoNLL-U file list;
- diagnostic counts by level and code;
- grouped diagnostic details with all locations;
- `UPOS-DEPREL Combinations for Review`;
- `Non-UD DEPREL Labels`.

The `UPOS-DEPREL Combinations for Review` table is a combination-level view of highlighted labels. `suspicious` and `atypical` rows are already counted above as token-level diagnostics. `rare` rows are corpus-frequency review candidates and are shown when such combinations exist.

Example cells in the review table use one surface form only: `sent_id:token_id:form->hHEAD`.

## Summary Report

`generate_summary_report()` is separate from error checking. It includes descriptive statistics such as:

- `UPOS Distribution`;
- `DEPREL Distribution`;
- `Root UPOS Distribution`;
- `Full UPOS-DEPREL Frequency Table`.

## UPOS-DEPREL Review Status

- `non_ud`: the `DEPREL` label is outside the UD relation inventory.
- `suspicious`: the `(UPOS, DEPREL)` pair is in the built-in strong-warning list.
- `atypical`: the pair is not in the built-in expected mapping for that `UPOS`.
- `rare`: the pair is valid and expected but appears at or below the corpus frequency threshold.

Priority is `non_ud` > `suspicious` > `atypical` > `rare`. If a pair matches more than one status, the higher-priority status is shown.

## Current Checks

Format and raw CoNLL-U checks:

- `COLUMN_COUNT`
- `MISSING_SENTENCE_SEPARATOR`
- `MALFORMED_METADATA`
- `MALFORMED_ID`
- `MALFORMED_HEAD`
- `MALFORMED_FEATS`
- `MALFORMED_DEPS`

Basic tree checks:

- `DUPLICATE_SENT_ID`: the same `sent_id` appears more than once in the checked corpus, including parsed sentence lists or multiple files loaded from one directory.
- `DUPLICATE_ID`
- `WORD_ID_SEQUENCE`
- `HEAD_TYPE`
- `HEAD_NOT_FOUND`
- `SELF_LOOP`
- `CYCLE`
- `DISCONNECTED_TREE`
- `NO_ROOT`
- `NO_ROOT_DEPREL`
- `MULTIPLE_ROOTS`
- `MULTIPLE_ROOT_DEPRELS`
- `HEAD_ROOT_MISMATCH`
- `DEPREL_ROOT_MISMATCH`

Multiword token and empty node checks:

- `MWT_RANGE_ORDER`
- `MWT_MISSING_CONSTITUENT`
- `MWT_OVERLAP`
- `MWT_EMPTY_HEAD`
- `EMPTY_NODE_BASE_MISSING`
- `EMPTY_NODE_ID`
- `EMPTY_NODE_SEQUENCE`

UD label and feature checks:

- `INVALID_UPOS`
- `INVALID_DEPREL`
- `UPOS_FEATS_MISMATCH`
- `MUTUALLY_SUSPICIOUS_FEATS`

Punctuation checks:

- `PUNCT_NON_PUNCT`
- `NON_PUNCT_PUNCT`
- `PUNCT_AS_ROOT`
- `PUNCT_AS_HEAD`

UPOS-DEPREL combination checks:

- `SUSPICIOUS_UPOS_DEPREL`
- `ATYPICAL_UPOS_DEPREL`
- `RARE_UPOS_DEPREL`

Enhanced dependency checks:

- `INVALID_DEPS_DEPREL`
- `DEPS_HEAD_NOT_FOUND`
- `ENHANCED_SELF_LOOP`
- `DUPLICATE_DEPS_EDGE`
- `MULTIPLE_DEPS_RELS_SAME_HEAD`
- `BASIC_EDGE_MISSING_FROM_DEPS`
- `ENHANCED_ROOT_NONROOT_TOKEN`
- `OVER_SPECIFIED_DEPS_DEPREL`
- `ENHANCED_CONJ_CHAIN_HEAD`
- `ENHANCED_CONJ_RECIPROCAL`

Coordination and text reconstruction checks:

- `CONJ_CHAIN_HEAD`
- `TEXT_FORM_MISMATCH`

## Notes On Removed Or Limited Checks

- `root` with `NOUN` or `ADJ` is not treated as a default error.
- `FORM_UPOS_INCONSISTENCY`, `LEMMA_UPOS_INCONSISTENCY`, and `LEMMA_DEPREL_INCONSISTENCY` were removed from default checking because minority annotation by form or lemma is not reliable enough as an error signal.
- `BASIC_EDGE_MISSING_FROM_DEPS` is corpus-gated. It is emitted only when most tokens that have non-empty `DEPS` preserve their corresponding basic edge. Corpora without meaningful `DEPS` annotation are not checked for this condition.
