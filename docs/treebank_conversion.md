# Treebank Conversion

Applicable version: `asymgram 0.2.3`.

## F9 UD to SUD

- Entry: `ag.ud_to_sud(data, output_path=None, output_dir=None, recursive=False, overwrite=False, in_place=False, max_steps=10000, trace=None)`.
- Trigger: Convert a UD sentence or corpus to the universal SUD scheme.
- Behavior: Interpret the bundled official GRS rules with the embedded Python matcher and executor. No Grew executable, model, or additional dependency is required.
- Inputs: `TokenList`, `SentenceList`, `SentenceLists`, a file, or a directory. Memory returns corresponding copies unless `in_place=True`; paths return `Path` or a list of `Path` objects.
- Output: File defaults to `name.sud.conllu`; directory defaults to sibling `name_sud`. Preserve relative filenames; recursion and overwrite are explicit. Output directories must be outside input directories.
- File handling: Preserve original comments, write atomically on success, and leave source files untouched. Batch failures stop with earlier completed outputs retained.
- Limits: Preserve ordinary tokens and MWT rows; remove enhanced DEPS and empty nodes. No lossless round-trip guarantee. `trace` applies only to memory inputs. Explicit `target_deprels` selects the older experimental local mode, not complete SUD.

## F10 SUD to UD

- Entry: `ag.sud_to_ud(data, ...)`; options and shapes follow F9.
- Trigger: Convert SUD annotations to the universal UD-oriented output scheme.
- Behavior: Interpret the bundled official reverse GRS rules. Default output names are `name.ud.conllu` and sibling `name_ud`.
- Limits: Does not recover all original UD distinctions, enhanced dependencies, or empty nodes. Conversion direction must be supplied by the caller; annotations are not auto-detected.
- File-specific APIs: `convert_ud_to_sud(input_conllu, output_conllu, overwrite=True)` and `convert_sud_to_ud(...)`. Only the Python backend is supported.

## Reference And Verification

- Reference: [official SUD conversions](https://github.com/surfacesyntacticud/conversions), revision `7cd14fd1a5e106c18f2ddc9a597e02fe86fddb78`.
- The bundled rules are from the pinned official SUD conversion repository, revision `7cd14fd1a5e106c18f2ddc9a597e02fe86fddb78`; the Python implementation interprets them at runtime.
- Python 3.14 source test suite: 20 tests passed, including a coordinated-adposition regression found during full validation. All 278 command blocks and 12,752 condition comparisons matched the previous implementation during translation checks.
- English-PUD test (1000 sentences) and English-EWT test (2077 sentences): full forward output byte-identical to official SUD2.18 files.
- Full reverse checks cover the same English corpora and Chinese-GSDSimp test (500 sentences). These check compatibility with the prior Python GRS implementation, not an independent official Grew executable.

## Packaging

- Conversion components including public dispatch: direct Python ZIP 33,451 bytes; previous interpreter plus minimal GRS closure 41,392 bytes; previous interpreter plus all GRS resources 74,549 bytes.
- Uncompressed direct Python source is larger than the minimal reference design (237,672 versus 158,199 bytes); compression and installed source size are different measures.
- A twenty-sentence warm-run benchmark gave medians 2.603 seconds for direct Python and 2.987 seconds for the previous implementation. Concurrent full checks may affect timing; this is not a general performance guarantee.
- Wheel and sdist must contain `resources/sud/grs`, including imported `.grs` rules and `.lp` lexicon files; conversion has no Grewpy or OCaml runtime dependency.
