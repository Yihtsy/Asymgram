"""Dependency annotation scheme conversion helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .conllu import SentenceList, SentenceLists, TokenList
from .format_changer import read_conllu, write_conllu
from .tree_manipulation import convert_ud_function_dependents, restore_ud_function_dependents


SUD_CORE_FUNCTION_DEPRELS = ("case", "aux", "aux:pass", "mark", "cop")


def ud_to_sud(
    data,
    *,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    recursive: bool = False,
    overwrite: bool = False,
    target_deprels: Iterable[str] = SUD_CORE_FUNCTION_DEPRELS,
    in_place: bool = False,
):
    """Convert UD function-dependent structures to a SUD-style head scheme.

    This is a direct Python rule conversion, not a GRS interpreter. It currently
    covers the core function-head alternations that are explicit in asymgram:
    `case`, `aux`, `aux:pass`, `mark`, and `cop`.
    """
    return _convert_input(
        data,
        direction="ud_to_sud",
        transform=lambda item: convert_ud_function_dependents(
            item,
            target_deprels=tuple(target_deprels),
            in_place=in_place,
        ),
        output_path=output_path,
        output_dir=output_dir,
        recursive=recursive,
        overwrite=overwrite,
    )


def sud_to_ud(
    data,
    *,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    recursive: bool = False,
    overwrite: bool = False,
    target_deprels: Iterable[str] = SUD_CORE_FUNCTION_DEPRELS,
    in_place: bool = False,
):
    """Restore SUD-style function-head structures to UD dependencies."""
    return _convert_input(
        data,
        direction="sud_to_ud",
        transform=lambda item: restore_ud_function_dependents(
            item,
            target_deprels=tuple(target_deprels),
            in_place=in_place,
        ),
        output_path=output_path,
        output_dir=output_dir,
        recursive=recursive,
        overwrite=overwrite,
    )


def convert_ud_to_sud(
    input_conllu: str | Path,
    output_conllu: str | Path,
    *,
    overwrite: bool = True,
    target_deprels: Iterable[str] = SUD_CORE_FUNCTION_DEPRELS,
):
    """Convert one CoNLL-U file from UD to the current SUD-style scheme."""
    return ud_to_sud(
        input_conllu,
        output_path=output_conllu,
        overwrite=overwrite,
        target_deprels=target_deprels,
    )


def convert_sud_to_ud(
    input_conllu: str | Path,
    output_conllu: str | Path,
    *,
    overwrite: bool = True,
    target_deprels: Iterable[str] = SUD_CORE_FUNCTION_DEPRELS,
):
    """Convert one CoNLL-U file from the current SUD-style scheme to UD."""
    return sud_to_ud(
        input_conllu,
        output_path=output_conllu,
        overwrite=overwrite,
        target_deprels=target_deprels,
    )


def _convert_input(
    data,
    *,
    direction: str,
    transform,
    output_path: str | Path | None,
    output_dir: str | Path | None,
    recursive: bool,
    overwrite: bool,
):
    if output_path == "":
        output_path = None
    if output_dir == "":
        output_dir = None

    if isinstance(data, (TokenList, SentenceList, SentenceLists)):
        if output_path is not None or output_dir is not None or recursive or overwrite:
            raise ValueError("Output and directory options apply only to path inputs.")
        return transform(data)

    if not isinstance(data, (str, Path)):
        raise TypeError("data must be a TokenList, SentenceList, SentenceLists, file path, or directory path")

    source = Path(data).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input not found: {source}")

    suffix = "sud" if direction == "ud_to_sud" else "ud"
    if source.is_file():
        if output_dir is not None or recursive:
            raise ValueError("Use output_path for a single file.")
        destination = (
            Path(output_path).expanduser().resolve()
            if output_path is not None
            else source.with_name(source.stem + "." + suffix + ".conllu")
        )
        pairs = [(source, destination)]
    elif source.is_dir():
        if output_path is not None:
            raise ValueError("Use output_dir for a directory.")
        destination = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else source.with_name(source.name + "_" + suffix)
        )
        if destination == source or source in destination.parents:
            raise ValueError("Output directory must be outside the input directory.")
        files = sorted(
            path for path in (source.rglob("*") if recursive else source.iterdir())
            if path.is_file() and path.suffix.lower() == ".conllu"
        )
        pairs = [(path, destination / path.relative_to(source)) for path in files]
    else:
        raise ValueError(f"Unsupported input path: {source}")

    for path, target in pairs:
        if path == target or (target.exists() and path.samefile(target)):
            raise ValueError("Input and output must be different files.")
        if target.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {target}")
        if target.exists() and not target.is_file():
            raise ValueError(f"Output is not a file: {target}")

    results = []
    for path, target in pairs:
        target.parent.mkdir(parents=True, exist_ok=True)
        converted = transform(read_conllu(str(path)))
        write_conllu(converted, str(target))
        results.append(target)
    return results[0] if source.is_file() else results
