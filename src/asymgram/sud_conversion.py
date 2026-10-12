"""UD/SUD file conversion using Python rules or the optional Grew CLI."""

import os
from pathlib import Path
import subprocess
import tempfile
from io import StringIO
from itertools import chain


def convert_ud_to_sud(input_conllu, output_conllu, grs_path=None,
                      grew_exe="grew", config="sud", *, timeout=300,
                      backend="python", max_steps=10000):
    """Convert a file using official rules; return the absolute output Path.

    The Python backend executes the bundled pinned GRS without Grew.
    backend='grew' selects the external official CLI for comparison.

    Set grs_path or ASYMGRAM_UD_TO_SUD_GRS to UD_to_SUD.grs. Keep the
    complete official rules directory, including imported rule files.
    grew_exe is a native executable path or a command prefix. For WSL,
    run this Python function inside Linux so paths share the same runtime.
    Output is replaced only after a successful conversion. No model or rules
    are downloaded implicitly.
    """
    return _convert_file(input_conllu, output_conllu, grs_path, grew_exe, config,
                         timeout=timeout, backend=backend, max_steps=max_steps)


def convert_sud_to_ud(input_conllu, output_conllu, grs_path=None,
                      grew_exe="grew", config="sud", *, timeout=300,
                      backend="python", max_steps=10000):
    """Convert SUD to UD with bundled official rules or the optional Grew CLI.

    Set grs_path or ASYMGRAM_SUD_TO_UD_GRS to override bundled rules.
    Output is replaced atomically on success; conversion is not lossless.
    """
    return _convert_file(input_conllu, output_conllu, grs_path, grew_exe, config,
                         direction='SUD_to_UD', timeout=timeout, backend=backend,
                         max_steps=max_steps)


def _convert_file(input_conllu, output_conllu, grs_path=None, grew_exe="grew",
                  config="sud", *, direction='UD_to_SUD', timeout=300,
                  backend="python", max_steps=10000):
    source = Path(input_conllu).expanduser().resolve()
    destination = Path(output_conllu).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input CoNLL-U file not found: {source}")
    if source == destination:
        raise ValueError("Input and output must be different files.")
    if backend not in {'python', 'grew'}:
        raise ValueError("backend must be 'python' or 'grew'.")
    environment_key = 'ASYMGRAM_' + direction.upper() + '_GRS'
    rules = grs_path or os.environ.get(environment_key)
    if not rules and backend == 'python':
        rules = Path(__file__).parent / 'resources' / 'sud' / 'grs' / (direction + '.grs')
    if not rules:
        raise ValueError(f"Specify grs_path or {environment_key}.")
    rules = Path(rules).expanduser().resolve()
    if not rules.is_file():
        raise FileNotFoundError(f"Grew rules not found: {rules}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"Output directory not found: {destination.parent}")
    prefix = [os.fspath(grew_exe)] if isinstance(grew_exe, (str, os.PathLike)) else list(grew_exe)
    if not prefix:
        raise ValueError("grew_exe must specify an executable.")
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        temporary_output = Path(directory) / "converted.conllu"
        if backend == 'python':
            if config != 'sud':
                raise ValueError("Python backend supports config='sud' only.")
            from .conllu import parse_incr
            from .sud_python import _convert_sentence
            with source.open(encoding='utf-8') as reader, temporary_output.open('w', encoding='utf-8', newline='\n') as writer:
                block = []
                for line in chain(reader, ('\n',)):
                    if line.strip():
                        block.append(line.rstrip('\r\n'))
                        continue
                    if not block:
                        continue
                    sentence = next(parse_incr(StringIO('\n'.join(block) + '\n\n')))
                    try:
                        converted = _convert_sentence(sentence, rules, direction=direction, max_steps=max_steps).serialize()
                    except Exception as exc:
                        identifier = sentence.metadata.get('sent_id', '<missing sent_id>')
                        raise RuntimeError(f'{direction} failed at {identifier} in {source}: {exc}') from exc
                    comments = [line for line in block if line.startswith('#')]
                    rows = [line for line in converted.splitlines() if line and not line.startswith('#')]
                    writer.write('\n'.join(comments + rows) + '\n\n')
                    block = []
            os.replace(temporary_output, destination)
            return destination
        command = prefix + ["transform", "-grs", str(rules), "-config", config,
                            "-i", str(source), "-o", str(temporary_output)]
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=timeout,
                                    check=False)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Grew executable not found: {prefix[0]}. Install Grew or set grew_exe.") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Grew conversion timed out after {timeout} seconds.") from exc
        if result.returncode:
            raise RuntimeError(f"Grew failed ({result.returncode}): {result.stderr.strip() or result.stdout.strip()}")
        if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
            raise RuntimeError("Grew reported success without a non-empty output file.")
        os.replace(temporary_output, destination)
    return destination
