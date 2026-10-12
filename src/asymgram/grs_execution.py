"""Compatibility shim for the embedded GRS executor.

The implementation now lives in asymgram.scheme_changer.
"""
from . import scheme_changer as _implementation

def __getattr__(name):
    return getattr(_implementation, name)

def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))

