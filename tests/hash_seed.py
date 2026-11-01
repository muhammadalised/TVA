"""Run a builder like the tokenizer notebook does, with PYTHONHASHSEED=42.

TVA's BigramTokenizer breaks frequency ties in set() order, which depends on
the hash seed, so frozen bigram files only reproduce under the same seed.
"""
import importlib
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SEED = '42'
CODE = ('import importlib, pickle, sys; module, name = sys.argv[1].split(":"); '
        'args = pickle.loads(open(sys.argv[2], "rb").read()); '
        'result = getattr(importlib.import_module(module), name)(*args); '
        'open(sys.argv[3], "wb").write(pickle.dumps(result))')


def call_with_hash_seed(target: str, *args):
    """Call 'module:function'(*args) under PYTHONHASHSEED=42 and return its result."""
    if os.environ.get('PYTHONHASHSEED') == SEED:
        module, name = target.split(':')
        return getattr(importlib.import_module(module), name)(*args)
    with tempfile.TemporaryDirectory() as directory:
        path_args, path_result = Path(directory, 'args'), Path(directory, 'result')
        path_args.write_bytes(pickle.dumps(args))
        subprocess.run(
            [sys.executable, '-c', CODE, target, str(path_args), str(path_result)],
            cwd=ROOT, env=dict(os.environ, PYTHONHASHSEED=SEED, PYTHONPATH=str(ROOT)),
            check=True, capture_output=True,
        )
        return pickle.loads(path_result.read_bytes())
