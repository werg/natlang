#!/usr/bin/env python3
"""CLI compatibility entrypoint for the shared exact capacity verifier."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'training/neuralese'))
from natlang_neuralese.train import native_capacity_continuation as _mechanism

# Preserve callers importing the historical script's verifier and policy names.
globals().update({name: value for name, value in vars(_mechanism).items()
                  if not name.startswith('__')})

if __name__ == '__main__':
    _mechanism.main()
