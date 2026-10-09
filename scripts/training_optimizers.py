"""Compatibility shim: the optimizer moved to natlang_neuralese.train.optim (plans/ARCHITECTURE_IMPROVEMENT.md C2).

Running and frozen jobs import `scripts.training_optimizers`; keep these names stable.
"""
import sys
from pathlib import Path

try:
    from natlang_neuralese.train.optim import MuonWithAdamW, make_muon_optimizer
except ImportError:  # launched without training/neuralese on the path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "neuralese"))
    from natlang_neuralese.train.optim import MuonWithAdamW, make_muon_optimizer

__all__ = ["MuonWithAdamW", "make_muon_optimizer"]
