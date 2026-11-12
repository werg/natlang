"""`scripts/neuralese_data/lfm3_fast_train` (banded sliding attention, shifted-add conv) against the stock HF forward and
backward on a tiny LFM3 twin whose window is shorter than the sequence (FP32, CPU: the same math, not the same kernels)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/neuralese_data'))
import lfm3_fast_train  # noqa: E402
from test_lfm3_port import tiny_lfm3_hf  # noqa: E402


def test_fast_forward_and_gradients_match_stock():
    stock = tiny_lfm3_hf(window=4)
    fast = tiny_lfm3_hf(window=4)
    fast.load_state_dict(stock.state_dict())
    for model in (stock, fast):
        model.train()
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    lfm3_fast_train.prepare(fast, compile_layers=False)
    for length in (3, 8, 11):                       # inside one window, a whole number of windows, a partial last window
        ids = torch.randint(0, 60, (length,))
        expected = stock.model(input_ids=ids[None], use_cache=False).last_hidden_state[0]
        actual = lfm3_fast_train.hidden(fast, ids)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        weights = torch.randn_like(expected)
        for model, value in ((stock, expected), (fast, actual)):
            model.zero_grad()
            (value * weights).sum().backward()
        for (name, a), (_, b) in zip(stock.named_parameters(), fast.named_parameters()):
            assert (b.grad - a.grad).norm() <= 1e-4 * a.grad.norm() + 1e-12, (length, name)
