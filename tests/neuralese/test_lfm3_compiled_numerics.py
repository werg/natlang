"""Compiled `lfm3_fast_train` numerics (CUDA): the compiled layers round where eager does in both grad modes, and the
same compiled forward gives the same bits in every process (`COMPILE_OPTIONS`, 2026-11-17: compiled no-grad
log-probabilities had drifted from eager and from the grad-mode graph, and differed between processes).

The real-model check over the 48 behaviour-probe calls runs with NATLANG_REAL_MODEL_TESTS=1 (start checkpoint, ~10 GB)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts/neuralese_data'), str(ROOT / 'tests/neuralese')]
from natlang_neuralese.train import lfm3_fast as lfm3_fast_train  # noqa: E402
from natlang_neuralese.common.paths import resolve_str  # noqa: E402

# the start policy 5ff07638 (grpo-v3's --checkpoint)
START = resolve_str('data_hdd', 'natlang-development-data', 'archive', 'data', 'lfm3-rl-dgx-20261029', 'fresh-init',
                    'checkpoint.pt')
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA kernel check')


def _model(seed=0):
    """A random LFM3 wide enough for the flash kernel and the Inductor reductions of the real one, in BF16 on CUDA."""
    from natlang_neuralese.third_party import lfm3
    layer_types = ['conv', 'full_attention', 'conv', 'sliding_attention', 'conv', 'full_attention', 'conv', 'conv']
    config = lfm3.config_class()(vocab_size=4096, hidden_size=512, intermediate_size=1536,
                                 num_hidden_layers=len(layer_types), num_attention_heads=8, num_key_value_heads=2,
                                 head_dim=64, layer_types=layer_types, sliding_window=128, max_position_embeddings=8192,
                                 pad_token_id=None, bos_token_id=1, eos_token_id=2, use_sandwich_norm=False)
    config._attn_implementation = 'eager'
    torch.manual_seed(seed)
    model = lfm3.causal_lm_class()(config)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith('norm.weight'):
                parameter.copy_(1 + 0.2 * torch.randn_like(parameter))
    return model.to('cuda', torch.bfloat16).train()


def _logp(model, ids, grad):
    positions = torch.arange(ids.shape[0] - 1, device=ids.device)
    with torch.set_grad_enabled(grad):
        h = lfm3_fast_train.hidden(model, ids, lengths=[1200, 1800])
        return h.detach(), -lfm3_fast_train.linear_nll(h[positions], model.lm_head.weight, ids[1:]).detach()


@cuda
def test_compiled_logprobs_match_eager_unbiased_in_both_grad_modes():
    assert lfm3_fast_train.COMPILE_OPTIONS == {'emulate_precision_casts': True, 'deterministic': True}
    model = _model()
    ids = torch.randint(0, 4096, (3000,), device='cuda')
    lfm3_fast_train.prepare(model, compile_layers=False)
    eager = _logp(model, ids, False)[1]
    lfm3_fast_train.prepare(model, compile_layers=True)
    runs = {grad: _logp(model, ids, grad)[1] for grad in (False, True)}
    # Compiled and eager differ by kernel-level FP32 order (fusion, reductions; 2026-11-17: ~0.006 nats a token here,
    # 0.014 on the real model, against 0.031 between eager and the SGLang actor); what may not appear is a one-sided
    # shift, nor a gap between the two grad modes beyond that floor.
    for label, d in (('nograd', runs[False] - eager), ('grad', runs[True] - eager), ('modes', runs[True] - runs[False])):
        se = (d.std() / len(d) ** .5).item()
        assert d.abs().mean() < 0.02, (label, d.abs().mean().item())
        assert abs(d.mean().item()) < max(5 * se, 5e-4), (label, d.mean().item(), se)


_CHILD = """
import sys, torch
sys.path[:0] = {paths!r}
from natlang_neuralese.common.cuda_context import ensure_context
ensure_context()   # a fresh process's CUDA context: retried after compaction on GB10 fragmentation
from test_lfm3_compiled_numerics import _model, _logp
from natlang_neuralese.train import lfm3_fast as lfm3_fast_train
model = _model()
torch.manual_seed(1)
ids = torch.randint(0, 4096, (3000,), device='cuda')
lfm3_fast_train.prepare(model, compile_layers=True)
torch.save([_logp(model, ids, grad)[1].cpu() for grad in (False, True)], sys.argv[1])
"""


@cuda
def test_compiled_forward_is_bitwise_identical_across_processes(tmp_path):
    paths = [str(ROOT / 'tests/neuralese'), str(ROOT / 'scripts/neuralese_data'), str(ROOT / 'training/neuralese'),
             str(ROOT)]
    outs = []
    for index in range(2):
        out = tmp_path / f'{index}.pt'
        env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=str(tmp_path / f'cache-{index}'))   # each compiles afresh
        subprocess.run([sys.executable, '-c', _CHILD.format(paths=paths), str(out)], check=True, env=env)
        outs.append(torch.load(out))
    for a, b in zip(*outs):
        assert torch.equal(a, b)


@pytest.mark.skipif(os.environ.get('NATLANG_REAL_MODEL_TESTS') != '1' or not torch.cuda.is_available(),
                    reason='real start checkpoint on CUDA (NATLANG_REAL_MODEL_TESTS=1)')
def test_behaviour_probe_calls_show_no_one_sided_compiled_shift():
    import exp0_sft
    from grpo import BEHAVIOUR_PROBE, BASE
    calls = json.loads(Path(BEHAVIOUR_PROBE).read_text())['calls']
    model, _, _ = exp0_sft.load_model(START, BASE, dtype=torch.bfloat16)
    model.train()

    def score(compiled, grad):
        lfm3_fast_train.prepare(model, compile_layers=compiled)
        out = []
        for call in calls:
            seq = call['prompt'] + call['think']
            ids = torch.tensor(seq, device='cuda')
            positions = torch.arange(len(call['prompt']) - 1, len(seq) - 1, device='cuda')
            lfm3_fast_train.set_checkpointing(model, len(seq), 3.)
            with torch.set_grad_enabled(grad):
                h = lfm3_fast_train.hidden(model, ids)
                out.append(-exp0_sft.token_nll(model, h, positions, torch.tensor(call['think'], device='cuda'))
                           .detach().float().cpu())
        return torch.cat(out)
    eager = score(False, False)
    for grad in (False, True):
        d = score(True, grad) - eager
        se = (d.std() / len(d) ** .5).item()
        assert abs(d.mean().item()) < max(4 * se, 1e-3), (grad, d.mean().item(), se)
        assert d.abs().mean() < 0.025, (grad, d.abs().mean().item())     # 0.014 measured (2026-11-17)
