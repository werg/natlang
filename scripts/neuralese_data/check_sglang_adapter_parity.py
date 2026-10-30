"""Native adapters on the SGLang actor against the native engine's adapted forward (run_native + native_step), BF16.
usage: check_sglang_adapter_parity.py CHECKPOINT BASE WORK_DIR PORT  (NATLANG_SGLANG_PYTHONPATH: a fork worktree under test)

One in-process SglangEngine front (its native model stays resident for the reference) with the fork's adapter slots.
Three random adapter stacks (xs + free-lora on attention, conv and MLP matrices of several layers) and unadapted
requests run concurrently through 2 slots (slot reuse), sampling at T=1 with behaviour capture. Each sample's logp is
compared with the native engine's log-softmax of the same token after the same tokens under the same stack. Control:
the native unadapted forward on the adapted samples."""
import json, sys, types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training/neuralese')); sys.path.insert(0, str(ROOT / 'scripts/neuralese_data'))
import torch
from natlang_neuralese.common.hashing import sha256_file_hex_cached
from natlang_neuralese.serve import load_native_engine
from natlang_neuralese.serve.engine import GenerationRequest
from natlang_neuralese.serve.sglang_actor import SglangEngine
from natlang_neuralese.serve.store import TensorStore
from natlang_neuralese.model.native_adapters import create_adapter, target_parameters
from natlang_neuralese.model.shared_state import native_step
from serve_native_checkpoint import launch_sglang

ck, base, work, port = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])
work.mkdir(parents=True, exist_ok=True)
front = load_native_engine(str(ck), base=base, device='cuda', dtype=torch.bfloat16, engine_class=SglangEngine,
                           store=TensorStore(work / 'store'))
args = types.SimpleNamespace(sglang_python=None, hf_export_dir=work / 'export', store_dir=work, checkpoint_sha256=sha256_file_hex_cached(ck),
    sglang_port=port + 1000, port=port, sglang_context=16384, sglang_max_total_tokens=65536, sglang_max_running=8,
    sglang_mem_fraction=0.3, sglang_arg=[], sglang_adapter_slots=2, sglang_adapter_rank=32)
process = launch_sglang(front, args, ck)
front.start()
available = target_parameters(front.shared_model)
names = [n for n in available if any(f'.layers.{i}.' in n for i in (0, 2, 4, 9))]
stacks = []
for seed in range(3):
    made = []
    for kind in ('xs', 'free-lora'):
        artifact = create_adapter(front, {'kind': kind, 'rank': 4, 'seed': seed * 2 + (kind == 'xs'), 'parameter_names': names}).artifact
        g = torch.Generator().manual_seed(100 + seed)
        values = {}
        for key, value in artifact.tensors.items():
            scale = 0.6 if kind == 'xs' else (0.02 if key.startswith('lora_B.') else 1.0)
            values[key] = torch.randn(value.shape, generator=g) * scale if kind == 'xs' or key.startswith('lora_B.') else value
        made.append(front.store.put_native_tensor_artifact(replace(artifact, tensors=values)).id)
    stacks.append([{'id': made[0], 'scale': 1.0}, {'id': made[1], 'scale': 0.5 + seed}])
prompts = ['Explain in two sentences why greenhouses need ventilation.', 'List three prime numbers and add them.',
           'Write a TypeScript function that returns the largest element of an array.', 'What does a conv layer do?']
jobs = [(p, s) for p in prompts for s in (*stacks, None)]
def run(job):
    prompt, stack = job
    request = GenerationRequest(messages=[{'role': 'user', 'content': prompt}], temperature=1.0, max_tokens=48, rl_capture=True,
                                rl_top_k=20, native_recording=True, seed=7, adapters=stack if stack else [])
    return front.generate(request)['x_natlang_rl_capture']
with ThreadPoolExecutor(8) as pool:
    captures = list(pool.map(run, jobs))
def reference(ids, stack, first):
    with torch.no_grad():
        out = front._native_run(lambda: front.shared_model.logits(native_step(front.shared_model, torch.tensor([ids], device='cuda'))['h_final'])[0].float(),
                                stack if stack else [])
    return torch.log_softmax(out, -1)[first - 1:-1]
rows = {'adapted': [], 'unadapted': [], 'control': []}
for (prompt, stack), capture in zip(jobs, captures):
    prompt_ids = [t for seg in capture['prompt'] for t in seg.get('token_ids', [])]
    fed = [x['token_id'] for x in capture['generated_inputs']]
    samples = [x for x in capture['samples'] if not x['terminal']][:len(fed)]
    ids = prompt_ids + fed
    ref = reference(ids, stack, len(prompt_ids))
    diffs = [abs(s['logp'] - float(ref[j, s['token_id']])) for j, s in enumerate(samples)]
    rows['adapted' if stack else 'unadapted'] += diffs
    if stack:
        plain = reference(ids, None, len(prompt_ids))
        rows['control'] += [abs(float(plain[j, s['token_id']]) - float(ref[j, s['token_id']])) for j, s in enumerate(samples)]
    assert (capture.get('adapter_binding') is not None) == bool(stack)
mean = lambda v: sum(v) / len(v)
report = {k: {'positions': len(v), 'mean_abs_logprob': mean(v), 'max_abs_logprob': max(v)} for k, v in rows.items()}
metrics = front.progress_metrics()
report['metrics'] = {k: v for k, v in metrics.items() if 'sglang' in k or 'fallback' in k}
report['failures'] = ([] if report['adapted']['mean_abs_logprob'] <= 1.5 * report['unadapted']['mean_abs_logprob'] + 0.005 else ['adapted'])
report['failures'] += [] if report['control']['mean_abs_logprob'] > 4 * report['adapted']['mean_abs_logprob'] else ['control_not_separated']
report['failures'] += [] if metrics.get('natlang_native_fallback_requests_total', 0) == 0 else ['fallbacks']
(work / 'report.json').write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1), flush=True)
front.stop()
import os, signal
os.killpg(process.pid, signal.SIGTERM)
