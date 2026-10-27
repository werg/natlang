"""Usage: check_tuned_lora_llama_parity.py CHECKPOINT OUT_DIR LLAMA_SERVER

Tuned LoRA on llama.cpp equals the native adapted model: PEFT -> GGUF LoRA on our merged native GGUF export.

Random free-LoRA factors on real LFM2 targets (attention, conv projection, MLP) of the pinned checkpoint. Reference:
the native model with the PEFT's dense deltas added to its weights. Candidate: llama-server on the f32 native export
with the converted GGUF LoRA. Compares greedy continuations and top-token probabilities; base runs show the adapter
changes the output.
"""
import json, subprocess, sys, time, urllib.request
from pathlib import Path
import torch

from natlang_neuralese.export.native_gguf import export as export_native_gguf
from natlang_neuralese.serve import load_native_engine
from natlang_neuralese.model.native_adapters import target_parameters
from natlang_neuralese.tune.targets import module_map
from natlang_neuralese.tune.parameterization import FreeLoRADelta
from natlang_neuralese.tune.stack import StackMember, stack_factors
from natlang_neuralese.tune.export import export_peft, export_gguf, load_peft_dense

checkpoint, out, server = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
out.mkdir(parents=True, exist_ok=True)
manifest = None if (out / 'base' / 'model-f32.gguf').exists() and (out / 'base' / 'hf-config').exists() else export_native_gguf(checkpoint, out / 'base', outtypes=('f32',))
engine = load_native_engine(str(checkpoint), device='cuda', dtype=torch.float32)
params = target_parameters(engine.shared_model)
targets, kinds = module_map(sorted(params))
chosen = {}
for kind in ('attention', 'conv-projection', 'mlp'):
    name = next(n for n in sorted(targets) if kinds[n] == kind)
    chosen[name] = tuple(params[name].shape)
g = torch.Generator().manual_seed(7)
deltas = {}
for name, (rows, cols) in chosen.items():
    scale = 0.5 * params[name].float().std().item()
    deltas[name] = FreeLoRADelta(torch.randn(rows, 4, generator=g) * scale, torch.randn(4, cols, generator=g), 4.)
stack = stack_factors([StackMember('lambda:parity', 'lambda', 1., deltas)], {}, chosen)
peft = out / 'lora.peft'
export_peft(stack, {n: targets[n] for n in chosen}, peft, base_identity=dict(engine.native_identity),
            base_model_name_or_path=str(out / 'base' / 'hf-config'))
export_gguf(peft, out / 'base' / 'hf-config', out / 'lora.gguf')
print('targets', {n: kinds[n] for n in chosen}, flush=True)

tokenizer = engine.tokenizer
prompt = tokenizer.encode('Write one sentence about the sea and ships:', add_special_tokens=False)

def native_greedy(steps=12):
    with torch.no_grad():
        ids, probs = list(prompt), []
        for _ in range(steps):
            h = engine._native_forward(ids, None)['h_final']
            logits = engine._native_logits(h)[0].float()
            p = torch.softmax(logits, -1)
            top = int(p.argmax())
            probs.append((top, float(p[top])))
            ids.append(top)
        return probs

base_native = native_greedy()
hf_params = dict(engine.shared_model.named_parameters())
dense = load_peft_dense(peft)
with torch.no_grad():
    for name in chosen:
        delta = dense[targets[name].hf_module].to(device=hf_params[name].device, dtype=hf_params[name].dtype)
        hf_params[name].add_(delta if targets[name].orientation == 'out_in' else delta.T)
tuned_native = native_greedy()

def llama(lora):
    port = 18977
    args = [server, '-m', str(out / 'base' / 'model-f32.gguf'), '--port', str(port), '-c', '1024', '-ngl', '99', '--no-warmup']
    if lora: args += ['--lora', str(out / 'lora.gguf')]
    process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1); break
            except Exception: time.sleep(0.5)
        body = json.dumps({'prompt': prompt, 'n_predict': 12, 'temperature': 0, 'n_probs': 1, 'cache_prompt': False}).encode()
        reply = json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/completion', body,
                                                                       {'content-type': 'application/json'}), timeout=120))
        result = []
        (out / f'llama-{int(lora)}.json').write_text(json.dumps(reply, indent=1))
        for item in reply['completion_probabilities']:
            top = (item.get('top_probs') or item.get('top_logprobs') or item.get('probs'))[0]
            prob = top['prob'] if 'prob' in top else __import__('math').exp(top['logprob'])
            result.append((int(top['id']), float(prob)))
        return result
    finally:
        process.terminate(); process.wait(30)

base_llama, tuned_llama = llama(False), llama(True)
report = {'base_native': base_native, 'base_llama': base_llama, 'tuned_native': tuned_native, 'tuned_llama': tuned_llama,
          'base_tokens_equal': [a for a, _ in base_native] == [a for a, _ in base_llama],
          'tuned_tokens_equal': [a for a, _ in tuned_native] == [a for a, _ in tuned_llama],
          'adapter_changes_output': [a for a, _ in base_native] != [a for a, _ in tuned_native],
          'max_prob_gap_tuned': max(abs(a[1] - b[1]) for a, b in zip(tuned_native, tuned_llama))}
(out / 'report.json').write_text(json.dumps(report, indent=1))
print(json.dumps({k: v for k, v in report.items() if not isinstance(v, list)}), flush=True)
print('tuned native', [t for t, _ in tuned_native]); print('tuned llama ', [t for t, _ in tuned_llama])
