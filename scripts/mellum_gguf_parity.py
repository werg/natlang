#!/usr/bin/env python3
"""Forward parity of a Mellum2.1 GGUF in our llama.cpp fork against the transformers reference (S8 serving).

Mellum 12B in BF16 (~24 GB) does not fit a ~12 GB ledger claim beside training, so the reference runs layer-streamed:
each decoder layer's weights are read from the safetensors, upcast to FP32, run with transformers' own Mellum layer
classes, rotary embeddings and mask builders (CPU), and freed. All prompts advance one layer at a time, so the 24 GB
are read once. The llama.cpp side is ``llama-logits-dump`` (fork, tools/neuralese/logits-dump.cpp): the same token ids,
logits at every position, weights memory-mapped (reclaimable page cache) and not repacked.

    # 1. prompts (HF tokenizer + chat template) and the GGUF tokenizer's ids for the same rendered text
    python scripts/mellum_gguf_parity.py prompts --model HF_DIR --out DIR [--gguf G --dump BIN]
    # 2. reference logits, FP32, layer-streamed (~10 GB)
    python scripts/mellum_gguf_parity.py reference --model HF_DIR --out DIR
    # 3. llama.cpp logits for a GGUF (tag e.g. bf16, q8_0)
    python scripts/mellum_gguf_parity.py llama --gguf G --dump BIN --out DIR --tag bf16 [--ubatch 512] [--greedy 24]
    # 4. metrics: top-1 agreement, KL(ref||gguf), max/mean |logit diff|, CE, split at the sliding window
    python scripts/mellum_gguf_parity.py compare --out DIR --tag bf16

Run every step under the memory ledger (AGENTS.md).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
WINDOW = 1024


def prompt_specs():
    """Short code, chat-template (thinking on/off, a tool turn) and two prompts longer than the sliding window."""
    code_file = (REPO / 'training/neuralese/natlang_neuralese/maple/model.py').read_text()
    ts_file = (REPO / 'ts-host/src/runtime/pluggable.ts').read_text()
    return [
        {'name': 'code-short', 'text': 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'},
        {'name': 'chat-thinking', 'messages': [
            {'role': 'system', 'content': 'You are a careful coding assistant.'},
            {'role': 'user', 'content': 'Write a TypeScript function that merges two sorted arrays.'}]},
        {'name': 'chat-nothink-tool', 'enable_thinking': False, 'messages': [
            {'role': 'user', 'content': 'What files are in the repo root?'},
            {'role': 'assistant', 'content': 'I will list them.\n<tool_call>\n{"name": "ls", "arguments": {"path": "."}}\n</tool_call>'},
            {'role': 'tool', 'content': 'README.md\nscripts\nts-host\ntraining'},
            {'role': 'user', 'content': 'Which one holds the runtime?'}]},
        {'name': 'code-long', 'text': code_file[:7000]},
        {'name': 'chat-long', 'messages': [
            {'role': 'user', 'content': 'Explain what this module does, briefly.\n\n```ts\n' + ts_file[:5200] + '\n```'}]},
    ]


def cmd_prompts(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in prompt_specs():
        if 'messages' in spec:
            kwargs = {} if 'enable_thinking' not in spec else {'enable_thinking': spec['enable_thinking']}
            text = tok.apply_chat_template(spec['messages'], tokenize=False, add_generation_prompt=True, **kwargs)
        else:
            text = spec['text']
        ids = tok(text, add_special_tokens=False).input_ids
        np.asarray(ids, dtype='<i4').tofile(out / f"{spec['name']}.ids.i32")
        (out / f"{spec['name']}.txt").write_text(text)
        row = {'name': spec['name'], 'n_tokens': len(ids)}
        if a.gguf:
            gpath = out / f"{spec['name']}.gguf-ids.i32"
            subprocess.run([a.dump, '-m', a.gguf, '--tokenize', str(out / f"{spec['name']}.txt"), '--ids-out',
                            str(gpath)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            gids = np.fromfile(gpath, dtype='<i4').tolist()
            mismatch = next((i for i, (x, y) in enumerate(zip(ids, gids)) if x != y), None)
            row['gguf_tokenizer'] = {'n_tokens': len(gids), 'identical': gids == ids, 'first_mismatch': mismatch}
        rows.append(row)
        print(json.dumps(row))
    (out / 'prompts.json').write_text(json.dumps(rows, indent=1) + '\n')


def cmd_reference(a):
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    from transformers.models.mellum import modeling_mellum as mm

    torch.set_num_threads(a.threads)
    dtype = {'fp32': torch.float32, 'bf16': torch.bfloat16}[a.dtype]
    out = Path(a.out)
    rows = json.loads((out / 'prompts.json').read_text())
    config = AutoConfig.from_pretrained(a.model)
    config._attn_implementation = 'sdpa'
    weight_map = json.loads((Path(a.model) / 'model.safetensors.index.json').read_text())['weight_map']
    handles = {}

    def get(name):
        f = weight_map[name]
        if f not in handles:
            handles[f] = safe_open(str(Path(a.model) / f), framework='pt')
        return handles[f].get_tensor(name).to(dtype)

    rotary = mm.MellumRotaryEmbedding(config)
    embed = get('model.embed_tokens.weight')
    states = []
    for row in rows:
        ids = torch.from_numpy(np.fromfile(out / f"{row['name']}.ids.i32", dtype='<i4').astype(np.int64))[None]
        h = embed[ids]
        pos = torch.arange(ids.shape[1])[None]
        kw = {'config': config, 'inputs_embeds': h, 'attention_mask': None, 'past_key_values': None, 'position_ids': pos}
        masks = {'full_attention': create_causal_mask(**kw), 'sliding_attention': create_sliding_window_causal_mask(**kw)}
        rope = {t: rotary(h, pos, t) for t in set(config.layer_types)}
        states.append({'h': h, 'pos': pos, 'masks': masks, 'rope': rope})
    del embed
    experts = config.num_experts
    taps = []
    with torch.no_grad():
        for i in range(config.num_hidden_layers):
            p = f'model.layers.{i}.'
            sd = {k[len(p):]: get(k) for k in weight_map if k.startswith(p) and '.experts.' not in k}
            sd['mlp.experts.gate_up_proj'] = torch.stack([torch.cat(
                [get(f'{p}mlp.experts.{e}.gate_proj.weight'), get(f'{p}mlp.experts.{e}.up_proj.weight')]) for e in range(experts)])
            sd['mlp.experts.down_proj'] = torch.stack([get(f'{p}mlp.experts.{e}.down_proj.weight') for e in range(experts)])
            with torch.device('meta'):
                layer = mm.MellumDecoderLayer(config, i)
            layer.load_state_dict(sd, assign=True, strict=True)
            layer.eval()
            kind = config.layer_types[i]
            for row, s in zip(rows, states):
                # MellumDecoderLayer.forward, split to tap the residual stream after attention and after the FFN
                attn, _ = layer.self_attn(hidden_states=layer.input_layernorm(s['h']), attention_mask=s['masks'][kind],
                                          position_embeddings=s['rope'][kind], position_ids=s['pos'])
                mid = s['h'] + attn
                s['h'] = mid + layer.mlp(layer.post_attention_layernorm(mid))
                if row['name'] == a.states:
                    taps.append(torch.stack([mid[0], s['h'][0]]).float().numpy())
            del layer, sd
            print(json.dumps({'layer': i, 'type': kind}), flush=True)
        if taps:
            np.stack(taps).astype('<f4').tofile(out / f'{a.states}.{a.tag}.states.f32')
        norm = mm.MellumRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        norm.weight = torch.nn.Parameter(get('model.norm.weight'))
        head = get('lm_head.weight')
        for row, s in zip(rows, states):
            h = norm(s['h'][0])
            logits = (h @ head.T).float().numpy().astype('<f4')
            logits.tofile(out / f"{row['name']}.{a.tag}.f32")
            print(json.dumps({'prompt': row['name'], 'tokens': int(h.shape[0])}), flush=True)


def cmd_llama(a):
    out = Path(a.out)
    rows = json.loads((out / 'prompts.json').read_text())
    results = {}
    for row in rows:
        if a.only and row['name'] not in a.only.split(','):
            continue
        cmd = [a.dump, '-m', a.gguf, '--ids', str(out / f"{row['name']}.ids.i32"),
               '--out', str(out / f"{row['name']}.{a.tag}.f32"), '--greedy', str(a.greedy), '-t', str(a.threads)]
        if a.ubatch:
            cmd += ['--ubatch', str(a.ubatch)]
        if a.ngl:
            cmd += ['-ngl', str(a.ngl)]
        if row['name'] == a.states:
            cmd += ['--states', str(out / f"{row['name']}.{a.tag}.states.f32")]
        r = subprocess.run(cmd, check=True, capture_output=True, text=True)
        results[row['name']] = json.loads(r.stdout.strip().splitlines()[-1])
        print(json.dumps({'prompt': row['name'], **results[row['name']]}), flush=True)
    (out / f'llama-{a.tag}.json').write_text(json.dumps(results, indent=1) + '\n')


def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def metrics(ref, got, ids, chunk=128):
    """Per-position metrics accumulated in row chunks (full FP64 [T, vocab] arrays would need ~2 GB each)."""
    kls, top1, max_diff, sum_diff, nll_ref, nll_got = [], 0, 0.0, 0.0, [], []
    for s in range(0, ref.shape[0], chunk):
        r, g = ref[s:s + chunk].astype(np.float64), got[s:s + chunk].astype(np.float64)
        lr, lg = log_softmax(r), log_softmax(g)
        kls.append((np.exp(lr) * (lr - lg)).sum(-1))
        top1 += int((r.argmax(-1) == g.argmax(-1)).sum())
        d = np.abs(r - g)
        max_diff, sum_diff = max(max_diff, float(d.max())), sum_diff + float(d.sum())
        nxt = ids[s + 1:s + 1 + r.shape[0]]
        rows = np.arange(len(nxt))
        nll_ref.append(-lr[rows, nxt])
        nll_got.append(-lg[rows, nxt])
    kl, nll_ref, nll_got = np.concatenate(kls), np.concatenate(nll_ref), np.concatenate(nll_got)
    return {
        'positions': int(ref.shape[0]),
        'top1_agreement': top1 / ref.shape[0],
        'kl_mean': float(kl.mean()), 'kl_max': float(kl.max()),
        'max_abs_logit_diff': max_diff, 'mean_abs_logit_diff': sum_diff / ref.size,
        'ce_ref': float(nll_ref.mean()) if len(nll_ref) else None,
        'ce_gguf': float(nll_got.mean()) if len(nll_got) else None,
    }


def cmd_compare(a):
    out = Path(a.out)
    rows = json.loads((out / 'prompts.json').read_text())
    report = {}
    for row in rows:
        ids = np.fromfile(out / f"{row['name']}.ids.i32", dtype='<i4')
        ref = np.memmap(out / f"{row['name']}.{a.ref}.f32", dtype='<f4', mode='r').reshape(len(ids), -1)
        got = np.memmap(out / f"{row['name']}.{a.tag}.f32", dtype='<f4', mode='r').reshape(len(ids), -1)
        entry = {'all': metrics(ref, got, ids)}
        if len(ids) > WINDOW:
            entry['before_window'] = metrics(ref[:WINDOW], got[:WINDOW], ids[:WINDOW])
            entry['after_window'] = metrics(ref[WINDOW:], got[WINDOW:], ids[WINDOW:])
        report[row['name']] = entry
        print(json.dumps({'prompt': row['name'], **{k: {m: round(v, 5) if isinstance(v, float) else v
                                                         for m, v in e.items()} for k, e in entry.items()}}))
    (out / f'compare-{a.tag}-vs-{a.ref}.json').write_text(json.dumps(report, indent=1) + '\n')


def cmd_states(a):
    out = Path(a.out)
    n = len(np.fromfile(out / f'{a.prompt}.ids.i32', dtype='<i4'))
    ref = np.fromfile(out / f'{a.prompt}.{a.ref}.states.f32', dtype='<f4').reshape(-1, 2, n, 2304)
    got = np.fromfile(out / f'{a.prompt}.{a.tag}.states.f32', dtype='<f4').reshape(-1, 2, n, 2304)
    rows = []
    for il in range(ref.shape[0]):
        row = {'layer': il}
        for k, name in enumerate(['after_attn', 'after_ffn']):
            r, g = ref[il, k].astype(np.float64), got[il, k].astype(np.float64)
            rel = np.linalg.norm(g - r, axis=-1) / np.maximum(np.linalg.norm(r, axis=-1), 1e-12)
            row[name] = {'rel_mean': float(rel.mean()), 'rel_max': float(rel.max()), 'argmax_pos': int(rel.argmax())}
        rows.append(row)
        print(json.dumps(row))
    (out / f'states-{a.prompt}-{a.tag}-vs-{a.ref}.json').write_text(json.dumps(rows, indent=1) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    dump_default = os.path.expanduser('~/llama.cpp-neuralese/build-cpu/bin/llama-logits-dump')
    s = sub.add_parser('prompts')
    s.add_argument('--model', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--gguf')
    s.add_argument('--dump', default=dump_default)
    s = sub.add_parser('reference')
    s.add_argument('--model', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--threads', type=int, default=16)
    s.add_argument('--dtype', choices=['fp32', 'bf16'], default='fp32',
                   help='bf16: the reference in the checkpoint dtype, as a precision noise floor for the GGUF gap')
    s.add_argument('--tag', default='ref', help='output tag (compare reads ref; another tag can be compared as --tag)')
    s.add_argument('--states', help='prompt name: also write its residual stream after each attention and FFN, '
                   '[n_layer][2][T][hidden] float32, to localize a divergence (llama: --states)')
    s = sub.add_parser('llama')
    s.add_argument('--gguf', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--tag', required=True)
    s.add_argument('--dump', default=dump_default)
    s.add_argument('--ubatch', type=int, default=0)
    s.add_argument('--greedy', type=int, default=24)
    s.add_argument('--threads', type=int, default=16)
    s.add_argument('--ngl', type=int, default=0)
    s.add_argument('--states', help='prompt name whose residual stream to write (as reference --states)')
    s.add_argument('--only', help='comma-separated prompt names to run (default: all)')
    s = sub.add_parser('states', help='per-layer relative error of the residual stream, tag against ref')
    s.add_argument('--out', required=True)
    s.add_argument('--prompt', required=True)
    s.add_argument('--tag', required=True)
    s.add_argument('--ref', default='ref')
    s = sub.add_parser('compare')
    s.add_argument('--out', required=True)
    s.add_argument('--tag', required=True)
    s.add_argument('--ref', default='ref', help='the reference tag (default: the FP32 transformers reference)')
    a = p.parse_args()
    {'prompts': cmd_prompts, 'reference': cmd_reference, 'llama': cmd_llama, 'compare': cmd_compare,
     'states': cmd_states}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
