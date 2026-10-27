"""Usage: check_llama_producer_parity.py CHECKPOINT OUT_DIR STATE_SPAN_DUMP [--export]

Native state production on the llama.cpp fork equals the native engine. One sequence: an ordinary prompt, the
produce-open marker, a producer span (the producer adapter applies and the span's state is captured), the
produce-close marker and an ordinary tail. Compares every position's logits and the captured embeddings, keys (after
the key norm, before RoPE), values and short-convolution gated inputs. Controls: the native engine without the
producer adapter on the span, and the fork's span decoded as ordinary tokens. With --export the f32 GGUF (with the
producer adapter) is exported to OUT_DIR/base first.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from natlang_neuralese.export.native_gguf import export as export_native_gguf
from natlang_neuralese.model.shared_state import native_step
from natlang_neuralese.serve import load_native_engine

checkpoint, out, dump = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
out.mkdir(parents=True, exist_ok=True)
if '--export' in sys.argv or not (out / 'base' / 'model-f32.gguf').exists():
    export_native_gguf(checkpoint, out / 'base', outtypes=('f32',))
engine = load_native_engine(str(checkpoint), device='cuda', dtype=torch.float32)
model, tok, roles = engine.shared_model, engine.tokenizer, engine.role_tokens
prompt = tok.encode('Summarize the ticket for the billing team:', add_special_tokens=False)
span = tok.encode(' refund request for order 4411, charged twice', add_special_tokens=False)
tail = tok.encode(' Thanks.', add_special_tokens=False)
open_, close = int(roles.produce_open), int(roles.produce_close)
adapter_norm = float(model.adapter.up.weight.norm())


def native(producing_span=True):
    def run(ids, cache, producing):
        with torch.no_grad():
            result = native_step(model, torch.tensor([ids], device='cuda'), cache, producing=producing)
            return result, model.logits(result['h_final'][0]).float().cpu()
    first, a = run(prompt + [open_], None, False)
    second, b = run(span, first['cache'], producing_span)
    third, c = run([close] + tail, second['cache'], False)
    return torch.cat([a, b, c]), second


def fork(produce):
    sequence = prompt + [open_] + ([{'produce': span}] if produce else span) + [close] + tail
    (out / 'seq.json').write_text(json.dumps(sequence))
    prefix = out / ('capture' if produce else 'plain')
    subprocess.run([dump, '-m', str(out / 'base' / 'model-f32.gguf'), '--input', str(out / 'seq.json'), '--from', '0',
                    '--out', str(prefix) + '.logits', '--f32-cache', '--capture', str(prefix)], check=True)
    n = len(prompt) + len(span) + len(tail) + 2
    logits = torch.from_numpy(np.fromfile(str(prefix) + '.logits', dtype=np.float32).reshape(n, -1))
    meta = json.loads(Path(str(prefix) + '.json').read_text())
    return logits, meta, np.fromfile(str(prefix) + '.f32', dtype=np.float32)


native_logits, captured = native()
control_native, _ = native(producing_span=False)
fork_logits, meta, flat = fork(True)
plain_logits, plain_meta, _ = fork(False)
vocab = native_logits.shape[1]
fork_logits, plain_logits = fork_logits[:, :vocab], plain_logits[:, :vocab]
n = meta['n_tokens']
report = {'adapter_up_norm': adapter_norm, 'producer_adapter_in_gguf': meta['producer_adapter'], 'span_tokens': len(span),
          'captured_tokens': n, 'plain_captured_tokens': plain_meta['n_tokens'],
          'max_dlogit': float((native_logits - fork_logits).abs().max()),
          'top1_agreement': float((native_logits.argmax(1) == fork_logits.argmax(1)).float().mean()),
          'control_native_without_adapter_vs_native': float((native_logits - control_native).abs().max()),
          'control_fork_without_production_vs_fork': float((plain_logits - fork_logits).abs().max())}
# Captured tensors in the native layout, flattened per token as the fork writes them.
offset = 0
def take(width):
    global offset
    part = flat[offset:offset + n * width].reshape(n, width)
    offset += n * width
    return torch.from_numpy(part)
# The token embeddings are compared with a fresh lookup: the native capture's own `embeddings` view can change after
# later native calls on the same model, which is reported separately.
with torch.no_grad():
    expected_embeddings = model.embed(torch.tensor([span], device='cuda'))[0].float().cpu()
fork_embeddings = take(meta['embd_width'])
gaps = {'embeddings': float((fork_embeddings - expected_embeddings).abs().max())}
report['native_capture_embeddings_vs_lookup'] = float((captured['embeddings'].float().cpu() - expected_embeddings).abs().max())
for il, (widths, layer) in enumerate(zip(meta['layers'], captured['layers'])):
    for kind, name in (('k', 'raw_k'), ('v', 'v'), ('conv', 'gated_inputs')):
        if widths[kind]:
            native_rows = layer[name].float().cpu().reshape(n, -1)
            gaps[f'{il}.{kind}'] = float((take(widths[kind]) - native_rows).abs().max())
report['max_capture_gap'] = max(gaps.values())
report['capture_gaps'] = gaps
(out / 'report.json').write_text(json.dumps(report, indent=1))
print(json.dumps({key: value for key, value in report.items() if key != 'capture_gaps'}, indent=1))
