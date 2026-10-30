"""decide on an --engine sglang actor against the native engine's decide (same checkpoint, BF16), plus FP32 native."""
# usage: check_sglang_decide_parity.py CHECKPOINT BASE OUT.json PORT
import json, subprocess, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, '/home/werg/natlang/training/neuralese')
import torch
from natlang_neuralese.common.hashing import sha256_file_hex_cached
from natlang_neuralese.serve import load_native_engine
from natlang_neuralese.serve.native_learning import decide_native

ck, base, out, port = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])
sha = sha256_file_hex_cached(ck)
cached = out.with_suffix('.sglang.json')
log = open(out.with_suffix('.actor.log'), 'w')
actor = None if cached.exists() else subprocess.Popen([sys.executable, '/home/werg/natlang/scripts/neuralese_data/serve_native_checkpoint.py',
    '--checkpoint', str(ck), '--checkpoint-sha256', sha, '--base', base, '--port', str(port),
    '--store-dir', str(out.parent / 'decide-store'), '--prompt-rendering', 'full', '--engine', 'sglang'],
    stdout=log, stderr=subprocess.STDOUT)
url = f'http://127.0.0.1:{port}'
cases = [
    ([{'role': 'system', 'content': 'You answer with one word.'}, {'role': 'user', 'content': 'Which is a primary colour: red, green-ish or beige?'}],
     ['red', 'green-ish', 'beige', 'Red.', 'I think red', 'none of them']),
    ([{'role': 'user', 'content': 'Sort these numbers ascending and reply with JSON: 5, 2, 9, 1.'}],
     ['[1, 2, 5, 9]', '[9, 5, 2, 1]', '[1,2,5,9]', '{"sorted": [1, 2, 5, 9]}', '1 2 5 9', '[2, 1, 5, 9]']),
    ([{'role': 'user', 'content': 'The greenhouse sensor reads 31 C and the limit is 28 C. Should the vents open? Answer yes or no.'}],
     ['yes', 'no', 'Yes.', 'No.', 'yes, open them', 'maybe']),
    ([{'role': 'user', 'content': 'Write a TypeScript expression that counts the items whose price is above 10 in `items`.'}],
     ['items.filter(i => i.price > 10).length', 'items.length', 'items.filter((item) => item.price > 10).length',
      'items.reduce((n, i) => n + (i.price > 10 ? 1 : 0), 0)', 'count(items)', 'items.map(i => i.price > 10)']),
    ([{'role': 'user', 'content': 'Summarize in five words: the north bank averaged 61 percent humidity and the south bank 58 percent.'}],
     ['North bank slightly more humid.', 'Both banks were equally humid.', 'South bank was more humid.',
      'Humidity: north 61, south 58.', 'Nothing to report today here.', 'North humid, south slightly drier.']),
]
try:
  if actor is None:
    sg, metrics = json.loads(cached.read_text())
  else:
    for _ in range(360):
        if actor.poll() is not None:
            raise RuntimeError('actor exited')
        try:
            urllib.request.urlopen(url + '/v1/neuralese/info', timeout=5); break
        except Exception:
            time.sleep(5)
    sg = []
    for messages, options in cases:
        req = urllib.request.Request(url + '/v1/neuralese/decide', data=json.dumps({'messages': messages, 'options': options}).encode(),
                                     headers={'Content-Type': 'application/json'})
        sg.append(json.loads(urllib.request.urlopen(req, timeout=600).read()))
    metrics = urllib.request.urlopen(url + '/metrics', timeout=30).read().decode()
    cached.write_text(json.dumps([sg, metrics]))
finally:
    if actor is not None:
        actor.terminate()
        try:
            actor.wait(timeout=120)
        except subprocess.TimeoutExpired:
            actor.kill()
ref = {}
for name, dtype in (('bf16', torch.bfloat16), ('fp32', torch.float32)):
    engine = load_native_engine(str(ck), device='cuda', dtype=dtype, base=base)
    ref[name] = [decide_native(engine, {'messages': m, 'options': o}) for m, o in cases]
    del engine; torch.cuda.empty_cache()
rows = []
for s, b, f in zip(sg, ref['bf16'], ref['fp32']):
    for i in range(len(s['log_probs'])):
        rows.append({'sglang': s['log_probs'][i], 'native_bf16': b['log_probs'][i], 'native_fp32': f['log_probs'][i],
                     'tokens': f['tokens'][i], 'same_tokens': s['tokens'][i] == f['tokens'][i]})
import statistics as st
per_tok = lambda key: [abs(r[key] - r['native_fp32']) / r['tokens'] for r in rows]
argmax = lambda key, res: [max(range(len(x['log_probs'])), key=lambda i: x['log_probs'][i]) for x in res]
report = {'options': len(rows), 'all_same_tokens': all(r['same_tokens'] for r in rows),
          'sglang_vs_fp32_per_token_mean': st.mean(per_tok('sglang')), 'sglang_vs_fp32_per_token_max': max(per_tok('sglang')),
          'native_bf16_vs_fp32_per_token_mean': st.mean(per_tok('native_bf16')), 'native_bf16_vs_fp32_per_token_max': max(per_tok('native_bf16')),
          'argmax_agree_sglang_fp32': sum(a == b for a, b in zip(argmax(0, sg), argmax(0, ref['fp32']))), 'cases': len(cases),
          'fallbacks': [l for l in metrics.splitlines() if 'fallback' in l], 'rows': rows}
out.write_text(json.dumps(report, indent=1))
print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=1))
