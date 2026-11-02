#!/usr/bin/env python3
"""Bounded two-actor evidence, without starting models or declaring qualification."""
import argparse
import hashlib
import json
import os
import tempfile
import time
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import load_file
from natlang_neuralese.serve.store import decode_native_state
from natlang_neuralese.common.sampling import decode_top_k_arrays


def request(url, body=None, method=None, content_type='application/json'):
    data = json.dumps(body).encode() if isinstance(body, dict) else body
    req = urllib.request.Request(url, data=data, method=method, headers={'Content-Type': content_type})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as response:
        data = response.read()
        kind = response.headers.get_content_type()
    return (json.loads(data) if kind == 'application/json' else data), time.monotonic() - t


def comparison(a, b):
    a, b = a.cpu(), b.cpu()
    if a.shape != b.shape:
        return {'shape_a': list(a.shape), 'shape_b': list(b.shape)}
    diff = (a.float() - b.float()).abs()
    return {'shape': list(a.shape), 'equal': bool(torch.equal(a, b)),
            'different': int((a != b).sum()), 'count': a.numel(),
            'max_abs': float(diff.max()) if diff.numel() else 0,
            'mean_abs': float(diff.mean()) if diff.numel() else 0}


def capture_case(native, backend, text):
    result, native_seconds = request(native + '/v1/neuralese/native-states/write-text', {'text': text})
    raw, _ = request(native + '/v1/neuralese/native-states/' + result['id'])
    block, decoded_id = decode_native_state(raw)
    if decoded_id != result['id']:
        raise ValueError('native state artifact hash differs from the write receipt')
    ids = sum((s['token_ids'] for s in result['source_capture']['segments']), [])
    start, end = result['source_capture']['payload_start'], result['source_capture']['payload_stop']
    fd, path = tempfile.mkstemp(prefix='natlang-live-capture-', suffix='.safetensors')
    os.close(fd)
    os.unlink(path)
    try:
        _, backend_seconds = request(backend + '/generate', {'input_ids': ids[:end], 'return_logprob': True,
            'logprob_start_len': max(start - 1, 0), 'sampling_params': {'temperature': 0.0, 'max_new_tokens': 1,
                'custom_params': {'natlang_capture': {'path': path, 'start': start, 'end': end}}}})
        data = load_file(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)
    fields = {'token_ids': comparison(block.token_ids, data['token_ids']),
              'positions': comparison(block.positions, data['positions']),
              'embeddings': comparison(block.embeddings, data['embeddings'])}
    for i, layer in enumerate(block.layers):
        for name, value in layer.items():
            if name != 'kind':
                fields[f'layers.{i}.{name}'] = comparison(value, data[f'layers.{i}.{name}'])
    return {'native_state_id': result['id'], 'source_text_sha256': result['source_text_sha256'],
            'source_capture': result['source_capture'], 'native_seconds': native_seconds,
            'backend_seconds': backend_seconds, 'fields': fields}, raw


def sample_diffs(a, b):
    sa, sb = a['x_natlang_rl_capture']['samples'], b['x_natlang_rl_capture']['samples']
    same_prefix = True
    out = []
    for x, y in zip(sa, sb):
        row = {'same_prefix': same_prefix, 'native_token': x['token_id'], 'sglang_token': y['token_id'],
               'same_token': x['token_id'] == y['token_id'], 'native_logp': x.get('logp'), 'sglang_logp': y.get('logp')}
        if same_prefix and x.get('top_k') and y.get('top_k'):
            ia, pa = decode_top_k_arrays(x['top_k']); ib, pb = decode_top_k_arrays(y['top_k'])
            left, right = dict(zip(map(int, ia), map(float, pa))), dict(zip(map(int, ib), map(float, pb)))
            shared = left.keys() & right.keys()
            row.update({'top1_equal': int(ia[0]) == int(ib[0]), 'top_k_overlap': len(shared),
                        'top_k_count': len(ia), 'shared_top_logp_max_abs': max((abs(left[k]-right[k]) for k in shared), default=None)})
        same_prefix &= row['same_token']
        out.append(row)
    return {'native_samples': len(sa), 'sglang_samples': len(sb), 'rows': out}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--native-url', required=True); p.add_argument('--sglang-url', required=True)
    p.add_argument('--backend-url', required=True); p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    evidence = {'schema': 'natlang.live-sglang-native-comparison/1', 'native_url': args.native_url,
                'sglang_url': args.sglang_url, 'backend_url': args.backend_url, 'producer': [], 'turns': []}
    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temp = args.out.with_suffix(args.out.suffix + '.partial')
        temp.write_text(json.dumps(evidence, indent=2) + '\n'); os.replace(temp, args.out)
    try:
        evidence['native_info'], _ = request(args.native_url + '/v1/neuralese/info')
        evidence['sglang_info'], _ = request(args.sglang_url + '/v1/neuralese/info')
        save()
        texts = ['A short test.', 'The north sensor reports a stable temperature of 19 degrees. ' * 8,
                 'One entry records an observation, its timestamp, and a verified result. ' * 32]
        artifacts = []
        for text in texts:
            result, raw = capture_case(args.native_url, args.backend_url, text)
            evidence['producer'].append(result); artifacts.append(raw); save()
        state_id = evidence['producer'][1]['native_state_id']
        request(args.sglang_url + '/v1/neuralese/native-states/' + state_id, artifacts[1], method='PUT',
                content_type='application/octet-stream')
        prompts = [ [{'role':'user','content':'What is 2 + 3? Give only the answer.'}],
                    [{'role':'user','content':[{'type':'text','text':'Summarize this recorded observation: '},
                                                {'type':'native_state','id':state_id}]}],
                    [{'role':'user','content':[{'type':'text','text':'Compare these two observations: '},
                                                {'type':'native_state','id':state_id},
                                                {'type':'text','text':' and '},
                                                {'type':'native_state','id':state_id}]}] ]
        for messages in prompts:
            for temperature in (0.0, 1.0):
                body = {'messages':messages,'temperature':temperature,'max_tokens':8,'seed':42,
                        'x_natlang_rl_capture':True,'x_natlang_native_recording':True,'x_natlang_rl_top_k':16}
                native, _ = request(args.native_url + '/v1/chat/completions', body)
                sglang, _ = request(args.sglang_url + '/v1/chat/completions', body)
                evidence['turns'].append({'request':body, 'native':native, 'sglang':sglang,
                                          'comparison':sample_diffs(native, sglang)}); save()
    except Exception as e:
        evidence['error'] = {'type':type(e).__name__,'message':str(e)}; save(); raise
    print(args.out)

if __name__ == '__main__':
    main()
