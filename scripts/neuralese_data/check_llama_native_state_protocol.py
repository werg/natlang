"""Usage: check_llama_native_state_protocol.py LLAMA_SERVER MODEL.gguf SERVER_LOG

llama-server's native-state protocol on a natlang GGUF export (natlang_neuralese.export.native_gguf): /v1/neuralese/info
declares the export's native identity; a grammar forces a producer span (produce-open, text, produce-close) and the
reply carries it as a native_state part; the stored artifact decodes with the reference codec (serve/store.py) to the
same content ID and re-encodes byte for byte; a re-upload is accepted, a wrong ID refused; consuming the state answers an
exact-native-state receipt; an unknown state fails with neuralese-unknown-native-state.
"""
import base64, json, math, os, struct, subprocess, sys, time, urllib.request
from pathlib import Path
from natlang_neuralese.serve.store import decode_native_state, encode_native_state
server, gguf, port = sys.argv[1], sys.argv[2], 18991
base = f'http://127.0.0.1:{port}'
def call(path, body=None, method=None, raw=False, ctype='application/json'):
    data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data, {'content-type': ctype}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            out = r.read(); return r.status, (out if raw else json.loads(out))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:400]
proc = subprocess.Popen([server, '-m', gguf, '--port', str(port), '-c', '2048', '-np', '1', '--no-warmup', '--jinja',
                         *(['-ngl', '0'] if '--cpu' in sys.argv else [])],
                        env={**os.environ, **({'CUDA_VISIBLE_DEVICES': ''} if '--cpu' in sys.argv else {})},
                        stdout=open(sys.argv[3], 'w'), stderr=subprocess.STDOUT)
try:
    for _ in range(300):
        if proc.poll() is not None:
            raise RuntimeError(f'llama-server exited before readiness: {proc.returncode}; see {sys.argv[3]}')
        try: urllib.request.urlopen(base + '/health', timeout=1); break
        except Exception: time.sleep(0.5)
    st, info = call('/v1/neuralese/info'); print('info', st, json.dumps(info)[:600])
    if '--expect-rl' in sys.argv:
        checkpoint_sha = sys.argv[sys.argv.index('--expect-rl') + 1]
        binding = info['rl_policy_binding']
        assert binding['schema'] == 'natlang.rl-policy-binding/2'
        assert binding['checkpoint_sha256'] == checkpoint_sha
        assert binding['native_identity']['checkpoint_sha256'] == checkpoint_sha
        manifest = json.loads((Path(gguf).parent / 'manifest.json').read_text())
        exported = [item for item in manifest['files'] if item['file'] == Path(gguf).name]
        assert len(exported) == 1 and binding['actor_export']['sha256'] == exported[0]['sha256']
        if '--expect-quantized' in sys.argv:
            assert binding['actor_export']['kind'] == 'quantized'
            assert 'nvfp4' in binding['actor_export']['format']
        assert 'rl.capture' in info['capabilities']
        status, sampled = call('/v1/chat/completions', {
            'messages': [{'role': 'user', 'content': 'What is 2 plus 2?'}],
            'max_tokens': 4, 'temperature': 0.8, 'seed': 31,
            'x_natlang_rl_capture': True, 'x_natlang_rl_top_k': 128})
        assert status == 200, sampled
        capture = sampled['x_natlang_rl_capture']
        assert capture['policy'] == binding['native_identity']
        assert capture['actor_export'] == binding['actor_export']
        assert capture['behavior_top_k']['k'] == 128 and capture['samples']
        for sample in capture['samples']:
            top = sample['top_k']
            ids = struct.unpack('<128i', base64.b64decode(top['ids'], validate=True))
            logps = struct.unpack('<128e', base64.b64decode(top['logps'], validate=True))
            assert len(set(ids)) == 128 and all(math.isfinite(x) and x <= 0 for x in logps)
            assert all(a >= b for a, b in zip(logps, logps[1:]))
            if sample['token_id'] in ids:
                assert abs(sample['logp'] - logps[ids.index(sample['token_id'])]) < 0.01
        print('RL binding and top-128 capture', len(capture['samples']))
    po, pc = info['native_state']['role_marker_ids'][:2]
    grammar = f'root ::= "Billing: " <[{po}]> " refund request for order 4411" <[{pc}]> " done"'
    st, reply = call('/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'Summarize the ticket.'}],
        'max_tokens': 40, 'temperature': 0, 'grammar': grammar})
    assert st == 200, reply
    print('produce', st, json.dumps(reply)[:900] if st == 200 else reply)
    blocks = reply['native_state']['blocks']; sid = blocks[0]['id']
    msg = reply['choices'][0]['message']; print('content', msg['content'])
    st, body = call(f'/v1/neuralese/native-states/{sid}', raw=True); print('get', st, len(body))
    assert st == 200
    block, pid = decode_native_state(body); print('python decode id matches', pid == sid, 'text', repr(block.text), 'tokens', len(block.token_ids))
    assert encode_native_state(block) == body, 'python re-encoding differs byte for byte'
    print('python re-encoding identical bytes')
    st, meta = call(f'/v1/neuralese/native-states/{sid}/meta'); print('meta', st, {k: meta[k] for k in ('id', 'text', 'layer_kinds')} if st == 200 else meta)
    st, put = call(f'/v1/neuralese/native-states/{sid}', body, method='PUT', ctype='application/octet-stream'); print('put again', st)
    assert st in (200, 201)
    st, bad = call('/v1/neuralese/native-states/nzs1_' + 'a' * 52, body, method='PUT', ctype='application/octet-stream'); print('put wrong id', st, str(bad)[:120])
    assert st == 400
    st, reply2 = call('/v1/chat/completions', {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'Which order is this about? '},
        {'type': 'native_state', 'id': sid}]}], 'max_tokens': 20, 'temperature': 0})
    print('consume', st, json.dumps(reply2.get('native_state') if st == 200 else reply2)[:400], reply2['choices'][0]['message']['content'][:120] if st == 200 else '')
    assert st == 200 and any(item['id'] == sid and item['mode'] == 'exact-native-state'
                             for item in reply2['native_state']['imports'])
    st, unknown = call('/v1/chat/completions', {'messages': [{'role': 'user', 'content': [{'type': 'native_state', 'id': 'nzs1_' + 'b' * 52}]}], 'max_tokens': 5})
    print('unknown', st, str(unknown)[:200])
    assert st == 400
finally:
    proc.terminate(); proc.wait(30)
