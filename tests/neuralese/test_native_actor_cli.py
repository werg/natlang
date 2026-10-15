"""The managed actor entry point serves exact native checkpoints without heads."""
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

import torch

from natlang_neuralese.common.hashing import sha256_file_hex_cached
from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA, load_native_model, model_state
from natlang_neuralese.rl.production import validate_actor_receipts
from test_native_serve import _tiny_source


def test_native_actor_cli_restores_actual_checkpoint_and_exposes_exact_policy(tmp_path):
    base = tmp_path/'base'
    base.mkdir()
    _tiny_source(base)
    model, _, _, config = load_native_model(base, dtype=torch.float32, rank=4)
    checkpoint = tmp_path/'native.pt'
    torch.save({'schema': CHECKPOINT_SCHEMA, 'model_config': config, **model_state(model)}, checkpoint)
    digest = sha256_file_hex_cached(checkpoint)
    root = Path(__file__).resolve().parents[2]
    argv = [sys.executable, str(root/'scripts/neuralese_data/serve_direct_checkpoint.py'),
            '--checkpoint', str(checkpoint), '--checkpoint-sha256', digest,
            '--native-state', '--device', 'cpu', '--port', '0',
            '--store-dir', str(tmp_path/'actor-store')]
    written = None
    raw = None
    for attempt in range(2):
        with (tmp_path/'actor-stderr.log').open('w') as errors:
            process = subprocess.Popen(argv,
                cwd=root, stdout=subprocess.PIPE, stderr=errors, text=True)
            try:
                line = process.stdout.readline()
                assert line, (tmp_path/'actor-stderr.log').read_text()
                ready = json.loads(line)
                assert ready['native_state'] is True and ready['heads_sha256'] is None
                endpoint = ready['listening']
                with urllib.request.urlopen(endpoint+'/v1/neuralese/info', timeout=30) as response:
                    info = json.load(response)
                assert info['rl_policy_binding'] == ready['rl_policy_binding']
                validate_actor_receipts({endpoint: info['rl_policy_binding']}, policy_id=digest, neuralese_server=True)
                if attempt:
                    from natlang_neuralese.rl.graph_replay import save_bundle, verified_bundle_bytes
                    from natlang_neuralese.serve.store import decode_native_state
                    def fetch(block_id):
                        with urllib.request.urlopen(endpoint+'/v1/neuralese/native-states/'+block_id, timeout=30) as response:
                            return response.read()
                    assert fetch(written['id']) == raw
                    save_bundle([{'native_block': written['id']}], tmp_path/'bundle', policy_id=digest, fetch=fetch)
                    archived = verified_bundle_bytes(tmp_path/'bundle/manifest.json', written['id'], policy_id=digest)
                    block, block_id = decode_native_state(archived)
                    assert archived == raw and block_id == written['id']
                    assert block.checkpoint_sha256 == digest
                    continue
                request = urllib.request.Request(endpoint+'/v1/neuralese/native-states/write-text',
                    data=json.dumps({'text': 'hello'}).encode(), headers={'content-type': 'application/json'})
                with urllib.request.urlopen(request, timeout=30) as response:
                    written = json.load(response)
                assert written['id'].startswith('nzs1_')
                assert written['native_state']['checkpoint_sha256'] == digest
                with urllib.request.urlopen(endpoint+'/v1/neuralese/native-states/'+written['id'], timeout=30) as response:
                    raw = response.read()
            finally:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
