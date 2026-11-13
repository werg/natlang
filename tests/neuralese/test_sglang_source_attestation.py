import importlib.util
import subprocess
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('serve_native_checkpoint',
    Path(__file__).resolve().parents[2] / 'scripts/neuralese_data/serve_native_checkpoint.py')
serve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve)


def test_attestation_records_the_actual_fork_commit_and_warns_never_refuses(tmp_path, capsys):
    repo = tmp_path / 'fork'
    package = repo / 'python/sglang'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'add', 'python/sglang/__init__.py'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=test', '-c', 'user.email=test@example.org',
                    'commit', '-qm', 'fixture'], check=True)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    python = tmp_path / 'python'
    python.write_text('#!/bin/sh\nprintf "%s\\n" ' + str(package / '__init__.py') + '\n')
    python.chmod(0o755)
    record = serve.attest_sglang_source(str(python), {'commit': commit})
    assert record['commit'] == commit and record['matches_declared'] is True and record['dirty'] is False
    assert capsys.readouterr().err == ''
    other = serve.attest_sglang_source(str(python), {'commit': '0' * 40})   # a different fork head still launches
    assert other['commit'] == commit and other['declared_commit'] == '0' * 40 and other['matches_declared'] is False
    assert 'sglang_fork_differs_from_declared' in capsys.readouterr().err
    (package / 'injected.py').write_text('')
    dirty = serve.attest_sglang_source(str(python), {'commit': commit})
    assert dirty['dirty'] is True and dirty['commit'] == commit
    assert dirty['hybrid_swa_kv'] is False   # the fixture fork has no LFM3 model file


LFM3_KINDS = ['conv', 'conv', 'full_attention', 'conv', 'sliding_attention'] * 2 + ['full_attention']


def test_hybrid_swa_pool_counts_the_sliding_layers_at_the_swa_ratio(tmp_path):
    import argparse, json
    (tmp_path / 'config.json').write_text(json.dumps({'layer_types': LFM3_KINDS, 'sliding_window': 256, 'head_dim': 128,
                                                      'num_key_value_heads': 4, 'num_hidden_layers': len(LFM3_KINDS)}))
    args = argparse.Namespace(sglang_kv_cache_dtype='fp8_e4m3', sglang_max_total_tokens=None, sglang_kv_gb=10.0,
                              sglang_mem_fraction=0.5, sglang_swa_ratio=0.1, sglang_full_kv=False,
                              sglang_max_running=128)
    assert serve.kv_layout(tmp_path, args, {'hybrid_swa_kv': True}) == 'hybrid-swa'
    assert serve.kv_layout(tmp_path, args, {'hybrid_swa_kv': False}) == 'full'   # an older fork keeps full KV
    full, _ = serve.sglang_pool(tmp_path, args, 1024, 'full')
    hybrid, _ = serve.sglang_pool(tmp_path, args, 1024, 'hybrid-swa')
    per_layer = 4 * 128 * 2   # FP8 K and V
    assert full == int(10e9 // (5 * per_layer)) and hybrid == int(10e9 // ((3 + 0.1 * 2) * per_layer))
    small = argparse.Namespace(**{**vars(args), 'sglang_kv_gb': 0.5})   # 0.1 x 156k tokens < 128 windows + 2 chunks
    tokens, _ = serve.sglang_pool(tmp_path, small, 1024, 'hybrid-swa')
    assert small.sglang_swa_ratio * tokens >= 128 * (256 + 130) + 2 * 16384 > 0.1 * tokens
    args.sglang_full_kv = True
    assert serve.kv_layout(tmp_path, args, {'hybrid_swa_kv': True}) == 'full'
