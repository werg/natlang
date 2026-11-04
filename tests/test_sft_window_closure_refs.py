import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sft_window', ROOT / 'scripts' / 'neuralese_data' / 'sft_window.py')
window = importlib.util.module_from_spec(spec)
spec.loader.exec_module(window)


def test_admission_references_are_followed_transitively_but_not_source_manifests(tmp_path):
    selected = tmp_path / 'selected-publication.json'
    selected.write_text('{}')
    admission = tmp_path / 'admission.json'
    admission.write_text(json.dumps({'selected_execution': {'path': str(selected), 'sha256': 'a' * 64}}))
    source_controller = tmp_path / 'source-controller.json'
    source_controller.write_text(json.dumps({'admission': {'path': str(admission), 'sha256': 'b' * 64},
                                             'evidence': str(tmp_path / 'unbound.json')}))
    (tmp_path / 'unbound.json').write_text('{}')
    shard = tmp_path / 'shard-receipt.json'
    shard.write_text('{}')
    manifest = tmp_path / 'sources.json'
    manifest.write_text(json.dumps({'schema': 'natlang.native-graph-sources/1',
                                    'sources': [{'path': str(shard), 'sha256': 'c' * 64}]}))
    history = tmp_path / 'history.json'
    history.write_text(json.dumps({'snapshot': {'path': str(manifest), 'sha256': 'd' * 64}}))
    config = {'execution_transition_source_controller': {'path': str(source_controller), 'sha256': 'e' * 64},
              'trainer_parameters': {'native_graphs': str(manifest)}, 'task_selection_history': str(history)}
    roots = window.admission_roots(config)
    assert roots == sorted([str(source_controller), str(history)])
    found = window.nested_json_references(roots)
    # Bound references are shipped transitively; unbound paths are evidence; a source manifest is not expanded.
    assert found == sorted([str(admission), str(selected), str(manifest)])


def test_historical_checkpoints_bound_in_admission_records_are_reported(tmp_path):
    parent = tmp_path / 'old' / 'checkpoint.pt'
    parent.parent.mkdir()
    parent.write_bytes(b'x')
    admission = tmp_path / 'admission.json'
    admission.write_text(json.dumps({'transition': {'parent': {'path': str(parent), 'sha256': 'f' * 64},
                                                    'child': {'path': '/absent/checkpoint.pt', 'sha256': 'e' * 64}}}))
    found = window.nested_checkpoint_references([str(admission)], [])
    assert found == {'/absent/checkpoint.pt': 'e' * 64, str(parent): 'f' * 64}


def test_sealed_collection_directories_are_admission_payloads(tmp_path):
    out = tmp_path / 'collection'
    (out / 'blocks').mkdir(parents=True)
    proof = tmp_path / 'sealed-collection.json'
    proof.write_text(json.dumps({'schema': 'natlang.sealed-collection-continuation/1', 'out': str(out),
                                 'files': {'rollouts.jsonl': {'path': str(out / 'rollouts.jsonl'), 'sha256': 'a' * 64}}}))
    other = tmp_path / 'source.jsonl'
    other.write_text('{}')
    record = tmp_path / 'record.json'
    record.write_text(json.dumps({'source': {'path': str(other), 'sha256': 'b' * 64}}))
    assert window.admission_payload_references([str(record)], [str(proof)]) == [str(out)]


def test_protection_roots_named_by_admission_records_ship_their_untracked_sources(tmp_path):
    import subprocess
    root = tmp_path / 'checkout'
    (root / 'training').mkdir(parents=True)
    (root / 'runs' / 'held').mkdir(parents=True)
    (root / 'runs' / 'held' / 'cases.jsonl').write_text('{}\n')
    (root / 'groups.json').write_text('{}')
    (root / 'training' / 'neuralese_corpora.json').write_text(json.dumps({'corpora': [
        {'id': 'held', 'path': 'runs/held', 'admission': 'evaluation-only-never-training', 'include': ['cases.jsonl']}]}))
    (root / 'training' / 'protected-worlds.json').write_text(json.dumps({
        'schema': 'natlang.protected-worlds-registry/1',
        'sources': [{'id': 'packets', 'kind': 'evaluation-packets', 'registry': 'training/neuralese_corpora.json'},
                    {'id': 'groups', 'kind': 'source-groups', 'path': 'groups.json'}]}))
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    subprocess.run(['git', '-C', str(root), 'add', 'training', 'groups.json'], check=True)
    record = tmp_path / 'qualification.json'
    record.write_text(json.dumps({'report': {'protection_root': str(root)}}))
    # The registry files tracked by the checkout arrive with the code bundle; the untracked held cases are shipped.
    # The canonical checkout (REPO) is always a protection root too (the parent's folded admission history can name it).
    shipped = window.protection_root_inputs([str(record)], [])
    assert str(root / 'runs' / 'held' / 'cases.jsonl') in shipped
    assert not any(p.startswith(str(root)) and not p.endswith('cases.jsonl') for p in shipped)


def test_traced_reads_keep_successful_regular_file_accesses_outside_system_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(window, 'TRACE_SYSTEM', ('/usr/', '/lib/'))  # pytest's tmp_path lives under /tmp
    source = tmp_path / 'source.jsonl.zst'
    source.write_bytes(b'x')
    (tmp_path / 'dir').mkdir()
    trace = tmp_path / 'strace.txt'
    trace.write_text('\n'.join([
        f'24    openat(AT_FDCWD, "{source}", O_RDONLY) = 3',
        f'24    newfstatat(AT_FDCWD, "{tmp_path / "dir"}", {{st_mode=S_IFDIR|0755, ...}}, 0) = 0',
        f'24    openat(AT_FDCWD, "{tmp_path / "absent.json"}", O_RDONLY) = -1 ENOENT (No such file or directory)',
        '24    openat(AT_FDCWD, "/usr/lib/python3.12/json/__init__.py", O_RDONLY) = 3',
        '24    openat(AT_FDCWD, "/lib/aarch64-linux-gnu/liblz4.so.1", O_RDONLY|O_CLOEXEC) = 3',
        f'24    statx(AT_FDCWD, "{source}", AT_STATX_SYNC_AS_STAT, STATX_ALL, {{...}}) = 0',
    ]) + '\n')
    assert window.traced_reads(trace) == [str(source)]
