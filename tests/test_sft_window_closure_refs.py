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
