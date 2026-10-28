import hashlib
import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location('student_rerender', Path(__file__).resolve().parents[2] / 'scripts/neuralese_data/student_rerender.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, 'REPO', tmp_path)
    item = {'id': 'example', 'out': 'new', 'path': 'old', 'kind': 'direct',
            'views': ['program-header'], 'status': 'verify-failed'}
    queue = {'student': 'lfm3', 'reference': 'lfm25'}
    for base, tok in [('old', 'lfm25'), ('new', 'lfm3')]:
        data = tmp_path / base / 'program-header' / tok / 'text-data.jsonl'
        data.parent.mkdir(parents=True); data.write_text('source')
        receipt = {'sources': {'input': {'sha256': '123'}}, 'selection': {'selected': 1},
                   'render_omissions': {f'{tok}/program-header/native_chat_render_failed': 1},
                   'renders': {f'program-header/{tok}': {'text_data_sha256': hashlib.sha256(data.read_bytes()).hexdigest()}}}
        put(tmp_path / base / 'receipt.json', receipt)
    put(tmp_path / 'new/student-verification.json',
        {'id': 'example', 'student': 'lfm3', 'reference': 'lfm25', 'passed': True, 'identical_documents': True,
         'render_receipt_sha256': hashlib.sha256((tmp_path / 'new/receipt.json').read_bytes()).hexdigest(),
         'reference_receipt_sha256': hashlib.sha256((tmp_path / 'old/receipt.json').read_bytes()).hexdigest()})
    return item, queue


def test_late_verification_requires_bound_identity_and_source(tmp_path, monkeypatch):
    item, queue = setup(tmp_path, monkeypatch)
    assert mod.late_receipt_status(item, queue) == 'verified'
    put(tmp_path / 'new/receipt.json', {'sources': {'other': {}}, 'selection': {'selected': 1}})
    assert mod.late_receipt_status(item, queue) is None


def test_late_subset_requires_exact_typed_omissions(tmp_path, monkeypatch):
    item, queue = setup(tmp_path, monkeypatch)
    report_path = tmp_path / 'new/student-verification.json'
    report = json.loads(report_path.read_text())
    report.update(identical_documents=False, views={'program-header': {
        'only_reference': 1, 'only_reference_sample': ['x'], 'only_student': 0,
        'split_or_group_changed': 0}})
    put(report_path, report)
    assert mod.late_receipt_status(item, queue) is None
    omission = tmp_path / 'new/render-omissions/part.jsonl'
    omission.parent.mkdir(parents=True)
    omission.write_text(json.dumps({'rendering': 'program-header', 'id': 'x',
                                    'reason': 'native_chat_render_failed',
                                    'detail': 'tool-call-unrepresentable'}) + '\n')
    assert mod.late_receipt_status(item, queue) == 'verified'
    omission.write_text(omission.read_text().replace('"x"', '"other"'))
    assert mod.late_receipt_status(item, queue) is None
    put(tmp_path / 'new/student-verification.json', {'id': 'other', 'student': 'lfm3', 'reference': 'lfm25', 'passed': True})
    assert mod.late_receipt_status(item, queue) is None


def test_late_graph_requires_actual_source_and_outputs(tmp_path, monkeypatch):
    item, queue = setup(tmp_path, monkeypatch)
    item['status'] = 'graph-failed'
    source = tmp_path / 'new/program-header/lfm3/text-data.jsonl'
    source.write_text('source')
    target = tmp_path / 'new/native-graphs/program-header'
    target.mkdir(parents=True); output = target / 'train.jsonl'; output.write_text('graph')
    import student_tokenizers
    monkeypatch.setattr(student_tokenizers, 'model_id', lambda _: 'model')
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    put(target / 'receipt.json', {'model_id': 'model', 'source': {'path': str(source), 'sha256': digest(source)},
                                  'outputs': [{'path': str(output), 'sha256': digest(output)}]})
    assert mod.late_receipt_status(item, queue) == 'graphed'
    output.write_text('changed')
    assert mod.late_receipt_status(item, queue) is None
