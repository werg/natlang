"""Provenance-pinned cohort assembly (no model loads)."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path



REPO = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_assembler(*args):
    return subprocess.run([sys.executable, str(REPO / 'scripts/assemble_neuralese_cohorts.py'), *map(str, args)],
                          capture_output=True, text=True)


def test_assembler_labels_cohorts_merges_pieces_and_checks_pins(tmp_path):
    native = tmp_path / 'native.jsonl'
    native.write_text(''.join(json.dumps({'id': f'n{i}', 'split': 'test' if i == 0 else 'train',
                                          'source_groups': [f'g{i}']}) + '\n' for i in range(3)))
    harness = tmp_path / 'harness.jsonl'
    harness.write_text(''.join(json.dumps({'id': f'h{i}', 'split': 'train', 'source_groups': ['repo']}) + '\n'
                               for i in range(4)))
    pieces_a, pieces_b = tmp_path / 'pa.jsonl', tmp_path / 'pb.jsonl'
    pieces_a.write_text(json.dumps({'name': 'prompt:a', 'text': 'A'}) + '\n')
    pieces_b.write_text(json.dumps({'name': 'prompt:a', 'text': 'A'}) + '\n' +
                        json.dumps({'name': 'prompt:b', 'text': 'B'}) + '\n')
    out = tmp_path / 'out'
    result = run_assembler('records', '--cohort', f'native=path:{native}@{sha(native)},{pieces_a}@{sha(pieces_a)}',
                           '--cohort', f'harness_bench=path:{harness}@{sha(harness)},{pieces_b}@{sha(pieces_b)}',
                           '--authority', 'native=recipe pin (test)', '--authority', 'harness_bench=test',
                           '--limit', 'harness_bench=2', '--out', out)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in (out / 'records.jsonl').read_text().splitlines()]
    assert [(r['id'], r['cohort']) for r in rows] == [('n0', 'native'), ('n1', 'native'), ('n2', 'native'),
                                                      ('h0', 'harness_bench'), ('h1', 'harness_bench')]
    assert {json.loads(l)['name'] for l in (out / 'pieces.jsonl').read_text().splitlines()} == {'prompt:a', 'prompt:b'}
    receipt = json.loads((out / 'receipt.json').read_text())
    assert receipt['admission_granted'] is False and receipt['cohorts']['harness_bench']['limit'] == 2
    assert receipt['outputs']['records.jsonl'] == sha(out / 'records.jsonl')
    # A wrong pin, a missing authority, or a piece with two texts is refused.
    assert run_assembler('records', '--cohort', f'native=path:{native}@{"0" * 64},{pieces_a}@{sha(pieces_a)}',
                         '--authority', 'native=x', '--out', tmp_path / 'bad1').returncode != 0
    assert run_assembler('records', '--cohort', f'native=path:{native}@{sha(native)},{pieces_a}@{sha(pieces_a)}',
                         '--out', tmp_path / 'bad2').returncode != 0
    pieces_b.write_text(json.dumps({'name': 'prompt:a', 'text': 'other'}) + '\n')
    assert run_assembler('records', '--cohort', f'native=path:{native}@{sha(native)},{pieces_a}@{sha(pieces_a)}',
                         '--cohort', f'harness_bench=path:{harness}@{sha(harness)},{pieces_b}@{sha(pieces_b)}',
                         '--authority', 'native=x', '--authority', 'harness_bench=x',
                         '--out', tmp_path / 'bad3').returncode != 0


def test_assembler_text_mode_requires_one_tokenizer_and_disjoint_held_groups(tmp_path):
    def rows(path, tokenizer, splits):
        path.write_text(''.join(json.dumps({'id': f'{path.stem}{i}', 'split': split, 'tokenizer_sha256': tokenizer,
                                            'source_groups': [f'{path.stem}-g{i}'], 'text': 't'}) + '\n'
                                for i, split in enumerate(splits)))
        return f'path:{path}@{sha(path)}'
    a = rows(tmp_path / 'a.jsonl', 't1', ['train', 'test'])
    b = rows(tmp_path / 'b.jsonl', 't1', ['train'])
    ok = run_assembler('text', '--cohort', 'native=' + a, '--cohort', 'harness_bench=' + b, '--authority', 'native=x',
                       '--authority', 'harness_bench=x', '--out', tmp_path / 'ok')
    assert ok.returncode == 0, ok.stderr
    labelled = [json.loads(l) for l in (tmp_path / 'ok/text.jsonl').read_text().splitlines()]
    assert [r['text_cohort'] for r in labelled] == ['native', 'native', 'harness_bench']
    c = rows(tmp_path / 'c.jsonl', 't2', ['train'])
    assert run_assembler('text', '--cohort', 'native=' + a, '--cohort', 'other=' + c, '--authority', 'native=x',
                         '--authority', 'other=x', '--out', tmp_path / 'mixed').returncode != 0
    leak = tmp_path / 'leak.jsonl'
    leak.write_text(json.dumps({'id': 'l0', 'split': 'train', 'tokenizer_sha256': 't1', 'source_groups': ['g']}) + '\n'
                    + json.dumps({'id': 'l1', 'split': 'test', 'tokenizer_sha256': 't1', 'source_groups': ['g']}) + '\n')
    assert run_assembler('text', '--cohort', f'native=path:{leak}@{sha(leak)}', '--authority', 'native=x',
                         '--out', tmp_path / 'leak').returncode != 0
