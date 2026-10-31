import json
import os
from pathlib import Path
import shutil
import subprocess


def test_verifier_reuses_only_same_input_and_executable_runtime(tmp_path):
    runtime = tmp_path / 'runtime'
    native = runtime / 'dist/native'
    teacher = runtime / 'dist/teacher'
    native.mkdir(parents=True)
    teacher.mkdir()
    (runtime / 'package.json').write_text('{"type":"module"}')
    prompt = native / 'prompt.js'
    prompt.write_text('export const TOOLS_PROMPT = "good";')
    (native / 'external.js').write_text('export function externalModule() { throw Error("unused"); }')
    (teacher / 'curriculum.js').write_text('''
      import fs from 'node:fs';
      export async function verifyCases(rows, prompt) {
        fs.appendFileSync(process.env.VERIFIER_TEST_TRACE, 'checked\\n');
        return rows.map(() => ({ok: prompt === 'good', problems: []}));
      }
      export async function replayReference(row) {
        const value = row.curriculum.reference.root.at(-1)[1].value;
        return {run: {outcome: {accepted: value === row.semantics.expected}}};
      }
    ''')
    row = {'id': 'same-world', 'family': 'test', 'split': 'train',
           'semantics': {'expected': 1},
           'curriculum': {'reference': {'root': [['return_result', {'status': 'success', 'value': 1}]]}}}
    source = tmp_path / 'cases.jsonl'
    source.write_text(json.dumps(row) + '\n')
    out = tmp_path / 'verified'
    trace = tmp_path / 'trace'
    script = Path(__file__).resolve().parents[1] / 'scripts/rl_tasks/verify.mjs'
    command = [shutil.which('node'), str(script), str(runtime), str(source), str(out), '1']

    def run():
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=30,
                       env={**os.environ, 'VERIFIER_TEST_TRACE': str(trace)})
        return json.loads((out / 'verification.json').read_text())

    assert run()['summary']['kept'] == 1
    assert run()['summary']['kept'] == 1
    assert trace.read_text().splitlines() == ['checked']
    # Same case ID and input, different executable semantics: old outcomes must
    # not acquire a fresh runtime's qualification simply by resuming the job.
    prompt.write_text('export const TOOLS_PROMPT = "bad";')
    assert run()['summary']['kept'] == 0
    assert len(trace.read_text().splitlines()) == 2
    row['semantics']['expected'] = 2
    row['curriculum']['reference']['root'][0][1]['value'] = 2
    source.write_text(json.dumps(row) + '\n')
    run()
    assert len(trace.read_text().splitlines()) == 3
    assert len(list(out.glob('partial.jsonl.previous-*'))) == 2
