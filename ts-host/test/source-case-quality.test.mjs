import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { sourceCases } from '../scripts/code-corpus/source-cases.mjs';

const record = (body, type = 'number') => ({
  id: 'test:function', kind: 'function', language: 'javascript',
  function: { name: 'candidate', parameters: [{ name: 'value', type }], body },
  cases: [], verification: { status: 'unverified', reasons: [] },
});

test('source case generation has deterministic duplicate, negative, empty, tie, and Unicode boundaries', () => {
  const scalar = sourceCases(record('{ return value; }', 'string'));
  assert.equal(scalar.eligible, true);
  assert.ok(scalar.cases.some(item => item.args[0] === ''));
  assert.ok(scalar.cases.some(item => item.args[0] === '🙂'));
  assert.ok(scalar.cases.some(item => item.args[0] === 'aa'));
  assert.deepEqual(sourceCases(record('{ return value; }', 'string')), scalar);

  const array = sourceCases(record('{ return value; }', 'number[]'));
  assert.ok(array.cases.some(item => item.args[0].length === 0));
  assert.ok(array.cases.some(item => item.args[0].includes(-1)));
  assert.ok(array.cases.some(item => item.args[0].filter(value => value === 2).length > 1));
});

test('upstream return expectations remain separately identifiable beside source-derived outputs', async t => {
  const root = await mkdtemp(join(tmpdir(), 'source-case-evidence-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const input = join(root, 'tasks.jsonl'), output = join(root, 'observed.jsonl');
  const task = { ...record('{ return value + 1; }'), cases: [
    { args: [4], expected: 999, outcome: 'return' },
    { args: [5], expected: 6, outcome: 'return' },
    { args: [-7], expected: 'not portable', outcome: 'throw' },
  ], case_provenance:'upstream_tests', verification:{ status:'unverified', reasons:[] } };
  await writeFile(input, `${JSON.stringify(task)}\n`);
  const result = spawnSync(process.execPath, ['scripts/code-corpus/source-cases.mjs', '--input', input, '--output', output, '--execute'], { cwd: new URL('..', import.meta.url), encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const row = JSON.parse((await readFile(output, 'utf8')).trim());
  assert.deepEqual(row.cases, []);
  assert.equal(row.verification.status, 'rejected');
  assert.ok(row.verification.reasons.includes('upstream_assertion_mismatch'));
  const mismatch = row.observation.diagnostic_cases.find(item => item.upstream_case_index === 0);
  const match = row.observation.diagnostic_cases.find(item => item.upstream_case_index === 1);
  assert.equal(mismatch.expected, 5);
  assert.equal(mismatch.upstream_asserted_expected, 999);
  assert.equal(mismatch.upstream_assertion_status, 'mismatched_source_observation');
  assert.equal(match.expected, 6);
  assert.equal(match.upstream_asserted_expected, 6);
  assert.equal(match.upstream_assertion_status, 'matched_source_observation');
  assert.ok(row.observation.case_rejections.some(item => item.upstream_case_index === 2 && item.rejection === 'unsupported_upstream_case_shape_or_outcome'));
  assert.equal(row.observation.behavioral_evidence.kind, 'source_observed');
  assert.equal(row.observation.behavioral_evidence.upstream_assertions_checked, 2);
});

test('one source exception rejects only its case while valid observations survive', async t => {
  const root = await mkdtemp(join(tmpdir(), 'source-case-partial-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const input = join(root, 'tasks.jsonl'), output = join(root, 'observed.jsonl');
  const task = { ...record('{ if (value < 0) throw "negative"; return value * 2; }'), cases: [{ args: [-1], expected: 0, outcome: 'return' }], verification:{ status:'captured', reasons:[] } };
  await writeFile(input, `${JSON.stringify(task)}\n`);
  const result = spawnSync(process.execPath, ['scripts/code-corpus/source-cases.mjs', '--input', input, '--output', output, '--execute'], { cwd: new URL('..', import.meta.url), encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const row = JSON.parse((await readFile(output, 'utf8')).trim());
  assert.ok(row.cases.length > 0);
  assert.ok(row.observation.case_rejections.some(item => item.upstream_case_index === 0 && /negative/.test(item.rejection)));
  assert.equal(row.cases.some(item => item.upstream_case_index === 0), false);
});

test('conflicting expectations for duplicate upstream arguments quarantine the task', async t => {
  const root = await mkdtemp(join(tmpdir(), 'source-case-conflicting-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const input = join(root, 'tasks.jsonl'), output = join(root, 'observed.jsonl');
  const task = { ...record('{ return value; }'), cases: [
    { args:[7], expected:7, outcome:'return' }, { args:[7], expected:8, outcome:'return' },
  ], case_provenance:'upstream_tests', verification:{ status:'unverified', reasons:[] } };
  await writeFile(input, `${JSON.stringify(task)}\n`);
  const result = spawnSync(process.execPath, ['scripts/code-corpus/source-cases.mjs', '--input', input, '--output', output, '--execute'], { cwd: new URL('..', import.meta.url), encoding:'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const row = JSON.parse((await readFile(output, 'utf8')).trim());
  assert.deepEqual(row.cases, []);
  assert.equal(row.verification.status, 'rejected');
  assert.ok(row.verification.reasons.includes('conflicting_upstream_assertions'));
  assert.equal(row.observation.behavioral_evidence.conflicting_duplicate_input_groups, 1);
  assert.equal(row.observation.diagnostic_cases.filter(item => item.upstream_case_index !== undefined).length, 2);
});

test('an explicit upstream return assertion quarantines when source execution throws', async t => {
  const root = await mkdtemp(join(tmpdir(), 'source-case-assertion-throw-'));
  t.after(() => rm(root, { recursive:true, force:true }));
  const input = join(root, 'tasks.jsonl'), output = join(root, 'observed.jsonl');
  const task = { ...record('{ if (value < 0) throw "negative"; return value; }'),
    cases:[{ args:[-1], expected:-1, outcome:'return' }], case_provenance:'upstream_tests', verification:{ status:'unverified', reasons:[] } };
  await writeFile(input, `${JSON.stringify(task)}\n`);
  const result = spawnSync(process.execPath, ['scripts/code-corpus/source-cases.mjs', '--input', input, '--output', output, '--execute'], { cwd:new URL('..', import.meta.url), encoding:'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const row = JSON.parse((await readFile(output, 'utf8')).trim());
  assert.deepEqual(row.cases, []);
  assert.equal(row.verification.status, 'rejected');
  assert.ok(row.verification.reasons.includes('upstream_assertion_execution_failure'));
  assert.equal(row.behavioral_evidence.upstream_assertion_execution_failures, 1);
  assert.ok(row.observation.case_rejections.some(item => item.upstream_case_index === 0 && /negative/.test(item.rejection)));
});

test('existing cases on generated tasks are not labeled as upstream assertions', () => {
  const generated = { ...record('{ return value; }'), generation:{ generator:'fixture' }, cases:[{ args:[1], expected:10, outcome:'return' }] };
  const selected = sourceCases(generated).cases[0];
  assert.equal(selected.input_source, 'existing_task_case');
  assert.equal(Object.hasOwn(selected, 'upstream_asserted_expected'), false);
  assert.equal(selected.existing_expected, 10);
  const wrongType = sourceCases({ ...record('{ return value; }'), cases:[{ args:['1'], expected:1, outcome:'return' }] }).cases[0];
  assert.equal(wrongType.rejected, true);
  assert.equal(wrongType.rejection, 'upstream_argument_type_mismatch');
});

test('captured return snapshots remain capture evidence rather than upstream assertions', async t => {
  const root = await mkdtemp(join(tmpdir(), 'source-case-capture-'));
  t.after(() => rm(root, { recursive:true, force:true }));
  const input = join(root, 'tasks.jsonl'), output = join(root, 'observed.jsonl');
  const task = { ...record('{ return value * 2; }'), cases:[{ args:[3], expected:6, input_after:[3], outcome:'return' }],
    verification:{ status:'captured', reasons:[] } };
  const selected = sourceCases(task).cases[0];
  assert.equal(selected.input_source, 'runtime_capture');
  assert.equal(selected.captured_expected, 6);
  assert.equal(Object.hasOwn(selected, 'upstream_asserted_expected'), false);
  await writeFile(input, `${JSON.stringify(task)}\n`);
  const result = spawnSync(process.execPath, ['scripts/code-corpus/source-cases.mjs', '--input', input, '--output', output, '--execute'], { cwd:new URL('..', import.meta.url), encoding:'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const row = JSON.parse((await readFile(output, 'utf8')).trim());
  const capture = row.cases.find(item => item.input_source === 'runtime_capture');
  assert.equal(capture.captured_output_status, 'matched_source_observation');
  assert.equal(row.behavioral_evidence.upstream_assertions_checked, 0);
  assert.deepEqual(row.behavioral_evidence, row.observation.behavioral_evidence);
});

test('CLI appends keyed captures, checks declared argument types, and quarantines capture mismatches', async t => {
  const root = await mkdtemp(join(tmpdir(), 'source-case-cli-captures-'));
  t.after(() => rm(root, { recursive:true, force:true }));
  const input = join(root, 'tasks.jsonl'), capturesPath = join(root, 'captures.jsonl'), output = join(root, 'observed.jsonl');
  const task = { ...record('{ return value * 2; }'), cases:[{ args:[1], expected:2, outcome:'return' }], verification:{ status:'unverified', reasons:[] } };
  const captureKey = task.id;
  const captures = [
    { key:captureKey, args:[2], expected:99, input_after:[2], outcome:'return', portable:true, reasons:[] },
    { key:captureKey, args:['2'], expected:4, input_after:['2'], outcome:'return', portable:true, reasons:[] },
    { key:'missing:function', args:[4], expected:8, input_after:[4], outcome:'return', portable:true, reasons:[] },
  ];
  await writeFile(input, `${JSON.stringify(task)}\n`);
  await writeFile(capturesPath, `${captures.map(JSON.stringify).join('\n')}\n`);
  const result = spawnSync(process.execPath, ['scripts/code-corpus/source-cases.mjs', '--input', input, '--captures', capturesPath, '--output', output, '--execute'],
    { cwd:new URL('..', import.meta.url), encoding:'utf8', timeout:15000 });
  assert.equal(result.status, 0, result.stderr);
  const row = JSON.parse((await readFile(output, 'utf8')).trim());
  const report = JSON.parse(result.stdout.trim());
  const manifest = JSON.parse(await readFile(`${output}.manifest.json`, 'utf8'));
  assert.equal(report.matched_capture_cases, 2);
  assert.equal(report.unmatched_capture_cases, 1);
  assert.ok(manifest.config.inputs.some(([path]) => path === capturesPath));
  assert.deepEqual(row.cases, []);
  assert.equal(row.verification.status, 'rejected');
  assert.ok(row.verification.reasons.includes('capture_fidelity_mismatch'));
  const capture = row.observation.diagnostic_cases.find(item => item.input_source === 'runtime_capture');
  assert.equal(capture.capture_key, captureKey);
  assert.equal(capture.captured_expected, 99);
  assert.equal(capture.captured_output_status, 'different_source_observation');
  assert.ok(row.observation.case_rejections.some(item => item.rejection === 'runtime_capture_argument_type_mismatch'));
  assert.ok(row.observation.diagnostic_cases.some(item => item.input_source === 'existing_task_case' && item.existing_expected === 2));
});

test('mutation, asynchronous syntax, and unsupported outcomes stay rejected', () => {
  assert.equal(sourceCases(record('{ value.x = 1; return value; }')).eligible, false);
  assert.equal(sourceCases(record('{ return await value; }')).eligible, false);
  assert.equal(sourceCases({ ...record('{ return value; }'), verification: { status: 'unverified', reasons: ['async function'] } }).eligible, false);
  const pair = record('{ return left - right; }');
  pair.function.parameters = [{ name:'left', type:'number' }, { name:'right', type:'number' }];
  assert.ok(sourceCases(pair).cases.some(item => item.args[0] !== item.args[1]));
});

test('a mismatching capture quarantines conversion without claiming a specification failure', async t => {
  const root = await mkdtemp(join(tmpdir(), 'source-case-capture-mismatch-'));
  t.after(() => rm(root, { recursive:true, force:true }));
  const input = join(root, 'tasks.jsonl'), output = join(root, 'observed.jsonl');
  const task = { ...record('{ return value * 2; }'), cases:[{ args:[3], expected:99, outcome:'return' }],
    verification:{ status:'captured', reasons:[] } };
  await writeFile(input, `${JSON.stringify(task)}\n`);
  const result = spawnSync(process.execPath, ['scripts/code-corpus/source-cases.mjs', '--input', input, '--output', output, '--execute'],
    { cwd:new URL('..', import.meta.url), encoding:'utf8', timeout:15000 });
  assert.equal(result.status, 0, result.stderr);
  const row = JSON.parse((await readFile(output, 'utf8')).trim());
  assert.equal(row.verification.status, 'rejected');
  assert.ok(row.verification.reasons.includes('capture_fidelity_mismatch'));
  assert.deepEqual(row.cases, []);
  assert.equal(row.behavioral_evidence.upstream_assertions_checked, 0);
});
