import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { buildTaskSources, digest, replacement, safePath } from '../scripts/inline-curriculum/directory-sources.mjs';
import { buildTrajectorySources, fullView } from '../scripts/inline-curriculum/trajectory-sources.mjs';
import { buildSourceBundle } from '../scripts/inline-curriculum/build-source-backed.mjs';
import { staticBundleInput } from '../scripts/inline-curriculum/static-bundle-input.mjs';
import { admitRow } from '../dist/teacher/curriculum.js';
import { materializeNativeRows } from '../dist/teacher/native-materializer.js';
import { renderSftTurn } from '../scripts/export-native-sft.mjs';

const info = { license: 'MIT', original_split: 'tool', sha256: 'a'.repeat(64), dataset: 'fixture/tool-traces', files: [] };
const call = (index, args, observation, name = 'str_replace_editor') => [
  { role: 'assistant', tool_calls: [{ id: `c${index}`, function: { name, arguments: JSON.stringify(args) } }] },
  { role: 'tool', tool_call_ids: [`c${index}`], content: [{ type: 'text', text: observation }] },
];
const view = "Here's the result of running `cat -n` on /testbed/notes.txt:\n     1\told\n     2\tkeep";
const terminal = { role: 'assistant', tool_calls: [{ id: 'terminal', function: { name: 'submit', arguments: '{}' } }] };
const trace = (changes = {}) => ({ resolved: true, instance_id: 'sample__repo.12345678.task', traj_id: 'trace-1',
  messages: JSON.stringify([{ role: 'user', content: 'Original task' },
    ...call(0, { command: 'view', path: '/testbed/notes.txt' }, view),
    ...call(1, { command: 'str_replace', path: '/testbed/notes.txt', old_str: 'old', new_str: 'new' },
      'The file /testbed/notes.txt has been edited.\n     1\tnew\n     2\tkeep'), terminal]), ...changes });
const licenseResolver = async () => ({ spdx: 'MIT', pinned: true, revision: '12345678', sha256: 'b'.repeat(64) });

test('replacement is exact, unique and preserves newline bytes', () => {
  for (const [before, after] of [['a\nb\n', 'a\nc\n'], ['a\n', 'a\nb\n'], ['a', 'b']]) {
    const edit = replacement(before, after);
    assert.equal(before.replace(edit.find, edit.replace_with), after);
  }
  for (const path of ['/outside', '../escape', 'a/../b', 'a\\b', 'a//b']) assert.throws(() => safePath(path));
});

test('partial and clipped views cannot restore a complete source file', () => {
  assert.equal(fullView(view, {}), 'old\nkeep\n');
  assert.throws(() => fullView(view, { view_range: [1, 2] }), /partial_view/);
  assert.throws(() => fullView(view + '\n<response clipped>', {}));
  assert.throws(() => fullView('cat -n\n 2\told', {}));
});

test('known successful editor slice retains source actions and native wrapper scope', async () => {
  const result = await buildTrajectorySources({ swesmith: { info, rows: [trace()] } }, 1, licenseResolver);
  assert.equal(result.records.length, 1);
  const record = result.records[0];
  assert.equal(record.semantics.expected_files['notes.txt'], 'new\nkeep\n');
  assert.equal(record.external_source.trajectory.whole_issue_replayed, false);
  assert.deepEqual(record.external_source.trajectory.source_message_indices, [1, 3]);
  assert.equal(record.external_source.trajectory.source_calls[1].args.new_str, 'new');
  assert.equal(result.audits[0].unsupported_tools.length, 0);
});

test('failed, unknown and unlicensed source traces are excluded', async () => {
  const result = await buildTrajectorySources({ swesmith: { info, rows: [trace({ resolved: false }), trace({ resolved: -1 }), trace()] } }, 3, async () => null);
  assert.equal(result.records.length, 0);
  assert.deepEqual(result.rejected.map(r => r.reason), ['source_task_failed', 'unknown_source_success', 'pinned_repository_license_missing']);
});

test('opaque shell commands invalidate prior reconstructed file state', async () => {
  const row = trace(); const ms = JSON.parse(row.messages);
  ms.splice(3, 0, ...call(8, { command: 'modify notes.txt' }, 'done', 'bash'));
  row.messages = JSON.stringify(ms);
  const result = await buildTrajectorySources({ swesmith: { info, rows: [row] } }, 1, licenseResolver);
  assert.equal(result.records.length, 0);
  assert.equal(result.rejected[0].reason, 'no_independent_verified_editor_slice');
});

test('complete successful create can become an explicit-content task without pretending to replay the issue', async () => {
  const row = trace({ messages: JSON.stringify([
    ...call(0, { command: 'create', path: '/testbed/probe.py', file_text: 'print(1)\n' }, 'File created successfully at: /testbed/probe.py'), terminal]) });
  const result = await buildTrajectorySources({ swesmith: { info, rows: [row] } }, 1, licenseResolver);
  assert.equal(result.records[0].external_source.trajectory.conversion_scope, 'independent_file_creation');
  assert.equal(result.records[0].semantics.expected_files['probe.py'], 'print(1)\n');
  assert.equal(result.records[0].semantics.folder_files['probe.py'], undefined);
});

test('WorkBench latest-date source disagreement and CommitPack unsafe paths are rejected', () => {
  const workbench = { info: { ...info, original_split: 'unsplit' }, rows: [{ tasks: [{ task: 'Delete my last email from alex',
    base_template: 'Delete my last email from {name}', outcome: 'email.delete_email.func(email_id="1")' }], emails: [
    { email_id: '1', 'inbox/outbox': 'inbox', 'sender/recipient': 'alex.user@example.org', sent_datetime: '2020-01-01' },
    { email_id: '2', 'inbox/outbox': 'inbox', 'sender/recipient': 'alex.user@example.org', sent_datetime: '2021-01-01' }]}] };
  const commitpack = { info, rows: [{ commit: '1', license: 'MIT', old_file: '../bad', new_file: '../bad' }] };
  const result = buildTaskSources({ workbench, commitpack });
  assert.equal(result.records.length, 0);
  assert.deepEqual(result.rejected.map(r => r.reason), ['source_latest_email_disagreement', 'unsafe_source_path']);
});

test('source bundle replays, admits, materializes and preserves attribution through rendering; tampering fails closed', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-source-bundle-'));
  try {
    const raw = JSON.stringify([trace()]) + '\n';
    await writeFile(join(root, 'rows.json'), raw);
    await writeFile(join(root, 'manifest.json'), JSON.stringify({ version: 'natlang.directory_sources/1', sources: {
      swesmith: { ...info, path: 'rows.json', sha256: digest(raw) } } }));
    const out = join(root, 'output');
    const report = await buildSourceBundle({ cache: root, out, limit: 1, trajectoryLimit: 1, licenseResolver });
    assert.equal(report.cases, 1); assert.equal(report.model_calls, 0);
    const input = await staticBundleInput(join(out, 'static.manifest.json'));
    const row = JSON.parse((await readFile(input, 'utf8')).trim());
    assert.equal(admitRow(row).admitted, true);
    assert.equal(row.provenance.collection_role, 'external_replay');
    const turns = materializeNativeRows([row]);
    assert.equal(turns.unlinked.length, 0);
    const turn = turns.turns.find(t => t.training_admission.approved);
    assert.match(turn.license, /MIT/);
    // Deterministic template fixture verifies export metadata without a model/server.
    const render = async messages => messages.map(m => m.role === 'assistant' && m.tool_calls ?
      `assistant:${JSON.stringify(m.tool_calls)}<end>` : `${m.role}:${m.content}`).join('\n') + '\n';
    const rendered = await renderSftTurn({ ...turn, teacher_reasoning: null }, render, '<end>');
    assert.equal(rendered.license, turn.license);
    assert.equal(rendered.source_conversion.source_success, true);
    row.task.program_ir.semantics.expected = 'tampered';
    assert.equal(admitRow(row).admitted, false);
    assert.equal(materializeNativeRows([row]).acceptedRows, 0);
    await writeFile(input, JSON.stringify(row) + '\n');
    await assert.rejects(staticBundleInput(join(out, 'static.manifest.json')), /checksum_mismatch/);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('static bundle optional absence differs from explicitly missing bundle', async () => {
  assert.equal(await staticBundleInput('/missing/natlang/static.manifest.json', { optional: true }), null);
  await assert.rejects(staticBundleInput('/missing/natlang/static.manifest.json'));
});

test('TAT-QA reference reads evidence before choosing numeric operations', () => {
  const result = buildTaskSources({ tatqa: { info: { ...info, original_split: 'train' }, rows: [{
    table: { uid: 'table1', table: [['Revenue from subscriptions', '25'], ['Previous revenue', '20']] },
    paragraphs: [{ uid: 'note1', text: 'Revenue is expressed in millions.' }],
    questions: [{ uid: 'q1', question: 'What is the revenue increase?', answer: 5,
      answer_type: 'arithmetic', derivation: '25-20', scale: 'million' }] }] } });
  assert.equal(result.records.length, 1);
  assert.equal(result.records[0].curriculum.mode, 'followup');
  assert.deepEqual(result.records[0].curriculum.reference.root.slice(0, 2).map(call => call[0]), ['read_file', 'read_file']);
});

test('MuSiQue training cases sharing a development seed are held out', () => {
  const result = buildTaskSources({ musique: { info: { ...info, original_split: 'train', held_out_seed_ids: ['dev1'] }, rows: [{
    id: 'q1', answerable: true, answer: 'Paris', question: 'Where?', question_decomposition: [{ id: 'dev1' }],
    paragraphs: [{ idx: 0, title: 'Location', paragraph_text: 'Paris', is_supporting: true }] }] } });
  assert.equal(result.records.length, 0);
  assert.equal(result.rejected[0].reason, 'held_out_musique_seed');
});

test('streaming materializer reports aggregate counters for an empty corpus without leaving staged files', async () => {
  const { spawnSync } = await import('node:child_process');
  const { readdir } = await import('node:fs/promises');
  const root = await mkdtemp(join(tmpdir(), 'natlang-stream-empty-'));
  try {
    const input = join(root, 'input.jsonl'), output = join(root, 'output.jsonl');
    await writeFile(input, '');
    const run = spawnSync(process.execPath, [new URL('../scripts/materialize-native-teacher.mjs', import.meta.url).pathname, input, output], {encoding: 'utf8'});
    assert.equal(run.status, 0, run.stderr);
    assert.deepEqual(JSON.parse(run.stdout), {output, accepted_rows: 0, rejected_rows: 0, training_decisions: 0});
    assert.equal(await readFile(output, 'utf8'), '');
    assert.deepEqual((await readdir(root)).sort(), ['input.jsonl', 'output.jsonl']);
  } finally { await rm(root, {recursive: true, force: true}); }
});
