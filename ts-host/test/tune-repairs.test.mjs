import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as host from '../dist/index.js';
import { WeightBinding } from '../dist/adaptation/weights/binding.js';
import { currentSplit, tuneProgram } from '../dist/tune/pipeline.js';
import { scaffoldManifest } from '../dist/tune/collect.js';
import { readRepairRows, runRepairs } from '../dist/tune/repairs.js';
import { roundsSelector } from '../dist/tune/schedule.js';
import { scriptedModel } from './support/natlang.mjs';

const base = { architecture: 'lfm2', checkpoint_sha256: 'c'.repeat(64), tokenizer_fingerprint: 't', embedding_weight_sha256: 'e'.repeat(64) };
const named = body => `---\nargs: { value: string }\nreturns: string\n---\n${body}\n`;
const teacherRespond = opening => opening.includes('Pass value') ? 'return await inner(value)' : 'return value.toUpperCase()';
// The student delegates correctly but forgets to upper-case inside the inner call.
const studentRespond = opening => opening.includes('Pass value') ? 'return await inner(value)' : 'return value';

function project() {
  const root = mkdtempSync(join(tmpdir(), 'natlang-repairs-'));
  mkdirSync(join(root, 'outer'));
  writeFileSync(join(root, 'outer.nl'), named('Pass value to inner.'));
  writeFileSync(join(root, 'outer', 'inner.nl'), named('Return value in uppercase.'));
  writeFileSync(join(root, 'package.json'), '{"type":"module"}');
  const examples = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta'].map(value => ({ function: 'outer.nl', input: { value }, expected: value.toUpperCase() }));
  examples.push({ function: 'outer.nl', input: { value: 'held' }, expected: 'HELD', protected: true });
  writeFileSync(join(root, 'tune.examples.jsonl'), examples.map(example => JSON.stringify(example)).join('\n') + '\n');
  writeFileSync(join(root, 'trainer.mjs'), `import { readFileSync, writeFileSync } from 'node:fs';
const args = process.argv.slice(2), plan = JSON.parse(readFileSync(args[args.indexOf('--plan') + 1], 'utf8')), out = args[args.indexOf('--out') + 1];
const rows = readFileSync(plan.rows, 'utf8').split('\\n').filter(Boolean).map(line => JSON.parse(line));
if (rows.some(row => row.partition !== 'train')) throw new Error('non-training row reached the trainer');
const artifacts = plan.manifest.tree.filter(node => node.level === 'lambda').map(node => ({ node: node.id, artifact: 'nnt1:' + node.id + ':' + rows.length, rank: 2, parameterization: 'native-subspace', requires: [] }));
writeFileSync(out + '/result.json', JSON.stringify({ artifacts, receipt: { rows: rows.length, corrections: rows.filter(row => row.correction).length } }));`);
  return root;
}
const programOf = root => {
  const built = host.buildProject({ project: root, emit: false, write: false });
  assert.equal(built.ok, true, host.formatDiagnostics(built.diagnostics));
  return built.manifest.adaptation;
};

test('a verified teacher repair of the student prefix is admitted with provenance; unverified ones are rejected', async () => {
  const root = project();
  try {
    const program = programOf(root), examples = currentSplit(root);
    const release = scaffoldManifest(program, base);
    const student = scriptedModel(studentRespond), teacher = scriptedModel(teacherRespond);
    const out = join(root, 'out');
    const summary = await runRepairs({ root, program, scaffold: release, release, weights: new WeightBinding(release, { program, strict: true }),
      student: student.driver, teacher: teacher.driver, teacherId: 'teacher:test', examples, round: 1,
      records: join(out, 'records.jsonl'), rows: join(out, 'rows.jsonl'), checkpointSha256: base.checkpoint_sha256 });
    const train = examples.filter(example => example.partition === 'train');
    assert.equal(summary.roots, train.length);
    assert.equal(summary.studentCorrect, 0);
    assert.equal(summary.admitted, train.length, 'every wrong training root gets one verified repair');
    assert.ok(summary.rejected >= train.length, 'later boundaries (after the mistake) cannot repair it and are rejected');
    // Only training roots were ever run, by the student or the teacher.
    const held = examples.filter(example => example.partition !== 'train').map(example => String(example.input.value)).concat('held');
    for (const opening of [...student.openings, ...teacher.openings]) for (const value of held) assert.ok(!opening.includes(`"${value}"`), `${value} reached collection`);
    const records = readFileSync(join(out, 'records.jsonl'), 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line));
    const accepted = records.filter(record => record.status === 'admitted');
    for (const record of accepted) {
      assert.equal(record.kind, 'repair');
      assert.deepEqual(record.turns.map(turn => turn.role), ['student-prefix', 'correction', 'correction', 'correction'],
        'the whole trajectory is kept: the replayed outer turn, then the teacher from the inner call on');
    }
    assert.ok(records.some(record => record.status === 'rejected' && /root oracle/.test(record.reason)));
    const rows = readRepairRows(join(out, 'rows.jsonl'));
    assert.equal(rows.length, 3 * train.length, 'teacher turns only; the student prefix is never a row');
    for (const row of rows) {
      assert.equal(row.partition, 'train');
      assert.match(row.call, /^repair:r1:b1:/);
      assert.equal(row.correction.kind, 'repair');
      assert.equal(row.correction.boundary.index, 1);
      assert.equal(row.correction.student.release, release.digest);
      assert.equal(row.correction.student.checkpointSha256, base.checkpoint_sha256);
      assert.ok(row.correction.student.stack, 'the boundary call\'s requested/effective stack is recorded');
      assert.equal(row.correction.teacher, 'teacher:test');
      assert.deepEqual(row.correction.verification, { studentRootOracle: false, rootOracle: true, prefix: 'observed-identical', replayedTurns: 1 });
      assert.equal(row.evidence.rootOracle, true);
    }
    // The boundary turn's prompt holds what the student saw; its reply is the teacher's fix.
    const fix = rows.find(row => row.reply.calls?.[0]?.[1]?.code === 'return value.toUpperCase()');
    assert.ok(fix);
    // A rerun resumes: recorded roots are not run again.
    const before = student.openings.length;
    await runRepairs({ root, program, scaffold: release, release, weights: new WeightBinding(release, { program, strict: true }),
      student: student.driver, teacher: teacher.driver, teacherId: 'teacher:test', examples, round: 1,
      records: join(out, 'records.jsonl'), rows: join(out, 'rows.jsonl'), checkpointSha256: base.checkpoint_sha256 });
    assert.equal(student.openings.length, before);
    // A teacher that repeats the mistake never gets admitted.
    const wrong = await runRepairs({ root, program, scaffold: release, release, weights: new WeightBinding(release, { program, strict: true }),
      student: student.driver, teacher: scriptedModel(studentRespond).driver, teacherId: 'teacher:bad', examples, round: 2,
      records: join(out, 'records2.jsonl'), rows: join(out, 'rows2.jsonl'), checkpointSha256: base.checkpoint_sha256 });
    assert.equal(wrong.admitted, 0);
    assert.equal(readRepairRows(join(out, 'rows2.jsonl')).length, 0);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('repair rounds train on corrections with the teacher rows and are selected on development roots', async () => {
  const root = project();
  try {
    const teacher = scriptedModel(teacherRespond);
    // The served student is wrong on the inner call until a release trained with corrections serves.
    const serve = async ({ manifest }) => {
      const corrected = manifest.receipts?.training?.corrections > 0;
      const model = scriptedModel(corrected ? teacherRespond : studentRespond);
      return { driver: model.driver, loaded: new Set(manifest.artifacts.map(item => item.artifact)), close: async () => {} };
    };
    const options = { root, program: programOf(root), settings: { student: 'lfm', students: {}, teacher: 'teacher', backend: 'managed-local' },
      student: { name: 'lfm', base, checkpoint: '/unused', trainer: [process.execPath, join(root, 'trainer.mjs')], moduleMap: 'lfm2', baseGgufSha256: 'f'.repeat(64) },
      teacher: teacher.driver, teacherId: 'teacher:test', serve };
    const report = await tuneProgram(options, roundsSelector(options, { schedule: ['sft', 'repair'], patience: 1 }));
    assert.equal(report.activated, true, report.reason);
    const dir = join(root, '.natlang/tune/rounds', readdirSync(join(root, '.natlang/tune/rounds'))[0]);
    const journal = JSON.parse(readFileSync(join(dir, 'journal.json'), 'utf8'));
    assert.equal(journal.rounds[1].kind, 'repair');
    assert.ok(journal.rounds[1].repairs.admitted > 0);
    assert.equal(journal.rounds[1].parent.round, 0, 'the repair round continues the SFT build');
    assert.equal(journal.best.round, 1, 'the corrected release wins on development roots');
    const plans = readdirSync(join(root, '.natlang/tune/jobs')).map(job => JSON.parse(readFileSync(join(root, '.natlang/tune/jobs', job, 'plan.json'), 'utf8')));
    const repairPlan = plans.find(plan => plan.label === 'round-1-repair');
    const rows = readFileSync(repairPlan.rows, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line));
    assert.ok(rows.some(row => row.correction) && rows.some(row => !row.correction), 'corrections mix with the earlier teacher rows');
    assert.ok(repairPlan.parent);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
