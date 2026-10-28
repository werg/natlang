// Teacher-free collection: the student teaches itself with privileged training references (tune/selfteach.ts).
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync, readFileSync, readdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as host from '../dist/index.js';
import { prepareData } from '../dist/tune/pipeline.js';
import { readRootExamples, splitExamples } from '../dist/tune/collect.js';
import { PRIVILEGED_MARKER, PRIVILEGED_SCHEMA, leaksPrivileged, privilegedText, selfTeacherId, withPrivileged } from '../dist/tune/selfteach.js';
import { scriptedModel } from './support/natlang.mjs';

const base = { architecture: 'lfm3', checkpoint_sha256: 'c'.repeat(64), tokenizer_fingerprint: 'd'.repeat(64), embedding_weight_sha256: 'e'.repeat(64) };
const named = body => `---\nargs: { value: string }\nreturns: string\n---\n${body}\n`;
const values = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa', 'lambda', 'mu'];

function project() {
  const root = mkdtempSync(join(tmpdir(), 'natlang-selfteach-'));
  mkdirSync(join(root, 'outer'));
  writeFileSync(join(root, 'outer.nl'), named('Pass value to inner.'));
  writeFileSync(join(root, 'outer', 'inner.nl'), named('Return value in uppercase.'));
  writeFileSync(join(root, 'package.json'), '{"type":"module"}');
  const examples = values.map(value => ({ function: 'outer.nl', input: { value }, expected: value.toUpperCase(), group: `w:${value}` }));
  writeFileSync(join(root, 'tune.examples.jsonl'), examples.map(example => JSON.stringify(example)).join('\n') + '\n');
  return root;
}
const program = root => {
  const built = host.buildProject({ project: root, emit: false, write: false });
  assert.equal(built.ok, true, host.formatDiagnostics(built.diagnostics));
  return built.manifest.adaptation;
};
const settings = { student: 'lfm3', students: {}, teacher: 'self', backend: 'managed-local' };
const student = { name: 'lfm3', base, checkpoint: '/unused', trainer: ['/unused'], moduleMap: 'lfm3', baseGgufSha256: 'f'.repeat(64) };
const split = root => splitExamples(readRootExamples(root));
function writeReferences(root, examples) {
  writeFileSync(join(root, 'tune.privileged.jsonl'), examples.map(example => JSON.stringify({ schema: PRIVILEGED_SCHEMA,
    function: example.function, group: example.group, input: example.input, result: example.expected,
    calls: [{ function: 'inner', value: example.expected }] })).join('\n') + '\n');
}
/** A student that solves the task only when its turn carries a privileged reference; `leak` roots echo the marker. */
function student_(leak = new Set()) {
  const seen = [];
  const model = scriptedModel(opening => {
    const privileged = opening.includes(PRIVILEGED_MARKER), value = /"value":\s*"(\w+)"|value[^a-z]*([a-z]+)/.exec(opening);
    seen.push({ privileged, opening });
    if (opening.includes('Pass value')) return 'return await inner(value)';
    const echo = [...leak].some(item => opening.includes(item)) ? ` // ${PRIVILEGED_MARKER}` : '';
    return privileged ? `return value.toUpperCase()${echo}` : 'return value';
  });
  return { driver: model.driver, seen };
}

test('privileged self-teaching collects only training roots and records the task without the reference', async () => {
  const root = project();
  try {
    const examples = split(root);
    const train = examples.filter(example => example.partition === 'train');
    assert.ok(train.length >= 4 && examples.some(example => example.partition === 'development'));
    writeReferences(root, train);
    const leaked = train[0].input.value;
    const self = student_(new Set([`"${leaked}"`]));
    const teacherId = selfTeacherId(student, 'digest');
    const data = await prepareData({ root, program: program(root), settings, student, teacher: self.driver, teacherId, serve: async () => { throw new Error('unused'); } });
    // Every model turn of a training root saw the reference; no development root was run at all.
    assert.ok(self.seen.length > 0 && self.seen.every(item => item.privileged), 'self-teaching turns carry the privileged reference');
    for (const example of examples.filter(item => item.partition !== 'train'))
      assert.ok(!self.seen.some(item => item.opening.includes(`"${example.input.value}"`)), `${example.partition} root ${example.input.value} was never run`);
    // Recorded rows hold the plain task: training removes the reference, so the student learns to act without it.
    // (The leaking root's own later turns hold its echoing reply in their assistant history, never the reference.)
    assert.ok(data.turns.length > 0 && data.turns.every(turn => turn.root === train[0].id ||
      !JSON.stringify(turn.messages).includes(PRIVILEGED_MARKER)));
    assert.ok(data.turns.every(turn => turn.messages.every(message => message.role === 'assistant' ||
      !JSON.stringify(message).includes(PRIVILEGED_MARKER))));
    assert.ok(data.turns.every(turn => turn.teacher === teacherId && turn.partition === 'train'));
    // Oracle-verified roots are targets; the root whose reply echoed the reference is not.
    const rows = new Set(data.trainRows.map(turn => turn.root));
    assert.equal(rows.size, train.length - 1);
    assert.ok(!rows.has(train[0].id) && data.turns.some(turn => turn.root === train[0].id && leaksPrivileged(turn)));
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('a privileged reference for a development root is a custody error', async () => {
  const root = project();
  try {
    const development = split(root).filter(example => example.partition === 'development');
    writeReferences(root, development.slice(0, 1));
    await assert.rejects(prepareData({ root, program: program(root), settings, student, teacher: student_().driver,
      teacherId: selfTeacherId(student, 'digest'), serve: async () => { throw new Error('unused'); } }), /privileged context comes only from training worlds/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('existing verified teacher data is reused before the self teacher runs a root', async () => {
  const root = project();
  try {
    const teacher = scriptedModel(opening => opening.includes('Pass value') ? 'return await inner(value)' : 'return value.toUpperCase()');
    const first = await prepareData({ root, program: program(root), settings: { ...settings, teacher: 'teacher' }, student, teacher: teacher.driver,
      teacherId: 'teacher:test', serve: async () => { throw new Error('unused'); } });
    assert.ok(first.trainRows.length > 0);
    writeReferences(root, split(root).filter(example => example.partition === 'train'));
    const data = await prepareData({ root, program: program(root), settings, student, teacher: () => { throw new Error('the self teacher must not run reused roots'); },
      teacherId: selfTeacherId(student, 'digest'), serve: async () => { throw new Error('unused'); } });
    assert.deepEqual(data.trainRows.map(turn => turn.teacher), first.trainRows.map(turn => turn.teacher));
    assert.ok(readdirSync(join(root, '.natlang/tune/data/roots')).every(file =>
      JSON.parse(readFileSync(join(root, '.natlang/tune/data/roots', file), 'utf8')).teacher === 'teacher:test'));
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('the privileged block is appended to the first user message without mutating the request', () => {
  const request = { messages: [{ role: 'system', content: 's' }, { role: 'user', content: 'task' }, { role: 'user', content: 'later' }], tools: [], seed: null, max_tokens: null };
  const text = privilegedText({ schema: PRIVILEGED_SCHEMA, function: 'main.nl', input: {}, result: 'grant', calls: [{ function: 'riskLevel', value: 'low' }] });
  assert.equal(text, `${PRIVILEGED_MARKER}: riskLevel returns "low"; the task's final result is "grant". Follow the instructions step by step so that each step reaches its value.`);
  const changed = withPrivileged(request, text);
  assert.equal(changed.messages[1].content, `task\n\n${text}`);
  assert.equal(changed.messages[2].content, 'later');
  assert.equal(request.messages[1].content, 'task');
  const parts = withPrivileged({ ...request, messages: [{ role: 'user', content: [{ type: 'text', text: 'x' }] }] }, text);
  assert.deepEqual(parts.messages[0].content.at(-1), { type: 'text', text: `\n\n${text}` });
});
