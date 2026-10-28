// Teacher-free collection: the student teaches itself with privileged training references (tune/selfteach.ts).
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync, readFileSync, readdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as host from '../dist/index.js';
import { prepareData } from '../dist/tune/pipeline.js';
import { readRootExamples, splitExamples } from '../dist/tune/collect.js';
import { PRIVILEGED_HEADER, PRIVILEGED_SCHEMA, leaksPrivileged, privilegedDriver, privilegedText, selectFacts, selfTeacherId, withPrivileged } from '../dist/tune/selfteach.js';
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

test('privileged self-teaching supplies only invocation-visible facts to scored decisions', async () => {
  const seen = [];
  const student = Object.assign(async () => ({}), {decide: async request => { seen.push(request); return {log_probs:[-1,-2]}; }});
  const reference = {facts:[{id:'root',kind:'expected_result',text:'root answer'},
    {id:'child',kind:'child_value',text:'child answer',match:['child question']}]};
  const driver = privilegedDriver(student, reference);
  const request = {messages:[{role:'system',content:'s'},{role:'user',content:'child question'}],
    options:['"yes"','"no"'],invocation_id:'child',invocation_role:'child'};
  await driver.decide(request);
  assert.match(JSON.stringify(seen[0].messages), /child answer/);
  assert.doesNotMatch(JSON.stringify(seen[0].messages), /root answer/);
  assert.equal(request.messages.length, 2);
  await driver.decide({...request,invocation_id:'root',invocation_role:'root'});
  assert.match(JSON.stringify(seen[1].messages), /root answer/);
  assert.doesNotMatch(JSON.stringify(seen[1].messages), /child answer/);
});

test('verified scored decisions become training turns without privileged prompts', async () => {
  const root = project();
  try {
    writeFileSync(join(root, 'outer.nl'), '---\nargs: { value: string }\nreturns: "yes" | "no"\nreadout: decision\n---\nChoose yes.\n');
    writeFileSync(join(root, 'tune.examples.jsonl'), values.map(value => JSON.stringify({function:'outer.nl',
      input:{value},expected:'yes',group:`w:${value}`})).join('\n')+'\n');
    const examples = split(root);
    writeReferences(root, examples.filter(example => example.partition === 'train'));
    const seen = [];
    const teacher = Object.assign(async () => { throw new Error('tool loop was not expected'); },
      {decide:async request => { seen.push(request); return {log_probs:[-1,-2]}; }});
    const teacherId = selfTeacherId(student, 'decision-digest');
    const data = await prepareData({root,program:program(root),settings,student,teacher,teacherId,
      serve:async () => { throw new Error('unused'); }});
    assert.ok(seen.length && seen.every(request => JSON.stringify(request.messages).includes(PRIVILEGED_HEADER)));
    assert.ok(seen.every(request => request.invocation_role === 'root'), 'a callable task root has no parent call');
    assert.ok(data.trainRows.length && data.trainRows.every(turn => turn.decision?.chosen === 0));
    assert.ok(data.trainRows.every(turn => !JSON.stringify(turn.messages).includes(PRIVILEGED_HEADER)));
  } finally { rmSync(root, {recursive:true,force:true}); }
});
function writeReferences(root, examples) {
  // The shape `natlang_neuralese.tune.supply privileged` writes from the shared `case_facts` (rl/privileged_context.py).
  writeFileSync(join(root, 'tune.privileged.jsonl'), examples.map(example => JSON.stringify({ schema: PRIVILEGED_SCHEMA,
    function: example.function, group: example.group, input: example.input, result: example.expected, case_id: `case:${example.group}`,
    custody: { schema: 'natlang.jolt-privileged-custody/1', split: 'train', protected_sha256: 'p'.repeat(64) },
    facts: [{ id: 'expected_result:0', kind: 'expected_result', text: `The verified result of this program is ${JSON.stringify(example.expected)}.` },
      { id: 'child_value:0', kind: 'child_value', text: `The verified answer of this call is ${JSON.stringify(example.expected)}.`,
        match: ['Return value in uppercase.'] }] })).join('\n') + '\n');
}
/** A student that solves the task only when its turn carries a privileged block; `leak` roots echo the header. */
function student_(leak = new Set()) {
  const seen = [];
  const model = scriptedModel(opening => {
    if (opening.includes('Pass value')) return 'return await inner(value)';
    const echo = [...leak].some(item => opening.includes(item)) ? ` // ${PRIVILEGED_HEADER}` : '';
    return seen.at(-1).privileged ? `return value.toUpperCase()${echo}` : 'return value';
  });
  const driver = async (request, ...rest) => {
    const block = request.messages.find(message => message.role === 'system' && String(message.content).startsWith(PRIVILEGED_HEADER));
    seen.push({ privileged: !!block, block: block?.content, opening: request.messages.map(message => String(message.content)).join('\n') });
    return model.driver({ ...request, messages: request.messages.filter(message => message !== block) }, ...rest);
  };
  return { driver, seen };
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
    // The root sees its world-level result; a child sees only its own matched verified value.
    assert.ok(self.seen.some(item => item.block.includes('The verified result of this program')));
    assert.ok(self.seen.every(item => !(item.block.includes('verified result') && item.block.includes('answer of this call'))));
    for (const example of examples.filter(item => item.partition !== 'train'))
      assert.ok(!self.seen.some(item => item.opening.includes(`"${example.input.value}"`)), `${example.partition} root ${example.input.value} was never run`);
    // Recorded rows hold the plain task: training removes the reference, so the student learns to act without it.
    // (The leaking root's own later turns hold its echoing reply in their assistant history, never the reference.)
    assert.ok(data.turns.length > 0 && data.turns.every(turn => turn.root === train[0].id ||
      !JSON.stringify(turn.messages).includes(PRIVILEGED_HEADER)));
    assert.ok(data.turns.every(turn => turn.messages.every(message => message.role === 'assistant' ||
      !JSON.stringify(message).includes(PRIVILEGED_HEADER))));
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

test('the shared crisp policy selects facts per invocation and the block is a system turn after the leading one', () => {
  const facts = [{ id: 'child_value:0', kind: 'child_value', text: 'The verified answer of this call is "low".', match: ['Rate the risk'] },
    { id: 'decisive:0', kind: 'decisive', text: 'Decisive evidence: total(order) returns 9500' },
    { id: 'expected_result:0', kind: 'expected_result', text: 'The verified result of this program is "grant".' }];
  assert.deepEqual(selectFacts(facts, { role: 'root', text: 'Decide an access request.' }).map(fact => fact.id), ['expected_result:0', 'decisive:0']);
  assert.deepEqual(selectFacts(facts, { role: 'child', text: 'Rate the risk of the request.' }).map(fact => fact.id), ['child_value:0']);
  assert.deepEqual(selectFacts(facts, { role: 'child', text: 'Apply in order' }), []);
  // Two different verified values match one child: no answer is given (ambiguous).
  assert.deepEqual(selectFacts([...facts, { ...facts[0], id: 'child_value:1', text: 'The verified answer of this call is "high".' }],
    { role: 'child', text: 'Rate the risk' }), []);
  const text = privilegedText(selectFacts(facts, { role: 'root', text: '' }));
  assert.equal(text, `${PRIVILEGED_HEADER}\n\nThe verified result of this program is "grant".\n\nDecisive evidence: total(order) returns 9500`);
  assert.equal(privilegedText([]), null);
  const request = { messages: [{ role: 'system', content: 's' }, { role: 'user', content: 'task' }], tools: [], seed: null, max_tokens: null };
  const changed = withPrivileged(request, text);
  assert.deepEqual(changed.messages.map(message => message.role), ['system', 'system', 'user']);
  assert.equal(changed.messages[1].content, text);
  assert.equal(request.messages.length, 2);
  assert.deepEqual(withPrivileged({ ...request, messages: [{ role: 'user', content: 'x' }] }, text).messages[0], { role: 'system', content: text });
});
