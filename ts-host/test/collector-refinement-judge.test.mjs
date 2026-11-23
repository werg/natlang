import assert from 'node:assert/strict';
import { test } from 'node:test';
import { executeProgram } from '../dist/teacher/collector.js';

// A helper with a refined return, called from the root's eval: the collector (agent-driven, no model config) must
// decide it on its own driver, by decision scoring or by a scripted `holds` child.
const record = { version: 'natlang.program/2', id: 'collector-refinement-judge', kind: 'lambda_source', split: 'test',
  source_ids: ['collector-refinement-judge'], source_groups: ['synthetic-only'], semantics: { root: 'summary.nl',
    files: {
      'summary.nl': '---\nargs:\n  text: string\nreturns: string\n---\nSummarize text with describe.\n',
      'summary/describe.nl': '---\nargs:\n  text: string\nreturns: "Is<string, \\"one sentence that says whose order the writer is talking about\\">"\n---\n' +
        'Say in one sentence whose order the writer of text is talking about.\n' },
    inputs: { text: 'My neighbour cancelled hers, but I am keeping mine.' },
    expected: 'The writer talks about their own order.', operation: 'exact' } };
const callName = request => /You are inside this call: ([\w@$.:-]+)\(/.exec(String(request.messages?.[1]?.content ?? ''))?.[1];
const root = { calls: [['eval', { code: 'return await describe(text);', finish: true }]] };
const answer = { calls: [['return_result', { status: 'success', value: 'The writer talks about their own order.' }]] };
const options = runId => ({ rootSeed: 0, runId, maxTurns: 6, contextTokens: 16384, systemPrompt: 'Use the tools.' });
const traceText = run => JSON.stringify(run.outcome);

for (const verdict of [true, false]) test(`a refined helper return is decided by the driver's decision scorer (${verdict})`, async () => {
  const scored = [], names = [];
  const driver = Object.assign(async request => {
    const name = callName(request); names.push(name);
    return name === 'summary' ? root : answer;
  }, { model: 'test-policy', decide: async request => {
    scored.push(request);
    return { log_probs: verdict ? [0, -6] : [-6, 0] };
  } });
  const run = await executeProgram(record, driver, options(`refine-scored-${verdict}`));
  assert.ok(scored.length >= 1, 'the judge scored on the policy driver');
  assert.deepEqual(scored[0].options, ['true', 'false']);
  assert.match(JSON.stringify(scored[0].messages), /whose order the writer is talking about/);
  assert.ok(!names.includes('holds'), 'a scoring driver needs no holds call');
  assert.doesNotMatch(traceText(run), /refinement-undecided/);
  assert.equal(run.outcome.accepted, verdict);
  if (!verdict) assert.match(traceText(run), /refinement-unsatisfied/);
});

for (const verdict of [true, false]) test(`a refined helper return is decided by a scripted holds child (${verdict})`, async () => {
  const holds = [];
  const driver = async request => {
    const name = callName(request);
    if (name === 'summary') return root;
    if (name === 'holds') {
      holds.push(request);
      // A finite (boolean) call without a scorer generates its choice as JSON text on a tool-free request.
      return Array.isArray(request.tools) && request.tools.length === 0 ? { text: JSON.stringify(verdict), calls: [] } :
        { calls: [['return_result', { status: 'success', value: verdict }]] };
    }
    return answer;
  };
  const run = await executeProgram(record, driver, options(`refine-scripted-${verdict}`));
  assert.ok(holds.length >= 1, 'the judge ran as a holds call through the scripted driver');
  assert.match(JSON.stringify(holds[0].messages), /Decide whether value satisfies predicate/);
  assert.doesNotMatch(traceText(run), /refinement-undecided/);
  assert.equal(run.outcome.accepted, verdict);
});

test('NATLANG_COLLECTOR_REFINEMENT_JUDGE=0 restores the judge-less collector', async () => {
  process.env.NATLANG_COLLECTOR_REFINEMENT_JUDGE = '0';
  try {
    const driver = async request => callName(request) === 'summary' ? root : answer;
    const run = await executeProgram(record, driver, options('refine-opt-out'));
    assert.equal(run.outcome.accepted, false);
    assert.match(traceText(run), /refinement-undecided/);
  } finally { delete process.env.NATLANG_COLLECTOR_REFINEMENT_JUDGE; }
});

// Every place a helper's result can carry a refinement (held-v4 B1 lost hv4-untr-008/012 to these shapes before the
// fix): a field of a record result, the elements of an array result, an inline `nl` lambda made in eval, and a
// callable-folder TypeScript helper over a refined `.nl` function. Each is judged (true admits, false rejects with
// refinement-unsatisfied); none ends undecided.
const VERBATIM = 'copied verbatim from the note';
const shapeRecord = (id, files, code) => ({ code, record: { version: 'natlang.program/2', id, kind: 'lambda_source', split: 'test',
  source_ids: [id], source_groups: ['synthetic-only'], semantics: { root: 'r.nl', files,
    inputs: { note: 'Order A-1 ships Monday.' }, expected: 'A-1', operation: 'exact' } } });
const rootNl = '---\nargs:\n  note: string\nreturns: string\n---\nGive the order id of note.\n';
const refinedShapes = {
  'a record field': shapeRecord('refine-field', { 'r.nl': rootNl,
    'r/pick.nl': `---\nargs:\n  note: string\nreturns: "{ id: Is<string, \\"${VERBATIM}\\">; n: number }"\n---\nPick the order id of note.\n` },
  'return (await pick(note)).id;'),
  'array elements': shapeRecord('refine-array', { 'r.nl': rootNl,
    'r/pick.nl': `---\nargs:\n  note: string\nreturns: "Is<string, \\"${VERBATIM}\\">[]"\n---\nList the order ids of note.\n` },
  'return (await pick(note))[0];'),
  'an inline nl lambda': shapeRecord('refine-inline', { 'r.nl': rootNl },
    `const pick = nl<(note: string) => Is<string, "${VERBATIM}">>\`Pick the order id of note.\`;\nreturn await pick(note);`),
  'a TypeScript helper over a refined .nl': shapeRecord('refine-ts', { 'r.nl': rootNl,
    'r/pick.ts': `import type { Is } from "@natlang/node";\nimport ask from "./ask.nl";\n` +
      `export default async function pick(note: string): Promise<Is<string, "${VERBATIM}">> { return await ask(note); }\n`,
    'r/ask.nl': `---\nargs:\n  note: string\nreturns: "Is<string, \\"${VERBATIM}\\">"\n---\nPick the order id of note.\n` },
  'return await pick(note);'),
};
const helperValue = id => id === 'refine-field' ? { id: 'A-1', n: 1 } : id === 'refine-array' ? ['A-1'] : 'A-1';
for (const [shape, { record: shaped, code }] of Object.entries(refinedShapes)) for (const scorer of [false, true]) for (const verdict of [true, false])
  test(`a refined helper result in ${shape} is judged by ${scorer ? 'the scorer' : 'a holds child'} (${verdict})`, async () => {
    let judged = 0;
    const reply = request => {
      const name = callName(request);
      if (name === 'r') return { calls: [['eval', { code, finish: true }]] };
      if (name === 'holds') {
        judged += 1;
        return Array.isArray(request.tools) && request.tools.length === 0 ? { text: JSON.stringify(verdict), calls: [] } :
          { calls: [['return_result', { status: 'success', value: verdict }]] };
      }
      return { calls: [['return_result', { status: 'success', value: helperValue(shaped.id) }]] };
    };
    const driver = scorer ? Object.assign(async request => reply(request), { model: 'test-policy',
      decide: async () => { judged += 1; return { log_probs: verdict ? [0, -6] : [-6, 0] }; } }) : async request => reply(request);
    const run = await executeProgram(shaped, driver, options(`${shaped.id}-${scorer}-${verdict}`));
    assert.ok(judged >= 1, 'a judge ran');
    assert.doesNotMatch(traceText(run), /refinement-undecided/);
    assert.equal(run.outcome.accepted, verdict);
    if (!verdict) assert.match(traceText(run), /refinement-unsatisfied/);
  });
