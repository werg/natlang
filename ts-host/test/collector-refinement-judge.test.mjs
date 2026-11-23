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
