import test from 'node:test';
import assert from 'node:assert/strict';
import { finalPairGate, pairedReplayMatches, supportTaskDefinition, recordedRequestTurn, materializeVerifiedTrajectory } from '../scripts/skills/export-training.mjs';
import { materializeNativeRows } from '../dist/teacher/native-materializer.js';

test('paired quality is a gate only; failures and regressions cannot export SFT', () => {
  const artifact = { disposition: 'evaluated', positive: true,
    query: { selected: { gatesPassed: true }, effect: 0.2, cases: [{ expected: 'sealed query answer' }] },
    transfer: { selected: { gatesPassed: true }, effect: 0, cases: [{ expected: 'sealed transfer answer' }] } };
  assert.equal(finalPairGate(artifact), null);
  assert.equal(finalPairGate({ ...artifact, query: { selected: { gatesPassed: true }, effect: 0 } }), 'final_query_gate_not_positive');
  assert.equal(finalPairGate({ ...artifact, transfer: { selected: { gatesPassed: true }, effect: -0.01 } }), 'final_transfer_gate_regressed');
  assert.equal(finalPairGate({ ...artifact, transfer: { selected: { gatesPassed: false }, effect: 0 } }), 'final_transfer_gate_regressed');
  assert.equal(finalPairGate({ ...artifact, positive: false }), 'episode_not_positive');
});

test('offline pair replay rejects mutated support identity, query gain and transfer gate', () => {
  const saved = { disposition: 'evaluated', positive: true, identity: 'i', baseline: 'b', selected: 's', selectedFiles: { 'skills/a.md': 'good' },
    query: { selected: { gatesPassed: true }, effect: 1 }, transfer: { selected: { gatesPassed: true }, effect: 0 } };
  assert.equal(pairedReplayMatches(saved, structuredClone(saved)), true);
  const queryMutation = structuredClone(saved); queryMutation.query.effect = -1;
  assert.equal(pairedReplayMatches(saved, queryMutation), false);
  const transferMutation = structuredClone(saved); transferMutation.transfer.selected.gatesPassed = false;
  assert.equal(pairedReplayMatches(saved, transferMutation), false);
  const sourceMutation = structuredClone(saved); sourceMutation.selectedFiles['skills/a.md'] = 'different';
  assert.equal(pairedReplayMatches(saved, sourceMutation), false);
});

test('training task copies only support cases and groups', () => {
  const def = { version: 'natlang.improvement-case/1', id: 'episode-1', seed: 0,
    files: { 'target.nl': 'baseline' }, authoredFiles: { 'improveStep/rewriteProgram.nl': 'instructions' },
    authoredDigest: 'pinned', sourceGroups: ['support-a', 'support-b'],
    cases: [{ id: 's1', group: 'support-a', split: 'train', expected: 1 },
      { id: 's2', group: 'support-b', split: 'validation', expected: 2 }],
    query: { expected: 'must never copy' }, transfer: { expected: 'must never copy' }, executorExchanges: ['must never copy'] };
  const task = supportTaskDefinition(def, { 'target.nl': 'selected' });
  assert.deepEqual(task.source_groups, ['support-a', 'support-b']);
  assert.deepEqual(task.source_ids, ['s1', 's2']);
  assert.deepEqual(task.semantics.expected_files, { 'target.nl': 'selected' });
  assert.equal(JSON.stringify(task).includes('must never copy'), false);
  assert.equal(JSON.stringify(task).includes('query'), false);
  assert.equal(JSON.stringify(task).includes('transfer'), false);
});

test('test split cases and mismatched group declarations are rejected', () => {
  const base = { version: 'natlang.improvement-case/1', id: 'episode-1', seed: 0,
    files: {}, authoredFiles: { 'rewrite.nl': 'source' }, authoredDigest: 'pin',
    sourceGroups: ['a', 'b'], cases: [{ id: 'a1', group: 'a', split: 'train' }, { id: 'b1', group: 'b', split: 'validation' }] };
  assert.throws(() => supportTaskDefinition({ ...base, cases: [...base.cases, { id: 'q', group: 'q', split: 'test' }], sourceGroups: ['a', 'b', 'q'] }, {}), /non-support/);
  assert.throws(() => supportTaskDefinition({ ...base, sourceGroups: ['a', 'wrong'] }, {}), /do not match/);
  assert.throws(() => supportTaskDefinition({ ...base, seed: 1 }, {}), /seed/);
});

test('recorded author exchange keeps the exact prompt, tools and raw call target', () => {
  const exchange = { request: { invocation_id: 'call/1', messages: [{ role: 'user', content: 'support only' }], tools: [{ type: 'function' }] },
    wireResponse: { choices: [{ message: { role: 'assistant', content: null, reasoning_content: 'check support facts',
      tool_calls: [{ id: 'c1', function: { name: 'write_file', arguments: '{"path":"skills/a/SKILL.md","text":"procedure"}' } }] } }] } };
  const turn = recordedRequestTurn(exchange);
  assert.deepEqual(turn.context, exchange.request.messages);
  assert.deepEqual(turn.tools_offered, exchange.request.tools);
  assert.deepEqual(turn.assistant.calls[0], { tool: 'write_file', source_tool: 'write_file',
    arguments: { path: 'skills/a/SKILL.md', text: 'procedure' }, call_id: 'c1' });
  assert.throws(() => recordedRequestTurn({ ...exchange, wireResponse: { choices: [{ message: { tool_calls: [
    { function: { name: 'write_file', arguments: 'not json' } }] } }] } }), /invalid JSON/);
});

test('general native materializer dry run admits support-only turns and excludes sealed fixtures', () => {
  const definition = { version: 'natlang.improvement-case/1', id: 'episode-materializer', seed: 0,
    files: { 'target.nl': 'baseline' }, authoredFiles: { 'improveStep/rewriteProgram.nl': '---\nkind: directory-reducer\n---\ninstructions' },
    authoredDigest: 'pinned', sourceGroups: ['support-a', 'support-b'],
    cases: [{ id: 'support-1', group: 'support-a', split: 'train' }, { id: 'support-2', group: 'support-b', split: 'validation' }] };
  const task = supportTaskDefinition(definition, { 'target.nl': 'selected support procedure' });
  const row = { version: 'natlang.teacher_trajectory.native/1', id: 'trajectory-1', task: { kind: 'whole_program', program_ir: {
      ...task, semantics: { ...task.semantics, inputs: { request: { sourceFiles: [{ path: 'target.nl', text: 'baseline' }] } }, expected: { summary: 'support procedure' } } } },
    provenance: { collection_role: 'teacher', support_only: true, parent_candidate_verified: true },
    outcome: { accepted: true, status: 'completed', oracle: 'exact' }, outcome_actions: [],
    trajectory: [{ invocation_id: 'call-1', context: [{ role: 'system', content: 'system' }, { role: 'user', content: 'Use support evidence only.' }],
      tools_offered: [], assistant: { content: 'The reusable procedure is ready.', reasoning: 'Summarize the support evidence.', calls: [], raw_calls: [] }, model_response: { raw_calls: [] } }] };
  const turns = materializeVerifiedTrajectory(row, { materializeNativeRows });
  assert.equal(turns.length, 1);
  assert.equal(turns[0].training_admission.approved, true);
  assert.deepEqual(turns[0].source_groups.filter(group => group.startsWith('support-')).sort(), ['support-a', 'support-b']);
  const exported = JSON.stringify(turns);
  for (const withheld of ['sealed query answer', 'sealed transfer answer', 'evaluation_ticket', 'executorExchanges'])
    assert.equal(exported.includes(withheld), false);
});
