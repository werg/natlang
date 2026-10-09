import test from 'node:test';
import assert from 'node:assert/strict';
import { reserveExportDirectory, finalPairGate, negativeArtifactReason, traceFailureKind, pairedReplayMatches, parentSelectionProof, stateHistorySelectionProof, supportTaskDefinition, recordedRequestTurn, materializeVerifiedTrajectory, objectiveKinds } from '../scripts/skills/export-training.mjs';
import { materializeNativeRows } from '../dist/teacher/native-materializer.js';
import { stageFields, stageOf } from '../scripts/self-improvement/improver-stages.mjs';

test('paired quality is a gate only; failures and regressions cannot export SFT', () => {
  const artifact = { disposition: 'evaluated', positive: true,
    query: { selected: { gatesPassed: true }, effect: 0.2, cases: [{ expected: 'sealed query answer' }] },
    transfer: { selected: { gatesPassed: true }, effect: 0, cases: [{ expected: 'sealed transfer answer' }] } };
  assert.equal(finalPairGate(artifact), null);
  assert.equal(finalPairGate({ ...artifact, query: { selected: { gatesPassed: true }, effect: 0 } }), 'final_query_gate_not_positive');
  assert.equal(finalPairGate({ ...artifact, transfer: { selected: { gatesPassed: true }, effect: -0.01 } }), 'final_transfer_gate_regressed');
  assert.equal(finalPairGate({ ...artifact, transfer: { selected: { gatesPassed: false }, effect: 0 } }), 'final_transfer_gate_regressed');
  assert.equal(finalPairGate({ ...artifact, positive: false }), 'episode_not_positive');
  for (const failureKind of ['fixture', 'timeout']) {
    assert.equal(finalPairGate({...artifact, query:{...artifact.query,
      baseline:{outcomes:[{failureKind,quality:0,error:'unscored diagnostic'}]}}}), 'final_query_contains_unscored_failure');
  }
  assert.equal(finalPairGate({...artifact, query:{...artifact.query,
    baseline:{outcomes:[{failureKind:'target',quality:0,error:'target produced an invalid value'}]}}}), null);
});

test('negative sidecars classify interrupted incomplete searches without copying provider content', () => {
  assert.equal(negativeArtifactReason({ disposition: 'incomplete', search: { disposition: 'interrupted' } }), 'incomplete_search_interrupted');
  assert.equal(negativeArtifactReason({ disposition: 'incomplete', search: { disposition: 'incomplete' } }), 'incomplete_search');
  assert.equal(traceFailureKind({ outcome: 'failed', detail: 'SIGTERM' }), 'interrupted_by_sigterm');
  assert.equal(traceFailureKind({ outcome: 'failed', detail: 'request deadline exceeded' }), 'timeout_or_deadline');
  assert.equal(traceFailureKind({ outcome: 'failed', detail: '0 steps' }), 'trace_failed');
});

test('parent selection proof accepts only explicit trace redaction plus matching replayed host state', () => {
  const files = { 'skills/a/SKILL.md': 'selected procedure' };
  const state = { incumbent: 'selected', history: [{ source: 'selected', accepted: true, selected: true }],
    lastExperiment: { sourceFiles: [{ path: 'skills/a/SKILL.md', text: files['skills/a/SKILL.md'] }] } };
  const redacted = { callId: 'parent', events: [{ kind: 'state', phase: 'final', value: { $lambda: { return: {
    $diagnostic_preview: 'result omitted', complete: false, holder: 'result' } } } }] };
  assert.equal(parentSelectionProof(redacted, 'selected', files, state), true);
  assert.equal(parentSelectionProof(redacted, 'other', files, state), false);
  assert.equal(parentSelectionProof(redacted, 'selected', { ...files, 'skills/a/SKILL.md': 'forged' }, state), false);
  const contradictory = structuredClone(redacted);
  contradictory.events[0].value.$lambda.return = { incumbent: 'other', history: state.history, lastExperiment: state.lastExperiment };
  assert.equal(parentSelectionProof(contradictory, 'selected', files, state), false);
  assert.equal(parentSelectionProof({ callId: 'parent', events: [] }, 'selected', files, state), false);
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

test('objective validation follows the sealed runtime exports, with a narrow older-runtime fallback', () => {
  assert.deepEqual(objectiveKinds({ OBJECTIVE_KINDS: ['graph-coloring', 'tsp'] }), ['graph-coloring', 'tsp']);
  assert.deepEqual(objectiveKinds({}), ['knapsack', 'bin-packing', 'weighted-tardiness']);
});

test('training task copies only support cases and groups', () => {
  const def = { version: 'natlang.improvement-case/1', id: 'episode-1', seed: 0,
    files: { 'target.nl': 'baseline' }, authoredFiles: { 'improveStep/rewriteProgram.nl': 'instructions' },
    authoredDigest: 'pinned', sourceGroups: ['support-a', 'support-b'],
    cases: [{ id: 's1', group: 'support-a', split: 'train', expected: 1 },
      { id: 's2', group: 'support-b', split: 'validation', expected: 2 }],
    query: { expected: 'must never copy' }, transfer: { expected: 'must never copy' }, executorExchanges: ['must never copy'] };
  const task = supportTaskDefinition(def, { 'target.nl': 'selected' }, 'improveStep/rewriteProgram.nl');
  assert.equal(task.semantics.root, 'improveStep/rewriteProgram.nl');
  assert.deepEqual(task.source_groups, ['support-a', 'support-b']);
  assert.deepEqual(task.source_ids, ['s1', 's2']);
  assert.deepEqual(task.semantics.expected_files, { 'target.nl': 'selected' });
  assert.equal(JSON.stringify(task).includes('must never copy'), false);
  assert.equal(JSON.stringify(task).includes('query'), false);
  assert.equal(JSON.stringify(task).includes('transfer'), false);
});

test('the task names the editor that was recorded, for both generations, and refuses any other definition', () => {
  const def = { version: 'natlang.improvement-case/1', id: 'episode-1', seed: 0, files: { 'target.nl': 'baseline' }, authoredFiles: {},
    authoredDigest: 'pinned', sourceGroups: ['a', 'b'], cases: [{ id: 'a1', group: 'a', split: 'train' }, { id: 'b1', group: 'b', split: 'validation' }] };
  for (const root of ['improveStep/rewriteProgram.nl', 'improveStep/editSource.nl', 'improveStep/editSourceStructural.nl'])
    assert.equal(supportTaskDefinition(def, { 'target.nl': 'selected' }, root).semantics.root, root);
  assert.throws(() => supportTaskDefinition(def, {}, 'improveStep/diagnose.nl'), /unexpected editor definition.*improveStep\/editSource\.nl/);
  assert.throws(() => supportTaskDefinition(def, {}, undefined), /unexpected editor definition/);
});

test('with no step invocation the run state vouches for the selected edit', () => {
  const state = { incumbent: 'sel', history: [{ source: 'sel', accepted: true, selected: true }, { source: 'x', accepted: false, selected: false }] };
  assert.equal(stateHistorySelectionProof(state, 'sel', 'sel'), true);
  assert.equal(stateHistorySelectionProof(state, 'sel', 'other'), false, 'the edit must produce the selected source');
  assert.equal(stateHistorySelectionProof({ ...state, incumbent: 'x' }, 'sel', 'sel'), false);
  assert.equal(stateHistorySelectionProof({ ...state, history: [{ source: 'sel', accepted: true, selected: false }] }, 'sel', 'sel'), false);
  assert.equal(stateHistorySelectionProof(undefined, 'sel', 'sel'), false);
});

test('test split cases and mismatched group declarations are rejected', () => {
  const base = { version: 'natlang.improvement-case/1', id: 'episode-1', seed: 0,
    files: {}, authoredFiles: { 'rewrite.nl': 'source' }, authoredDigest: 'pin',
    sourceGroups: ['a', 'b'], cases: [{ id: 'a1', group: 'a', split: 'train' }, { id: 'b1', group: 'b', split: 'validation' }] };
  assert.throws(() => supportTaskDefinition({ ...base, cases: [...base.cases, { id: 'q', group: 'q', split: 'test' }], sourceGroups: ['a', 'b', 'q'] }, {}, 'improveStep/editSource.nl'), /non-support/);
  assert.throws(() => supportTaskDefinition({ ...base, sourceGroups: ['a', 'wrong'] }, {}, 'improveStep/editSource.nl'), /do not match/);
  assert.throws(() => supportTaskDefinition({ ...base, seed: 1 }, {}, 'improveStep/editSource.nl'), /seed/);
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

test('effective turn replay preserves normalized tool choice, truncation, and usage', () => {
  const exchange = { recording_version: 'natlang.effective-model-turn/1',
    request: { invocation_id: 'call/2', messages: [{ role: 'user', content: 'support only' }], tools: [{ type: 'function' }] },
    turn: { calls: [['read_file', { path: 'skills/a/SKILL.md' }]], text: '', reasoning: 'inspect the selected support file',
      raw_calls: [{ id: 'wire-1', function: { name: 'read_text', arguments: '{"path":"skills/a/SKILL.md"}' } }],
      prompt_tokens: 17, completion_tokens: 8, truncated: true }, wireExchanges: [] };
  const turn = recordedRequestTurn(exchange);
  assert.deepEqual(turn.assistant.calls[0], { tool: 'read_file', source_tool: 'read_file',
    arguments: { path: 'skills/a/SKILL.md' }, call_id: null });
  assert.equal(turn.assistant.truncated, true);
  assert.equal(turn.model_response.truncated, true);
  assert.equal(turn.model_response.prompt_tokens, 17);
  assert.equal(turn.model_response.completion_tokens, 8);
  assert.equal(turn.assistant.raw_calls[0].function.name, 'read_text');
  assert.throws(() => recordedRequestTurn({ ...exchange, recording_version: 'natlang.effective-model-turn/2' }), /unsupported recorded/);
  assert.throws(() => recordedRequestTurn({ ...exchange, wireExchanges: undefined }), /malformed effective/);
});

test('general native materializer dry run admits support-only turns and excludes sealed fixtures, for both editor generations', () => {
  for (const root of ['improveStep/rewriteProgram.nl', 'improveStep/editSource.nl', 'improveStep/editSourceStructural.nl']) {
    const definition = { version: 'natlang.improvement-case/1', id: 'episode-materializer', seed: 0,
      files: { 'target.nl': 'baseline' }, authoredFiles: { [root]: '---\nkind: directory-reducer\n---\ninstructions' },
      authoredDigest: 'pinned', sourceGroups: ['support-a', 'support-b'],
      cases: [{ id: 'support-1', group: 'support-a', split: 'train' }, { id: 'support-2', group: 'support-b', split: 'validation' }] };
    const task = supportTaskDefinition(definition, { 'target.nl': 'selected support procedure' }, root);
    const row = { version: 'natlang.teacher_trajectory.native/1', id: 'trajectory-1', task: { kind: 'whole_program', program_ir: {
        ...task, semantics: { ...task.semantics, inputs: { request: { sourceFiles: [{ path: 'target.nl', text: 'baseline' }] } }, expected: { summary: 'support procedure' } } } },
      provenance: { collection_role: 'teacher', support_only: true, parent_candidate_verified: true, ...stageFields(stageOf(root)) },
      outcome: { accepted: true, status: 'completed', oracle: 'exact' }, outcome_actions: [],
      trajectory: [{ invocation_id: 'call-1', context: [{ role: 'system', content: 'system' }, { role: 'user', content: 'Use support evidence only.' }],
        tools_offered: [], assistant: { content: 'The reusable procedure is ready.', reasoning: 'Summarize the support evidence.', calls: [], raw_calls: [] }, model_response: { raw_calls: [] } }] };
    const turns = materializeVerifiedTrajectory(row, { materializeNativeRows });
    assert.equal(turns.length, 1);
    assert.equal(turns[0].training_admission.approved, true);
    assert.deepEqual(turns[0].source_groups.filter(group => group.startsWith('support-')).sort(), ['support-a', 'support-b']);
    // The row names its stage and generation, and declares whole-trajectory supervision.
    assert.equal(turns[0].improver_stage.definition_source, root);
    assert.equal(turns[0].improver_stage.generation, root === 'improveStep/rewriteProgram.nl' ? 'rewriteProgram/1' : 'stages/1');
    assert.equal(turns[0].supervision.scope, 'whole-trajectory');
    assert.deepEqual(turns[0].supervision.masked, []);
    assert.equal(turns[0].supervision.classes.length, turns[0].messages.length);
    assert.equal(turns[0].supervision.classes[0], 'instructions-inputs');
    const exported = JSON.stringify(turns);
    for (const withheld of ['sealed query answer', 'sealed transfer answer', 'evaluation_ticket', 'executorExchanges'])
      assert.equal(exported.includes(withheld), false);
    assert.throws(() => materializeVerifiedTrajectory({ ...row, provenance: { ...row.provenance, improver_stage: undefined } }, { materializeNativeRows }), /does not name its improver stage/);
  }
});


test('export reserves one whole fresh output bundle before concurrent replay writers run', async()=>{
 const {mkdtemp,rm,writeFile,readFile}=await import('node:fs/promises');
 const {tmpdir}=await import('node:os');const {join}=await import('node:path');
 const parent=await mkdtemp(join(tmpdir(),'skill-export-exclusive-'));
 try {
  const output=join(parent,'nested','bundle');
  const attempts=await Promise.allSettled([reserveExportDirectory(output),reserveExportDirectory(output)]);
  assert.equal(attempts.filter(item=>item.status==='fulfilled').length,1);
  assert.match(String(attempts.find(item=>item.status==='rejected').reason),/fresh directory/);
  await writeFile(join(output,'evidence.txt'),'original');
  await assert.rejects(()=>reserveExportDirectory(output),/already exists/);
  assert.equal(await readFile(join(output,'evidence.txt'),'utf8'),'original');
 } finally {await rm(parent,{recursive:true,force:true});}
});
