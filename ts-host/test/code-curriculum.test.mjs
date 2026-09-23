import test from 'node:test';
import assert from 'node:assert/strict';
import { codeProposalTurns, compileCurriculum, syntheticCodeTasks } from '../scripts/code-corpus/curriculum.mjs';
import { project, replayIsolated, materializeCorpus } from '../scripts/code-corpus/replay.mjs';
import { observeSourceCase, sourceCases } from '../scripts/code-corpus/source-cases.mjs';

test('curriculum keeps verified captures, proposals, and external calls distinct', () => {
  const rows = compileCurriculum([
    { id:'captured',group_id:'f',language:'javascript',instruction:'double',kind:'function',source:{name:'x'},
      function:{body:'{ return x * 2; }'},cases:[{args:[2],expected:4,outcome:'return',portable:true}],verification:{status:'captured'} },
    { id:'candidate',group_id:'c',language:'javascript',instruction:'triple',kind:'instruction',source:{name:'x'},
      function:{source:'function triple(x){ return x * 3; }'},cases:[],verification:{status:'unverified'} },
    { id:'api',group_id:'a',language:'typescript',instruction:'look up',kind:'tool_calls',source:{name:'x'},raw:{tools:[{name:'lookup'}],calls:[{name:'lookup'}]},verification:{status:'unverified'} },
  ]);
  assert.deepEqual(rows.map(row => row.kind), ['captured_function_replay','instruction_code_proposal','external_api_stub']);
  assert.equal(rows[1].completion_status, 'unverified');
  assert.equal(rows[2].expected, null);
  assert.match(rows[2].note, /no tool execution/);
});

test('synthetic tasks are deterministic, source grouped, and replay to accepted native turns', async () => {
  const tasks = syntheticCodeTasks(91, 0, 6);
  assert.deepEqual(tasks, syntheticCodeTasks(91, 0, 6));
  assert.equal(tasks[0].group_id, syntheticCodeTasks(927, 0, 1)[0].group_id);
  const ir = project(tasks[0], 0).program;
  assert.equal(ir.version, 'natlang.program/1');
  assert.deepEqual(ir.source_groups, [tasks[0].group_id]);
  const trajectory = await replayIsolated(tasks[0], 0);
  assert.equal(trajectory.outcome.accepted, true, JSON.stringify(trajectory.outcome));
  const materialized = await materializeCorpus([trajectory]);
  assert.ok(materialized.turns.length >= 1);
  assert.ok(materialized.turns.every(turn => turn.source_groups.includes(tasks[0].group_id)));
});

test('general code proposal turns use no tools and exclude rejected records', async () => {
  const source = { name:'fixture', license:'MIT' };
  const turns = await codeProposalTurns([
    { id:'ok',group_id:'function-family',language:'typescript',instruction:'Implement twice.',kind:'instruction',source,
      function:{source:'function twice(x: number): number { return x * 2; }'},verification:{status:'unverified'} },
    { id:'rejected',group_id:'bad',language:'typescript',instruction:'Do not train.',kind:'instruction',source,
      function:{source:'broken {'},verification:{status:'rejected'} },
  ]);
  assert.equal(turns.length, 1);
  assert.deepEqual(Object.keys(turns[0].target).sort(), ['content','role']);
  assert.deepEqual(turns[0].training_admission, { approved:true, reason:'unverified-direct-code-proposal' });
  assert.equal(turns[0].execution_verified, false);
  assert.deepEqual(turns[0].source_groups, ['function-family']);
  assert.deepEqual(turns[0].tools, []);
  assert.equal(turns[0].family, 'general_code_proposal');
  assert.equal((await codeProposalTurns([{ id:'bad',group_id:'bad',language:'javascript',instruction:'bad',function:{source:'function f( {'} }])).length, 0);
});

test('source observations require explicit portable types and return actual source outputs', async () => {
  const source = { id:'source:add',group_id:'source:add',kind:'function',language:'typescript',instruction:'Add two numbers.',
    function:{name:'add',parameters:[{name:'a',type:'number'},{name:'b',type:'number'}],body:'{ return a + b; }',helpers:[],imports:[]},
    cases:[],verification:{status:'inventory',reasons:[]} };
  const candidates = sourceCases(source);
  assert.equal(candidates.eligible, true);
  const observedCase = await observeSourceCase(source, candidates.cases[0].args);
  assert.equal(observedCase.expected, 6);
  assert.deepEqual(observedCase.input_after,candidates.cases[0].args);
  const mutating={...source,function:{...source.function,parameters:[{name:'items',type:'number[]'}],body:'{ items[0] = 0; return items[0]; }'}};
  await assert.rejects(observeSourceCase(mutating,[[3]]),/mutated invocation inputs/);
  const observed = { ...source, observation:{ kind:'source_derived_observations', not_upstream_tests:true },
    cases:[{args:[3,3],expected:6,outcome:'return',portable:true}] };
  assert.equal(compileCurriculum([observed])[0].kind, 'source_observed_replay');
  assert.match(compileCurriculum([observed])[0].evidence, /not_upstream_tests/);
  assert.equal(sourceCases({ ...source, function:{ ...source.function, parameters:[{name:'x',type:'any'}] } }).reason, 'parameter_type_not_portable');
  assert.equal(sourceCases({ ...source, function:{ ...source.function, body:'{ return process.version; }' } }).reason, 'unsafe_global_or_dynamic_code');
});

test('second synthetic boundary cases are consistent with the generated implementation', () => {
  const rows = syntheticCodeTasks(7, 0, 30);
  assert.equal(rows.length, 30);
  assert.ok(rows.every(row => row.cases.length === 2 && row.cases.every(c => c.portable)));
  assert.ok(new Set(rows.map(row => row.generation.difficulty)).size >= 3);
});
