import test from 'node:test';import assert from 'node:assert/strict';
import {executeProgram} from '../dist/teacher/collector.js';
import {TOOLS_PROMPT} from '../dist/native/prompt.js';

test('direct Program IR collection discovers and discloses companion skills',async()=>{
 const record={version:'natlang.program/2',id:'skill-collection-test',kind:'lambda_source',split:'train',semantics:{root:'solve.nl',files:{'solve.nl':'---\nargs: {}\nreturns: number\n---\nCompute two plus two.\n','solve/skills/exact-computation/SKILL.md':'---\nname: exact-computation\ndescription: Use eval to compute exact results.\n---\nCalculate from the bound inputs and return the computed value.\n'},inputs:{},expected:4}};
 let i=0;const requests=[];
 const run=await executeProgram(record,async r=>{requests.push(structuredClone(r));return [
  {calls:[['read_code',{name:'skills.exact-computation'}]]},
  {calls:[['eval',{code:'2 + 2',finish:true}]]},
 ][i++]??{calls:[['return_result',{status:'success',value:4}]]};},{systemPrompt:TOOLS_PROMPT,contextTokens:8192,maxTurns:4,rootSeed:42,runId:'skill-collection-test'});
 assert.equal(run.outcome.accepted,true);
 assert.match(JSON.stringify(requests[0].messages),/exact-computation/);
 assert.match(requests[1].messages.at(-1).content,/Calculate from the bound inputs/);
});

test('delegated pre-existing functions discover their own skills during direct collection',async()=>{
 const record={version:'natlang.program/2',id:'child-skill-collection-test',kind:'lambda_source',split:'train',semantics:{root:'solve.nl',files:{
  'solve.nl':'---\nargs: {}\nreturns: number\n---\nCall compute and return its result.\n',
  'solve/compute.nl':'---\nargs: {}\nreturns: number\n---\nCompute two plus two.\n',
  'solve/compute/skills/exact-computation/SKILL.md':'---\nname: exact-computation\ndescription: Compute exact values using eval.\n---\nCalculate using the bound inputs.\n',
 },inputs:{},expected:4}};
 let i=0;const requests=[];
 const run=await executeProgram(record,async r=>{requests.push(structuredClone(r));return [
  {calls:[['eval',{code:'await compute()',finish:true}]]},
  {calls:[['read_code',{name:'skills.exact-computation'}]]},
  {calls:[['eval',{code:'2 + 2',finish:true}]]},
 ][i++]??{calls:[['return_result',{status:'failed',reason:'Unexpected extra model turn.'}]]};},{systemPrompt:TOOLS_PROMPT,contextTokens:8192,maxTurns:6,rootSeed:42,runId:'child-skill-collection-test'});
 assert.equal(run.outcome.accepted,true,JSON.stringify(run.outcome));
 assert.doesNotMatch(JSON.stringify(requests[0].messages),/skills.exact-computation/,'child skills are scoped to the child');
 assert.match(JSON.stringify(requests[1].messages),/exact-computation/);
 assert.match(requests[2].messages.at(-1).content,/Calculate using the bound inputs/);
});
