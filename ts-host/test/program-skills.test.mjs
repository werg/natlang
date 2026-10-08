import test from 'node:test';import assert from 'node:assert/strict';
import {resolve} from 'node:path';import {pathToFileURL} from 'node:url';
const dist=process.env.NATLANG_TEST_DIST?resolve(process.env.NATLANG_TEST_DIST):resolve('dist');
const {executeProgram}=await import(pathToFileURL(resolve(dist,'teacher/collector.js')));
const {TOOLS_PROMPT}=await import(pathToFileURL(resolve(dist,'native/prompt.js')));

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
  'solve/skills/exact-computation/SKILL.md':'---\nname: exact-computation\ndescription: Parent-only arithmetic guidance.\n---\nDo not use this instruction in a file-backed child; it is owned by the caller.\n',
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
 assert.match(JSON.stringify(requests[0].messages),/Parent-only arithmetic guidance/,'root sees its own companion skill');
 assert.match(JSON.stringify(requests[1].messages),/exact-computation/);
 assert.doesNotMatch(JSON.stringify(requests[1].messages),/Parent-only arithmetic guidance/,'file-backed child keeps its own skill scope');
 assert.match(requests[2].messages.at(-1).content,/Calculate using the bound inputs/);
});

test('repeated skill reads retain instructions and give an application reminder',async()=>{
 const record={version:'natlang.program/2',id:'repeat-skill-test',kind:'lambda_source',split:'train',semantics:{root:'solve.nl',files:{
  'solve.nl':'---\nargs: {}\nreturns: number\n---\nCompute two plus two.\n',
  'solve/skills/exact-computation/SKILL.md':'---\nname: exact-computation\ndescription: Compute exact values.\n---\nCalculate using the bound inputs.\n',
 },inputs:{},expected:4}};
 let i=0;const requests=[];
 const run=await executeProgram(record,async r=>{requests.push(structuredClone(r));return [
  {calls:[['read_code',{name:'skills.exact-computation'}]]},
  {calls:[['read_code',{name:'skills.exact-computation'}]]},
  {calls:[['eval',{code:'2 + 2',finish:true}]]},
 ][i++]??{calls:[['return_result',{status:'failed',reason:'Unexpected turn.'}]]};},{systemPrompt:TOOLS_PROMPT,contextTokens:8192,maxTurns:4,rootSeed:42,runId:'repeat-skill-test'});
 assert.equal(run.outcome.accepted,true);
 assert.doesNotMatch(requests[1].messages.at(-1).content,/unchanged since/);
 assert.match(requests[2].messages.at(-1).content,/unchanged since your earlier read/);
 assert.match(requests[2].messages.at(-1).content,/Calculate using the bound inputs/);
});
