import assert from 'node:assert/strict';
import { test } from 'node:test';
import { checkAuthoring } from '../dist/teacher/authoring.js';
import { createNatlangRuntime } from '../dist/runtime/node.js';
import { loadVirtualNatlang } from '../dist/runtime/virtual-project.js';

const files = {'app.ts':'export async function judge(note:string):Promise<boolean> { const policy="A source-owned check"; return await nl.with<boolean>({policy})`Judge note against policy.`(note); }'};
const spec = {module:'app.ts',export:'judge',requires:{nl:true},runs:[{args:['known'],expected:true}]};

test('a Decision-only source oracle is uncovered for a boolean child and terminates',async()=>{
  const result=await checkAuthoring(files,{...spec,oracle:[{match:['"known"'],value:'hold'}]});
  assert.equal(result.ok,false);assert.deepEqual(result.results,[null]);
  assert.match(result.problems[0],/authoring oracle has no answer/);
});

test('source answers route by exact native type without coercing string decisions',async()=>{
  const oracle=[{match:['"known"'],value:'hold'},{match:['"known"'],value:true}];
  assert.deepEqual(await checkAuthoring(files,{...spec,oracle}),{ok:true,problems:[],results:[true]});
  const decision={'app.ts':files['app.ts'].replaceAll('boolean','"hold"|"rollback"')};
  assert.deepEqual(await checkAuthoring(decision,{...spec,runs:[{args:['known'],expected:'hold'}],oracle}),{ok:true,problems:[],results:['hold']});
  const opposite={'app.ts':files['app.ts'].replace('return await','return !await')};
  assert.equal((await checkAuthoring(opposite,{...spec,oracle})).ok,false);
});

test('literal unions and complete record contracts exclude invalid fixed answers',async()=>{
  const source={'app.ts':'export async function judge(note:string):Promise<{ok:boolean,label:"keep"|"drop"}> { return await nl<{ok:boolean,label:"keep"|"drop"}>`Judge note.`(note); }'};
  const oracle=[{match:['"known"'],value:{ok:true}},{match:['"known"'],value:{ok:true,label:'wrong'}},{match:['"known"'],value:{ok:true,label:'keep'}}];
  assert.deepEqual(await checkAuthoring(source,{...spec,runs:[{args:['known'],expected:{ok:true,label:'keep'}}],oracle}),{ok:true,problems:[],results:[{ok:true,label:'keep'}]});
  assert.equal((await checkAuthoring(source,{...spec,runs:[{args:['known'],expected:{ok:true,label:'keep'}}],oracle:oracle.slice(0,2)})).ok,false);
});

test('parallel different child contracts retain their own session checks',async()=>{
  const source={'app.ts':'export async function judge(note:string):Promise<{flag:boolean,decision:"hold"|"rollback"}> { const [flag,decision]=await Promise.all([nl<boolean>`Judge note.`(note),nl<"hold"|"rollback">`Decide note.`(note)]); return {flag,decision}; }'};
  const expected={flag:true,decision:'hold'};
  assert.deepEqual(await checkAuthoring(source,{...spec,runs:[{args:['known'],expected}],oracle:[{match:['"known"'],value:'hold'},{match:['"known"'],value:true}]}),{ok:true,problems:[],results:[expected]});
});

test('ordinary drivers receive unchanged options and request fields',async()=>{
  const seen=[];
  const runtime=createNatlangRuntime({model:(request,_signal,options)=>{seen.push({request,options});return{calls:[['return_result',{status:'success',value:true}]]};},seed:{mode:'backend'}});
  const fn=loadVirtualNatlang({'judge.nl':'---\nargs:\n  note: string\nreturns: boolean\n---\nJudge note.\n'},'judge.nl');
  assert.equal(await runtime.run(()=>fn('known')),true);
  assert.equal(seen.length,1);assert.equal(seen[0].options,undefined);assert.equal('acceptsResult'in seen[0].request,false);
});
