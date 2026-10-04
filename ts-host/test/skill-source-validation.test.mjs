import test from 'node:test';
import assert from 'node:assert/strict';
import {SourceEvaluator} from '../dist/improvement/host.js';
import {Folder} from '../dist/native/scoped-fs.js';
import {UsageGateway} from '../dist/evaluation/usage.js';
test('source checker validates skill metadata before any model evaluation',async()=>{
 let calls=0;
 const evaluator=new SourceEvaluator({entry:'main.ts',exportName:'solve',programId:'skill-yaml'},[],()=>{calls++;throw Error('unexpected inference');},new UsageGateway({maxModelCalls:2,maxRollouts:2,maxProposals:2}),{executorId:'fixture'});
 const folder=Folder.fromFiles({'main.ts':'export function solve(x:number):number{return x;}', 'main/skills/bookkeeping/SKILL.md':'---\nname: bookkeeping\ndescription: Bookkeeping: track exact sources\n---\nRecord sources.\n'});
 const bad=await evaluator.check(folder.snapshot());assert.equal(bad.valid,false);assert.ok(bad.diagnostics.some(d=>d.includes('skill-frontmatter')&&d.includes('main/skills/bookkeeping/SKILL.md')));
 folder.writeText('main/skills/bookkeeping/SKILL.md','---\nname: bookkeeping\ndescription: "Bookkeeping: track exact sources"\n---\nRecord sources.\n');assert.equal((await evaluator.check(folder.snapshot())).valid,true);assert.equal(calls,0);
});
