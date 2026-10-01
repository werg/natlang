import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { AssignmentBudget,OperationJournal,FrozenImprover,Folder,finiteCounterexamples } from '../dist/index.js';
import { migrateCode,migrateRow } from '../scripts/self-improvement/migrate-folder-api.mjs';
test('assignment capacity cannot be minted by batches, and final reserves stay protected',()=>{
 const ceiling={collection:2,modelCalls:3,caseExecutions:4,trainingJobs:4,trainingUpdates:8,confirmation:1,cost:10};
 const budget=new AssignmentBudget(ceiling,{trainingJobs:4,confirmation:1});
 budget.allocate({collection:2}); assert.throws(()=>budget.allocate({collection:1}),/exhausted/);
 assert.throws(()=>budget.allocate({trainingJobs:1}),/exhausted/);budget.allocate({trainingJobs:4,confirmation:1},true);
 for(let i=0;i<3;i++)budget.attempt('unreconstructable');assert.throws(()=>budget.attempt('unreconstructable'),/coverage-exhausted/);
});
test('assignment operation capacity and identity survive a crash without charging twice',()=>{
 const journal=new OperationJournal(mkdtempSync(join(tmpdir(),'natlang-charge-once-')));
 const ceiling={collection:1,modelCalls:3,caseExecutions:4,trainingJobs:1,trainingUpdates:8,confirmation:1,cost:10};
 const first=new AssignmentBudget(ceiling,{},3,journal);first.allocateOnce('frozen-confirmation',{confirmation:1,modelCalls:3});
 const restarted=new AssignmentBudget(ceiling,{},3,journal);restarted.allocateOnce('frozen-confirmation',{confirmation:1,modelCalls:3});
 assert.equal(restarted.used.confirmation,1);assert.equal(restarted.used.modelCalls,3);
 assert.throws(()=>restarted.allocateOnce('frozen-confirmation',{confirmation:1,modelCalls:2}),/identity changed/);
});
test('durable results are reused, checkpoint revisions reject stale publication',async()=>{
 const journal=new OperationJournal(mkdtempSync(join(tmpdir(),'natlang-journal-')));let executions=0;
 assert.equal(await journal.run('evaluate:snapshot:suite:seed',async()=>++executions),1);
 assert.equal(await journal.run('evaluate:snapshot:suite:seed',async()=>++executions),1);
 assert.equal(journal.commit(0,{source:'I',state:{iteration:0}}),1);
 assert.throws(()=>journal.commit(0,{source:'C'}),/stale/);
});
test('frozen improver adoption is between runs and finite counterexample search is bounded',async()=>{
 const a=Folder.fromFiles({'step.nl':'original'}).snapshot(),b=a.branch();b.writeText('step.nl','candidate');
 const frozen=new FrozenImprover(a);await frozen.run(0,async source=>{assert.equal(source.digest,a.digest);assert.throws(()=>frozen.adopt(b.snapshot()),/between/);});
 frozen.adopt(b.snapshot());assert.equal(frozen.snapshot().digest,b.snapshot().digest);
 await assert.rejects(()=>frozen.run(2,async()=>{}),/depth/);
 assert.deepEqual(await finiteCounterexamples([1,2,3],async n=>n<2,2),{checked:2,complete:false,counterexamples:[2]});
});
test('migration rewrites identified reducer calls and invalidates old observations',()=>{
 const migrated=migrateCode('const value = await tidy(folder, "x");',new Set(['tidy']));assert.match(migrated.source,/folder.propose\(tidy, "x"\)/);
 const row=migrateRow({id:'a',helper:{name:'tidy',kind:'directory-reducer'},trajectory:[{code:'await tidy(folder, "x")'}],outcome:{accepted:true}});
 assert.equal(row.changes,1);assert.equal(row.row.outcome.accepted,false);assert.equal(row.row.migration.disposition,'replay-required');
});

test('a crashed operation owner is reconciled without minting a new operation',async()=>{
 const {writeFileSync}=await import('node:fs');const {fingerprint}=await import('../dist/adaptation/identity.js');
 const directory=mkdtempSync(join(tmpdir(),'natlang-crash-journal-')),journal=new OperationJournal(directory);
 journal.record('pending-check',null);
 writeFileSync(join(directory,fingerprint('pending-check')+'.json'),JSON.stringify({status:'pending'}));
 writeFileSync(join(directory,fingerprint('pending-check')+'.lock'),'2147483647');
 assert.equal(await journal.run('pending-check',async()=>{throw Error('must reconcile');},async()=>42),42);
 assert.equal(await journal.run('pending-check',async()=>0),42);
});
