import test from 'node:test';
import assert from 'node:assert/strict';
import { Folder, FolderConflictError, createNatlangRuntime, IterationLimitError } from '../dist/index.js';
import { APPLY_TO_FOLDER } from '../dist/native/scoped-fs.js';
const reducer = run => ({ [APPLY_TO_FOLDER]: run });
const runtime = () => createNatlangRuntime({ model: async () => { throw Error('unexpected model request'); } });

test('proposal acceptance rejects stale and forged drafts; snapshots freeze actual bytes', async () => {
  const folder = Folder.fromFiles({ 'entry.ts': 'baseline' });
  const snapshot = folder.snapshot();
  const proposal = await folder.propose(reducer(async draft => { await draft.file('entry.ts').writeText('candidate'); return 'explanation'; }));
  assert.equal(await folder.readText('entry.ts'), 'baseline');
  assert.equal((await folder.accept(proposal)).digest, proposal.folder.digest);
  assert.equal(proposal.value, 'explanation');
  assert.equal(await snapshot.readText('entry.ts'), 'baseline');
  await assert.rejects(() => snapshot.file('entry.ts').writeText('mutate'), /read-only/);
  await assert.rejects(() => folder.accept(proposal), FolderConflictError);
  await assert.rejects(() => folder.accept({ ...proposal }), FolderConflictError);
  const stale = await folder.propose(reducer(async () => 'old'));
  folder.writeText('other.ts', 'concurrent');
  await assert.rejects(() => folder.accept(stale), /stale/);
});

test('population child acceptance does not select it; exact non-incumbent snapshot can be selected', async () => {
  const incumbent = Folder.fromFiles({ 'entry.ts': 'I', 'incumbent-only.ts': 'delete on selection' });
  const parent = incumbent.snapshot().branch();
  parent.writeText('entry.ts', 'P'); parent.remove('incumbent-only.ts');
  const child = await parent.propose(reducer(async draft => { await draft.file('entry.ts').writeText('C'); return true; }));
  await parent.accept(child);
  assert.equal(await incumbent.readText('entry.ts'), 'I');
  await incumbent.select(parent.snapshot());
  assert.equal(await incumbent.readText('entry.ts'), 'C');
  assert.equal(incumbent.isFile('incumbent-only.ts'), false);
  await assert.rejects(() => incumbent.select(Folder.fromFiles({ 'entry.ts': 'foreign' }).snapshot()), /foreign/);
});

test('folder iteration returns joint checkpoints, leaves caller unchanged, and rolls back failed edits', async () => {
  const folder = Folder.fromFiles({ 'entry.ts': '0' });
  const step = reducer(async (draft, [state]) => { await draft.file('entry.ts').writeText(String(state.n + 1)); return { n: state.n + 1 }; });
  const events = [];
  const result = await runtime().run(() => folder.iterateOn(step, { n: 0 }).checkProgress('off').withLimit({ maxSteps: 2 })
    .onStep(event => events.push(event)).until(state => state.n === 2));
  assert.equal(await folder.readText('entry.ts'), '0');
  assert.equal(await result.folder.readText('entry.ts'), '2');
  assert.equal(result.state.n, 2);
  assert.ok(Object.isFrozen(result.state));
  for (const event of events.filter(event => event.state)) assert.equal(await event.state.folder.readText('entry.ts'), String(event.state.state.n));
  const failure = reducer(async draft => { await draft.file('entry.ts').writeText('failed'); throw Error('bad state'); });
  await assert.rejects(() => runtime().run(() => result.folder.iterateOn(failure, result.state).withLimit({maxSteps:1}).checkProgress('off').until(() => false)), error => {
    assert.equal(error.lastState.folder.digest, result.folder.digest); assert.deepEqual(error.lastState.state, { n: 2 }); return true;
  });
  await assert.rejects(() => runtime().run(() => folder.iterateOn(step, { n: 0 }).checkProgress('off').withLimit({ maxSteps: 1 }).until(() => false)), IterationLimitError);
});

test('proposal authority survives distinct interpreter eval calls',async()=>{
 const {loadVirtualNatlang}=await import('../dist/runtime/virtual-project.js');
 const source=Folder.fromFiles({'entry.ts':'before'});
 const step=loadVirtualNatlang({
  'step.nl':'---\nkind: directory-reducer\nargs: {}\nreturns: boolean\n---\nPropose a rewrite, then accept it in a later eval.',
  'step/rewrite.nl':'---\nkind: directory-reducer\nargs: {}\nreturns: string\n---\nRewrite entry.ts.'},'step.nl');
 const runtime=createNatlangRuntime({agent:async session=>{
  if(session.lam.functionName==='rewrite'){
   const event=await session.applyAsync('eval',{code:'await folder.file("entry.ts").writeText("after"); return "changed";'});assert.equal(event.kind,'ok',event.text);
  }else{
   const first=await session.applyAsync('eval',{code:'const parent=folder.snapshot().branch(); const proposal=await parent.propose(rewrite);'});assert.equal(first.kind,'ok',first.text);
   const second=await session.applyAsync('eval',{code:'await parent.accept(proposal); await folder.select(parent.snapshot()); return true;'});assert.equal(second.kind,'ok',second.text);
  }
 }});
 assert.equal(await runtime.run(()=>source.apply(step)),true);assert.equal(await source.readText('entry.ts'),'after');
});


test('eval helper results resolve conflicting module-local return aliases',async()=>{
 const {loadVirtualNatlang}=await import('../dist/runtime/virtual-project.js');
 const step=loadVirtualNatlang({
  'step.nl':'---\nargs: {}\nreturns: string\n---\nCall both helpers.',
  'step/left.ts':'export type Member={label:string}; export function make():Member{return {label:"left"};}',
  'step/right.ts':'export type Member={count:number}; export function make():Member{return {count:2};}'},'step.nl');
 const rt=createNatlangRuntime({agent:async session=>{
  const event=await session.applyAsync('eval',{code:'const a=await left.make(); const b=await right.make(); return a.label+":"+b.count;'});
  assert.equal(event.kind,'ok',event.text);
 }});
 assert.equal(await rt.run(()=>step()),'left:2');
});

test('folder revisions are automatic, immutable, lineage-scoped and available after branching',()=>{
 const source=Folder.fromFiles({'main.ts':'baseline'}),baseline=source.snapshot(),branch=baseline.branch();
 branch.writeText('main.ts','candidate');const candidate=branch.snapshot();
 assert.equal(source.at(candidate.digest),candidate);assert.equal(branch.at(baseline.digest),baseline);
 assert.equal(new TextDecoder().decode(source.at(baseline.digest).readBytesSync('main.ts')),'baseline');
 assert.throws(()=>Folder.fromFiles({'main.ts':'baseline'}).at(candidate.digest),/unknown source revision/);
 assert.throws(()=>source.root().dir('nested').at(candidate.digest),/entry not found|unknown source revision/);
});

test('invalid iteration state names the field and sparse arrays cannot normalize silently',async()=>{
 const folder=Folder.fromFiles({'entry.ts':'baseline'});
 const step=reducer(async()=>({done:true,lastExperiment:{training:[{value:undefined}]}}));
 await assert.rejects(()=>runtime().run(()=>folder.iterateOn(step,{done:false}).checkProgress('off').withLimit({maxSteps:1}).until(s=>s.done)),/\$\["lastExperiment"\]\["training"\]\[0\]\["value"\]/);
 await assert.rejects(()=>runtime().run(()=>folder.iterateOn(step,{rows:Array(2)})),/\$\["rows"\]\[0\]/);
 assert.equal(await folder.readText('entry.ts'),'baseline');
});
