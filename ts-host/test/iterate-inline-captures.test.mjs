import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { test } from 'node:test';
import { createNatlangRuntime, defineNatlang } from '../dist/index.js';
import { Folder } from '../dist/native/scoped-fs.js';
import { scriptedModel } from './support/natlang.mjs';

// V51 case 00 is preserved at
// runs/luna-semantic-v51-action-quality-review-20261007-v1/raw/case-00/result.jsonl
// (SHA-256 2eb78b7148139dfd09a2d56f21ad7502cf473703379825da6a0a6ac42039cc37).
// Its seq-38 creator code is SHA-256 635c4a285f871229a85c43e88cceb39ad9f37bd3f4af9c3e45ea12ac2126187c;
// frozen lowered.js/lower.js/iterate.js hashes are 44bb8682…488a9, 36573198…947f, 240db79b…59bc8.
// In that trace `/9` logs `draft`, which is the carried wrapper, not the sibling captures. This test
// checks actual per-iteration scalar snapshots; it does not treat that wrapper log as proof of stale captures.
test('nl.with snapshots created inside iterateOn use each step’s current values', async () => {
  const code = `type Draft={stop:string;route:string;repair:string;inspection:string}; type Pass={constraint:string;source_scope:string;allowed_fields:string[];evidence_role:string;evidence_path:string}; type Task={initialDraft:Draft;passes:Pass[]}; type Progress={index:number;draft:unknown}; const task=JSON.parse(await folder.file('task.json').readText()) as Task; const initial:Progress={index:0,draft:{...task.initialDraft}}; const step=async(state:Progress,passes:typeof task.passes):Promise<Progress>=>{const pass=passes[state.index];if(!pass)return state;const ev=folder.file(pass.evidence_path);const nextDraft=await nl.with<{draft:unknown;constraint:string;scope:string;allowed:string[];evidenceRole:string}>({draft:state.draft,constraint:pass.constraint,scope:pass.source_scope,allowed:pass.allowed_fields??[],evidenceRole:pass.evidence_role})\`Read only this current-pass evidence FileHandle. Reconcile its authority and scope using the constraint. Return full draft, copying exact source wording for updates only in allowed fields; preserve other fields exactly.\`(ev);return{index:state.index+1,draft:nextDraft};};const done=await iterateOn(step,initial,task.passes).withLimit({maxSteps:task.passes.length}).until(s=>s.index===task.passes.length);return done;`;
  const root = defineNatlang('---\nargs: {}\nreturns: unknown\nkind: directory-reducer\n---\nProcess the ordered evidence passes, using the current pass values.\n');
  let modelCalls = 0;
  const model = scriptedModel(() => modelCalls++ === 0 ? code : 'return { draft, constraint, scope, allowed, evidenceRole };');
  const passes = [
    { constraint: 'route-only constraint', source_scope: 'one work order', allowed_fields: ['route'], evidence_role: 'draft slip', evidence_path: 'evidence-01.md' },
    { constraint: 'stop-only constraint', source_scope: 'one work order', allowed_fields: ['stop'], evidence_role: 'active stop registry', evidence_path: 'evidence-02.md' },
    { constraint: 'repair-only constraint', source_scope: 'one work order', allowed_fields: ['repair'], evidence_role: 'completed work ticket', evidence_path: 'evidence-03.md' },
  ];
  const folder = Folder.fromFiles({
    'task.json': JSON.stringify({ initialDraft: { stop: 'unknown', route: 'R-16', repair: 'pending', inspection: 'unchecked' }, passes }),
    'evidence-01.md': 'unsigned route proposal',
    'evidence-02.md': 'active stop registry',
    'evidence-03.md': 'completed repair ticket',
  });

  const traces = [];
  const result = await createNatlangRuntime({ model: model.driver, trace: trace => traces.push(trace) })
    .run(() => folder.apply(root));

  assert.equal(result.index, passes.length);
  // The helper result is carried as the next draft, matching the V51 wrapper shape.
  // Nested values are old state; sibling captures must still use the current pass.
  let carried = result.draft;
  for (const pass of [...passes].reverse()) {
    assert.equal(carried.constraint, pass.constraint);
    assert.equal(carried.scope, pass.source_scope);
    assert.deepEqual(carried.allowed, pass.allowed_fields);
    assert.equal(carried.evidenceRole, pass.evidence_role);
    carried = carried.draft;
  }
  assert.equal(model.openings.length, passes.length + 1);

  const parent = traces.find(trace => !trace.parentCallId);
  const evalAction = parent.events.find(event => event.kind === 'action' && event.name === 'eval');
  const children = traces.filter(trace => trace.parentCallId === parent.callId &&
    trace.events.some(event => event.kind === 'manifest' && event.inline_instruction_site));
  const sites = children.map(trace => trace.events.find(event => event.kind === 'manifest').inline_instruction_site);
  assert.equal(sites.length, passes.length);
  assert.equal(new Set(sites.map(site => site.definition_id)).size, 1);
  assert.ok(sites.every(site => site.origin.parentInvocationId === parent.callId));
  const sourceHash = createHash('sha256').update(evalAction.arguments.code).digest('hex');
  assert.ok(sites.every(site => site.origin.writtenCodeSha256 === sourceHash));
});
