import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Folder } from '../dist/index.js';
import { session as open } from './support/natlang.mjs';

test('successful evals recompile source-backed helpers against current inputs and retained type aliases', async () => {
  let childEvents;
  const { lam, session } = open({ type: '(value: number) => number', instructions: 'Return the revised score.', args: { value: 5 } }, {
    agent: async child => { childEvents = child.runtime.trace.events; child.apply('return_result', { status: 'success', value: 41 }); },
  });
  session.runtime.currentCallId = 'test-invocation';
  const declared = await session.applyAsync('eval', { code:
    'type Progress = { done: number }; const revise = async (progress: Progress) => { const result: number = await nl<number>`Read the progress.`(); return result + progress.done + value; };' });
  assert.equal(declared.kind, 'ok', declared.text);
  assert.equal(Object.hasOwn(lam.let, 'revise'), false, 'the runtime keeps source, not a function value');
  assert.match(session.scopeGuide(), /revise/);
  const called = await session.applyAsync('eval', { code: 'return await revise({done: 2});' }, 'eval-call-2');
  assert.equal(called.kind, 'ok', called.text);
  assert.equal(lam.return, 48);
  const inlineSite = childEvents.map(event => event.inline_instruction_site).find(site => site?.source_backed_helper);
  assert.ok(inlineSite, 'the newly compiled inline call records its authenticated helper source');
  assert.equal(inlineSite.source_backed_helper.name, 'revise');
  assert.match(inlineSite.source_backed_helper.sourceHash, /^[a-f0-9]{64}$/);
  assert.ok(inlineSite.source_backed_helper.declaredAction?.actionOrdinal >= 0);
  assert.equal(inlineSite.origin?.toolCallId, 'eval-call-2', 'execution is attributed to the current eval action');
  assert.equal(inlineSite.origin?.sourceTemplateSpan, undefined, 'a prior-eval template gets no fabricated current-source span');
});

test('helpers recompile from fresh source after a successful redeclaration', async () => {
  const { lam, session } = open({ type: '(value: number) => number', instructions: 'Return a score.', args: { value: 3 } });
  assert.equal((await session.applyAsync('eval', { code: 'function revise(n: number) { return n + 1; }' })).kind, 'ok');
  assert.equal((await session.applyAsync('eval', { code: 'function revise(n: number) { return n + 2; } return revise(value);' })).kind, 'ok');
  assert.equal(lam.return, 5);
  assert.equal((await session.applyAsync('eval', { code: 'return revise(value);' })).kind, 'ok');
  assert.equal(lam.return, 5);
});

test('failed evals do not persist new helpers or replace an earlier helper', async () => {
  const { lam, session } = open({ type: '(value: number) => number', instructions: 'Return a score.', args: { value: 3 } });
  assert.equal((await session.applyAsync('eval', { code: 'function revise(n: number) { return n + 1; }' })).kind, 'ok');
  assert.equal((await session.applyAsync('eval', { code: 'function revise(n: number) { return n + 9; } throw new Error("rollback");' })).kind, 'error');
  assert.equal((await session.applyAsync('eval', { code: 'return revise(value);' })).kind, 'ok');
  assert.equal(lam.return, 4);
  assert.equal((await session.applyAsync('eval', { code: 'function fresh(n: number) { return n; } throw new Error("rollback fresh");' })).kind, 'error');
  const missing = await session.applyAsync('eval', { code: 'return fresh(value);' });
  assert.equal(missing.kind, 'error'); assert.match(missing.text, /fresh is not defined/);
});

test('helpers that refer to eval-local functions stay eval-local with an explicit explanation', async () => {
  const { session } = open({ type: '(value: number) => number', instructions: 'Return a score.', args: { value: 3 } });
  const declared = await session.applyAsync('eval', { code:
    'const step = (n: number) => n + 1; function revise(n: number) { return step(n); } revise(value);' });
  assert.equal(declared.kind, 'ok', declared.text);
  assert.match(declared.text, /revise is available only in this eval because it refers to eval-local function step/);
  const missing = await session.applyAsync('eval', { code: 'return revise(value);' });
  assert.equal(missing.kind, 'error'); assert.match(missing.text, /revise is not defined/);
});

test('helper free names include transient references in parameter defaults', async () => {
  const { session } = open({ type: '(value: number) => number', instructions: 'Return a score.', args: { value: 3 } });
  const declared = await session.applyAsync('eval', { code:
    'const step = (n: number) => n + 1; function revise(n: number = step(value)) { return n; } revise();' });
  assert.equal(declared.kind, 'ok', declared.text);
  assert.match(declared.text, /revise is available only in this eval because it refers to eval-local function step/);
});

test('nested shadowing does not hide a real helper dependency on the outer saved scope', async () => {
  const { lam, session } = open({ type: '(value: number) => number', instructions: 'Return a score.', args: { value: 3 } });
  assert.equal((await session.applyAsync('eval', { code:
    'const outer = value + 4; function revise() { function inner(outer: number) { return outer * 2; } return outer + inner(1); }' })).kind, 'ok');
  assert.equal((await session.applyAsync('eval', { code: 'return revise();' })).kind, 'ok');
  assert.equal(lam.return, 9, 'revise retains the persisted outer value while its nested parameter shadows it locally');
});

test('a source-backed helper uses the child current folder fence on every invocation', async () => {
  const folder = Folder.fromFiles({ 'records/packet.md': 'approved evidence', 'records/decision.json': 'parent result' });
  let declaration, denied, recovered, childTrace;
  const { lam, session } = open({ type: '() => boolean', subtype: 'directory-reducer',
    instructions: 'Inspect the supplied packet and report whether it has content.' }, { agent: async child => {
    declaration = await child.applyAsync('eval', { code:
      'async function writeSibling() { await folder.file("decision.json").writeText("unauthorized"); return true; }' });
    denied = await child.applyAsync('eval', { code: 'await writeSibling(); return true;' });
    childTrace = child.runtime.trace.events;
    recovered = await child.applyAsync('eval', { code:
      'return (await folder.file("packet.md").readText()).length > 0;' });
    child.apply('return_result', { status: 'success', value: recovered.value });
  } });
  lam.projectTransaction = await folder.beginTransaction(); lam.reducerMode = 'apply';
  const result = await session.applyAsync('eval', { code:
    'const file = folder.file("records/packet.md"); return await nl<boolean>`Inspect this packet.`(file);' });
  assert.equal(declaration.kind, 'ok', declaration.text);
  assert.equal(denied.kind, 'error', denied.text); assert.match(denied.text, /outside the supplied FileHandle scope/);
  assert.equal(recovered.kind, 'ok', recovered.text);
  assert.equal(result.kind, 'ok', result.text); assert.equal(result.value, true);
  assert.equal(await lam.projectTransaction.folder.readText('records/decision.json'), 'parent result');
  assert.ok(childTrace.some(event => event.kind === 'scope_failure' && String(event.message ?? '').includes('outside the supplied FileHandle scope')));
  lam.projectTransaction.abort();
});
