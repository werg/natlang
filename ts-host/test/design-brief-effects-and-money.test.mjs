import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, readdir, mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import ts from 'typescript';
import { loadApplications } from '../../applications/design-briefs-host/runtime.mjs';
import { createApplicationService } from '../../applications/design-briefs-host/service.mjs';
import { openDesignBriefStore } from '../../applications/design-briefs-host/store.mjs';
import worldAdapter from '../../applications/design-briefs-host/adapters/098-persistent-world-actions.mjs';

const root = new URL('../../', import.meta.url);

test('slice identity collisions withhold every affected expense regardless of input order', async () => {
  const name = (await readdir(new URL('codebases/design_briefs/', root))).find(name => name.startsWith('025-'));
  const body = await readFile(new URL(`codebases/design_briefs/${name}/grantDesk/grantMath/checkPeriodSliceTotals.ts`, root), 'utf8');
  const compiled = ts.transpileModule(body, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } });
  const { default: verify } = await import('data:text/javascript;base64,' + Buffer.from(compiled.outputText).toString('base64'));
  const slice = (expenseId, sliceId) => ({ expenseId, sliceId, periodId: 'period', basisRef: 'basis',
    amount: { minorUnits: '100', currency: 'USD', scale: 2 },
    originalAmount: { minorUnits: '100', currency: 'USD', scale: 2 } });
  const a = slice('A', 'collision'), b = slice('B', 'collision'), c = slice('C', 'unique');
  for (const input of [[a, b, c], [b, a, c], [c, b, a]]) {
    const result = verify(input);
    assert.deepEqual(result.withheldExpenseIds, ['A', 'B']);
    assert.deepEqual(result.verifiedSlices, [c]);
    assert(result.blockers.includes('duplicate_slice:collision'));
  }
});

test('real effect-producing roots declare complete outbox boundaries; missing adapters cannot advance pending state', async t => {
  const catalog = await loadApplications();
  for (const [id, operation] of [['092', 'assembleDocumentary'], ['097', 'directLiveEvent'], ['098', 'directPersistentWorld']])
    assert.deepEqual(catalog.get(id).interfaces.get(operation).effectsPath, ['effects']);
  const app = catalog.get('098');
  const folder = (await readdir(new URL('training/semantic_applications/design_briefs/', root))).find(name => name.startsWith('098-'));
  const cases = JSON.parse(await readFile(new URL(`training/semantic_applications/design_briefs/${folder}/cases.json`, root), 'utf8'));
  const inputs = cases.find(row => row.root === 'directPersistentWorld.nl').inputs;
  const principal = { id: 'operator', tenantId: 'tenant', roles: ['operator'] };
  const capabilities = { '098': { read: ['operator'], write: ['operator'], dispatch: ['operator'], 'invoke:directPersistentWorld': ['operator'] } };
  const directory = await mkdtemp(join(tmpdir(), 'real-root-effects-'));
  const store = openDesignBriefStore({ filename: join(directory, 'state.sqlite'), capabilities });
  t.after(async () => { store.close(); await rm(directory, { recursive: true, force: true }); });
  const effect = { kind: 'apply-player-actions', worldId: inputs.world.id, priorRevision: inputs.world.revision,
    ruleVersion: inputs.world.ruleVersion, clientActionId: 'new-action', actions: [], idempotencyKey: 'new-action' };
  const result = { intent: [], questions: [], views: [], actions: [], receipts: [], beliefUpdates: [],
    continuityNotes: [], checkpoint: { ...inputs.checkpoint, pendingActionIds: ['new-action'] },
    status: 'awaiting-simulator', effects: [effect] };
  const options = { catalog, store, capabilities, execute: async () => structuredClone(result) };
  const service = createApplicationService(options);
  service.create(principal, '098', 'case', { interface: 'directPersistentWorld', inputs, idempotencyKey: 'create' });
  const request = { interface: 'directPersistentWorld', expectedVersion: 1, idempotencyKey: 'invoke' };
  await assert.rejects(service.invoke(principal, '098', 'case', request), error => error.code === 'effect_adapter_required');
  const key = { tenantId: principal.tenantId, appId: app.id, caseId: 'case' };
  assert.equal(store.readCase(principal, key).version, 1);
  assert.equal(store.readCase(principal, key).domainRecords.length, 0);
  const configured = createApplicationService({ ...options, effectAdapters: { '098': { directPersistentWorld: worldAdapter } } });
  const committed = await configured.invoke(principal, '098', 'case', request);
  assert.equal(committed.outbox.length, 1);
  assert.deepEqual(store.readOutbox(principal, committed.outbox[0].id).payload, effect);
  assert.deepEqual(store.readCase(principal, key).domainRecords[0].value.state.pendingActionIds, ['new-action']);
});
