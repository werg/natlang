import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { rootServices } from '../dist/tune/services.js';

test('a root gets its program\'s own tune services and declarations, or the host\'s with the same declarations', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-tune-services-'));
  assert.deepEqual(await rootServices(root, undefined, { id: 'r' }), {});
  writeFileSync(join(root, 'tune.services.mjs'), [
    'export const declarations = { clock: "export function now(): Promise<number>;" };',
    'export default async example => ({ clock: { now: async () => example.reference.at } });'].join('\n'));
  const provided = await rootServices(root, undefined, { id: 'r', reference: { at: 7 } });
  assert.equal(await provided.services.clock.now(), 7);
  assert.deepEqual(provided.serviceDeclarations, { clock: 'export function now(): Promise<number>;' });
  const given = await rootServices(root, () => ({ clock: { now: async () => 1 } }), { id: 'r' });
  assert.equal(await given.services.clock.now(), 1);
  assert.deepEqual(given.serviceDeclarations, provided.serviceDeclarations);
  writeFileSync(join(root, 'tune.services.mjs'), 'export default 3;\n');
  await assert.rejects(rootServices(root, undefined, { id: 'r' }), /must default-export/);
});

test('the learned optimizer serves whole adapt roots from their reference context', async () => {
  const root = new URL('../../applications/learned-optimizer/', import.meta.url).pathname;
  await assert.rejects(rootServices(root, undefined, { id: 'r', reference: {} }), /reference \{ plan, out \}/);
  const { services, serviceDeclarations } = await rootServices(root, undefined, { id: 'r', reference: { plan: '/p.json', out: '/o' } });
  assert.deepEqual(Object.keys(services.optimizer).sort(), ['evidence', 'propose', 'refine', 'retrieve']);
  assert.match(serviceDeclarations.optimizer, /operations: Record<string, RevisionOperation>/);
});
