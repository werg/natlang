import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { rootServices, servicesDigest } from '../dist/tune/services.js';
import { sharedSourceHash } from '../dist/tune/keys.js';

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
  await assert.rejects(rootServices(root, undefined, { id: 'r', function: 'adapt.nl', reference: {} }), /reference \{ plan, out \}/);
  assert.deepEqual((await rootServices(root, undefined, { id: 'r', function: 'adapt/chooseOperator.nl', reference: {} })).services, {});
  const { services, serviceDeclarations } = await rootServices(root, undefined, { id: 'r', function: 'adapt.nl', reference: { plan: '/p.json', out: '/o' } });
  assert.deepEqual(Object.keys(services.optimizer).sort(), ['evidence', 'propose', 'refine', 'retrieve']);
  assert.match(serviceDeclarations.optimizer, /operations: Record<string, RevisionOperation>/);
});

test('a program\'s tune services bind its teacher data; programs without them keep their hash', () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-tune-services-'));
  const program = { sources: { 'helpers.ts': 'a', 'f.nl': 'b' } };
  const without = sharedSourceHash(program, root);
  assert.equal(servicesDigest(root), null);
  writeFileSync(join(root, 'tune.services.mjs'), 'export default () => ({});\n');
  const first = sharedSourceHash(program, root);
  assert.notEqual(first, without);
  writeFileSync(join(root, 'tune.services.mjs'), 'export default () => ({ clock: {} });\n');
  assert.notEqual(sharedSourceHash(program, root), first);
  writeFileSync(join(root, 'helper.mjs'), 'export const value = 1;\n');
  writeFileSync(join(root, 'tune.services.mjs'), "import { value } from './helper.mjs'; export default () => ({ value });\n");
  const bound = servicesDigest(root);
  writeFileSync(join(root, 'helper.mjs'), 'export const value = 2;\n');
  assert.notEqual(servicesDigest(root), bound);
});
