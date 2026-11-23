// The standard library reaches callable-folder TypeScript through the platform alone: a host that loads only the
// runtime's Node platform (the collector, a held-eval runner) and never the package entry still serves
// `@natlang/node/std` as its own instance, so the combinators see the calling task. Records built in the module realm
// (which has its own Object.prototype) merge field by field.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualCallables } from '../dist/runtime/node.js';
import { isPlainObject } from '../dist/runtime/plain-object.js';
import vm from 'node:vm';

test('callable-folder semanticMerge through the Node platform alone merges realm records field by field', async () => {
  const tree = loadVirtualCallables({ 'merge_case.ts': `import { semanticMerge } from '@natlang/node/std';
export default async function merge_case(): Promise<{ value: Record<string, string>; unresolved: string[] }> {
  const base = { hours: 'Mon-Fri 9-5', note: 'rear', phone: '1' };
  const ours = { hours: 'Mon-Fri 9-5, Sat 10-1', note: 'rear, two spaces', phone: '1' };
  const theirs = { hours: 'Mon-Fri 9-5, Sun 11-3', note: 'front', phone: '2' };
  const merged = await semanticMerge(base, ours, theirs, {
    intentOf: async (before: unknown, after: unknown) => 'changes ' + String(after),
    compatible: async (_base: unknown, a: { after: unknown }, b: { after: unknown }) => /Sat/.test(String(a.after)) && /Sun/.test(String(b.after)),
    combine: async () => 'Mon-Fri 9-5, Sat 10-1, Sun 11-3',
  });
  return { value: merged.value, unresolved: merged.unresolved.map(item => item.path.join('.')) };
}
` });
  const result = await createNatlangRuntime({}).run(() => tree.merge_case());
  assert.deepEqual(JSON.parse(JSON.stringify(result)), {
    value: { hours: 'Mon-Fri 9-5, Sat 10-1, Sun 11-3', note: 'rear', phone: '2' }, unresolved: ['note'] });
});

test('isPlainObject: plain records from any realm, nothing else', () => {
  assert.equal(isPlainObject({}), true);
  assert.equal(isPlainObject(Object.create(null)), true);
  assert.equal(isPlainObject(vm.runInNewContext('({ a: 1 })')), true);
  for (const value of [[], null, 'x', new Date(), new (class A {})(), vm.runInNewContext('[1]'), vm.runInNewContext('new Date()')])
    assert.equal(isPlainObject(value), false);
});
