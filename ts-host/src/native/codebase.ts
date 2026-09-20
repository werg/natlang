import { hexDigest } from './hash.js';
import { TypeEnv, parseType } from './types.js';
import { Reject, buildPending, type LambdaNode } from './values.js';

export type NativeDefinition = { args?: Record<string, string>; returns: string;
  instructions?: string; code?: string; engine?: string; types?: Record<string, string>;
  uses?: Record<string, string>; effects?: string[]; description?: string };
export type NativeGraph = { root: string; definitions: Record<string, NativeDefinition>;
  revision: string; instantiate(inputs?: Record<string, unknown>): LambdaNode };
const id = /^[A-Za-z_][A-Za-z0-9_]*$/;
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
    .map(([key, child]) => `${JSON.stringify(key)}:${canonical(child)}`).join(',')}}`;
  return JSON.stringify(value);
}

export function checkedDefinitions(entries: Record<string, NativeDefinition>, root: string): NativeGraph {
  const supplied = structuredClone(entries);
  if (!Object.hasOwn(supplied, root)) throw new Reject([{ path: root, code: 'no-such-path', expected: 'a root definition' }]);
  for (const [name, def] of Object.entries(supplied)) {
    if (!id.test(name)) throw new Reject([{ path: name, code: 'type-mismatch', expected: 'a function identifier' }]);
    if (typeof def.returns !== 'string' || (typeof def.code === 'string') === (typeof def.instructions === 'string'))
      throw new Reject([{ path: name, code: 'type-mismatch', expected: 'returns and exactly one source body' }]);
    const env = new TypeEnv(Object.fromEntries(Object.entries(def.types ?? {}).map(([key, value]) => [key, parseType(value)])));
    const signature = parseType(`Lambda<{ ${Object.entries(def.args ?? {}).map(([key, value]) =>
      `${key.replace(/\?$/, '')}${key.endsWith('?') ? '?' : ''}: ${value}`).join(', ')} }, ${def.returns}>`);
    env.checkNames(signature);
    for (const type of Object.values(env.names)) env.checkNames(type);
    for (const [alias, target] of Object.entries(def.uses ?? {})) {
      if (!id.test(alias) || !Object.hasOwn(supplied, target))
        throw new Reject([{ path: `${name}/uses/${alias}`, code: 'no-such-path', expected: 'a supplied definition', got: target }]);
    }
  }
  const visited = new Set<string>(), active = new Set<string>();
  function visit(name: string): void {
    if (active.has(name)) throw new Reject([{ path: name, code: 'recursion', expected: 'an acyclic source graph' }]);
    if (visited.has(name)) return;
    active.add(name);
    for (const target of Object.values(supplied[name]!.uses ?? {})) visit(target);
    active.delete(name); visited.add(name);
  }
  for (const name of Object.keys(supplied)) visit(name);
  function inline(name: string): Record<string, unknown> {
    const def = supplied[name]!;
    const kind = def.code !== undefined ? 'code' : 'instructions';
    return { args: def.args ?? {}, returns: def.returns, [kind]: def[kind],
      ...(kind === 'code' && def.engine && def.engine !== 'quickjs-isolated' ? { engine: def.engine } : {}),
      types: def.types ?? {}, effects: def.effects ?? [], description: def.description ?? '',
      codebase: Object.fromEntries(Object.entries(def.uses ?? {}).map(([alias, target]) => [alias, inline(target)])) };
  }
  const stable = canonical(supplied);
  const revision = hexDigest(stable);
  return { root, definitions: supplied, revision, instantiate(inputs = {}) {
    const doc = inline(root), def = supplied[root]!;
    const params = Object.entries(def.args ?? {}).map(([name, type]) =>
      `${name.replace(/\?$/, '')}${name.endsWith('?') ? '?' : ''}: ${type}`).join(', ');
    const kind = def.code !== undefined ? 'code' : 'instructions';
    const node = buildPending({ $lambda: { type: `Lambda<{ ${params} }, ${def.returns}>`, [kind]: def[kind],
      ...(kind === 'code' && def.engine && def.engine !== 'quickjs-isolated' ? { engine: def.engine } : {}),
      args: inputs, types: def.types ?? {}, effects: def.effects ?? [],
      codebase: doc.codebase, function: root } });
    if (node.nodeKind !== 'lambda') throw new Error('internal graph root error');
    return node;
  } };
}
