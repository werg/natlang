import { fitsType, formatType, parseType, TypeEnv } from './types.js';
import type { Type } from './types.js';

export const MISSING = Symbol('natlang-missing');
export type Missing = typeof MISSING;
export type Value = null | boolean | number | string | Value[] | { [key: string]: Value } | Pending | Missing;
export type Status = 'unreduced' | 'running' | 'quiesced' | 'waiting' | 'done';

type Base = { type: Type; types: Record<string, Type>; typesSrc: Record<string, string>;
  status: Status; note: string; attempts: number; steps: number };
export type LambdaNode = Base & { nodeKind: 'lambda'; kind: 'instructions' | 'code'; engine: string;
  body: string; args: Record<string, Value>; return: Value; effects: string[];
  journal: unknown[]; originalBody?: string; let: Record<string, Value>;
  letTypes: Record<string, Type>; codebase: Record<string, unknown>; functionName: string;
  marks: Record<number, string>; fnCopies: Record<string, unknown> };
export type MapNode = Base & { nodeKind: 'map'; over: Value; fn: Value; slots?: Value[]; itemName: string };
export type FoldNode = Base & { nodeKind: 'fold'; over: Value; init: Value; step: Value;
  acc: Value; at: number; current: Value | null };
export type IterateNode = Base & { nodeKind: 'iterate'; init: Value; step: Value; check: Value;
  max: Value; state: Value; iteration: number; recent: Value[]; seenHashes: string[];
  current: Value | null; stateName: string; checkName: string };
export type Pending = LambdaNode | MapNode | FoldNode | IterateNode;
export type Diagnostic = { path: string; code: string; expected?: string; got?: string };

export class Reject extends Error {
  constructor(readonly diagnostics: Diagnostic[]) {
    super(diagnostics.map(d => `${d.path}: ${d.code}${d.expected ? ` (expected ${d.expected})` : ''}`).join('; '));
  }
}
const reject = (path: string, code: string, expected?: string, got?: string): never => {
  throw new Reject([{ path, code, expected, got }]);
};
const plain = (value: unknown): value is Record<string, unknown> => value !== null &&
  typeof value === 'object' && !Array.isArray(value) && Object.getPrototypeOf(value) === Object.prototype;
export const isPending = (value: unknown): value is Pending => plain(value) &&
  ['lambda', 'map', 'fold', 'iterate'].includes(String(value.nodeKind));

export function cloneValue<T extends Value>(value: T): T {
  if (value === MISSING || value === null || typeof value !== 'object') return value;
  if (Array.isArray(value)) return value.map(x => cloneValue(x)) as T;
  if (isPending(value)) {
    // Immutable source definitions can be shared; execution state must not be.
    const out: Record<string, unknown> = { ...value };
    for (const [key, item] of Object.entries(value)) {
      if (key === 'codebase' || key === 'fnCopies' || key === 'types' || key === 'typesSrc' || key === 'letTypes') continue;
      out[key] = cloneValue(item as Value);
    }
    return out as T;
  }
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, cloneValue(item as Value)])) as T;
}

function wrapper(raw: unknown): string | undefined {
  if (!plain(raw)) return;
  const keys = Object.keys(raw).filter(k => ['$lambda', '$map', '$fold', '$iterate'].includes(k));
  return keys.length ? Object.keys(raw).length === 1 ? keys[0] : 'invalid' : undefined;
}

function preview(value: unknown): string {
  if (isPending(value)) return formatType(value.type);
  if (value === MISSING) return 'missing';
  const text = JSON.stringify(value);
  return (text ?? String(value)).slice(0, 45);
}

export function coerce(raw: unknown, type: Type, env: TypeEnv, path = 'value'): Value {
  const wanted = env.resolve(type);
  if (wanted.kind === 'lambda') {
    let value = raw;
    if (!isPending(value)) {
      if (wrapper(value) !== '$lambda') return reject(path, 'type-mismatch', formatType(type), preview(raw));
      value = buildPending(value, env, path);
    }
    if (!isPending(value) || value.nodeKind !== 'lambda' || !bodyLambdaFits(value.type, wanted, env))
      return reject(path, 'type-does-not-fit-slot', formatType(type), isPending(value) ? formatType(value.type) : preview(value));
    return value;
  }
  if (isPending(raw)) {
    if (!fitsType(raw.type, type, env)) return reject(path, 'type-does-not-fit-slot', formatType(type), formatType(raw.type));
    return raw;
  }
  const key = wrapper(raw);
  if (key === 'invalid') return reject(path, 'reserved-key', 'a single wrapper key');
  if (key) {
    const node = buildPending(raw, env, path);
    if (!fitsType(node.type, type, env)) return reject(path, 'type-does-not-fit-slot', formatType(type), formatType(node.type));
    return node;
  }
  if (wanted.kind === 'union') {
    for (const member of wanted.members) {
      try { return coerce(raw, member, env, path); } catch (error) { if (!(error instanceof Reject)) throw error; }
    }
    return reject(path, 'type-mismatch', formatType(type), preview(raw));
  }
  if (wanted.kind === 'prim') {
    if ((wanted.name === 'Text' || wanted.name === 'Blob') && typeof raw === 'string') return raw;
    if (wanted.name === 'Num' && typeof raw === 'number' && Number.isFinite(raw) &&
        (!Number.isInteger(raw) || Number.isSafeInteger(raw))) return raw;
    if (wanted.name === 'Bool' && typeof raw === 'boolean') return raw;
    if (wanted.name === 'Null' && raw === null) return null;
    return reject(path, 'type-mismatch', wanted.name, preview(raw));
  }
  if (wanted.kind === 'lit') {
    if (raw === wanted.value) return raw as Value;
    return reject(path, 'type-mismatch', formatType(wanted), preview(raw));
  }
  if (wanted.kind === 'list') {
    if (!Array.isArray(raw)) return reject(path, 'type-mismatch', formatType(type), preview(raw));
    return raw.map((item, i) => coerce(item, wanted.element, env, `${path}/${i}`));
  }
  if (wanted.kind === 'dict' || wanted.kind === 'record') {
    if (!plain(raw)) return reject(path, 'type-mismatch', formatType(type), preview(raw));
    const out: Record<string, Value> = {};
    for (const [key, value] of Object.entries(raw)) {
      if (key.startsWith('$')) return reject(`${path}/${key}`, 'reserved-key');
      if (wanted.kind === 'dict') out[key] = coerce(value, wanted.element, env, `${path}/${key}`);
      else {
        const field = wanted.fields.find(f => f.name === key);
        if (!field) return reject(`${path}/${key}`, 'unknown-field', formatType(wanted));
        if (value === null && field.optional && !fitsType(parseType('Null'), field.type, env)) continue;
        out[key] = coerce(value, field.type, env, `${path}/${key}`);
      }
    }
    return wanted.kind === 'record' ? Object.fromEntries(wanted.fields.filter(f => f.name in out).map(f => [f.name, out[f.name]!])) : out;
  }
  return reject(path, 'type-mismatch', formatType(type), preview(raw));
}

export function bodyLambdaFits(actual: Type, wanted: Extract<Type, { kind: 'lambda' }>, env: TypeEnv): boolean {
  if (actual.kind !== 'lambda' || !fitsType(actual.returns, wanted.returns, env)) return false;
  return wanted.params.fields.every(field => {
    const got = actual.params.fields.find(f => f.name === field.name);
    return !!got && fitsType(field.type, got.type, env);
  });
}

export function partType(node: Pending, part: string): Type {
  const type = node.type;
  if (node.nodeKind === 'map' && type.kind === 'map') {
    if (part === 'over') return { kind: 'list', element: type.a };
    if (part === 'fn') return { kind: 'lambda', params: { kind: 'record', fields: [{ name: node.itemName, type: type.a, optional: false }] }, returns: type.b };
  }
  if (node.nodeKind === 'fold' && type.kind === 'fold') {
    if (part === 'over') return { kind: 'list', element: type.a };
    if (part === 'init' || part === 'acc') return type.s;
    if (part === 'step') return { kind: 'lambda', params: { kind: 'record', fields: [
      { name: 'acc', type: type.s, optional: false }, { name: 'item', type: type.a, optional: false }] }, returns: type.s };
  }
  if (node.nodeKind === 'iterate' && type.kind === 'iterate') {
    if (part === 'init' || part === 'state') return type.s;
    if (part === 'max') return parseType('Num');
    if (part === 'step') return { kind: 'lambda', params: { kind: 'record', fields: [
      { name: node.stateName, type: type.s, optional: false }] }, returns: type.s };
    if (part === 'check') return { kind: 'lambda', params: { kind: 'record', fields: node.checkName ? [
      { name: node.checkName, type: type.s, optional: false }] : [
      { name: 'recent', type: { kind: 'list', element: type.s }, optional: false },
      { name: 'iteration', type: parseType('Num'), optional: false }] },
      returns: node.checkName ? parseType('Bool') : parseType('LoopVerdict') };
  }
  throw new Error(`unknown part ${part}`);
}

export function buildPending(raw: unknown, env = new TypeEnv(), path = ''): Pending {
  const key = wrapper(raw);
  if (!key || key === 'invalid') return reject(path, 'type-mismatch', 'a single pending wrapper');
  const body = (raw as Record<string, unknown>)[key];
  if (!plain(body)) return reject(path, 'type-mismatch', `a mapping for ${key}`);
  const typesSrc = (body.types ?? {}) as Record<string, string>;
  const types = Object.fromEntries(Object.entries(typesSrc).map(([name, text]) => [name, parseType(text)]));
  const inner = env.child(types);
  if (typeof body.type !== 'string') return reject(path, 'type-mismatch', `${key} with a type`);
  const type = parseType(body.type);
  inner.checkNames(type);
  for (const localType of Object.values(types)) inner.checkNames(localType);
  const expected = key.slice(1);
  if (type.kind !== expected) return reject(path, 'type-mismatch', `a ${expected} type`, formatType(type));
  const common = { type, types, typesSrc, status: (body.status ?? 'unreduced') as Status,
    note: String(body.note ?? ''), attempts: 0, steps: 0 };
  if (key === '$lambda' && type.kind === 'lambda') {
    const hasInstructions = Object.hasOwn(body, 'instructions'), hasCode = Object.hasOwn(body, 'code');
    if (hasInstructions === hasCode) return reject(path, 'type-mismatch', 'exactly one of instructions / code');
    const text = body[hasInstructions ? 'instructions' : 'code'];
    if (typeof text !== 'string') return reject(path, 'type-mismatch', 'Text body');
    const node: LambdaNode = { ...common, nodeKind: 'lambda', kind: hasInstructions ? 'instructions' : 'code',
      engine: String(body.engine ?? 'quickjs-isolated'), body: text && !text.endsWith('\n') ? text + '\n' : text,
      args: {}, return: MISSING, effects: (body.effects ?? []) as string[], journal: (body.effects_journal ?? []) as unknown[],
      let: {}, letTypes: {}, codebase: (body.codebase ?? {}) as Record<string, unknown>,
      functionName: String(body.function ?? ''), marks: (body.marks ?? {}) as Record<number, string>, fnCopies: {} };
    for (const [name, value] of Object.entries((body.args ?? {}) as Record<string, unknown>)) {
      const field = type.params.fields.find(f => f.name === name);
      if (!field) return reject(`${path}/args/${name}`, 'unknown-field');
      node.args[name] = coerce(value, field.type, inner, `${path}/args/${name}`);
    }
    if (Object.hasOwn(body, 'return')) node.return = coerce(body.return, type.returns, inner, `${path}/return`);
    for (const [name, text] of Object.entries((body.let_types ?? {}) as Record<string, string>)) {
      node.letTypes[name] = parseType(text);
      if (Object.hasOwn((body.let ?? {}) as object, name))
        node.let[name] = coerce(((body.let ?? {}) as Record<string, unknown>)[name], node.letTypes[name]!, inner, `${path}/let/${name}`);
    }
    return node;
  }
  if (key === '$map' && type.kind === 'map') {
    const node: MapNode = { ...common, nodeKind: 'map', itemName: String(body.item_name ?? 'item'), over: MISSING, fn: MISSING };
    for (const part of ['over', 'fn'] as const) if (part in body) node[part] = coerce(body[part], partType(node, part), inner, `${path}/${part}`);
    return node;
  }
  if (key === '$fold' && type.kind === 'fold') {
    const node: FoldNode = { ...common, nodeKind: 'fold', over: MISSING, init: MISSING, step: MISSING,
      acc: MISSING, at: Number(body.at ?? 0), current: null };
    for (const part of ['over', 'init', 'step'] as const) if (part in body) node[part] = coerce(body[part], partType(node, part), inner, `${path}/${part}`);
    if ('acc' in body) node.acc = coerce(body.acc, type.s, inner, `${path}/acc`);
    return node;
  }
  if (key === '$iterate' && type.kind === 'iterate') {
    const node: IterateNode = { ...common, nodeKind: 'iterate', init: MISSING, step: MISSING, check: MISSING,
      max: MISSING, state: MISSING, iteration: Number(body.iteration ?? 0), recent: [], seenHashes: [], current: null,
      stateName: String(body.state_name ?? 'state'), checkName: String(body.check_name ?? '') };
    for (const part of ['init', 'step', 'check', 'max'] as const) if (part in body) node[part] = coerce(body[part], partType(node, part), inner, `${path}/${part}`);
    if ('state' in body) node.state = coerce(body.state, type.s, inner, `${path}/state`);
    return node;
  }
  return reject(path, 'type-mismatch', `a ${expected} type`);
}

export function loadProgram(raw: unknown): Pending { return buildPending(raw); }

export function problems(value: Value, type: Type, env: TypeEnv, path: string): { holes: Diagnostic[]; pending: string[] } {
  const holes: Diagnostic[] = [], pending: string[] = [];
  function walk(item: Value, current: Type, at: string): void {
    if (item === MISSING) { holes.push({ path: at, code: 'hole', expected: formatType(current) }); return; }
    if (isPending(item)) { pending.push(at); return; }
    const resolved = env.resolve(current);
    if (resolved.kind === 'union') {
      for (const member of resolved.members) {
        try { coerce(item, member, env, at); walk(item, member, at); return; }
        catch (error) { if (!(error instanceof Reject)) throw error; }
      }
    } else if (resolved.kind === 'record' && plain(item)) {
      for (const field of resolved.fields) {
        if (!(field.name in item)) { if (!field.optional) holes.push({ path: `${at}/${field.name}`, code: 'hole', expected: field.name }); }
        else walk(item[field.name] as Value, field.type, `${at}/${field.name}`);
      }
    } else if (resolved.kind === 'list' && Array.isArray(item)) item.forEach((child, i) => walk(child, resolved.element, `${at}/${i}`));
    else if (resolved.kind === 'dict' && plain(item)) for (const [key, child] of Object.entries(item)) walk(child as Value, resolved.element, `${at}/${key}`);
  }
  walk(value, type, path);
  return { holes, pending };
}

export function unboundParts(node: Pending, env: TypeEnv, path: string): Diagnostic[] {
  const inner = env.child(node.types), out: Diagnostic[] = [];
  if (node.nodeKind === 'lambda' && node.type.kind === 'lambda') {
    for (const field of node.type.params.fields) {
      const at = `${path}/args/${field.name}`;
      if (!(field.name in node.args)) { if (!field.optional) out.push({ path: at, code: 'unbound-param', expected: formatType(field.type) }); }
      else out.push(...problems(node.args[field.name]!, field.type, inner, at).holes.map(d => ({ ...d, code: 'unbound-param' })));
    }
  } else {
    const parts = node.nodeKind === 'map' ? ['over', 'fn'] : node.nodeKind === 'fold' ? ['over', 'init', 'step'] : ['init', 'step', 'check', 'max'];
    for (const part of parts) if ((node as unknown as Record<string, Value>)[part] === MISSING)
      out.push({ path: `${path}/${part}`, code: 'unbound-part', expected: formatType(partType(node, part)) });
  }
  return out;
}

export function dump(value: Value): unknown {
  if (value === MISSING) return null;
  if (Array.isArray(value)) return value.map(dump);
  if (value && typeof value === 'object' && !isPending(value))
    return Object.fromEntries(Object.entries(value).map(([key, child]) => [key, dump(child)]));
  if (!isPending(value)) return value;
  const body: Record<string, unknown> = { type: formatType(value.type) };
  if (Object.keys(value.typesSrc).length) body.types = value.typesSrc;
  if (value.status !== 'unreduced') body.status = value.status;
  if (value.note) body.note = value.note;
  if (value.nodeKind === 'lambda') {
    body[value.kind] = value.body;
    if (value.engine !== 'quickjs-isolated') body.engine = value.engine;
    if (Object.keys(value.args).length) body.args = dump(value.args);
    if (value.return !== MISSING) body.return = dump(value.return);
    if (value.effects.length) body.effects = value.effects;
    if (value.journal.length) body.effects_journal = value.journal;
    if (Object.keys(value.let).length) { body.let = dump(value.let); body.let_types = Object.fromEntries(Object.entries(value.letTypes).map(([k, t]) => [k, formatType(t)])); }
    if (value.functionName) body.function = value.functionName;
    if (Object.keys(value.marks).length) body.marks = value.marks;
  } else {
    const parts = value.nodeKind === 'map' ? ['over', 'fn'] : value.nodeKind === 'fold' ? ['over', 'init', 'step'] : ['init', 'step', 'check', 'max'];
    for (const part of parts) {
      const item = (value as unknown as Record<string, Value>)[part];
      if (item !== undefined && item !== MISSING) body[part] = dump(item);
    }
    if (value.nodeKind === 'fold') { if (value.acc !== MISSING) body.acc = dump(value.acc); if (value.at) body.at = value.at; }
    if (value.nodeKind === 'iterate') { if (value.state !== MISSING) body.state = dump(value.state); if (value.iteration) body.iteration = value.iteration; }
  }
  return { [`$${value.nodeKind}`]: body };
}
