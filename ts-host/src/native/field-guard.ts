/**
 * Field guards: a value with a declared record type reaches eval code behind a view that refuses to read a field the
 * type does not declare and the value does not have. Such a read would be `undefined`, which code then takes for
 * "none" and acts on (an abort that answers no calls because it read `entry.content` instead of
 * `entry.model[0].content`). The static check (compiler/eval-check.ts checkEvalFields) catches the reads it can see;
 * this view catches the rest, such as reads through a callback parameter annotated `any`.
 *
 * Fields a value has are always readable, declared or not (a provider value keeps fields its type does not list), and
 * so are optional fields it lacks. Writes go to the value. The view reads like the value, so copying it out of eval
 * (hostCopy, JSON) yields plain data.
 */
import ts from 'typescript';
import { evalTypeDeclarations } from './eval-types.js';
import { formatType, parseType, type Type, type TypeEnv } from './types.js';

/** Keys any object may be asked for without declaring them: protocol probes of await, JSON and inspection. */
const PROBES = new Set(['then', 'toJSON', 'constructor', 'asymmetricMatch', 'nodeType', 'length']);

type Shape = { kind: 'record'; fields: Map<string, Type> } | { kind: 'list' | 'dict'; element: Type };

/** How a guard reads types: a type's definition, and its name as the code's author knows it. */
export type GuardTypes = { resolve(type: Type): Type; show(type: Type): string };
/** The call's own types. */
export const callTypes = (env: TypeEnv): GuardTypes => ({ resolve: type => env.resolve(type), show: formatType });

/** What a value of `type` may hold, as far as a guard can tell, or undefined when it holds anything. */
function shapeOf(type: Type, types: GuardTypes): Shape | undefined {
  const resolved = types.resolve(type);
  if (resolved.kind === 'record') return { kind: 'record', fields: new Map(resolved.fields.map(field => [field.name, field.type])) };
  if (resolved.kind === 'list' || resolved.kind === 'dict') return { kind: resolved.kind, element: resolved.element };
  if (resolved.kind !== 'union') return;
  // Nullish members do not hold fields; every other member must be a record, so the fields are those of any of them.
  const members = resolved.members.map(member => types.resolve(member))
    .filter(member => !(member.kind === 'prim' && member.name === 'null'));
  if (!members.length || members.some(member => member.kind !== 'record')) return;
  const fields = new Map<string, Type[]>();
  for (const member of members as Extract<Type, { kind: 'record' }>[])
    for (const field of member.fields) fields.set(field.name, [...fields.get(field.name) ?? [], field.type]);
  return { kind: 'record', fields: new Map([...fields].map(([name, types]) => [name, types.length === 1 ? types[0]! : { kind: 'union', members: types }])) };
}

/** `type` without its nullish members: the type of a value that is there. */
function present(type: Type): Type {
  if (type.kind !== 'union') return type;
  const members = type.members.filter(member => !(member.kind === 'prim' && member.name === 'null'));
  return members.length === 1 ? members[0]! : { ...type, members };
}

/**
 * Views by value and declared type (as text). One value seen as one type has one view, however it is reached, so data
 * with cycles stays finite for whatever walks it (hostCopy's seen-set, JSON, inspection): a fresh view per path would
 * never meet itself again.
 */
const views = new WeakMap<object, Map<string, object>>();

/** `value` as eval code should see it given its declared `type`; `path` names it in errors (`facts.task`). */
export function guardFields(value: unknown, type: Type, types: GuardTypes, path: string): unknown {
  if (!value || typeof value !== 'object') return value;
  const key = formatType(type);
  const known = views.get(value)?.get(key);
  if (known) return known;
  const view = guardedView(value, type, types, path);
  if (view !== value) {
    let byType = views.get(value);
    if (!byType) views.set(value, byType = new Map());
    byType.set(key, view as object);
  }
  return view;
}

function guardedView(value: object, type: Type, types: GuardTypes, path: string): unknown {
  // Plain data of any realm only: live host objects keep their own behavior.
  const prototype = Object.getPrototypeOf(value);
  if (!Array.isArray(value) && prototype !== null && Object.getPrototypeOf(prototype) !== null) return value;
  let shape: Shape | undefined;
  try { shape = shapeOf(type, types); } catch { return value; }
  if (!shape || (shape.kind === 'list') !== Array.isArray(value)) return value;
  const target = value as Record<PropertyKey, unknown>;
  const childType = (key: string): Type | undefined => shape!.kind === 'record' ? shape!.fields.get(key) :
    shape!.kind === 'dict' || /^\d+$/.test(key) ? shape!.element : undefined;
  const childPath = (key: string) => /^\d+$/.test(key) ? `${path}[${key}]` : `${path}.${key}`;
  // The view's own target is an empty shell, so the value may be frozen: a frozen value's fields would otherwise have
  // to read as themselves, unguarded.
  return new Proxy(Array.isArray(value) ? [] : {}, {
    get(_shell, key) {
      if (typeof key === 'symbol') return Reflect.get(target, key);
      if (!(key in target) && shape!.kind === 'record' && !shape!.fields.has(key) && !PROBES.has(key)) {
        const declared = [...shape!.fields.keys()];
        throw new TypeError(`${path} has no field ${key}: its type ${types.show(present(type))} has ${declared.length ? declared.join(', ') : 'no fields'}. ` +
          'Read the field that holds what you need.');
      }
      const raw = Reflect.get(target, key);
      const inner = childType(key);
      return inner === undefined || !raw || typeof raw !== 'object' ? raw : guardFields(raw, inner, types, childPath(key));
    },
    set: (_shell, key, item) => Reflect.set(target, key, item),
    has: (_shell, key) => key in target,
    deleteProperty: (_shell, key) => Reflect.deleteProperty(target, key),
    ownKeys: () => Reflect.ownKeys(target).filter(key => key !== 'length' || !Array.isArray(target)).concat(Array.isArray(target) ? ['length'] : []),
    getOwnPropertyDescriptor(shell, key) {
      // An array shell's own length is real and must be reported as it is.
      if (Array.isArray(target) && key === 'length') { (shell as unknown[]).length = (target as unknown[]).length; return Reflect.getOwnPropertyDescriptor(shell, key); }
      const descriptor = Reflect.getOwnPropertyDescriptor(target, key);
      return descriptor && { ...descriptor, configurable: true, ...('value' in descriptor ? { writable: true } : {}) };
    },
    defineProperty: (_shell, key, descriptor) => Reflect.defineProperty(target, key, descriptor),
    getPrototypeOf: () => Reflect.getPrototypeOf(target),
  });
}

/**
 * The declared result types of a service's functions, from its declaration (`declare namespace name { … }` or module
 * text), and how to read them: names the declaration defines resolve to its definitions, others in the call's types.
 * A function declared more than once (overloads) or with type parameters has none: its result is not guarded.
 */
export function serviceResultTypes(name: string, declaration: string): { results: Map<string, Type>; own: Record<string, Type> } {
  const results = new Map<string, Type>();
  const file = ts.createSourceFile('service.ts', declaration, ts.ScriptTarget.ES2022, true);
  const own = evalTypeDeclarations(/^\s*declare (?:namespace|const) /.test(declaration) ? declaration : `declare namespace ${name} {\n${declaration}\n}`);
  const seen = new Set<string>();
  const visit = (statements: ts.NodeArray<ts.Statement>) => {
    for (const statement of statements) {
      if (ts.isModuleDeclaration(statement) && statement.body && ts.isModuleBlock(statement.body)) { visit(statement.body.statements); continue; }
      if (!ts.isFunctionDeclaration(statement) || !statement.name) continue;
      const method = statement.name.text;
      if (seen.has(method) || statement.typeParameters?.length || !statement.type) { results.delete(method); seen.add(method); continue; }
      seen.add(method);
      let node: ts.TypeNode = statement.type;
      if (ts.isTypeReferenceNode(node) && ts.isIdentifier(node.typeName) && node.typeName.text === 'Promise' && node.typeArguments?.length === 1) node = node.typeArguments[0]!;
      try { results.set(method, qualifyNames(parseType(node.getText(file)), name, own)); }
      catch { /* a result type the type language cannot express is not guarded */ }
    }
  };
  visit(file.statements);
  return { results, own };
}

/** Types as a service's guards read them: its own names first (shown without the namespace), then the call's. */
export function serviceTypes(name: string, own: Record<string, Type>, env: TypeEnv): GuardTypes {
  const resolve = (type: Type): Type => type.kind === 'name' && Object.hasOwn(own, type.name) ? resolve(own[type.name]!) : env.resolve(type);
  return { resolve, show: type => formatType(type).split(`${name}.`).join('') };
}

/** Names `namespace` defines, as `declared` keys them (`namespace.Name`). */
function qualifyNames(type: Type, namespace: string, declared: Record<string, Type>): Type {
  const child = (inner: Type) => qualifyNames(inner, namespace, declared);
  switch (type.kind) {
    case 'name': return Object.hasOwn(declared, `${namespace}.${type.name}`) ? { ...type, name: `${namespace}.${type.name}` } : type;
    case 'record': return { ...type, fields: type.fields.map(field => ({ ...field, type: child(field.type) })) };
    case 'list': case 'dict': return { ...type, element: child(type.element) };
    case 'union': return { ...type, members: type.members.map(child) };
    default: return type;
  }
}

/** `service` as eval code calls it: results of functions with a declared type come back guarded. */
export function guardedService(service: object, name: string, results: Map<string, Type>, types: GuardTypes): object {
  if (!results.size) return service;
  return new Proxy(service, {
    get(object, property, receiver) {
      const value = Reflect.get(object, property, receiver);
      const type = typeof property === 'string' ? results.get(property) : undefined;
      if (typeof value !== 'function' || !type) return value;
      const path = `${name}.${String(property)}(…)`;
      return (...args: unknown[]) => {
        const result = (value as Function).apply(object, args);
        return result && typeof (result as PromiseLike<unknown>).then === 'function' ?
          Promise.resolve(result).then(settled => guardFields(settled, type, types, path)) : guardFields(result, type, types, path);
      };
    },
  });
}
