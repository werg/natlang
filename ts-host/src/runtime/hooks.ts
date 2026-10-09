/** What the invocation kernel gives each interpreter run: callables, inline lambdas, iteration, guards, analysis. */
import type { NativeRuntimeHooks, NativeSession } from '../native/runtime.js';
import { formatType } from '../native/types.js';
import { analyzeEvalSnippet, checkEvalFields, type EvalImport, type EvalScopeDeclarations } from '../compiler/eval-check.js';
import { guard } from './context.js';
import { callableTree } from './callable.js';
import { finite, finiteArrayIterator, finiteAsync, inline, type CaptureAccessors } from './lowered.js';
import { iterateOn } from './iterate.js';
import type { ItemRecord } from './loader.js';

export const kernelHooks: NativeRuntimeHooks = {
  callables: (codebase, session) => callableTree(codebase as Record<string, ItemRecord>, session.runtime.frame),
  // Inherited callable namespaces are already in the child's codebase. Capturing them again creates duplicate
  // injected bindings and competes with their declarations in the opening. Ordinary lexical captures stay live.
  inline: (session, plan, values, accessors, origin) => inline(plan, values,
    Object.fromEntries(Object.entries(accessors).filter(([name]) => !Object.hasOwn(session.lam.codebase, name))) as CaptureAccessors, session.lam.codebase,
    undefined, session.runtime.frame, origin),
  iterateOn: (session, step, initial, ...args) => iterateOn(step as never, initial, ...args).inFrame(session.runtime.frame),
  finite: (source, label) => finite(source as Iterable<unknown>, label),
  finiteArrayIterator: (source, method, label) => finiteArrayIterator(source, method, label),
  finiteAsync: (source, label) => finiteAsync(source, label),
  guard: (id, fn, args) => guard(id, fn, args),
  analyze: (session, source) => analyzeEvalSnippet(source, evalDeclarations(session)),
  checkFields: (session, source) => checkEvalFields(source, evalDeclarations(session)),
};

function importOf(name: string, record: ItemRecord): EvalImport {
  const children = Object.entries(record.codebase).map(([child, item]) => importOf(child, item));
  if (record.kind === 'namespace') return { name, params: [], returns: 'null', async: false, kind: 'module', children };
  if (record.kind === 'module') {
    const exports = Object.entries(record.exports).filter(([key, spec]) => key !== 'default' && spec.kind === 'function')
      .map(([key, spec]) => ({ name: key, params: Object.entries(spec.kind === 'function' ? spec.args : {}).map(([raw, type]) =>
        ({ name: raw.replace(/\?$/, ''), type, optional: raw.endsWith('?') })), returns: spec.kind === 'function' ? spec.returns : 'null',
        async: spec.kind === 'function' && spec.async, kind: 'TypeScript' as const, children: [] }));
    const main = record.exports.default;
    if (main?.kind === 'function') return { name, params: Object.entries(main.args).map(([raw, type]) =>
      ({ name: raw.replace(/\?$/, ''), type, optional: raw.endsWith('?') })), returns: main.returns, async: main.async,
      kind: 'TypeScript', children: [...exports, ...children] };
    return { name, params: [], returns: 'null', async: false, kind: 'module', children: [...exports, ...children] };
  }
  const params = Object.entries(record.args).map(([raw, type]) => ({ name: raw.replace(/\?$/, ''), type, optional: raw.endsWith('?') }));
  if (record.subtype === 'directory-reducer') params.unshift({ name: 'folder', type: 'Folder', optional: false });
  return { name, params, returns: record.returns, async: true,
    kind: record.subtype === 'directory-reducer' ? 'directory reducer' : 'natural language', children };
}

/** Describe a session's typed scope for the eval checker. */
export function evalDeclarations(session: NativeSession): EvalScopeDeclarations {
  const lam = session.lam;
  const codebase = lam.codebase as Record<string, ItemRecord>;
  const types: Record<string, string> = { ...lam.typesSrc, ...session.analysisTypeAliases() };
  // Items of the folder may each define a type of the same name differently (two modules' own `Member`). The checker
  // has one namespace, so such a name cannot stand for either definition: it is declared as `any`.
  const own = new Set(Object.keys(types)), conflicting = new Set<string>();
  const collect = (level: Record<string, ItemRecord>) => {
    for (const record of Object.values(level)) {
      if ('types' in record) for (const [name, text] of Object.entries(record.types)) {
        if (!Object.hasOwn(types, name)) types[name] = text;
        else if (!own.has(name) && types[name] !== text) conflicting.add(name);
      }
      collect(record.codebase);
    }
  };
  collect(codebase);
  for (const name of conflicting) types[name] = 'any';
  const inputs = lam.type.kind === 'lambda' ?
    lam.type.params.fields.map(field => ({ name: field.name, type: formatType(field.type) })) : [];
  // A directory reducer's transaction injects a real Folder capability. Declaring
  // it as opaque `any` erased folder.file()'s FileHandle type when inferring the
  // parameters of an inline NL child. Use the intrinsic contract, just as for a
  // declared Folder parameter; never infer capabilities from a value's shape.
  if (lam.projectTransaction && !inputs.some(input => input.name === 'folder'))
    inputs.push({ name: 'folder', type: 'Folder' });
  return { types, scopeIdentity: session.runtime.options.seedId ?? String(session.runtime.frame?.adHocDepth ?? 0),
    inputs,
    locals: Object.entries(lam.letTypes).filter(([name]) => Object.hasOwn(lam.let, name))
      .map(([name, type]) => ({ name, type: formatType(type), mutable: true })),
    captures: Object.values(lam.captures ?? {}).map(cell => ({ name: cell.name, type: cell.type, mutable: cell.mutable })),
    imports: Object.entries(codebase).map(([name, record]) => importOf(name, record)),
    services: Object.keys(session.runtime.services),
    serviceDeclarations: session.runtime.declarations,
    returns: lam.type.kind === 'lambda' ? formatType(lam.type.returns) : undefined,
    opaque: [] };
}
