/** What the invocation kernel gives each interpreter run: callables, inline lambdas, iteration, guards, analysis. */
import type { NativeRuntimeHooks, NativeSession } from '../native/runtime.js';
import { formatType } from '../native/types.js';
import { analyzeEvalSnippet, type EvalImport, type EvalScopeDeclarations } from '../compiler/eval-check.js';
import { guard } from './context.js';
import { callableTree } from './callable.js';
import { finite, inline, type CaptureAccessors } from './lowered.js';
import { iterateOn } from './iterate.js';
import type { ItemRecord } from './loader.js';

export const kernelHooks: NativeRuntimeHooks = {
  callables: (codebase, session) => callableTree(codebase as Record<string, ItemRecord>, session.runtime.frame),
  inline: (session, plan, values, accessors) => inline(plan, values, accessors as CaptureAccessors, session.lam.codebase,
    undefined, session.runtime.frame),
  iterateOn: (session, step, initial, ...args) => iterateOn(step as never, initial, ...args).inFrame(session.runtime.frame),
  finite: source => finite(source as Iterable<unknown>),
  guard: (id, fn) => guard(id, fn),
  analyze: (session, source) => analyzeEvalSnippet(source, evalDeclarations(session)),
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
  const types: Record<string, string> = { ...lam.typesSrc };
  const collect = (level: Record<string, ItemRecord>) => {
    for (const record of Object.values(level)) {
      if ('types' in record) for (const [name, text] of Object.entries(record.types)) types[name] ??= text;
      collect(record.codebase);
    }
  };
  collect(codebase);
  return { types,
    inputs: lam.type.kind === 'lambda' ? lam.type.params.fields.map(field => ({ name: field.name, type: formatType(field.type) })) : [],
    locals: Object.entries(lam.letTypes).filter(([name]) => Object.hasOwn(lam.let, name))
      .map(([name, type]) => ({ name, type: formatType(type), mutable: true })),
    captures: Object.values(lam.captures ?? {}).map(cell => ({ name: cell.name, type: cell.type, mutable: cell.mutable })),
    imports: Object.entries(codebase).map(([name, record]) => importOf(name, record)),
    services: Object.keys(session.runtime.services),
    result: lam.type.kind === 'lambda' ? formatType(lam.type.returns) : undefined,
    opaque: lam.projectTransaction ? ['folder'] : [] };
}
