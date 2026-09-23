/**
 * Playground projects: virtual files edited in a page or an IDE, run through the same compiler and
 * kernel as applications. The root is either a named `.nl` function (called with `inputs`) or a
 * TypeScript entry module whose exported `main(inputs)` is called.
 */
import { NativeTraceRecorder } from '../native/trace.js';
import ts from 'typescript';
import { parseType, type Type } from '../native/types.js';
import { readTypeAliases } from '../native/type-aliases.js';
import { admitNativeTrace, type NativeScenarioContract } from '../native/scenario.js';
import { compileProject, formatDiagnostics } from '../compiler/project.js';
import { loadNamedFunction, NatlangSourceError } from '../runtime/loader.js';
import { namedCallable } from '../runtime/callable.js';
import { compileVirtualProject, virtualProjectFiles, virtualSourceFiles } from '../runtime/virtual-project.js';
import type { InvocationTrace, NatlangRuntime } from '../runtime/runtime.js';

export type PlaygroundProject = {
  schema: 'natlang.playground.project/2'; id: string; name: string; root: string;
  files: Record<string, string>; inputs: Record<string, unknown>; expected?: unknown;
  revision: string; updatedAt: string;
};
export type PlaygroundDiagnostic = { file: string; severity: 'error' | 'warning'; message: string };
export type PlaygroundRun = {
  schema: 'natlang.playground.run/2'; id: string; projectId: string; projectName: string;
  revision: string; source: { root: string; files: Record<string, string> };
  inputs: Record<string, unknown>; startedAt: string; durationMs: number;
  outcome: { kind: 'done' | 'failed'; detail: string }; value: unknown;
  /** Events of the top-level natlang invocation; `invocations` holds every invocation's trace. */
  trace: Record<string, unknown>[]; invocations: InvocationTrace[]; expected?: unknown; correct?: boolean;
  model?: { id: string; diagnostics?: Record<string, unknown>; turns?: Array<Record<string, unknown>> };
};
export type TraceFrame = { cursor: number; event: Record<string, unknown> | null;
  state: Record<string, unknown> | null; actions: Record<string, unknown>[];
  effects: Record<string, unknown>[]; activeCalls: Record<string, unknown>[] };

const ROOT = '/project';
const pathPattern = /^(?!\/)(?!.*(?:^|\/)\.\.(?:\/|$))[A-Za-z0-9_./-]+\.(?:nl|ts|json|md)$/;
const id = () => globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`;

export function validProjectPath(path: string): boolean { return pathPattern.test(path) && !path.includes('//'); }

export function newPlaygroundProject(name: string, root: string, files: Record<string, string>,
  inputs: Record<string, unknown> = {}, expected?: unknown): PlaygroundProject {
  const project: PlaygroundProject = { schema: 'natlang.playground.project/2', id: id(), name, root,
    files: structuredClone(files), inputs: structuredClone(inputs), revision: id(), updatedAt: new Date().toISOString() };
  if (expected !== undefined) project.expected = structuredClone(expected);
  assertPlaygroundProject(project);
  return project;
}

export function assertPlaygroundProject(value: unknown): asserts value is PlaygroundProject {
  const p = value as PlaygroundProject;
  if (!p || p.schema !== 'natlang.playground.project/2' || typeof p.id !== 'string' || !p.id ||
      typeof p.name !== 'string' || !p.name.trim() || typeof p.revision !== 'string' || !p.revision ||
      !p.files || typeof p.files !== 'object' || Array.isArray(p.files) || !validProjectPath(p.root) ||
      !/\.(?:nl|ts)$/.test(p.root) || !Object.hasOwn(p.files, p.root) ||
      Object.entries(p.files).some(([path, body]) => !validProjectPath(path) || typeof body !== 'string') ||
      !p.inputs || typeof p.inputs !== 'object' || Array.isArray(p.inputs))
    throw new TypeError('invalid natlang playground project');
}

export function editPlaygroundProject(project: PlaygroundProject,
  update: Partial<Pick<PlaygroundProject, 'name' | 'root' | 'files' | 'inputs' | 'expected'>>): PlaygroundProject {
  const next = { ...structuredClone(project), ...structuredClone(update), revision: id(), updatedAt: new Date().toISOString() };
  assertPlaygroundProject(next);
  return next;
}

const sourceFiles = (files: Record<string, string>) => virtualSourceFiles(files, ROOT);

/** Check the project: named functions, callable folders, and TypeScript, without running anything. */
export function validatePlaygroundProject(project: PlaygroundProject): PlaygroundDiagnostic[] {
  assertPlaygroundProject(project);
  if (project.root.endsWith('.nl')) {
    try { loadNamedFunction(`${ROOT}/${project.root}`, sourceFiles(project.files)); return []; }
    catch (error) {
      return [{ file: error instanceof NatlangSourceError ? error.path.replace(`${ROOT}/`, '') : project.root, severity: 'error',
        message: error instanceof Error ? error.message : String(error) }];
    }
  }
  const result = compileProject({ project: ROOT, files: virtualProjectFiles(project.files, ROOT), emit: false, write: false, module: 'commonjs',
    surfaceSpecifiers: ['@natlang/browser', '@natlang/node'] });
  return result.diagnostics.map(item => ({ file: item.file, severity: item.severity, message: item.message }));
}

function sameValue(a: unknown, b: unknown): boolean {
  const stable = (value: unknown): unknown => Array.isArray(value) ? value.map(stable) :
    value && typeof value === 'object' ? Object.fromEntries(Object.entries(value)
      .sort(([x], [y]) => x.localeCompare(y)).map(([key, item]) => [key, stable(item)])) : value;
  return JSON.stringify(stable(a)) === JSON.stringify(stable(b));
}

/**
 * The declared inputs of a root, for building input forms: a `.nl` function's arguments, or the
 * fields of a TypeScript entry's `main(input: { ... })` parameter (types from a sibling `types.ts`).
 */
export function projectSignature(files: Record<string, string>, path: string):
  { types: Record<string, Type>; fields: { name: string; type: Type; optional: boolean }[]; returns: Type } {
  const parse = (types: Record<string, string>) => Object.fromEntries(Object.entries(types).map(([name, text]) => [name, parseType(text)]));
  if (path.endsWith('.nl')) {
    const record = loadNamedFunction(`${ROOT}/${path}`, sourceFiles(files));
    return { types: parse(record.types), returns: parseType(record.returns),
      fields: Object.entries(record.args).map(([raw, text]) => ({ name: raw.replace(/\?$/, ''), optional: raw.endsWith('?'), type: parseType(text) })) };
  }
  const source = ts.createSourceFile(path, files[path] ?? '', ts.ScriptTarget.ES2022, true);
  const main = source.statements.find((statement): statement is ts.FunctionDeclaration => ts.isFunctionDeclaration(statement) &&
    statement.name?.text === 'main' && !!ts.getModifiers(statement)?.some(modifier => modifier.kind === ts.SyntaxKind.ExportKeyword));
  const input = main?.parameters[0]?.type;
  if (!main || (input && !ts.isTypeLiteralNode(input))) throw new Error(`${path} does not export main(input: { ... })`);
  const typesPath = `${path.includes('/') ? path.slice(0, path.lastIndexOf('/') + 1) : ''}types.ts`;
  let returns = main.type?.getText(source) ?? 'null';
  const promised = /^Promise<([\s\S]*)>$/.exec(returns);
  if (promised) returns = promised[1]!;
  return { types: parse(files[typesPath] ? readTypeAliases(files[typesPath]!) : {}), returns: parseType(returns),
    fields: (input?.members ?? []).flatMap(member => ts.isPropertySignature(member) ? [{ name: member.name.getText(source),
      optional: !!member.questionToken, type: parseType(member.type?.getText(source) ?? 'string') }] : []) };
}

/**
 * The callable entry of a project file, taking a record of named inputs. A `.nl` function receives its
 * arguments from the record by name; a TypeScript module exports `main(inputs)`, and TypeScript entries
 * need the runtime namespace their `@natlang/browser` imports resolve to.
 */
export function projectEntry(files: Record<string, string>, path: string,
  runtimeNamespace?: Record<string, unknown>): (inputs: Record<string, unknown>) => Promise<unknown> {
  if (path.endsWith('.nl')) {
    const record = loadNamedFunction(`${ROOT}/${path}`, sourceFiles(files));
    const fn = namedCallable(record.name, record);
    return async inputs => fn(...Object.keys(record.args).map(name => inputs[name.replace(/\?$/, '')]));
  }
  if (!runtimeNamespace) throw new Error('running a TypeScript entry needs the runtime namespace');
  const compiled = compileVirtualProject({ files, root: ROOT }, runtimeNamespace);
  if (!compiled.ok) throw new Error(formatDiagnostics(compiled.diagnostics));
  const main = compiled.require(path).main;
  if (typeof main !== 'function') throw new Error(`${path} does not export main()`);
  return async inputs => main(inputs);
}

/** Pin source and inputs before execution; later edits cannot alter this run's identity. */
export async function runPlaygroundProject(runtime: NatlangRuntime, project: PlaygroundProject,
  options: { signal?: AbortSignal; model?: PlaygroundRun['model']; runtimeNamespace?: Record<string, unknown> } = {}): Promise<PlaygroundRun> {
  const snapshot = structuredClone(project);
  const diagnostics = validatePlaygroundProject(snapshot);
  if (diagnostics.some(item => item.severity === 'error'))
    throw new Error(`Source has ${diagnostics.length} diagnostic${diagnostics.length === 1 ? '' : 's'}:\n` +
      diagnostics.map(item => `${item.file}: ${item.message}`).join('\n'));
  const started = performance.now(), startedAt = new Date().toISOString();
  const traces: InvocationTrace[] = [];
  let value: unknown, outcome: PlaygroundRun['outcome'];
  try {
    value = await runtime.run(async () => projectEntry(snapshot.files, snapshot.root, options.runtimeNamespace)(snapshot.inputs),
      { signal: options.signal, trace: trace => { traces.push(trace); } });
    outcome = { kind: 'done', detail: '' };
  } catch (error) { value = null; outcome = { kind: 'failed', detail: error instanceof Error ? error.message : String(error) }; }
  const record: PlaygroundRun = { schema: 'natlang.playground.run/2', id: id(), projectId: snapshot.id, projectName: snapshot.name,
    revision: snapshot.revision, source: { root: snapshot.root, files: snapshot.files }, inputs: snapshot.inputs, startedAt,
    durationMs: Math.round(performance.now() - started), outcome, value: value ?? null,
    trace: traces.find(trace => trace.parentCallId === null)?.events ?? [], invocations: traces };
  if (snapshot.expected !== undefined) {
    record.expected = snapshot.expected;
    record.correct = outcome.kind === 'done' && sameValue(value, snapshot.expected);
  }
  if (options.model) record.model = structuredClone(options.model);
  return record;
}

/** Pure, read-only trace navigation. No source or recorded effect is executed. */
export function traceFrame(events: Record<string, unknown>[], cursor: number): TraceFrame {
  const end = Math.max(-1, Math.min(Math.trunc(cursor), events.length - 1));
  let state: Record<string, unknown> | null = null;
  const actions: Record<string, unknown>[] = [], effects: Record<string, unknown>[] = [];
  const active = new Map<string, Record<string, unknown>>();
  for (let index = 0; index <= end; index++) {
    const event = events[index]!;
    if (event.kind === 'state') state = event;
    if (event.kind === 'action') actions.push(event);
    if (event.kind === 'effect') effects.push(event);
    if (event.kind === 'invocation' && typeof event.call_id === 'string') {
      if (event.phase === 'start') active.set(event.call_id, event);
      if (event.phase === 'end') active.delete(event.call_id);
    }
  }
  return { cursor: end, event: events[end] ?? null, state, actions, effects, activeCalls: [...active.values()] };
}

/** Admission uses captured observations of one invocation and never calls a model, evaluator, or host effect. */
export function admitPlaygroundRun(run: PlaygroundRun, contract: NativeScenarioContract): Record<string, unknown> {
  if (run.schema !== 'natlang.playground.run/2' || !Array.isArray(run.trace)) throw new TypeError('invalid playground run');
  if (!run.trace.length) throw new Error('the run made no natural-language call, so there is no trace to admit');
  return admitNativeTrace(NativeTraceRecorder.fromEvents(run.trace), contract);
}
