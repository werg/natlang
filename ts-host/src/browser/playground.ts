import { loadFunctionFiles } from './source.js';
import { checkTypeScriptBody } from './environment.js';
import type { BrowserNatlangHost, BrowserRunOptions } from './host.js';
import type { BrowserNatlangClient } from './client.js';
import { NativeTraceRecorder } from '../native/trace.js';
import { admitNativeTrace, type NativeScenarioContract } from '../native/scenario.js';

export type PlaygroundProject = {
  schema: 'natlang.playground.project/1'; id: string; name: string; root: string;
  files: Record<string, string>; inputs: Record<string, unknown>; expected?: unknown;
  revision: string; updatedAt: string;
};
export type PlaygroundDiagnostic = { file: string; severity: 'error' | 'warning'; message: string };
export type PlaygroundRun = {
  schema: 'natlang.playground.run/1'; id: string; projectId: string; projectName: string;
  revision: string; source: { root: string; files: Record<string, string> };
  inputs: Record<string, unknown>; startedAt: string; durationMs: number;
  outcome: { kind: string; path: string; detail: string }; value: unknown; emitted: unknown[];
  trace: Record<string, unknown>[]; expected?: unknown; correct?: boolean;
  model?: { id: string; diagnostics?: Record<string, unknown>; turns?: Array<{
    durationMs: number; promptTokens: number | null; cachedTokens: number | null;
    completionTokens: number | null; toolSchemaBytes: number; retries: number;
    tokensPerSecond: number | null }> };
};
export type TraceFrame = { cursor: number; event: Record<string, unknown> | null;
  state: Record<string, unknown> | null; actions: Record<string, unknown>[];
  effects: Record<string, unknown>[]; activeCalls: Record<string, unknown>[] };

const pathPattern = /^(?!\/)(?!.*(?:^|\/)\.\.(?:\/|$))[A-Za-z0-9_./-]+\.(?:nl|ts)$/;
const id = () => globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`;

export function validProjectPath(path: string): boolean { return pathPattern.test(path) && !path.includes('//'); }

export function newPlaygroundProject(name: string, root: string,
  files: Record<string, string>, inputs: Record<string, unknown> = {}, expected?: unknown): PlaygroundProject {
  const project: PlaygroundProject = { schema: 'natlang.playground.project/1', id: id(), name,
    root, files: structuredClone(files), inputs: structuredClone(inputs), revision: id(),
    updatedAt: new Date().toISOString() };
  if (expected !== undefined) project.expected = structuredClone(expected);
  assertPlaygroundProject(project);
  return project;
}

export function assertPlaygroundProject(value: unknown): asserts value is PlaygroundProject {
  const p = value as PlaygroundProject;
  if (!p || p.schema !== 'natlang.playground.project/1' || typeof p.id !== 'string' ||
      !p.id || typeof p.name !== 'string' || !p.name.trim() || typeof p.revision !== 'string' ||
      !p.revision || !p.files || typeof p.files !== 'object' || Array.isArray(p.files) ||
      !validProjectPath(p.root) || !Object.hasOwn(p.files, p.root) ||
      Object.entries(p.files).some(([path, body]) => !validProjectPath(path) || typeof body !== 'string') ||
      !p.inputs || typeof p.inputs !== 'object' || Array.isArray(p.inputs))
    throw new TypeError('invalid natlang playground project');
}

export function editPlaygroundProject(project: PlaygroundProject,
  update: Partial<Pick<PlaygroundProject, 'name' | 'root' | 'files' | 'inputs' | 'expected'>>): PlaygroundProject {
  const next = { ...structuredClone(project), ...structuredClone(update), revision: id(),
    updatedAt: new Date().toISOString() };
  assertPlaygroundProject(next);
  return next;
}

/** Check the linked natlang source graph and parse TypeScript bodies without executing them. */
export function validatePlaygroundProject(project: PlaygroundProject): PlaygroundDiagnostic[] {
  assertPlaygroundProject(project);
  const diagnostics: PlaygroundDiagnostic[] = [];
  try { loadFunctionFiles(project.root, project.files); }
  catch (error) {
    const native = error as { diagnostics?: Array<{ path?: string; code?: string; expected?: string; got?: string }> };
    if (Array.isArray(native.diagnostics)) for (const item of native.diagnostics)
      diagnostics.push({ file: item.path ?? project.root, severity: 'error',
        message: [item.code, item.expected && `expected ${item.expected}`, item.got && `got ${item.got}`]
          .filter(Boolean).join(' · ') });
    else diagnostics.push({ file: project.root, severity: 'error',
      message: error instanceof Error ? error.message : String(error) });
  }
  for (const [file, source] of Object.entries(project.files)) {
    if (!file.endsWith('.ts')) continue;
    const body = /^\s*\/\*---\r?\n[\s\S]*?\r?\n---\*\/\r?\n?([\s\S]*)$/.exec(source)?.[1];
    if (body === undefined) continue; // The source loader reports malformed frontmatter.
    for (const message of checkTypeScriptBody(body))
      diagnostics.push({ file, severity: 'error', message: `TypeScript: ${message}` });
  }
  return diagnostics;
}

function sameValue(a: unknown, b: unknown): boolean {
  const stable = (value: unknown): unknown => Array.isArray(value) ? value.map(stable) :
    value && typeof value === 'object' ? Object.fromEntries(Object.entries(value)
      .sort(([a], [b]) => a.localeCompare(b)).map(([key, item]) => [key, stable(item)])) : value;
  return JSON.stringify(stable(a)) === JSON.stringify(stable(b));
}

/** Pin source and inputs before execution; later edits cannot alter this run's identity. */
export async function runPlaygroundProject(host: BrowserNatlangHost | BrowserNatlangClient, project: PlaygroundProject,
  options: { signal?: AbortSignal; timeoutMs?: number; runOptions?: BrowserRunOptions;
    model?: PlaygroundRun['model']; modelTurn?: Parameters<BrowserNatlangHost['run']>[0]['modelTurn'] } = {}):
  Promise<PlaygroundRun> {
  const snapshot = structuredClone(project);
  const diagnostics = validatePlaygroundProject(snapshot);
  if (diagnostics.some(item => item.severity === 'error'))
    throw new Error(`Source has ${diagnostics.length} diagnostic${diagnostics.length === 1 ? '' : 's'}`);
  const started = performance.now(), startedAt = new Date().toISOString();
  const result = await host.run({ source: { kind: 'files', root: snapshot.root, files: snapshot.files },
    inputs: snapshot.inputs, signal: options.signal, timeoutMs: options.timeoutMs,
    options: options.runOptions, modelTurn: options.modelTurn });
  const record: PlaygroundRun = { schema: 'natlang.playground.run/1', id: result.run_id,
    projectId: snapshot.id, projectName: snapshot.name, revision: snapshot.revision,
    source: { root: snapshot.root, files: snapshot.files }, inputs: snapshot.inputs, startedAt,
    durationMs: Math.round(performance.now() - started), outcome: result.outcome,
    value: result.value, emitted: result.emitted, trace: result.trace };
  if (snapshot.expected !== undefined) {
    record.expected = snapshot.expected;
    record.correct = result.outcome.kind === 'done' && sameValue(result.value, snapshot.expected);
  }
  if (options.model) record.model = structuredClone(options.model);
  else {
    const modelRun = result as Awaited<ReturnType<BrowserNatlangClient['run']>>;
    if (modelRun.model?.id) record.model = structuredClone(modelRun.model);
  }
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
      if (event.phase === 'start' || event.phase === 'enter') active.set(event.call_id, event);
      if (event.phase === 'end' || event.phase === 'exit') active.delete(event.call_id);
    }
  }
  return { cursor: end, event: events[end] ?? null, state, actions, effects,
    activeCalls: [...active.values()] };
}

/** Admission uses captured observations and never calls a model, evaluator, or host effect. */
export function admitPlaygroundRun(run: PlaygroundRun, contract: NativeScenarioContract): Record<string, unknown> {
  if (run.schema !== 'natlang.playground.run/1' || !Array.isArray(run.trace))
    throw new TypeError('invalid playground run');
  return admitNativeTrace(NativeTraceRecorder.fromEvents(run.trace), contract);
}
