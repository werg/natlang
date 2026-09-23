/**
 * IDE workbench over a playground project: revisioned text edits, checks, pinned runs with retained
 * traces, read-only trace inspection, scenario evaluation, and natlang-composed views rendered to
 * escaped HTML. Invalid source stays editable; stepping through a trace never re-executes anything.
 */
import { createHash } from 'node:crypto';
import * as natlang from '@natlang/node';
import { newPlaygroundProject, runPlaygroundProject, traceFrame, validatePlaygroundProject,
  type NatlangRuntime, type PlaygroundProject, type PlaygroundRun } from '@natlang/node';
import interpret from './interpret.nl';
import describe from './describe.nl';
import type { CheckReport, EditorSnapshot, EditorView, EditPatch, TraceView } from './types.js';

export type * from './types.js';
export type EditReport = { status: 'edited' | 'stale' | 'rejected', revision: string, detail: string };
export type RunReport = { status: 'done' | 'failed' | 'invalid-source', run_id: string, revision: string, value_text: string,
  trace_events: number, detail: string };

const hash = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const escape = (value: unknown) => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');

export class IdeWorkbench {
  current: string;
  private readonly revisions = new Map<string, Record<string, string>>();
  private readonly runs = new Map<string, PlaygroundRun & { sourceRevision: string }>();
  private readonly datasets = new Map<string, { id: string, revision: string, inputs: Record<string, unknown>, expected: unknown }>();
  private readonly events: Record<string, unknown>[] = [];

  constructor(files: Record<string, string>, readonly root: string, private readonly runtime: NatlangRuntime) {
    this.current = hash(files);
    this.revisions.set(this.current, structuredClone(files));
  }

  private project(revision: string, inputs: Record<string, unknown> = {}): PlaygroundProject {
    const files = this.revisions.get(revision);
    if (!files) throw new Error('unknown source revision');
    return newPlaygroundProject('ide', this.root, files, inputs);
  }

  snapshot(revision = this.current): EditorSnapshot {
    const files = this.revisions.get(revision);
    if (!files) throw new Error('unknown source revision');
    return { revision, root: this.root, files: Object.entries(files).map(([name, source]) => ({ name, source })) };
  }

  edit({ name, start, end, text, expected_revision }: EditPatch): EditReport {
    if (expected_revision !== this.current) return { status: 'stale', revision: this.current, detail: 'source changed' };
    const files = structuredClone(this.revisions.get(this.current)!), source = files[name];
    if (typeof source !== 'string' || !Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || start > end ||
        end > source.length || typeof text !== 'string') return { status: 'rejected', revision: this.current, detail: 'invalid edit range' };
    files[name] = source.slice(0, start) + text + source.slice(end);
    const revision = hash(files);
    this.revisions.set(revision, files); this.current = revision;
    this.events.push({ operation: 'ide.edit', name, revision, start, end });
    return { status: 'edited', revision, detail: '' };
  }

  check(revision = this.current): CheckReport {
    if (!this.revisions.has(revision)) return { status: 'invalid', revision, detail: 'unknown revision' };
    const diagnostics = validatePlaygroundProject(this.project(revision)).filter(item => item.severity === 'error');
    return diagnostics.length ? { status: 'invalid', revision, detail: diagnostics.map(item => `${item.file}: ${item.message}`).join('\n') } :
      { status: 'checked', revision, detail: '' };
  }

  /** Run a pinned revision; the trace is kept for inspection. */
  async run(inputs: Record<string, unknown>, revision = this.current): Promise<RunReport> {
    const checked = this.check(revision);
    if (checked.status !== 'checked')
      return { status: 'invalid-source', run_id: '', revision, value_text: '', trace_events: 0, detail: checked.detail };
    const run = await runPlaygroundProject(this.runtime, this.project(revision, inputs), { runtimeNamespace: natlang });
    this.runs.set(run.id, { ...run, sourceRevision: revision });
    this.events.push({ operation: 'ide.run', run_id: run.id, revision, status: run.outcome.kind });
    return { status: run.outcome.kind, run_id: run.id, revision, detail: run.outcome.detail,
      value_text: run.outcome.kind === 'done' ? JSON.stringify(run.value) : '', trace_events: run.trace.length };
  }

  inspect(runId: string, index: number): TraceView {
    const run = this.runs.get(runId);
    if (!run || !Number.isSafeInteger(index) || index < 0 || index >= run.trace.length) throw new Error('unknown run or trace index');
    return { run_id: runId, revision: run.sourceRevision, index, total: run.trace.length,
      event_json: JSON.stringify(traceFrame(run.trace, index).event) };
  }

  addScenario({ id, revision = this.current, inputs, expected }: { id: string, revision?: string, inputs: Record<string, unknown>, expected: unknown }) {
    if (!/^[A-Za-z][A-Za-z0-9_-]*$/.test(id) || this.datasets.has(id) || !this.revisions.has(revision))
      throw new Error('invalid or duplicate scenario');
    const scenario = { id, revision, inputs: structuredClone(inputs), expected: structuredClone(expected) };
    this.datasets.set(id, scenario);
    this.events.push({ operation: 'ide.scenario', id, revision });
    return { id, revision, identity: hash(scenario) };
  }

  async evaluate(ids: string[]) {
    const results = [];
    for (const id of ids) {
      const scenario = this.datasets.get(id);
      if (!scenario) throw new Error(`unknown scenario: ${id}`);
      const run = await this.run(scenario.inputs, scenario.revision);
      results.push({ id, run_id: run.run_id, revision: run.revision, status: run.status,
        matches: run.status === 'done' && run.value_text === JSON.stringify(scenario.expected) });
    }
    return { dataset_identity: hash(ids.map(id => this.datasets.get(id))), results, passed: results.every(row => row.matches) };
  }

  render(view: EditorView): string {
    if (!view || !view.title || !Array.isArray(view.panels)) throw new Error('invalid view');
    return '<!doctype html><meta charset="utf-8"><title>' + escape(view.title) + '</title><main><h1>' + escape(view.title) + '</h1>' +
      view.panels.map(panel => '<section><h2>' + escape(panel.heading) + '</h2><pre>' + escape(panel.body) + '</pre></section>').join('') +
      '</main>\n';
  }

  drainEvents(): Record<string, unknown>[] { return this.events.splice(0); }
}

/** Interpret one editor request as an exact patch against the current revision. */
export async function requestEdit(ide: IdeWorkbench, request: string): Promise<EditReport> {
  return ide.edit(await interpret(request, ide.snapshot()));
}

/** Compose and render an editor view around one recorded trace event. */
export async function viewTrace(ide: IdeWorkbench, runId: string, index: number): Promise<string> {
  const snapshot = ide.snapshot();
  return ide.render(await describe(snapshot, ide.check(snapshot.revision), ide.inspect(runId, index)));
}
