/** Revisioned source, child runs, trace inspection and scenario evaluation. */
import { createHash, randomUUID } from 'node:crypto';
import { Script } from 'node:vm';
import { NativeSourceWorkspace } from '../ts-host/dist/index.js';

const hash = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const escape = value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');

export class IdeWorkbench {
  constructor(definitions, root, { modelTurn } = {}) {
    this.root = root; this.modelTurn = modelTurn;
    this.revisions = new Map(); this.runs = new Map(); this.datasets = new Map();
    this.events = [];
    const revision = hash(definitions);
    this.revisions.set(revision, structuredClone(definitions));
    this.current = revision;
  }

  snapshot(revision = this.current) {
    const definitions = this.revisions.get(revision);
    if (!definitions) throw new Error('unknown source revision');
    return { revision, root: this.root,
      files: Object.entries(definitions).map(([name, row]) => ({ name,
        kind: row.code === undefined ? 'natlang' : 'crisp',
        source: row.code ?? row.instructions, returns: row.returns })) };
  }

  edit({ name, start, end, text, expected_revision }) {
    if (expected_revision !== this.current) return { status: 'stale',
      revision: this.current, detail: 'source changed' };
    const definitions = structuredClone(this.revisions.get(this.current));
    const row = definitions[name], field = row?.code === undefined ? 'instructions' : 'code';
    const source = row?.[field];
    if (typeof source !== 'string' || !Number.isSafeInteger(start) ||
        !Number.isSafeInteger(end) || start < 0 || start > end || end > source.length ||
        typeof text !== 'string') return { status: 'rejected',
      revision: this.current, detail: 'invalid edit range' };
    row[field] = source.slice(0, start) + text + source.slice(end);
    const revision = hash(definitions);
    this.revisions.set(revision, definitions); this.current = revision;
    this.events.push({ operation: 'ide.edit', name, revision, start, end });
    return { status: 'edited', revision, detail: '' };
  }

  check(revision = this.current) {
    const definitions = this.revisions.get(revision);
    if (!definitions) return { status: 'invalid', revision, source_revision: '',
      detail: 'unknown revision' };
    try {
      const workspace = new NativeSourceWorkspace(definitions, this.root);
      for (const [name, row] of Object.entries(definitions))
        if (typeof row.code === 'string' && !row.engine)
          try { new Script(`(async function(args) { ${row.code}\n })`); }
          catch (error) { throw new Error(`${name}: ${error.message}`); }
      return { status: 'checked', revision, source_revision: workspace.revision, detail: '' };
    } catch (error) {
      return { status: 'invalid', revision, source_revision: '', detail: String(error) };
    }
  }

  async run(inputs, revision = this.current) {
    const checked = this.check(revision);
    if (checked.status !== 'checked') return { status: 'invalid-source',
      run_id: '', revision, source_revision: '', value_text: '',
      trace_events: 0, detail: checked.detail };
    const workspace = new NativeSourceWorkspace(this.revisions.get(revision), this.root);
    const result = await workspace.invoke(this.root, inputs, { modelTurn: this.modelTurn });
    const runId = randomUUID();
    const record = { run_id: runId, revision, source_revision: result.source_revision,
      status: result.outcome, value_text: result.outcome === 'done' ?
        JSON.stringify(result.value) : '', trace_events: result.trace.length,
      detail: '' };
    this.runs.set(runId, { ...record, trace: result.trace });
    this.events.push({ operation: 'ide.run', run_id: runId, revision,
      source_revision: result.source_revision, status: result.outcome });
    return record;
  }

  inspect(runId, index) {
    const run = this.runs.get(runId);
    if (!run || !Number.isSafeInteger(index) || index < 0 || index >= run.trace.length)
      throw new Error('unknown run or trace index');
    const event = run.trace[index];
    return { run_id: runId, source_revision: run.source_revision,
      index, total: run.trace.length, event_json: JSON.stringify(event) };
  }

  addScenario({ id, revision = this.current, inputs, expected }) {
    if (!/^[A-Za-z][A-Za-z0-9_-]*$/.test(id) || this.datasets.has(id) ||
        !this.revisions.has(revision)) throw new Error('invalid or duplicate scenario');
    const scenario = { id, revision, inputs: structuredClone(inputs),
      expected: structuredClone(expected) };
    this.datasets.set(id, scenario);
    this.events.push({ operation: 'ide.scenario', id, revision });
    return { id, revision, identity: hash(scenario) };
  }

  async evaluate(ids) {
    const results = [];
    for (const id of ids) {
      const scenario = this.datasets.get(id);
      if (!scenario) throw new Error(`unknown scenario: ${id}`);
      const run = await this.run(scenario.inputs, scenario.revision);
      results.push({ id, run_id: run.run_id, source_revision: run.revision,
        status: run.status, matches: run.status === 'done' &&
          run.value_text === JSON.stringify(scenario.expected) });
    }
    return { dataset_identity: hash(ids.map(id => this.datasets.get(id))),
      results, passed: results.every(row => row.matches) };
  }

  render(view) {
    if (!view || !view.title || !Array.isArray(view.panels))
      throw new Error('invalid view');
    return '<!doctype html><meta charset="utf-8"><title>' + escape(view.title) +
      '</title><main><h1>' + escape(view.title) + '</h1>' +
      view.panels.map(panel => '<section><h2>' + escape(panel.heading) + '</h2><pre>' +
        escape(panel.body) + '</pre></section>').join('') + '</main>\n';
  }

  drainEvents() { return this.events.splice(0); }
}
