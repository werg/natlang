/**
 * A local collaborative wiki. The wiki's logic is natural language (merge.nl, maintain.nl, cells.nl: see
 * DECOMPOSITION.md); this file is the mechanism around it. The host decides on a snapshot (a model call that
 * returns data) and then commits the data in one bounded, synchronous step: transport identity, update coverage,
 * block shape and the atomic publish of a page revision. Pages hold cells: JavaScript cells run in a fresh VM
 * context, natlang cells are functions defined from the cell text. A result that finishes after the page changed
 * is reported stale.
 */
import { createHash } from 'node:crypto';
import vm from 'node:vm';
import { defineNatlang, type FolderHandle, type InvocationTrace, type NatlangRuntime } from '@natlang/node';
import mergePages from './merge.nl';
import maintainPage from './maintain.nl';
import planCells from './cells.nl';
import type { CellPlan, CellRecord, CellRequest, CellResult, Maintenance, MergeDraft, MergeProfile, MergeReport, PreparedMerge,
  Structure, WikiBlock, WikiPage, WikiSettings, WikiUpdate } from './types.js';

export type * from './types.js';

const hash = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const validId = (value: unknown): value is string => typeof value === 'string' && /^[A-Za-z][A-Za-z0-9_-]*$/.test(value);
const sourceRevision = (block: WikiBlock) => hash([block.language, block.returns, block.text]).slice(0, 16);
const CELL_TIMEOUT_MS = 1000;
export const defaultSettings: WikiSettings = { changes: 'natlang', staleness: 'natlang' };

export type WikiOptions = {
  profile: MergeProfile;
  /** Runs the pipelines and natlang cells; create it with `seed: { mode: 'derived', root: profile.seed }` to pin sampling. */
  runtime: NatlangRuntime;
  /** Project files given to natlang cells, read fresh for each run. */
  files?: () => FolderHandle;
  /** Which implementation each hot-path policy uses. */
  settings?: Partial<WikiSettings>;
};

type Output = CellResult & { trace: InvocationTrace[] };

export class WikiWorkspace {
  readonly profile: MergeProfile;
  readonly settings: WikiSettings;
  private readonly runtime: NatlangRuntime;
  private readonly files?: () => FolderHandle;
  private page: WikiPage;
  private readonly outputs = new Map<string, Output>();
  /** Results of the revision before the last merge, until maintenance has judged which of them survive. */
  private readonly retired = new Map<string, Output & { page_revision: string }>();
  private derived: Structure | null = null;
  private readonly events: Record<string, unknown>[] = [];

  constructor(page: { id: string, blocks: WikiBlock[] }, options: WikiOptions) {
    const { profile } = options;
    if (!validId(page.id) || !Array.isArray(page.blocks) || !profile?.model || !profile?.source || !Number.isSafeInteger(profile?.seed))
      throw new Error('invalid page or merge profile');
    this.profile = structuredClone(profile);
    this.settings = { ...defaultSettings, ...options.settings };
    this.runtime = options.runtime;
    this.files = options.files;
    checkBlocks(page.blocks);
    this.page = { id: page.id, blocks: structuredClone(page.blocks), revision: hash(page.blocks) };
  }

  snapshot(): WikiPage { return structuredClone(this.page); }
  structure(): Structure | null { return this.derived && structuredClone(this.derived); }

  /** Merge concurrent updates made against `base`, then maintain the page's structure. */
  async merge(base: WikiPage, updates: WikiUpdate[], profile: MergeProfile = this.profile): Promise<MergeReport> {
    const prepared = this.prepare(base, updates, profile);
    if (!prepared.valid) return { status: 'rejected', page: base, detail: prepared.detail };
    const draft = await this.runtime.run(() => mergePages(base, prepared.updates, this.settings), { name: 'wiki-merge' });
    const report = this.publish(base, prepared, draft, profile);
    if (report.status === 'rejected') return report;
    try {
      const maintenance = await this.maintain(base);
      return { ...report, structure: maintenance.structure, repairs: maintenance.repairs };
    } catch (error) {
      this.events.push({ operation: 'wiki.maintain', page_id: this.page.id, status: 'failed', detail: String(error) });
      return { ...report, detail: `maintenance failed: ${String(error).slice(0, 300)}` };
    }
  }

  /** Derive the current page's structure from the revision `before` and decide which remembered cell results survive. */
  async maintain(before: WikiPage): Promise<Maintenance> {
    const after = this.snapshot();
    const records: CellRecord[] = [...this.retired, ...this.outputs]
      .map(([block_id, output]) => ({ block_id, page_revision: output.page_revision, source_revision: output.source_revision }));
    const maintenance = await this.runtime.run(() => maintainPage(before, after, records, this.settings), { name: 'wiki-maintain' });
    return this.settle(after.revision, maintenance);
  }

  /** The commit of maintenance: keep a retired result only when its cell is fresh and its source is unchanged. */
  settle(revision: string, maintenance: Maintenance): Maintenance {
    if (revision !== this.page.revision || maintenance?.structure?.revision !== revision || !Array.isArray(maintenance.repairs))
      throw new Error('stale or invalid maintenance');
    const fresh = new Set(maintenance.structure.cells.filter(row => row.status === 'fresh').map(row => row.block_id));
    for (const [blockId, output] of this.retired) {
      const block = this.page.blocks.find(row => row.id === blockId && row.kind === 'cell');
      if (block && fresh.has(blockId) && output.source_revision === sourceRevision(block) && !this.outputs.has(blockId))
        this.outputs.set(blockId, { ...output, page_revision: revision });
    }
    this.retired.clear();
    this.derived = structuredClone(maintenance.structure);
    this.events.push({ operation: 'wiki.maintain', page_id: this.page.id, revision, status: 'done',
      fresh: [...fresh].sort(), repairs: maintenance.repairs.map(row => row.id) });
    return structuredClone(maintenance);
  }

  prepare(base: WikiPage, updates: WikiUpdate[], profile: MergeProfile): PreparedMerge {
    const bad = (detail: string): PreparedMerge => ({ valid: false, detail, updates: [], presentation: '' });
    if (JSON.stringify(profile) !== JSON.stringify(this.profile)) return bad('merge profile mismatch');
    if (base.id !== this.page.id || base.revision !== this.page.revision) return bad('stale base page');
    if (!Array.isArray(updates) || updates.length > 64) return bad('invalid update batch');
    const seen = new Map<string, WikiUpdate>();
    for (const update of updates) {
      if (!validId(update.id) || !validId(update.block_id) || update.base_revision !== base.revision ||
          typeof update.text !== 'string' || !base.blocks.some(row => row.id === update.block_id)) return bad('invalid update');
      const prior = seen.get(update.id);
      if (prior && JSON.stringify(prior) !== JSON.stringify(update)) return bad('conflicting duplicate update ID');
      seen.set(update.id, update);
    }
    const ordered = [...seen.values()].sort((a, b) => a.id.localeCompare(b.id));
    return { valid: true, detail: '', updates: ordered, presentation: hash(ordered.map(row => row.id)) };
  }

  publish(base: WikiPage, prepared: PreparedMerge, draft: MergeDraft, profile: MergeProfile): MergeReport {
    const rejected = (detail: string): MergeReport => ({ status: 'rejected', page: this.snapshot(), detail });
    const checked = this.prepare(base, prepared.updates, profile);
    if (!prepared.valid || !checked.valid || checked.presentation !== prepared.presentation) return rejected('stale or invalid merge');
    const expected = prepared.updates.map(row => row.id).sort();
    if (!draft || !Array.isArray(draft.accounted) || JSON.stringify([...draft.accounted].sort()) !== JSON.stringify(expected) ||
        new Set(draft.accounted).size !== expected.length || !Array.isArray(draft.blocks) || !Array.isArray(draft.unresolved))
      return rejected('updates not accounted for');
    try { checkBlocks(draft.blocks); } catch (error) { return rejected(String(error)); }
    if (draft.blocks.length !== base.blocks.length || draft.blocks.some((row, index) => row.id !== base.blocks[index]!.id) ||
        draft.unresolved.some(row => !expected.includes(row.update_id) ||
          row.block_id !== prepared.updates.find(update => update.id === row.update_id)!.block_id))
      return rejected('invalid block or conflict identity');
    const blocks = structuredClone(draft.blocks);
    for (const [blockId, output] of this.outputs) this.retired.set(blockId, { ...output });
    this.outputs.clear();
    this.page = { id: base.id, blocks, unresolved: structuredClone(draft.unresolved), revision: hash(blocks) };
    this.derived = null;
    this.events.push({ operation: 'wiki.merge', page_id: this.page.id, revision: this.page.revision,
      profile: this.profile, accounted: expected, unresolved: draft.unresolved.length });
    return { status: draft.unresolved.length ? 'unresolved' : 'merged', page: this.snapshot(), detail: '' };
  }

  /** Plan the requested cells with the evaluation policy, then run the plan batch by batch. */
  async schedule(requests: CellRequest[]): Promise<{ plan: CellPlan, results: Record<string, CellResult> }> {
    const page = this.snapshot(), fresh = [...this.outputs.keys()];
    const plan = await this.runtime.run(() => planCells(page, requests, fresh), { name: 'wiki-cells' });
    const known = new Map(requests.map(row => [row.block_id, row]));
    const results: Record<string, CellResult> = {};
    for (const batch of plan.batches) {
      const runs = batch.filter(run => page.blocks.some(row => row.id === run.block_id && row.kind === 'cell') && known.has(run.block_id));
      const done = await Promise.all(runs.map(run => this.runCell(run.block_id, run.input)));
      runs.forEach((run, index) => { results[run.block_id] = done[index]!; });
    }
    this.events.push({ operation: 'wiki.schedule', page_id: page.id, ran: Object.keys(results).sort(), refused: plan.refused.map(row => row.block_id) });
    return { plan, results };
  }

  /** Run a cell of the current page on `input`. */
  async runCell(blockId: string, input: string): Promise<CellResult> {
    const pageRevision = this.page.revision;
    const block = this.page.blocks.find(row => row.id === blockId && row.kind === 'cell');
    if (!block) throw new Error('unknown cell');
    const source = sourceRevision(block);
    const trace: InvocationTrace[] = [];
    let value: unknown, outcome = 'done';
    try {
      value = block.language === 'natlang' ? await this.runNatlangCell(block, input, trace) : runJavaScriptCell(block, input);
    } catch (error) { outcome = 'failed'; value = String(error); }
    const status = this.page.revision === pageRevision ? outcome : 'stale';
    const record: CellResult = { status, page_revision: pageRevision, source_revision: source,
      value_text: status === 'done' ? JSON.stringify(value) : '', trace_events: trace.reduce((sum, call) => sum + call.events.length, 0) };
    if (status === 'done') this.outputs.set(blockId, { ...record, trace });
    this.events.push({ operation: 'wiki.cell', block_id: blockId, status, page_revision: pageRevision, source_revision: source });
    return record;
  }

  private async runNatlangCell(block: WikiBlock, input: string, trace: InvocationTrace[]): Promise<unknown> {
    const files = this.files?.();
    const cell = defineNatlang(`---\nargs:\n  input: string\n${files ? '  files: Folder\n' : ''}returns: ${block.returns}\n---\n${block.text}\n`,
      { name: block.id.replace(/[^A-Za-z0-9_]/g, '_') });
    return this.runtime.run(() => files ? cell(input, files) : cell(input), { name: `wiki-cell-${block.id}`, trace: call => trace.push(call) });
  }

  output(blockId: string): CellResult | null {
    const value = this.outputs.get(blockId);
    if (!value) return null;
    const { trace: _, ...record } = value;
    return structuredClone(record);
  }
  trace(blockId: string): InvocationTrace[] { return this.outputs.get(blockId)?.trace ?? []; }
  drainEvents(): Record<string, unknown>[] { return this.events.splice(0); }
}

function checkBlocks(blocks: WikiBlock[]): void {
  if (!Array.isArray(blocks) || new Set(blocks.map(row => row.id)).size !== blocks.length) throw new Error('duplicate block IDs');
  for (const block of blocks) {
    if (!validId(block.id) || !['prose', 'cell'].includes(block.kind) || typeof block.text !== 'string') throw new Error('invalid block');
    if (block.kind === 'cell' && (!['javascript', 'natlang'].includes(block.language ?? '') ||
        !['string', 'number', 'boolean'].includes(block.returns ?? '')))
      throw new Error('invalid cell language or return type');
    if (block.kind === 'cell' && block.language === 'javascript') {
      try { new vm.Script(`(function (input) {\n${block.text}\n})`); } catch { throw new Error('a JavaScript cell does not compile'); }
    }
  }
}

/** A JavaScript cell is a function body over `input`, run in a fresh context with a time limit. */
function runJavaScriptCell(block: WikiBlock, input: string): unknown {
  const value = vm.runInNewContext(`(function (input) {\n${block.text}\n})(input)`, { input }, { timeout: CELL_TIMEOUT_MS });
  if (typeof value !== block.returns) throw new TypeError(`cell returned ${typeof value}, expected ${block.returns}`);
  return value;
}
