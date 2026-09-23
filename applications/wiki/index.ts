/**
 * A local collaborative wiki. Concurrent edits are reconciled semantically by `reconcile.nl` under a
 * pinned merge profile; the wiki checks transport identity, update coverage and block shape before
 * publishing. Pages hold cells: JavaScript cells run in a fresh VM context, natlang cells are functions
 * defined from the cell text. A result that finishes after the page changed is reported stale.
 */
import { createHash } from 'node:crypto';
import vm from 'node:vm';
import { defineNatlang, type FolderHandle, type InvocationTrace, type NatlangRuntime } from '@natlang/node';
import reconcile from './reconcile.nl';
import type { CellResult, MergeDraft, MergeProfile, MergeReport, PreparedMerge, WikiBlock, WikiPage, WikiUpdate } from './types.js';

export type * from './types.js';

const hash = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const validId = (value: unknown): value is string => typeof value === 'string' && /^[A-Za-z][A-Za-z0-9_-]*$/.test(value);
const CELL_TIMEOUT_MS = 1000;

export type WikiOptions = {
  profile: MergeProfile;
  /** Runs reconciliation and natlang cells; create it with `seed: { mode: 'derived', root: profile.seed }` to pin sampling. */
  runtime: NatlangRuntime;
  /** Project files given to natlang cells, read fresh for each run. */
  files?: () => FolderHandle;
};

type Output = CellResult & { trace: InvocationTrace[] };

export class WikiWorkspace {
  readonly profile: MergeProfile;
  private readonly runtime: NatlangRuntime;
  private readonly files?: () => FolderHandle;
  private page: WikiPage;
  private readonly outputs = new Map<string, Output>();
  private readonly events: Record<string, unknown>[] = [];

  constructor(page: { id: string, blocks: WikiBlock[] }, options: WikiOptions) {
    const { profile } = options;
    if (!validId(page.id) || !Array.isArray(page.blocks) || !profile?.model || !profile?.source || !Number.isSafeInteger(profile?.seed))
      throw new Error('invalid page or merge profile');
    this.profile = structuredClone(profile);
    this.runtime = options.runtime;
    this.files = options.files;
    checkBlocks(page.blocks);
    this.page = { id: page.id, blocks: structuredClone(page.blocks), revision: hash(page.blocks) };
  }

  snapshot(): WikiPage { return structuredClone(this.page); }

  /** Merge concurrent updates made against `base`. */
  async merge(base: WikiPage, updates: WikiUpdate[], profile: MergeProfile = this.profile): Promise<MergeReport> {
    const prepared = this.prepare(base, updates, profile);
    if (!prepared.valid) return { status: 'rejected', page: base, detail: prepared.detail };
    const draft = await this.runtime.run(() => reconcile(base, prepared.updates), { name: 'wiki-merge' });
    return this.publish(base, prepared, draft, profile);
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
    if (draft.blocks.length !== base.blocks.length || draft.blocks.some(row => !base.blocks.some(old => old.id === row.id)) ||
        draft.unresolved.some(row => !expected.includes(row.update_id)))
      return rejected('invalid block or conflict identity');
    const blocks = structuredClone(draft.blocks);
    this.page = { id: base.id, blocks, unresolved: structuredClone(draft.unresolved), revision: hash(blocks) };
    this.outputs.clear();
    this.events.push({ operation: 'wiki.merge', page_id: this.page.id, revision: this.page.revision,
      profile: this.profile, accounted: expected, unresolved: draft.unresolved.length });
    return { status: draft.unresolved.length ? 'unresolved' : 'merged', page: this.snapshot(), detail: '' };
  }

  /** Run a cell of the current page on `input`. */
  async runCell(blockId: string, input: string): Promise<CellResult> {
    const pageRevision = this.page.revision;
    const block = this.page.blocks.find(row => row.id === blockId && row.kind === 'cell');
    if (!block) throw new Error('unknown cell');
    const sourceRevision = hash([block.language, block.returns, block.text]).slice(0, 16);
    const trace: InvocationTrace[] = [];
    let value: unknown, outcome = 'done';
    try {
      value = block.language === 'natlang' ? await this.runNatlangCell(block, input, trace) : runJavaScriptCell(block, input);
    } catch (error) { outcome = 'failed'; value = String(error); }
    const status = this.page.revision === pageRevision ? outcome : 'stale';
    const record: CellResult = { status, page_revision: pageRevision, source_revision: sourceRevision,
      value_text: status === 'done' ? JSON.stringify(value) : '', trace_events: trace.reduce((sum, call) => sum + call.events.length, 0) };
    if (status === 'done') this.outputs.set(blockId, { ...record, trace });
    this.events.push({ operation: 'wiki.cell', block_id: blockId, status, page_revision: pageRevision, source_revision: sourceRevision });
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
  }
}

/** A JavaScript cell is a function body over `input`, run in a fresh context with a time limit. */
function runJavaScriptCell(block: WikiBlock, input: string): unknown {
  const value = vm.runInNewContext(`(function (input) {\n${block.text}\n})(input)`, { input }, { timeout: CELL_TIMEOUT_MS });
  if (typeof value !== block.returns) throw new TypeError(`cell returned ${typeof value}, expected ${block.returns}`);
  return value;
}
