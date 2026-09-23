/** Local collaborative wiki with semantic merge drafts and pinned child cells. */
import { createHash } from 'node:crypto';
import { NativeSourceWorkspace } from '../ts-host/dist/index.js';

const hash = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const validId = value => typeof value === 'string' && /^[A-Za-z][A-Za-z0-9_-]*$/.test(value);

export class WikiWorkspace {
  constructor(page, { profile, modelTurn, files } = {}) {
    if (!validId(page.id) || !Array.isArray(page.blocks) ||
        !profile?.model || !profile?.source || !Number.isSafeInteger(profile?.seed))
      throw new Error('invalid page or merge profile');
    this.profile = structuredClone(profile);
    this.modelTurn = modelTurn;
    this.files = files;
    this.page = structuredClone(page);
    this.page.revision = hash(this.page.blocks);
    this.outputs = new Map(); this.events = [];
    this.#checkBlocks(this.page.blocks);
  }

  #checkBlocks(blocks) {
    if (!Array.isArray(blocks) || new Set(blocks.map(row => row.id)).size !== blocks.length)
      throw new Error('duplicate block IDs');
    for (const block of blocks) {
      if (!validId(block.id) || !['prose', 'cell'].includes(block.kind) ||
          typeof block.text !== 'string') throw new Error('invalid block');
      if (block.kind === 'cell') this.#cell(block);
    }
  }

  #cell(block) {
    if (!['quickjs', 'natlang'].includes(block.language) ||
        !['string', 'number', 'boolean'].includes(block.returns))
      throw new Error('invalid cell language or return type');
    const withFiles = this.files && block.language === 'natlang';
    const definition = { args: { input: 'string', ...(withFiles ? { files: 'Record<string, File>' } : {}) },
      ...(withFiles ? { types: { File: '{ kind: "text", text: string, bytes: number } | { kind: "binary", bytes: number }' } } : {}),
      returns: block.returns,
      ...(block.language === 'natlang' ? { instructions: block.text } : { code: block.text }) };
    return new NativeSourceWorkspace({ cell: definition }, 'cell');
  }

  snapshot() { return structuredClone(this.page); }

  prepare(base, updates, profile) {
    const bad = reason => ({ valid: false, detail: reason, updates: [], presentation: '' });
    if (JSON.stringify(profile) !== JSON.stringify(this.profile))
      return bad('merge profile mismatch');
    if (base.id !== this.page.id || base.revision !== this.page.revision)
      return bad('stale base page');
    if (!Array.isArray(updates) || updates.length > 64) return bad('invalid update batch');
    const seen = new Map();
    for (const update of updates) {
      if (!validId(update.id) || !validId(update.block_id) ||
          update.base_revision !== base.revision || typeof update.text !== 'string' ||
          !base.blocks.some(row => row.id === update.block_id)) return bad('invalid update');
      const prior = seen.get(update.id);
      if (prior && JSON.stringify(prior) !== JSON.stringify(update))
        return bad('conflicting duplicate update ID');
      seen.set(update.id, update);
    }
    const ordered = [...seen.values()].sort((a, b) => a.id.localeCompare(b.id));
    return { valid: true, detail: '', updates: ordered,
      presentation: hash(ordered.map(row => row.id)) };
  }

  publish(base, prepared, draft, profile) {
    const checked = this.prepare(base, prepared.updates, profile);
    if (!prepared.valid || !checked.valid ||
        checked.presentation !== prepared.presentation ||
        JSON.stringify(profile) !== JSON.stringify(this.profile) ||
        base.revision !== this.page.revision) return { status: 'rejected',
      page: this.snapshot(), detail: 'stale or invalid merge' };
    const expected = prepared.updates.map(row => row.id).sort();
    if (!draft || !Array.isArray(draft.accounted) ||
        JSON.stringify([...draft.accounted].sort()) !== JSON.stringify(expected) ||
        new Set(draft.accounted).size !== expected.length ||
        !Array.isArray(draft.blocks) || !Array.isArray(draft.unresolved))
      return { status: 'rejected', page: this.snapshot(), detail: 'updates not accounted for' };
    try { this.#checkBlocks(draft.blocks); }
    catch (error) { return { status: 'rejected', page: this.snapshot(), detail: String(error) }; }
    if (draft.blocks.some(row => !base.blocks.some(old => old.id === row.id)) ||
        draft.blocks.length !== base.blocks.length ||
        draft.unresolved.some(row => !expected.includes(row.update_id)))
      return { status: 'rejected', page: this.snapshot(), detail: 'invalid block or conflict identity' };
    const next = { id: base.id, blocks: structuredClone(draft.blocks),
      unresolved: structuredClone(draft.unresolved) };
    next.revision = hash(next.blocks);
    this.page = next;
    this.outputs.clear();
    this.events.push({ operation: 'wiki.merge', page_id: next.id, revision: next.revision,
      profile: this.profile, accounted: expected, unresolved: next.unresolved.length });
    return { status: next.unresolved.length ? 'unresolved' : 'merged',
      page: this.snapshot(), detail: '' };
  }

  async runCell(blockId, input) {
    const pageRevision = this.page.revision;
    const block = this.page.blocks.find(row => row.id === blockId && row.kind === 'cell');
    if (!block) throw new Error('unknown cell');
    const source = this.#cell(block);
    const provider = block.language === 'natlang' && (typeof this.files === 'function' ? this.files() : this.files);
    const result = await source.invoke('cell', { input, ...(provider ? { files: provider } : {}) }, { modelTurn: this.modelTurn,
      seedPolicy: { mode: 'derived', root: this.profile.seed } });
    const status = this.page.revision === pageRevision ? result.outcome : 'stale';
    const record = { status, page_revision: pageRevision,
      source_revision: result.source_revision,
      value_text: status === 'done' ? JSON.stringify(result.value) : '',
      trace_events: result.trace.length };
    if (status === 'done') this.outputs.set(blockId, { ...record, trace: result.trace });
    this.events.push({ operation: 'wiki.cell', block_id: blockId, status,
      page_revision: pageRevision, source_revision: result.source_revision });
    return record;
  }

  output(blockId) {
    const value = this.outputs.get(blockId);
    if (!value) return null;
    const { trace, ...record } = value;
    return structuredClone(record);
  }
  trace(blockId) { return structuredClone(this.outputs.get(blockId)?.trace ?? []); }
  drainEvents() { return this.events.splice(0); }
}
