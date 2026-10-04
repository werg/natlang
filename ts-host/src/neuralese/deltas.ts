/**
 * Residual updates (plans/neuralese/LEARNING_CONTINUUM.md §5): `Delta<A>` values with `diff`, `apply` and `compose`.
 *
 * - A soft value's or adapter's delta is a block of the same shape in the `#delta` variant of the base's dialect, so a
 *   delta cannot be read as a value or bound as an adapter. `apply(base, delta, scale)` is `base + scale · delta`;
 *   `compose` sums deltas (with scales). Scale 0 removes an update; scale 0.5 halves it.
 * - A crisp artifact's delta is a file patch (`natlang.file-patch/1`): changed paths with their new text (or `null` for
 *   a removed file), naming the digest of the files it was taken against. Composition merges patches; two patches
 *   that change the same path differently do not compose.
 *
 * Deltas are computed on the host from stored blocks (f32, f16 or bf16) and written as f32 blocks. The `#delta`
 * dialect is bookkeeping for computed deltas only: no model writes it (decision 39). A learned updater writes an
 * ordinary block, and a trained delta projection maps it to a delta (§5).
 */
import { createHash } from 'node:crypto';
import { isNeuraleseRef, neuraleseRef, type NeuraleseRef } from '../native/neuralese.js';
import type { NeuraleseBlock, NeuraleseStore } from '../native/neuralese-store.js';

export const DELTA_SUFFIX = '#delta';
export const deltaDialect = (dialect: string) => dialect.split('#')[0] + DELTA_SUFFIX;
export const isDeltaDialect = (dialect: string) => dialect.endsWith(DELTA_SUFFIX);

export class DeltaError extends Error {
  constructor(readonly code: string, message: string) { super(`${code}: ${message}`); this.name = 'DeltaError'; }
}

/** Where blocks are read from (the runtime's store first, then a server's) and written to. */
export type DeltaStores = { readonly read: (id: string) => Promise<NeuraleseBlock | undefined>; readonly write: NeuraleseStore };

function half(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1, exponent = (bits >> 10) & 0x1f, fraction = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 0x1f) return fraction ? NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

/** A block's payload as f32 values. */
export function blockFloats(block: NeuraleseBlock): Float32Array {
  const { data, meta } = block, view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const count = meta.length * meta.width, out = new Float32Array(count);
  const scratch = new DataView(new ArrayBuffer(4));
  for (let i = 0; i < count; i++) {
    if (meta.dtype === 'f32') out[i] = view.getFloat32(i * 4, true);
    else if (meta.dtype === 'bf16') { scratch.setUint32(0, view.getUint16(i * 2, true) << 16); out[i] = scratch.getFloat32(0); }
    else out[i] = half(view.getUint16(i * 2, true));
  }
  return out;
}

const bytesOf = (values: Float32Array) => new Uint8Array(values.buffer.slice(values.byteOffset, values.byteOffset + values.byteLength));

async function load(stores: DeltaStores, ref: NeuraleseRef, what: string): Promise<NeuraleseBlock> {
  if (!isNeuraleseRef(ref)) throw new DeltaError('delta-arguments', `${what} must be a Neuralese value, an adapter or a delta`);
  const block = await stores.read(ref.$neuralese.id);
  if (!block) throw new DeltaError('neuralese-unknown-block', `${what} ${ref.$neuralese.id} is neither in the runtime's store nor on the server`);
  return block;
}

function sameShape(a: NeuraleseBlock, b: NeuraleseBlock, what: string) {
  if (a.meta.dialect.split('#')[0] !== b.meta.dialect.split('#')[0])
    throw new DeltaError('delta-dialect', `${what}: ${a.meta.dialect} and ${b.meta.dialect} are different dialects`);
  if (a.meta.length !== b.meta.length || a.meta.width !== b.meta.width)
    throw new DeltaError('delta-shape', `${what}: shapes ${a.meta.length}×${a.meta.width} and ${b.meta.length}×${b.meta.width} differ`);
}

async function put(stores: DeltaStores, like: NeuraleseBlock, values: Float32Array, dialect: string, type: string | undefined,
    producer: Record<string, unknown>): Promise<NeuraleseRef> {
  const meta = await stores.write.put({ dialect, length: like.meta.length, width: like.meta.width, dtype: 'f32',
    data: bytesOf(values), ...(type ? { type } : {}), producer });
  return neuraleseRef(type ?? 'Neuralese<unknown>', meta.id);
}

/** `after − base`: the update that turns `base` into `after`. */
export async function diff(stores: DeltaStores, after: NeuraleseRef, base: NeuraleseRef): Promise<NeuraleseRef> {
  const [a, b] = [await load(stores, after, 'after'), await load(stores, base, 'base')];
  if (isDeltaDialect(a.meta.dialect) || isDeltaDialect(b.meta.dialect)) throw new DeltaError('delta-arguments', 'diff takes values, not deltas');
  sameShape(a, b, 'diff');
  const x = blockFloats(a), y = blockFloats(b), out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i]! - y[i]!;
  return put(stores, a, out, deltaDialect(a.meta.dialect), `Delta<${base.$neuralese.type}>`, { kind: 'delta', of: base.$neuralese.id, to: after.$neuralese.id });
}

/** `base + scale · delta`, a value of the base's dialect and type. */
export async function apply(stores: DeltaStores, base: NeuraleseRef, delta: NeuraleseRef, scale = 1): Promise<NeuraleseRef> {
  const [b, d] = [await load(stores, base, 'base'), await load(stores, delta, 'delta')];
  if (!isDeltaDialect(d.meta.dialect)) throw new DeltaError('delta-arguments', 'apply needs a delta (from diff or compose) as its second argument');
  if (isDeltaDialect(b.meta.dialect)) throw new DeltaError('delta-arguments', 'apply needs a value as its base');
  sameShape(b, d, 'apply');
  if (scale === 0) return base;
  const x = blockFloats(b), y = blockFloats(d), out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i]! + scale * y[i]!;
  return put(stores, b, out, b.meta.dialect, b.meta.type ?? base.$neuralese.type,
    { kind: 'delta-apply', base: base.$neuralese.id, delta: delta.$neuralese.id, scale });
}

/** Σ scaleᵢ · deltaᵢ: deltas trained separately from one base, combined into one update. */
export async function compose(stores: DeltaStores, deltas: readonly (NeuraleseRef | { delta: NeuraleseRef; scale?: number })[]): Promise<NeuraleseRef> {
  if (!deltas.length) throw new DeltaError('delta-arguments', 'compose needs at least one delta');
  const items = deltas.map(item => isNeuraleseRef(item) ? { delta: item, scale: 1 } : { delta: item.delta, scale: item.scale ?? 1 });
  const blocks = await Promise.all(items.map((item, index) => load(stores, item.delta, `delta ${index}`)));
  for (const block of blocks) if (!isDeltaDialect(block.meta.dialect)) throw new DeltaError('delta-arguments', 'compose takes deltas');
  for (const block of blocks.slice(1)) sameShape(blocks[0]!, block, 'compose');
  const out = new Float32Array(blocks[0]!.meta.length * blocks[0]!.meta.width);
  blocks.forEach((block, index) => { const values = blockFloats(block); for (let i = 0; i < out.length; i++) out[i]! += items[index]!.scale * values[i]!; });
  return put(stores, blocks[0]!, out, blocks[0]!.meta.dialect, blocks[0]!.meta.type ?? items[0]!.delta.$neuralese.type,
    { kind: 'delta-compose', parts: items.map(item => ({ delta: item.delta.$neuralese.id, scale: item.scale })) });
}

// Crisp artifacts --------------------------------------------------------------------------------------------------
export const FILE_PATCH_SCHEMA = 'natlang.file-patch/1';
export type FilePatch = { readonly schema: typeof FILE_PATCH_SCHEMA; readonly base: string; readonly changes: Readonly<Record<string, string | null>> };

export const filesDigest = (files: Readonly<Record<string, string>>) => createHash('sha256')
  .update(JSON.stringify(Object.keys(files).sort().map(path => [path, files[path]]))).digest('hex');

/** The patch that turns `base` into `after`: changed and added paths with their text, removed paths as `null`. */
export function diffFiles(after: Readonly<Record<string, string>>, base: Readonly<Record<string, string>>): FilePatch {
  const changes: Record<string, string | null> = {};
  for (const path of new Set([...Object.keys(base), ...Object.keys(after)])) {
    if (!(path in after)) changes[path] = null;
    else if (base[path] !== after[path]) changes[path] = after[path]!;
  }
  return { schema: FILE_PATCH_SCHEMA, base: filesDigest(base), changes: Object.fromEntries(Object.entries(changes).sort(([a], [b]) => a < b ? -1 : 1)) };
}

/** `files` with the patch applied; `strict` refuses files other than the patch's base. */
export function applyPatch(files: Readonly<Record<string, string>>, patch: FilePatch, options: { strict?: boolean } = {}): Record<string, string> {
  if (options.strict && filesDigest(files) !== patch.base) throw new DeltaError('patch-base', 'the files are not the patch\'s base');
  const out: Record<string, string> = { ...files };
  for (const [path, text] of Object.entries(patch.changes)) { if (text === null) delete out[path]; else out[path] = text; }
  return out;
}

/** Patches taken against one base, merged; a path changed differently by two patches is a conflict. */
export function composePatches(patches: readonly FilePatch[]): FilePatch {
  if (!patches.length) throw new DeltaError('delta-arguments', 'composePatches needs at least one patch');
  const changes: Record<string, string | null> = {};
  for (const patch of patches) {
    if (patch.base !== patches[0]!.base) throw new DeltaError('patch-base', 'patches to compose must share their base');
    for (const [path, text] of Object.entries(patch.changes)) {
      if (path in changes && changes[path] !== text) throw new DeltaError('patch-conflict', `${path} is changed differently by two patches`);
      changes[path] = text;
    }
  }
  return { schema: FILE_PATCH_SCHEMA, base: patches[0]!.base, changes };
}
