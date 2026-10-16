import {verifyNativeStateDerivationRecords} from './native-state-derivation.js';
/**
 * `.nz` files (spec/NEURALESE_FILES.md): a safetensors container whose metadata key `natlang` holds a JSON header of
 * typed named exports, and whose tensors are the soft blocks those exports use, each named by its content ID.
 *
 * This module is the codec and the export loader. It knows nothing of callables: soft-function exports are returned
 * as `NzSoftFunctionSpec` values, which the runtime turns into callables (runtime/contexts.ts).
 */
import { decodeNativeTensorArtifact, encodeNativeTensorArtifact, nativeTensorArtifactBlockMeta, nativeTensorArtifactDependencies, nativeStateArtifactDependencies, importNativeArtifactBundle, type NativeTensorArtifact, decodeNativeState, encodeNativeState, nativeStateBlockMeta, type NativeStateArtifact, neuraleseContentId, type NeuraleseBlock, type NeuraleseBlockMeta, type NeuraleseDtype,
  type NeuraleseStore } from './neuralese-store.js';
import { isNeuraleseRef, neuraleseRef, type NeuraleseRef } from './neuralese.js';
import { readTypeAliases } from './type-aliases.js';
import { formatType, parseType, TypeEnv, type Type } from './types.js';
import { coerce, Reject } from './values.js';

export const NZ_FORMAT = 'natlang.neuralese-file/1';

/** Errors with the codes the specification names. */
export class NzFileError extends Error {
  constructor(readonly code: 'neuralese-block-integrity' | 'neuralese-file-format' | 'neuralese-file-type' |
    'neuralese-dialect-mismatch' | 'neuralese-ref-cycle' | 'neuralese-unknown-block' | 'neuralese-live-capture-save',
  message: string) { super(`${code}: ${message}`); this.name = 'NzFileError'; }
}

export type NzBlockHeader = { dialect: string; length: number; width: number; dtype: 'BF16' | 'F16' | 'F32';
  producer?: Record<string, unknown>; truncated?: boolean };
export type NzSkill = { name: string; description: string; body: string; natlang?: Record<string, unknown> };
export type NzHeader = {
  format: typeof NZ_FORMAT;
  dialect: string;
  types?: string;
  exports: Record<string, { type: string; value: unknown; description?: string }>;
  blocks: Record<string, NzBlockHeader>;
  native_states?: Record<string, { bytes: number }>;
  native_tensors?: Record<string, { bytes: number }>;
  skill?: NzSkill;
  provenance?: Record<string, unknown>;
};

/** A soft function as stored: its type, body block and snapshot captures (already resolved to values). */
export type NzSoftFunctionSpec = { readonly kind: 'soft-function'; readonly type: string; readonly body: string; readonly body_stop_gradient?: true; readonly body_source_occurrence?:string;
  readonly captures: Readonly<Record<string, unknown>> };
export const isSoftFunctionSpec = (value: unknown): value is NzSoftFunctionSpec =>
  !!value && typeof value === 'object' && (value as { kind?: unknown }).kind === 'soft-function';

/** A Gaussian block: mean and per-dimension log-scale, both stored blocks. Reading at temperature 0 gives the mean. */
export type NzDistribution = { kind: 'gaussian-diag'; mean: string; logScale: string };
const DISTRIBUTIONS = new Map<string, NzDistribution>();
/** The distribution a soft value was loaded from or saved as, keyed by its mean block. */
export const distributionOf = (id: string): NzDistribution | undefined => DISTRIBUTIONS.get(id);
/** Declare that the soft value with block `mean` is a Gaussian with log-scale block `logScale`. */
export function declareDistribution(mean: string, logScale: string): void {
  DISTRIBUTIONS.set(mean, { kind: 'gaussian-diag', mean, logScale });
}

const DTYPE_TO_FILE: Record<NeuraleseDtype, NzBlockHeader['dtype']> = { f32: 'F32', f16: 'F16', bf16: 'BF16' };
const DTYPE_FROM_FILE: Record<string, NeuraleseDtype> = { F32: 'f32', F16: 'f16', BF16: 'bf16' };
const BYTES: Record<NeuraleseDtype, number> = { f32: 4, f16: 2, bf16: 2 };

/** JSON with sorted keys, so equal content gives equal bytes. */
export function canonicalJson(value: unknown): string {
  const sort = (item: unknown): unknown => Array.isArray(item) ? item.map(sort) :
    item && typeof item === 'object' ? Object.fromEntries(Object.keys(item).sort().map(key =>
      [key, sort((item as Record<string, unknown>)[key])])) : item;
  return JSON.stringify(sort(value));
}

// --- Container ------------------------------------------------------------------------------------------------

/** Encode a header and its blocks as safetensors bytes. Tensors are ordered by name; offsets are contiguous. */
export function encodeNz(header: NzHeader, blocks: ReadonlyMap<string, Uint8Array>): Uint8Array {
  const names = [...Object.keys(header.blocks), ...Object.keys(header.native_states ?? {}), ...Object.keys(header.native_tensors ?? {})].sort();
  if (new Set(names).size !== names.length) throw new NzFileError('neuralese-file-format', 'native and legacy inventories overlap');
  const tensors: Record<string, unknown> = {};
  let offset = 0;
  for (const name of names) {
    const data = blocks.get(name);
    if (!data) throw new NzFileError('neuralese-unknown-block', `no bytes for block ${name}`);
    const meta = header.blocks[name]!;
    tensors[name] = header.native_states?.[name] || header.native_tensors?.[name] ?
      { dtype: 'U8', shape: [data.length], data_offsets: [offset, offset + data.length] } :
      { dtype: meta.dtype, shape: [meta.length, meta.width], data_offsets: [offset, offset + data.length] };
    offset += data.length;
  }
  const json = canonicalJson({ __metadata__: { natlang: canonicalJson(header) }, ...tensors });
  let headerBytes = new TextEncoder().encode(json);
  const padded = Math.ceil(headerBytes.length / 8) * 8;
  if (padded !== headerBytes.length) {
    const spaces = new Uint8Array(padded).fill(0x20);
    spaces.set(headerBytes);
    headerBytes = spaces;
  }
  const out = new Uint8Array(8 + headerBytes.length + offset);
  new DataView(out.buffer).setBigUint64(0, BigInt(headerBytes.length), true);
  out.set(headerBytes, 8);
  let at = 8 + headerBytes.length;
  for (const name of names) { const data = blocks.get(name)!; out.set(data, at); at += data.length; }
  return out;
}

/** Decode safetensors bytes into the natlang header and its blocks, checking that every block hashes to its name. */
export function decodeNz(bytes: Uint8Array): { header: NzHeader; blocks: Map<string, NeuraleseBlock>; nativeStates: Map<string, NativeStateArtifact>; nativeTensors: Map<string, NativeTensorArtifact> } {
  if (bytes.length < 8) throw new NzFileError('neuralese-file-format', 'not a safetensors file');
  const size = Number(new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getBigUint64(0, true));
  if (8 + size > bytes.length) throw new NzFileError('neuralese-file-format', 'header length exceeds the file');
  let outer: Record<string, unknown>;
  try { outer = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + size))); }
  catch { throw new NzFileError('neuralese-file-format', 'header is not JSON'); }
  const metadata = outer.__metadata__ as Record<string, unknown> | undefined;
  if (!metadata || typeof metadata.natlang !== 'string') throw new NzFileError('neuralese-file-format', 'no natlang header in the metadata');
  let header: NzHeader;
  try { header = JSON.parse(metadata.natlang); } catch { throw new NzFileError('neuralese-file-format', 'natlang header is not JSON'); }
  checkHeaderShape(header);
  const data = bytes.subarray(8 + size);
  const blocks = new Map<string, NeuraleseBlock>();
  for (const [id, meta] of Object.entries(header.blocks)) {
    const tensor = outer[id] as { dtype?: string; shape?: number[]; data_offsets?: [number, number] } | undefined;
    if (!tensor?.data_offsets) throw new NzFileError('neuralese-file-format', `block ${id} has no tensor`);
    const dtype = DTYPE_FROM_FILE[meta.dtype];
    if (!dtype || tensor.dtype !== meta.dtype || tensor.shape?.[0] !== meta.length || tensor.shape?.[1] !== meta.width)
      throw new NzFileError('neuralese-file-format', `block ${id}: tensor and header disagree on shape or dtype`);
    const [start, end] = tensor.data_offsets;
    if (end - start !== meta.length * meta.width * BYTES[dtype] || end > data.length)
      throw new NzFileError('neuralese-file-format', `block ${id}: wrong byte length`);
    const payload = data.slice(start, end);
    const actual = neuraleseContentId({ dialect: meta.dialect, length: meta.length, width: meta.width, dtype, data: payload });
    if (actual !== id) throw new NzFileError('neuralese-block-integrity', `block ${id} hashes to ${actual}`);
    blocks.set(id, { meta: { id, dialect: meta.dialect, length: meta.length, width: meta.width, dtype,
      ...(meta.producer ? { producer: meta.producer } : {}), ...(meta.truncated ? { truncated: true } : {}) }, data: payload });
  }
  const nativeStates = new Map<string, NativeStateArtifact>();
  for (const [id, meta] of Object.entries(header.native_states ?? {})) {
    const tensor = outer[id] as { dtype?: string; shape?: number[]; data_offsets?: [number, number] } | undefined;
    const offsets = tensor?.data_offsets;
    if (blocks.has(id) || !/^nzs1_[a-z2-7]{52}$/.test(id) || !Number.isSafeInteger(meta.bytes) || meta.bytes < 1 ||
        tensor?.dtype !== 'U8' || tensor.shape?.length !== 1 || tensor.shape[0] !== meta.bytes || !offsets ||
        offsets.some(value => !Number.isSafeInteger(value) || value < 0) || offsets[1] > data.length ||
        offsets[1] - offsets[0] !== meta.bytes)
      throw new NzFileError('neuralese-file-format', `native state ${id}: tensor and inventory disagree`);
    let artifact: NativeStateArtifact;
    try { artifact = decodeNativeState(data.slice(offsets[0], offsets[1])); }
    catch (error) { throw new NzFileError('neuralese-block-integrity', `native state ${id}: ${String(error)}`); }
    if (artifact.id !== id) throw new NzFileError('neuralese-block-integrity', `native state ${id} hashes to ${artifact.id}`);
    nativeStates.set(id, artifact);
  }
  const nativeTensors = new Map<string, NativeTensorArtifact>();
  for (const [id, meta] of Object.entries(header.native_tensors ?? {})) {
    const tensor = outer[id] as { dtype?: string; shape?: number[]; data_offsets?: [number, number] } | undefined;
    const offsets = tensor?.data_offsets;
    if (blocks.has(id) || nativeStates.has(id) || !/^nnt1_[a-z2-7]{52}$/.test(id) ||
        !Number.isSafeInteger(meta.bytes) || meta.bytes < 1 || tensor?.dtype !== 'U8' ||
        tensor.shape?.length !== 1 || tensor.shape[0] !== meta.bytes || !offsets ||
        offsets.some(value => !Number.isSafeInteger(value) || value < 0) || offsets[1] > data.length ||
        offsets[1] - offsets[0] !== meta.bytes)
      throw new NzFileError('neuralese-file-format', `native tensor ${id}: tensor and inventory disagree`);
    const artifact = decodeNativeTensorArtifact(data.slice(offsets[0], offsets[1]));
    if (artifact.id !== id) throw new NzFileError('neuralese-block-integrity', `native tensor ${id} changed`);
    nativeTensors.set(id, artifact);
  }
  for (const artifact of [...nativeTensors.values(),...nativeStates.values()]) for(const dependency of
      artifact.id.startsWith('nnt1_')?nativeTensorArtifactDependencies(artifact as NativeTensorArtifact):nativeStateArtifactDependencies(artifact as NativeStateArtifact))
    if(!nativeTensors.has(dependency)&&!nativeStates.has(dependency))throw new NzFileError('neuralese-unknown-block',`native artifact lacks dependency ${dependency}`);
  const verified=new Set<string>(),active=new Set<string>();
  const verify=(id:string):void=>{
    if(verified.has(id))return;
    if(active.has(id))throw new NzFileError('neuralese-file-format','native derivations must form a DAG');
    active.add(id);
    const state=nativeStates.get(id),value=state?.metadata.derivation;
    if(state&&value){
      verify(value.parent_state_id);
      const parent=nativeStates.get(value.parent_state_id);
      if(!parent)throw new NzFileError('neuralese-unknown-block','native derivation lacks original state');
      if(value.kind==='optimizer')verifyNativeStateDerivationRecords(state,parent,nativeTensors.get(value.gradient_id),nativeTensors.get(value.update_artifact_id),
        value.optimizer_state_parent_id?nativeTensors.get(value.optimizer_state_parent_id):undefined,value.program_id?nativeTensors.get(value.program_id):undefined);
      else verifyNativeStateDerivationRecords(state,parent,undefined,undefined);
    }
    active.delete(id);verified.add(id);
  };
  for(const id of nativeStates.keys())verify(id);
  return { header, blocks, nativeStates, nativeTensors };
}

function checkHeaderShape(header: NzHeader): void {
  const bad = (message: string) => { throw new NzFileError('neuralese-file-format', message); };
  if (!header || typeof header !== 'object') bad('header must be an object');
  if (header.format !== NZ_FORMAT) bad(`format must be ${NZ_FORMAT}`);
  if (typeof header.dialect !== 'string') bad('dialect must be a string');
  if (!header.exports || typeof header.exports !== 'object') bad('exports must be an object');
  if (!header.blocks || typeof header.blocks !== 'object') bad('blocks must be an object');
  for (const [name, entry] of Object.entries(header.exports)) {
    if (!/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(name)) bad(`export name ${JSON.stringify(name)} is not an identifier`);
    if (!entry || typeof entry.type !== 'string' || !('value' in entry)) bad(`export ${name} needs a type and a value`);
  }
  if (header.skill && (typeof header.skill.name !== 'string' || typeof header.skill.description !== 'string' ||
    !Object.hasOwn(header.exports, header.skill.body))) bad('skill needs name, description and a body naming an export');
}

// --- Loading -------------------------------------------------------------------------------------------------

export type LoadedNz = {
  header: NzHeader;
  /** Export values: plain data, `NeuraleseRef`s and `NzSoftFunctionSpec`s; `$ref`s resolved. */
  exports: Record<string, unknown>;
  /** Natlang type of every export. */
  types: Record<string, string>;
  blocks: Map<string, NeuraleseBlock>;
  nativeStates: Map<string, NativeStateArtifact>;
  nativeTensors: Map<string, NativeTensorArtifact>;
};

/** Resolve another file's export for a cross-file `$ref` (`./other.nz#name`). */
export type NzResolver = (path: string) => LoadedNz;

/**
 * Load a `.nz` file: check its blocks, resolve `$ref`s (a DAG), check every export against its declared type and every
 * soft value's dialect, and put the blocks into `store` when one is given.
 */
export async function loadNz(bytes: Uint8Array, options: { store?: NeuraleseStore; resolve?: NzResolver } = {}): Promise<LoadedNz> {
  const loaded = loadNzSync(bytes, options.resolve);
  if (options.store) {
    for (const block of loaded.blocks.values()) await options.store.put({ ...block.meta, data: block.data });
    await importNativeArtifactBundle(options.store,loaded.nativeStates,loaded.nativeTensors);
  }
  return loaded;
}

/** `loadNz` without storing blocks; compiled module code and declaration generation use it. */
const IMPORTED_BLOCKS = new Map<string, NeuraleseBlock>();
const IMPORTED_NATIVE_STATES = new Map<string, NativeStateArtifact>();
const IMPORTED_NATIVE_TENSORS = new Map<string, NativeTensorArtifact>();
/** Record blocks of `.nz` files loaded without a store (compiled imports, companion folders) for runtimes to adopt. */
export function registerImportedBlocks(blocks: Iterable<NeuraleseBlock>, nativeStates: Iterable<NativeStateArtifact> = [], nativeTensors: Iterable<NativeTensorArtifact> = []): void {
  for (const block of blocks) IMPORTED_BLOCKS.set(block.meta.id, block);
  for (const artifact of nativeStates) IMPORTED_NATIVE_STATES.set(artifact.id, structuredClone(artifact));
  for (const artifact of nativeTensors) IMPORTED_NATIVE_TENSORS.set(artifact.id, structuredClone(artifact));
}
/** Blocks of `.nz` files loaded without a store; a runtime with a tensor store puts them into it before a call. */
export function importedBlocks(): ReadonlyMap<string, NeuraleseBlock> { return IMPORTED_BLOCKS; }

/** Put every imported block a store lacks into it. Cheap when nothing new was imported since the last call. */
export async function adoptImportedBlocks(store: Pick<NeuraleseStore, 'has' | 'put' | 'putNativeState' | 'putNativeTensorArtifact'>): Promise<void> {
  const done = ADOPTED.get(store) ?? new Set<string>();
  ADOPTED.set(store, done);
  for (const [id, block] of IMPORTED_BLOCKS) {
    if (done.has(id)) continue;
    if (!(await store.has(id))) {
      const { id: _id, ...meta } = block.meta;
      void _id;
      await store.put({ ...meta, data: block.data });
    }
    done.add(id);
  }
  await importNativeArtifactBundle(store,IMPORTED_NATIVE_STATES,IMPORTED_NATIVE_TENSORS);
  for(const id of [...IMPORTED_NATIVE_STATES.keys(),...IMPORTED_NATIVE_TENSORS.keys()])done.add(id);
}
const ADOPTED = new WeakMap<object, Set<string>>();

export function loadNzSync(bytes: Uint8Array, resolve?: NzResolver): LoadedNz {
  const { header, blocks, nativeStates, nativeTensors } = decodeNz(bytes);
  const env = new TypeEnv(Object.fromEntries(Object.entries(header.types ? readTypeAliases(header.types) : {})
    .map(([name, text]) => [name, parseType(text)])));
  const types: Record<string, string> = {};
  const parsedTypes: Record<string, Type> = {};
  for (const [name, entry] of Object.entries(header.exports)) {
    try { parsedTypes[name] = parseType(entry.type); env.checkNames(parsedTypes[name]!); }
    catch (error) { throw new NzFileError('neuralese-file-type', `export ${name}: ${(error as Error).message}`); }
    types[name] = entry.type;
  }
  const values: Record<string, unknown> = {};
  const resolving = new Set<string>();
  const exportValue = (name: string): unknown => {
    if (Object.hasOwn(values, name)) return values[name];
    if (!Object.hasOwn(header.exports, name)) throw new NzFileError('neuralese-file-format', `$ref to unknown export ${name}`);
    if (resolving.has(name)) throw new NzFileError('neuralese-ref-cycle', `export ${name} refers to itself through $ref`);
    resolving.add(name);
    const value = decodeValue(header.exports[name]!.value, `export ${name}`);
    resolving.delete(name);
    return values[name] = value;
  };
  const block = (id: string, where: string) => {
    const named = nativeTensors.get(id);
    if (named) return { meta: nativeTensorArtifactBlockMeta(named), data: encodeNativeTensorArtifact(named) };
    const native = nativeStates.get(id);
    if (native) return { meta: nativeStateBlockMeta(native), data: encodeNativeState(native) };
    if (!blocks.has(id)) throw new NzFileError('neuralese-unknown-block', `${where} uses block ${id}, which the file does not hold`);
    return blocks.get(id)!;
  };
  const decodeValue = (raw: unknown, where: string): unknown => {
    if (Array.isArray(raw)) return raw.map((item, index) => decodeValue(item, `${where}[${index}]`));
    if (!raw || typeof raw !== 'object') return raw;
    const record = raw as Record<string, unknown>;
    if ('$ref' in record && Object.keys(record).length === 1) {
      const target = String(record.$ref);
      const hash = target.indexOf('#');
      if (hash < 0) return exportValue(target);
      if (!resolve) throw new NzFileError('neuralese-file-format', `${where}: cross-file reference ${target} needs a resolver`);
      const other = resolve(target.slice(0, hash));
      const name = target.slice(hash + 1);
      if (!Object.hasOwn(other.exports, name)) throw new NzFileError('neuralese-file-format', `${where}: ${target} is not an export`);
      return other.exports[name];
    }
    if ('$neuralese' in record && Object.keys(record).length === 1) {
      const body = record.$neuralese as { type?: unknown; id?: unknown; stop_gradient?: unknown; source_occurrence?:unknown; distribution?: unknown; logScale?: unknown };
      if (body.stop_gradient !== undefined && body.stop_gradient !== true) throw new NzFileError('neuralese-file-format', `${where}: invalid constant-use metadata`);
      if (body.source_occurrence !== undefined && (typeof body.source_occurrence !== 'string'||!body.source_occurrence)) throw new NzFileError('neuralese-file-format', `${where}: invalid source occurrence`);
      const id = String(body?.id);
      block(id, where);
      if (body.distribution !== undefined) {
        if (body.distribution !== 'gaussian-diag' || typeof body.logScale !== 'string')
          throw new NzFileError('neuralese-file-format', `${where}: unknown distribution`);
        if (nativeStates.has(id) || nativeStates.has(body.logScale))
          throw new NzFileError('neuralese-file-format', 'Gaussian vector parameters cannot be native KV state');
        const scale = block(body.logScale, where), mean = blocks.get(id)!;
        if (scale.meta.length !== mean.meta.length || scale.meta.width !== mean.meta.width)
          throw new NzFileError('neuralese-file-format', `${where}: log-scale and mean shapes differ`);
        declareDistribution(id, body.logScale);
      }
      return neuraleseRef(String(body.type), id, body.stop_gradient === true,body.source_occurrence as string|undefined);
    }
    if ('$neuralese-fn' in record && Object.keys(record).length === 1) {
      const body = record['$neuralese-fn'] as { type?: unknown; body?: unknown; body_stop_gradient?: unknown; body_source_occurrence?:unknown; captures?: unknown };
      if (body.body_stop_gradient !== undefined && body.body_stop_gradient !== true) throw new NzFileError('neuralese-file-format', `${where}: invalid body constant-use metadata`);
      if (body.body_source_occurrence!==undefined&&(typeof body.body_source_occurrence!=='string'||!body.body_source_occurrence)) throw new NzFileError('neuralese-file-format', `${where}: invalid body source occurrence`);
      block(String(body.body), where);
      const captures = Object.fromEntries(Object.entries((body.captures ?? {}) as Record<string, unknown>)
        .map(([name, value]) => [name, decodeValue(value, `${where} capture ${name}`)]));
      return Object.freeze({ kind: 'soft-function', type: String(body.type), body: String(body.body), ...(body.body_stop_gradient ? {body_stop_gradient:true as const} : {}), ...(body.body_source_occurrence?{body_source_occurrence:body.body_source_occurrence as string}:{}), captures: Object.freeze(captures) });
    }
    return Object.fromEntries(Object.entries(record).map(([key, item]) => [key, decodeValue(item, `${where}.${key}`)]));
  };
  for (const name of Object.keys(header.exports)) exportValue(name);
  for (const [name, type] of Object.entries(parsedTypes)) checkExport(name, type, values[name], env, header, blocks, nativeStates, nativeTensors);
  return { header, exports: values, types, blocks, nativeStates, nativeTensors };
}

/** Check one export against its declared type, and the dialect of every soft value it holds. */
function checkExport(name: string, type: Type, value: unknown, env: TypeEnv, header: NzHeader,
  blocks: Map<string, NeuraleseBlock>, nativeStates: Map<string, NativeStateArtifact>, nativeTensors: Map<string, NativeTensorArtifact>): void {
  const dialectOf = (id: string) => blocks.get(id)?.meta.dialect ??
    (nativeStates.has(id) ? nativeStateBlockMeta(nativeStates.get(id)!).dialect :
     nativeTensors.has(id) ? nativeTensorArtifactBlockMeta(nativeTensors.get(id)!).dialect : header.dialect);
  const softValues = (type: Type, value: unknown, path: string): void => {
    let resolved = type;
    if (type.kind === 'name') try { resolved = env.resolve(type); } catch { resolved = type; }
    if (resolved.kind === 'neuralese') {
      const id = isNeuraleseRef(value) ? value.$neuralese.id : isSoftFunctionSpec(value) ? value.body : undefined;
      if (!id) throw new NzFileError('neuralese-file-type', `${path} must be a ${formatType(resolved)}`);
      if (resolved.dialect !== 'DefaultDialect' && resolved.dialect !== dialectOf(id))
        throw new NzFileError('neuralese-dialect-mismatch', `${path} is ${dialectOf(id)}, its type says ${resolved.dialect}`);
      if (isSoftFunctionSpec(value) && resolved.element.kind !== 'lambda')
        throw new NzFileError('neuralese-file-type', `${path} is a soft function, but its type is not a function type`);
      return;
    }
    if (resolved.kind === 'record' && value && typeof value === 'object')
      for (const field of resolved.fields) softValues(field.type, (value as Record<string, unknown>)[field.name], `${path}.${field.name}`);
    if (resolved.kind === 'list' && Array.isArray(value)) value.forEach((item, index) => softValues(resolved.element, item, `${path}[${index}]`));
    if (resolved.kind === 'dict' && value && typeof value === 'object')
      for (const [key, item] of Object.entries(value)) softValues(resolved.element, item, `${path}.${key}`);
  };
  softValues(type, value, `export ${name}`);
  // Soft functions are checked structurally above; everything else goes through the ordinary value check.
  const plain = (type: Type, value: unknown): unknown => isSoftFunctionSpec(value) ?
    neuraleseRef(value.type, value.body, value.body_stop_gradient,value.body_source_occurrence) : value && typeof value === 'object' && !isNeuraleseRef(value) ?
      (Array.isArray(value) ? value.map(item => plain(type, item)) :
        Object.fromEntries(Object.entries(value).map(([key, item]) => [key, plain(type, item)]))) : value;
  try { coerce(plain(type, value), type, env, `export ${name}`); }
  catch (error) {
    if (error instanceof Reject) throw new NzFileError('neuralese-file-type', `export ${name} does not match ${formatType(type)}: ${error.message}`);
    throw error;
  }
}

// --- Saving --------------------------------------------------------------------------------------------------

/** A value to save: data, `NeuraleseRef`s, and soft functions (as specs; live captures are refused by the caller). */
export type NzSaveExport = { type: string; value: unknown; description?: string };

/**
 * Build `.nz` bytes from export values, fetching their blocks from `store`. Distributional values (declared with
 * `declareDistribution`, or loaded from a file) keep their log-scale block.
 */
export async function saveNz(exports: Record<string, NzSaveExport>, options: { store: NeuraleseStore; dialect: string;
  types?: string; skill?: NzSkill; provenance?: Record<string, unknown> }): Promise<Uint8Array> {
  const blocks = new Map<string, Uint8Array>();
  const headers: Record<string, NzBlockHeader> = {};
  const nativeHeaders: Record<string, { bytes: number }> = {};
  const nativeTensorHeaders: Record<string, { bytes: number }> = {};
  const use = async (id: string, where: string) => {
    if (headers[id] || nativeHeaders[id] || nativeTensorHeaders[id]) return;
    if (id.startsWith('nnt1_')) {
      const artifact = await options.store.getNativeTensorArtifact?.(id);
      if (!artifact) throw new NzFileError('neuralese-unknown-block', `${where} uses named native tensor ${id} not held by the store`);
      const wire = encodeNativeTensorArtifact(artifact);
      nativeTensorHeaders[id] = { bytes: wire.length };
      blocks.set(id, wire);
      for (const dependency of nativeTensorArtifactDependencies(artifact)) await use(dependency, 'native adapter dependency');
      return;
    }
    if (id.startsWith('nzs1_')) {
      const artifact = await options.store.getNativeState?.(id);
      if (!artifact) throw new NzFileError('neuralese-unknown-block', `${where} uses native state ${id} not held by the store`);
      const wire = encodeNativeState(artifact);
      nativeHeaders[id] = { bytes: wire.length };
      blocks.set(id, wire);
      for(const dependency of nativeStateArtifactDependencies(artifact))await use(dependency,'native state derivation dependency');
      return;
    }
    const block = await options.store.get(id);
    if (!block) throw new NzFileError('neuralese-unknown-block', `${where} uses block ${id}, which the store does not hold`);
    headers[id] = blockHeader(block.meta);
    blocks.set(id, block.data);
  };
  const encodeValue = async (value: unknown, where: string): Promise<unknown> => {
    if (isNeuraleseRef(value)) {
      await use(value.$neuralese.id, where);
      const distribution = distributionOf(value.$neuralese.id);
      if (!distribution) return { $neuralese: { ...value.$neuralese } };
      await use(distribution.logScale, where);
      return { $neuralese: { type: value.$neuralese.type, id: value.$neuralese.id, distribution: 'gaussian-diag', logScale: distribution.logScale } };
    }
    if (isSoftFunctionSpec(value)) {
      await use(value.body, where);
      const captures: Record<string, unknown> = {};
      for (const [name, item] of Object.entries(value.captures)) captures[name] = await encodeValue(item, `${where} capture ${name}`);
      return { '$neuralese-fn': { type: value.type, body: value.body, ...(value.body_stop_gradient ? {body_stop_gradient:true} : {}),...(value.body_source_occurrence?{body_source_occurrence:value.body_source_occurrence}:{}), captures } };
    }
    if (typeof value === 'function') throw new NzFileError('neuralese-file-type', `${where} holds a function that is not a soft function`);
    if (Array.isArray(value)) return Promise.all(value.map((item, index) => encodeValue(item, `${where}[${index}]`)));
    if (value && typeof value === 'object') {
      const out: Record<string, unknown> = {};
      for (const [key, item] of Object.entries(value)) out[key] = await encodeValue(item, `${where}.${key}`);
      return out;
    }
    return value;
  };
  // Contextual native bodies depend on actual imported state from their source capture.
  // Preserve only explicit source-capture references; arbitrary provenance strings are not edges.
  const sourceDependencies = async (value: unknown): Promise<void> => {
    if (!value || typeof value !== 'object') return;
    const object = value as Record<string, unknown>;
    for (const key of ['source_capture', 'native_source_capture']) {
      const capture = object[key] as { segments?: { kind?: string; id?: string }[];
        imports?: { id?: string; observed_native_state_id?: string }[] } | undefined;
      if (Array.isArray(capture?.segments)) for (const segment of capture.segments)
        if (segment.kind === 'native_state' && typeof segment.id === 'string') await use(segment.id, 'native source capture');
      for (const imported of capture?.imports ?? []) for (const id of [imported.id, imported.observed_native_state_id])
        if (typeof id === 'string') await use(id, 'native source import');
    }
    for (const nested of Object.values(object)) await sourceDependencies(nested);
  };
  await sourceDependencies(options.provenance);
  const header: NzHeader = { format: NZ_FORMAT, dialect: options.dialect,
    ...(Object.keys(nativeTensorHeaders).length ? { native_tensors: nativeTensorHeaders } : {}), exports: {}, blocks: headers, ...(Object.keys(nativeHeaders).length ? { native_states: nativeHeaders } : {}),
    ...(options.types ? { types: options.types } : {}), ...(options.skill ? { skill: options.skill } : {}),
    ...(options.provenance ? { provenance: options.provenance } : {}) };
  for (const [name, entry] of Object.entries(exports))
    header.exports[name] = { type: entry.type, value: await encodeValue(entry.value, `export ${name}`),
      ...(entry.description ? { description: entry.description } : {}) };
  if (Object.keys(nativeHeaders).length) header.native_states = nativeHeaders;
  if (Object.keys(nativeTensorHeaders).length) header.native_tensors = nativeTensorHeaders;
  const bytes = encodeNz(header, blocks);
  loadNzSync(bytes); // A file that does not load is never written.
  return bytes;
}

function blockHeader(meta: NeuraleseBlockMeta): NzBlockHeader {
  return { dialect: meta.dialect, length: meta.length, width: meta.width, dtype: DTYPE_TO_FILE[meta.dtype],
    ...(meta.producer ? { producer: meta.producer } : {}), ...(meta.truncated ? { truncated: true } : {}) };
}

// --- Declarations --------------------------------------------------------------------------------------------

/**
 * TypeScript declarations for a `.nz` file, read by the checker as `name.d.nz.ts`. Natlang types are written as
 * TypeScript through `typeText` (the compiler's natlang → TypeScript mapping); `Neuralese` is a global intrinsic type.
 */
export function nzDeclaration(header: NzHeader, source: string,
  typeText: (text: string, known: ReadonlySet<string>) => string): string {
  const aliases = header.types ? readTypeAliases(header.types) : {};
  const known = new Set(Object.keys(aliases));
  const lines = [`// Generated by natlang from ${source}; do not edit.`,
    ...Object.entries(aliases).map(([name, text]) => `type ${name} = ${typeText(text, known)};`),
    ...Object.entries(header.exports).map(([name, entry]) =>
      `${entry.description ? `/** ${entry.description.replace(/\*\//g, '* /')} */\n` : ''}export declare const ${name}: ${typeText(entry.type, known)};`)];
  return lines.join('\n') + '\n';
}

/** Base64 of bytes, for embedding a file in generated module code. */
export function toBase64(bytes: Uint8Array): string {
  let text = '';
  for (let index = 0; index < bytes.length; index += 0x8000) text += String.fromCharCode(...bytes.subarray(index, index + 0x8000));
  return btoa(text);
}
export function fromBase64(text: string): Uint8Array {
  const binary = atob(text);
  const out = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index++) out[index] = binary.charCodeAt(index);
  return out;
}
