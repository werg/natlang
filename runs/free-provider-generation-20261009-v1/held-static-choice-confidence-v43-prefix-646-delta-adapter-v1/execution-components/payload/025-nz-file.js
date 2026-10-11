/**
 * `.nz` files (spec/NEURALESE_FILES.md): a safetensors container whose metadata key `natlang` holds a JSON header of
 * typed named exports, and whose tensors are the soft blocks those exports use, each named by its content ID.
 *
 * This module is the codec and the export loader. It knows nothing of callables: soft-function exports are returned
 * as `NzSoftFunctionSpec` values, which the runtime turns into callables (runtime/contexts.ts).
 */
import { neuraleseContentId } from './neuralese-store.js';
import { isNeuraleseRef, neuraleseRef } from './neuralese.js';
import { readTypeAliases } from './type-aliases.js';
import { formatType, parseType, TypeEnv } from './types.js';
import { coerce, Reject } from './values.js';
export const NZ_FORMAT = 'natlang.neuralese-file/1';
/** Errors with the codes the specification names. */
export class NzFileError extends Error {
    code;
    constructor(code, message) {
        super(`${code}: ${message}`);
        this.code = code;
        this.name = 'NzFileError';
    }
}
export const isSoftFunctionSpec = (value) => !!value && typeof value === 'object' && value.kind === 'soft-function';
const DISTRIBUTIONS = new Map();
/** The distribution a soft value was loaded from or saved as, keyed by its mean block. */
export const distributionOf = (id) => DISTRIBUTIONS.get(id);
/** Declare that the soft value with block `mean` is a Gaussian with log-scale block `logScale`. */
export function declareDistribution(mean, logScale) {
    DISTRIBUTIONS.set(mean, { kind: 'gaussian-diag', mean, logScale });
}
const DTYPE_TO_FILE = { f32: 'F32', f16: 'F16', bf16: 'BF16' };
const DTYPE_FROM_FILE = { F32: 'f32', F16: 'f16', BF16: 'bf16' };
const BYTES = { f32: 4, f16: 2, bf16: 2 };
/** JSON with sorted keys, so equal content gives equal bytes. */
export function canonicalJson(value) {
    const sort = (item) => Array.isArray(item) ? item.map(sort) :
        item && typeof item === 'object' ? Object.fromEntries(Object.keys(item).sort().map(key => [key, sort(item[key])])) : item;
    return JSON.stringify(sort(value));
}
// --- Container ------------------------------------------------------------------------------------------------
/** Encode a header and its blocks as safetensors bytes. Tensors are ordered by name; offsets are contiguous. */
export function encodeNz(header, blocks) {
    const names = Object.keys(header.blocks).sort();
    const tensors = {};
    let offset = 0;
    for (const name of names) {
        const data = blocks.get(name);
        if (!data)
            throw new NzFileError('neuralese-unknown-block', `no bytes for block ${name}`);
        const meta = header.blocks[name];
        tensors[name] = { dtype: meta.dtype, shape: [meta.length, meta.width], data_offsets: [offset, offset + data.length] };
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
    for (const name of names) {
        const data = blocks.get(name);
        out.set(data, at);
        at += data.length;
    }
    return out;
}
/** Decode safetensors bytes into the natlang header and its blocks, checking that every block hashes to its name. */
export function decodeNz(bytes) {
    if (bytes.length < 8)
        throw new NzFileError('neuralese-file-format', 'not a safetensors file');
    const size = Number(new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getBigUint64(0, true));
    if (8 + size > bytes.length)
        throw new NzFileError('neuralese-file-format', 'header length exceeds the file');
    let outer;
    try {
        outer = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + size)));
    }
    catch {
        throw new NzFileError('neuralese-file-format', 'header is not JSON');
    }
    const metadata = outer.__metadata__;
    if (!metadata || typeof metadata.natlang !== 'string')
        throw new NzFileError('neuralese-file-format', 'no natlang header in the metadata');
    let header;
    try {
        header = JSON.parse(metadata.natlang);
    }
    catch {
        throw new NzFileError('neuralese-file-format', 'natlang header is not JSON');
    }
    checkHeaderShape(header);
    const data = bytes.subarray(8 + size);
    const blocks = new Map();
    for (const [id, meta] of Object.entries(header.blocks)) {
        const tensor = outer[id];
        if (!tensor?.data_offsets)
            throw new NzFileError('neuralese-file-format', `block ${id} has no tensor`);
        const dtype = DTYPE_FROM_FILE[meta.dtype];
        if (!dtype || tensor.dtype !== meta.dtype || tensor.shape?.[0] !== meta.length || tensor.shape?.[1] !== meta.width)
            throw new NzFileError('neuralese-file-format', `block ${id}: tensor and header disagree on shape or dtype`);
        const [start, end] = tensor.data_offsets;
        if (end - start !== meta.length * meta.width * BYTES[dtype] || end > data.length)
            throw new NzFileError('neuralese-file-format', `block ${id}: wrong byte length`);
        const payload = data.slice(start, end);
        const actual = neuraleseContentId({ dialect: meta.dialect, length: meta.length, width: meta.width, dtype, data: payload });
        if (actual !== id)
            throw new NzFileError('neuralese-block-integrity', `block ${id} hashes to ${actual}`);
        blocks.set(id, { meta: { id, dialect: meta.dialect, length: meta.length, width: meta.width, dtype,
                ...(meta.producer ? { producer: meta.producer } : {}), ...(meta.truncated ? { truncated: true } : {}) }, data: payload });
    }
    return { header, blocks };
}
function checkHeaderShape(header) {
    const bad = (message) => { throw new NzFileError('neuralese-file-format', message); };
    if (!header || typeof header !== 'object')
        bad('header must be an object');
    if (header.format !== NZ_FORMAT)
        bad(`format must be ${NZ_FORMAT}`);
    if (typeof header.dialect !== 'string')
        bad('dialect must be a string');
    if (!header.exports || typeof header.exports !== 'object')
        bad('exports must be an object');
    if (!header.blocks || typeof header.blocks !== 'object')
        bad('blocks must be an object');
    for (const [name, entry] of Object.entries(header.exports)) {
        if (!/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(name))
            bad(`export name ${JSON.stringify(name)} is not an identifier`);
        if (!entry || typeof entry.type !== 'string' || !('value' in entry))
            bad(`export ${name} needs a type and a value`);
    }
    if (header.skill && (typeof header.skill.name !== 'string' || typeof header.skill.description !== 'string' ||
        !Object.hasOwn(header.exports, header.skill.body)))
        bad('skill needs name, description and a body naming an export');
}
/**
 * Load a `.nz` file: check its blocks, resolve `$ref`s (a DAG), check every export against its declared type and every
 * soft value's dialect, and put the blocks into `store` when one is given.
 */
export async function loadNz(bytes, options = {}) {
    const loaded = loadNzSync(bytes, options.resolve);
    if (options.store)
        for (const block of loaded.blocks.values())
            await options.store.put({ ...block.meta, data: block.data });
    return loaded;
}
/** `loadNz` without storing blocks; compiled module code and declaration generation use it. */
const IMPORTED_BLOCKS = new Map();
/** Record blocks of `.nz` files loaded without a store (compiled imports, companion folders) for runtimes to adopt. */
export function registerImportedBlocks(blocks) {
    for (const block of blocks)
        IMPORTED_BLOCKS.set(block.meta.id, block);
}
/** Blocks of `.nz` files loaded without a store; a runtime with a tensor store puts them into it before a call. */
export function importedBlocks() { return IMPORTED_BLOCKS; }
/** Put every imported block a store lacks into it. Cheap when nothing new was imported since the last call. */
export async function adoptImportedBlocks(store) {
    const done = ADOPTED.get(store) ?? new Set();
    ADOPTED.set(store, done);
    for (const [id, block] of IMPORTED_BLOCKS) {
        if (done.has(id))
            continue;
        if (!(await store.has(id))) {
            const { id: _id, ...meta } = block.meta;
            void _id;
            await store.put({ ...meta, data: block.data });
        }
        done.add(id);
    }
}
const ADOPTED = new WeakMap();
export function loadNzSync(bytes, resolve) {
    const { header, blocks } = decodeNz(bytes);
    const env = new TypeEnv(Object.fromEntries(Object.entries(header.types ? readTypeAliases(header.types) : {})
        .map(([name, text]) => [name, parseType(text)])));
    const types = {};
    const parsedTypes = {};
    for (const [name, entry] of Object.entries(header.exports)) {
        try {
            parsedTypes[name] = parseType(entry.type);
            env.checkNames(parsedTypes[name]);
        }
        catch (error) {
            throw new NzFileError('neuralese-file-type', `export ${name}: ${error.message}`);
        }
        types[name] = entry.type;
    }
    const values = {};
    const resolving = new Set();
    const exportValue = (name) => {
        if (Object.hasOwn(values, name))
            return values[name];
        if (!Object.hasOwn(header.exports, name))
            throw new NzFileError('neuralese-file-format', `$ref to unknown export ${name}`);
        if (resolving.has(name))
            throw new NzFileError('neuralese-ref-cycle', `export ${name} refers to itself through $ref`);
        resolving.add(name);
        const value = decodeValue(header.exports[name].value, `export ${name}`);
        resolving.delete(name);
        return values[name] = value;
    };
    const block = (id, where) => {
        if (!blocks.has(id))
            throw new NzFileError('neuralese-unknown-block', `${where} uses block ${id}, which the file does not hold`);
        return blocks.get(id);
    };
    const decodeValue = (raw, where) => {
        if (Array.isArray(raw))
            return raw.map((item, index) => decodeValue(item, `${where}[${index}]`));
        if (!raw || typeof raw !== 'object')
            return raw;
        const record = raw;
        if ('$ref' in record && Object.keys(record).length === 1) {
            const target = String(record.$ref);
            const hash = target.indexOf('#');
            if (hash < 0)
                return exportValue(target);
            if (!resolve)
                throw new NzFileError('neuralese-file-format', `${where}: cross-file reference ${target} needs a resolver`);
            const other = resolve(target.slice(0, hash));
            const name = target.slice(hash + 1);
            if (!Object.hasOwn(other.exports, name))
                throw new NzFileError('neuralese-file-format', `${where}: ${target} is not an export`);
            return other.exports[name];
        }
        if ('$neuralese' in record && Object.keys(record).length === 1) {
            const body = record.$neuralese;
            const id = String(body?.id);
            block(id, where);
            if (body.distribution !== undefined) {
                if (body.distribution !== 'gaussian-diag' || typeof body.logScale !== 'string')
                    throw new NzFileError('neuralese-file-format', `${where}: unknown distribution`);
                const scale = block(body.logScale, where), mean = blocks.get(id);
                if (scale.meta.length !== mean.meta.length || scale.meta.width !== mean.meta.width)
                    throw new NzFileError('neuralese-file-format', `${where}: log-scale and mean shapes differ`);
                declareDistribution(id, body.logScale);
            }
            return neuraleseRef(String(body.type), id);
        }
        if ('$neuralese-fn' in record && Object.keys(record).length === 1) {
            const body = record['$neuralese-fn'];
            block(String(body.body), where);
            const captures = Object.fromEntries(Object.entries((body.captures ?? {}))
                .map(([name, value]) => [name, decodeValue(value, `${where} capture ${name}`)]));
            return Object.freeze({ kind: 'soft-function', type: String(body.type), body: String(body.body), captures: Object.freeze(captures) });
        }
        return Object.fromEntries(Object.entries(record).map(([key, item]) => [key, decodeValue(item, `${where}.${key}`)]));
    };
    for (const name of Object.keys(header.exports))
        exportValue(name);
    for (const [name, type] of Object.entries(parsedTypes))
        checkExport(name, type, values[name], env, header, blocks);
    return { header, exports: values, types, blocks };
}
/** Check one export against its declared type, and the dialect of every soft value it holds. */
function checkExport(name, type, value, env, header, blocks) {
    const dialectOf = (id) => blocks.get(id)?.meta.dialect ?? header.dialect;
    const softValues = (type, value, path) => {
        let resolved = type;
        if (type.kind === 'name')
            try {
                resolved = env.resolve(type);
            }
            catch {
                resolved = type;
            }
        if (resolved.kind === 'neuralese') {
            const id = isNeuraleseRef(value) ? value.$neuralese.id : isSoftFunctionSpec(value) ? value.body : undefined;
            if (!id)
                throw new NzFileError('neuralese-file-type', `${path} must be a ${formatType(resolved)}`);
            if (resolved.dialect !== 'DefaultDialect' && resolved.dialect !== dialectOf(id))
                throw new NzFileError('neuralese-dialect-mismatch', `${path} is ${dialectOf(id)}, its type says ${resolved.dialect}`);
            if (isSoftFunctionSpec(value) && resolved.element.kind !== 'lambda')
                throw new NzFileError('neuralese-file-type', `${path} is a soft function, but its type is not a function type`);
            return;
        }
        if (resolved.kind === 'record' && value && typeof value === 'object')
            for (const field of resolved.fields)
                softValues(field.type, value[field.name], `${path}.${field.name}`);
        if (resolved.kind === 'list' && Array.isArray(value))
            value.forEach((item, index) => softValues(resolved.element, item, `${path}[${index}]`));
        if (resolved.kind === 'dict' && value && typeof value === 'object')
            for (const [key, item] of Object.entries(value))
                softValues(resolved.element, item, `${path}.${key}`);
    };
    softValues(type, value, `export ${name}`);
    // Soft functions are checked structurally above; everything else goes through the ordinary value check.
    const plain = (type, value) => isSoftFunctionSpec(value) ?
        neuraleseRef(value.type, value.body) : value && typeof value === 'object' && !isNeuraleseRef(value) ?
        (Array.isArray(value) ? value.map(item => plain(type, item)) :
            Object.fromEntries(Object.entries(value).map(([key, item]) => [key, plain(type, item)]))) : value;
    try {
        coerce(plain(type, value), type, env, `export ${name}`);
    }
    catch (error) {
        if (error instanceof Reject)
            throw new NzFileError('neuralese-file-type', `export ${name} does not match ${formatType(type)}: ${error.message}`);
        throw error;
    }
}
/**
 * Build `.nz` bytes from export values, fetching their blocks from `store`. Distributional values (declared with
 * `declareDistribution`, or loaded from a file) keep their log-scale block.
 */
export async function saveNz(exports, options) {
    const blocks = new Map();
    const headers = {};
    const use = async (id, where) => {
        if (headers[id])
            return;
        const block = await options.store.get(id);
        if (!block)
            throw new NzFileError('neuralese-unknown-block', `${where} uses block ${id}, which the store does not hold`);
        headers[id] = blockHeader(block.meta);
        blocks.set(id, block.data);
    };
    const encodeValue = async (value, where) => {
        if (isNeuraleseRef(value)) {
            await use(value.$neuralese.id, where);
            const distribution = distributionOf(value.$neuralese.id);
            if (!distribution)
                return { $neuralese: { type: value.$neuralese.type, id: value.$neuralese.id } };
            await use(distribution.logScale, where);
            return { $neuralese: { type: value.$neuralese.type, id: value.$neuralese.id, distribution: 'gaussian-diag', logScale: distribution.logScale } };
        }
        if (isSoftFunctionSpec(value)) {
            await use(value.body, where);
            const captures = {};
            for (const [name, item] of Object.entries(value.captures))
                captures[name] = await encodeValue(item, `${where} capture ${name}`);
            return { '$neuralese-fn': { type: value.type, body: value.body, captures } };
        }
        if (typeof value === 'function')
            throw new NzFileError('neuralese-file-type', `${where} holds a function that is not a soft function`);
        if (Array.isArray(value))
            return Promise.all(value.map((item, index) => encodeValue(item, `${where}[${index}]`)));
        if (value && typeof value === 'object') {
            const out = {};
            for (const [key, item] of Object.entries(value))
                out[key] = await encodeValue(item, `${where}.${key}`);
            return out;
        }
        return value;
    };
    const header = { format: NZ_FORMAT, dialect: options.dialect, exports: {}, blocks: headers,
        ...(options.types ? { types: options.types } : {}), ...(options.skill ? { skill: options.skill } : {}),
        ...(options.provenance ? { provenance: options.provenance } : {}) };
    for (const [name, entry] of Object.entries(exports))
        header.exports[name] = { type: entry.type, value: await encodeValue(entry.value, `export ${name}`),
            ...(entry.description ? { description: entry.description } : {}) };
    const bytes = encodeNz(header, blocks);
    loadNzSync(bytes); // A file that does not load is never written.
    return bytes;
}
function blockHeader(meta) {
    return { dialect: meta.dialect, length: meta.length, width: meta.width, dtype: DTYPE_TO_FILE[meta.dtype],
        ...(meta.producer ? { producer: meta.producer } : {}), ...(meta.truncated ? { truncated: true } : {}) };
}
// --- Declarations --------------------------------------------------------------------------------------------
/**
 * TypeScript declarations for a `.nz` file, read by the checker as `name.d.nz.ts`. Natlang types are written as
 * TypeScript through `typeText` (the compiler's natlang → TypeScript mapping); `Neuralese` is a global intrinsic type.
 */
export function nzDeclaration(header, source, typeText) {
    const aliases = header.types ? readTypeAliases(header.types) : {};
    const known = new Set(Object.keys(aliases));
    const lines = [`// Generated by natlang from ${source}; do not edit.`,
        ...Object.entries(aliases).map(([name, text]) => `type ${name} = ${typeText(text, known)};`),
        ...Object.entries(header.exports).map(([name, entry]) => `${entry.description ? `/** ${entry.description.replace(/\*\//g, '* /')} */\n` : ''}export declare const ${name}: ${typeText(entry.type, known)};`)];
    return lines.join('\n') + '\n';
}
/** Base64 of bytes, for embedding a file in generated module code. */
export function toBase64(bytes) {
    let text = '';
    for (let index = 0; index < bytes.length; index += 0x8000)
        text += String.fromCharCode(...bytes.subarray(index, index + 0x8000));
    return btoa(text);
}
export function fromBase64(text) {
    const binary = atob(text);
    const out = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index++)
        out[index] = binary.charCodeAt(index);
    return out;
}
