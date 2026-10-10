/**
 * `natlang:neuralese`, the combinator library (spec/SPEC.md, Neuralese chapter; S0 §4): `map`, `zip`, `ap`, `combine`,
 * `empty`, `split`, `splitList`, `read`, `convert`, `gloss`, and the query operator `ask(block, question)`, defined as
 * `read(map(block, question))` (DECISIONS.md 2026-10-09, "one summarizer family"; its law in S0 §4.2).
 *
 * Each combinator except `empty` and `ask` is a system natural-language function whose instructions are a soft body: a block
 * initialised in context from its text description (text-init.ts, owner 2026-10-10: the soft call reproduces the
 * text-instructed call at initialisation) and trained later like any other block (S5). The bodies live in a
 * standard-library `.nz` file (`buildStandardLibrary` writes one for a server; `loadStandardLibrary` reads its bytes).
 * Every combinator call and readout is recorded as an execution-graph node.
 *
 * Combinators answer in the normal call trajectory (owner 2026-10-10): the call's opening is rendered as for any call
 * and the model may work in its turns before it returns. A Neuralese result (`map`, `zip`, `ap`, `combine`, `convert`)
 * is a block the model writes as the returned value; any other result is decoded. Template readout (the first reply
 * forced to `return_result`) stays only where a value must be forced: typed `read` (the readout behind text conversion
 * and coercions), a representation-generic call used for its Neuralese form (runtime/kernel.ts) and the builtin `view`
 * write.
 */
import { currentFrame } from '../runtime/context.js';
import { softFunction } from '../runtime/contexts.js';
import { graphNode, traceFor, valueInputs } from '../native/graph.js';
import { isNeuraleseRef, NeuraleseDialectError, neuraleseRef, type NeuraleseRef } from '../native/neuralese.js';
import { emptyBlock, type NeuraleseBlock, type NeuraleseStore } from '../native/neuralese-store.js';
import { decodeNz, isSoftFunctionSpec, saveNz } from '../native/nz-file.js';
import { fetchModel } from '../model/chat-completion.js';
import { HttpNeuraleseStore } from '../model/neuralese-server.js';
import { hexDigest } from '../native/hash.js';

type Json = Record<string, unknown>;

/** The combinators that have soft bodies, with their runtime signatures and the descriptions their bodies start from. */
export const COMBINATORS = {
  map: { type: '(v: Neuralese<unknown>, f: unknown) => Neuralese<unknown>',
    text: 'Apply the function f to the value that v holds, without reading v out first, and write the result as a new Neuralese value.' },
  zip: { type: '(a: Neuralese<unknown>, b: Neuralese<unknown>) => Neuralese<unknown>',
    text: 'Write one Neuralese value that holds the pair of the values a and b.' },
  ap: { type: '(f: Neuralese<unknown>, a: unknown) => Neuralese<unknown>',
    text: 'Apply the soft function f to the argument a and write the result as a Neuralese value instead of reading it out.' },
  combine: { type: '(vs: Neuralese<unknown>[]) => Neuralese<unknown>',
    text: 'Merge several views of the same kind of value into one Neuralese value that keeps what each of them holds.' },
  split: { type: '(v: Neuralese<unknown>) => unknown',
    text: 'Write one Neuralese value for each field of the record that v holds, as a record of Neuralese values.' },
  splitList: { type: '(v: Neuralese<unknown>) => Neuralese<unknown>[]',
    text: 'Write one Neuralese value for each element of the list that v holds, in order.' },
  read: { type: '(v: Neuralese<unknown>) => unknown',
    text: 'Read the value that v holds and return it exactly, as an ordinary value of its declared type.' },
  convert: { type: '(v: Neuralese<unknown>, dialect: string) => Neuralese<unknown>',
    text: 'Write the value that v holds again, as a Neuralese value in the named dialect.' },
  gloss: { type: '(v: Neuralese<unknown>) => string',
    text: 'Describe in plain words, for a person, what the value v holds. This is a diagnostic rendering, not the value.' },
} as const;
export type CombinatorName = keyof typeof COMBINATORS;

export type TextReadSource = { schema: 'natlang.text-read-source/1'; export: 'read'; bodyId: string; type: string;
  source: string; sourceSha256: string; learnedVectors: false };
export type StandardLibrary = { dialect: string; width: number; bodies: Record<CombinatorName, string>;
  /** Explicit provider-text implementation of the read instruction; it does not claim learned-vector semantics. */
  textReadSource?: TextReadSource };

async function postJson(endpoint: string, path: string, body: unknown, headers: Record<string, string> = {}): Promise<Json> {
  const response = await fetchModel(endpoint.replace(/\/$/, '') + path, { method: 'POST',
    headers: { 'content-type': 'application/json', ...headers }, body: JSON.stringify(body) });
  if (!response.ok) throw new Error(`neuralese standard library: ${path} failed (${response.status}): ${(await response.text()).slice(0, 500)}`);
  return await response.json() as Json;
}

/** Sample arguments that render each combinator's call for its in-context initialisation (the body's context). */
const SAMPLE_VALUE = 'The meeting moved to Tuesday at 3 pm in Lyon.';
const SAMPLE_PARTNER = 'Bring the signed contract.';
const SAMPLE_FUNCTION = 'Translate the text into French.';

/**
 * Initialise every combinator body in context from its description on a Neuralese server (text-init.ts: the soft call
 * reproduces the text-instructed call at initialisation; each body's gate goes into the library's provenance) and
 * return the library and its `.nz` bytes. On Node, `buildStandardLibrary` from `@natlang/node`
 * (neuralese/node-files.ts) also writes them to `path`. `acceptFailedGate` keeps bodies whose gate failed (heads whose
 * read transport cannot reproduce text, e.g. the legacy RMS profile), recorded as such.
 */
export async function buildStandardLibrary(options: { endpoint: string; headers?: Record<string, string>;
  store?: NeuraleseStore; textProviderRead?: boolean; acceptFailedGate?: boolean;
  /** Greedy reply tokens the gate compares (default 24). */
  gateReplyTokens?: number }): Promise<{ library: StandardLibrary; bytes: Uint8Array }> {
  const { initBodyInContext } = await import('./text-init.js');
  const { createNatlangRuntime } = await import('../runtime/runtime.js');
  const { MemoryNeuraleseStore } = await import('../native/neuralese-store.js');
  const headers = options.headers ?? {};
  const remote = new HttpNeuraleseStore(options.endpoint, options.headers);
  const info = await (await fetchModel(options.endpoint.replace(/\/$/, '') + '/v1/neuralese/info', { headers: options.headers })).json() as
    { dialects: string[]; width: number };
  const local = new MemoryNeuraleseStore();
  const fetchBlock = async (id: string) => {
    if (!(await local.has(id))) {
      const block = await remote.get(id);
      if (!block) throw new Error(`neuralese standard library: the server lost block ${id}`);
      const { id: _, ...rest } = block.meta;
      await local.put({ ...rest, data: block.data });
    }
    return id;
  };
  const value = async (text: string, type: string, path = '/v1/neuralese/encode') =>
    neuraleseRef(type, await fetchBlock(String((await postJson(options.endpoint, path, { text, type }, headers)).id)));
  const v = await value(SAMPLE_VALUE, 'Neuralese<string>');
  const w = await value(SAMPLE_PARTNER, 'Neuralese<string>');
  const record = await value(JSON.stringify({ city: 'Lyon', day: 'Tuesday' }), 'Neuralese<{ city: string; day: string }>');
  const list = await value(JSON.stringify(['Lyon', 'Paris']), 'Neuralese<string[]>');
  const fnBody = await value(SAMPLE_FUNCTION, 'Neuralese<(text: string) => string>', '/v1/neuralese/embed');
  const f = softFunction({ type: '(text: string) => string', body: fnBody.$neuralese.id });
  const samples: Record<CombinatorName, (lib: ReturnType<typeof createNeuraleseLibrary>) => Promise<unknown>> = {
    map: lib => lib.map(v, f), zip: lib => lib.zip(v, w), ap: lib => lib.ap(fnBody, SAMPLE_PARTNER),
    combine: lib => lib.combine(v, w), split: lib => lib.split(record), splitList: lib => lib.splitList(list),
    read: lib => lib.read(v), convert: lib => lib.convert(v, info.dialects[0]!), gloss: lib => lib.gloss(v) };
  const bodies = {} as Record<CombinatorName, string>;
  const gates: Record<string, unknown> = {};
  const exports: Record<string, { type: string; value: unknown; description: string }> = {};
  for (const [name, entry] of Object.entries(COMBINATORS) as [CombinatorName, (typeof COMBINATORS)[CombinatorName]][]) {
    const type = `Neuralese<${entry.type}>`;
    const result = await initBodyInContext({ endpoint: options.endpoint, headers, text: entry.text, type,
      acceptFailedGate: options.acceptFailedGate, replyTokens: options.gateReplyTokens,
      render: async (placeholder, driver) => {
        await fetchBlock(placeholder);
        const all = Object.fromEntries(Object.keys(COMBINATORS).map(key => [key, placeholder])) as Record<CombinatorName, string>;
        const lib = createNeuraleseLibrary({ dialect: info.dialects[0]!, width: info.width, bodies: all });
        const runtime = createNatlangRuntime({ model: driver as never, neuralese: { store: local } });
        return runtime.run(() => samples[name](lib));
      } });
    bodies[name] = result.id;
    gates[name] = { init: result.init, gate: result.gate };
    exports[name] = { type, description: entry.text,
      value: { kind: 'soft-function', type, body: result.id, captures: {} } };
  }
  const store = { async get(id: string): Promise<NeuraleseBlock | undefined> {
    return (await options.store?.get(id)) ?? (await remote.get(id)); } } as NeuraleseStore;
  const textReadSource = options.textProviderRead ? {
    schema: 'natlang.text-read-source/1' as const, export: 'read' as const, bodyId: bodies.read,
    type: `Neuralese<${COMBINATORS.read.type}>`, source: COMBINATORS.read.text,
    sourceSha256: hexDigest(COMBINATORS.read.text), learnedVectors: false as const,
  } : undefined;
  const bytes = await saveNz(exports, { store, dialect: info.dialects[0]!, provenance: { kind: 'natlang-standard-library',
    initialisation: 'text-in-context', init_gates: gates, ...(textReadSource ? { text_read_source: textReadSource } : {}) } });
  return { library: { dialect: info.dialects[0]!, width: info.width, bodies, ...(textReadSource ? { textReadSource } : {}) }, bytes };
}
/**
 * Read a standard-library `.nz` file from its bytes, registering its blocks in `store`. On Node, `loadStandardLibrary`
 * from `@natlang/node` (neuralese/node-files.ts) also takes a path.
 */
export async function loadStandardLibrary(bytes: Uint8Array, store?: NeuraleseStore): Promise<StandardLibrary> {
  if (typeof bytes === 'string')
    throw new TypeError('loadStandardLibrary takes the .nz file\'s bytes here; read the file first (fetch(url).then(r => r.arrayBuffer())), or use @natlang/node, which also takes a path');
  const { header, blocks } = decodeNz(bytes);
  if (store) for (const block of blocks.values()) if (!(await store.has(block.meta.id))) {
    const { id: _, ...rest } = block.meta;
    await store.put({ ...rest, data: block.data });
  }
  const bodies = {} as Record<CombinatorName, string>;
  const exports = header.exports as Record<string, { value?: unknown; description?: string }>;
  let width = 0, dialect = '';
  for (const name of Object.keys(COMBINATORS) as CombinatorName[]) {
    const value = exports[name]?.value as Json | undefined;
    const spec = value?.['$neuralese-fn'] as { body?: string } | undefined ?? (isSoftFunctionSpec(value) ? value : undefined);
    if (!spec?.body) throw new Error(`the standard library has no soft body for ${name}`);
    bodies[name] = spec.body;
    const block = blocks.get(spec.body);
    if (block) { width = block.meta.width; dialect = block.meta.dialect; }
  }
  let textReadSource: TextReadSource | undefined;
  const candidate = header.provenance?.text_read_source;
  if (candidate !== undefined) {
    const source = candidate as Partial<TextReadSource>;
    if (source.schema !== 'natlang.text-read-source/1' || source.export !== 'read' || source.bodyId !== bodies.read ||
        source.type !== `Neuralese<${COMBINATORS.read.type}>` || typeof source.source !== 'string' ||
        source.source !== exports.read?.description || source.sourceSha256 !== hexDigest(source.source) ||
        source.learnedVectors !== false || !blocks.has(bodies.read))
      throw new Error('the standard library text read source does not match its declared read body, type, description, or digest');
    textReadSource = source as TextReadSource;
  }
  return { dialect, width, bodies, ...(textReadSource ? { textReadSource } : {}) };
}

const record = (kind: 'combinator' | 'readout', data: Json, inputs: unknown) => {
  const trace = traceFor(currentFrame()?.parentCallId);
  return graphNode(trace, kind, data, valueInputs(inputs, 'args'));
};

/** The combinators over a standard library. */
export function createNeuraleseLibrary(library: StandardLibrary) {
  const functions = new Map<CombinatorName, ReturnType<typeof softFunction>>();
  const fn = (name: CombinatorName) => {
    let found = functions.get(name);
    if (!found) functions.set(name, found = softFunction({ type: COMBINATORS[name].type, body: library.bodies[name], name: `natlang.${name}` }));
    return found;
  };
  const call = async (name: CombinatorName, args: unknown[]) => {
    record('combinator', { combinator: name, call_id: currentFrame()?.parentCallId ?? null }, args);
    return (fn(name) as unknown as (...a: unknown[]) => Promise<unknown>)(...args);
  };
  const elementType = (ref: NeuraleseRef) => /^Neuralese<(.*)>$/.exec(ref.$neuralese.type)?.[1] ?? 'unknown';
  const lib = {
    map: (v: NeuraleseRef, f: unknown) => call('map', [v, f]),
    zip: (a: NeuraleseRef, b: NeuraleseRef) => call('zip', [a, b]),
    ap: (f: NeuraleseRef, a: unknown) => call('ap', [f, a]),
    combine: (...vs: NeuraleseRef[]) => vs.length === 1 ? Promise.resolve(vs[0]) : call('combine', [vs]),
    /** The zero-length value: the identity of `combine`. */
    empty: (): NeuraleseRef => neuraleseRef('Neuralese<unknown>', emptyId(library)),
    split: (v: NeuraleseRef) => call('split', [v]),
    splitList: (v: NeuraleseRef) => call('splitList', [v]),
    /** Typed readout; an ordinary string is already its own crisp value. */
    async read(v: NeuraleseRef | string): Promise<unknown> {
      if (typeof v === 'string') return v;
      if (!isNeuraleseRef(v)) throw new TypeError('read needs a Neuralese value');
      record('readout', { type: elementType(v), call_id: currentFrame()?.parentCallId ?? null }, [v]);
      // Typed readout must produce exactly a value of the declared type (JS coercions, text conversion and the text
      // transport's typed-readout provenance depend on it): it keeps template readout.
      const typed = softFunction({ type: `(v: ${v.$neuralese.type}) => ${elementType(v)}`, body: library.bodies.read, name: 'natlang.read',
        readout: 'template', adHoc: false });
      return (typed as unknown as (value: unknown) => Promise<unknown>)(v);
    },
    convert: (v: NeuraleseRef, dialect: string) => call('convert', [v, dialect]),
    gloss: (v: NeuraleseRef) => call('gloss', [v]) as Promise<string>,
    /**
     * The query operator: the answer to `question` from what `v` holds, by its definition `read(map(v, question))`:
     * `map` writes the answer as a `Neuralese<string>` and `read` reads it out. Its law (S0 §4.2) is its target:
     * answering from the block matches answering from the full text.
     */
    async ask(v: NeuraleseRef, question: string): Promise<string> {
      if (!isNeuraleseRef(v)) throw new TypeError('ask needs a Neuralese value and a question');
      if (typeof question !== 'string' || !question.trim()) throw new TypeError('ask needs a question (a non-empty string)');
      record('combinator', { combinator: 'map', call_id: currentFrame()?.parentCallId ?? null }, [v, question]);
      const map = softFunction({ type: `(v: ${v.$neuralese.type}, f: string) => Neuralese<string>`, body: library.bodies.map,
        name: 'natlang.map', adHoc: false });
      const answer = await (map as unknown as (value: unknown, f: string) => Promise<unknown>)(v, question);
      if (!isNeuraleseRef(answer)) throw new TypeError('ask: map did not write its answer as a Neuralese value');
      return await lib.read(answer) as string;
    },
  };
  return lib;
}

/** Raised when source-level text conversion has no typed readout available in this task. */
export class NeuraleseReadoutCapabilityError extends Error {
  readonly code = 'neuralese-readout-unavailable';
  readonly capability = 'typed-readout';
  constructor(detail = 'the task has no configured Neuralese standard library with a read body available in its block store') {
    super(`Converting a Neuralese value to text needs typed readout: ${detail}. ` +
      'Configure services.neuralese and its block store, or pass the Neuralese value to code that can handle it softly.');
    this.name = 'NeuraleseReadoutCapabilityError';
  }
}

/** Compiler and eval `read(value)` plus JavaScript text coercion delegate to this typed readout. */
export async function readNeuraleseForCurrentTask(value: unknown): Promise<unknown> {
  // A crisp string is already the exact readable value. Keep this idempotent without
  // treating numbers, records, or other arbitrary values as soft references.
  if (typeof value === 'string') return value;
  if (!isNeuraleseRef(value)) throw new TypeError('typed Neuralese readout requires a Neuralese reference');
  const task = currentFrame()?.task;
  const library = (task?.services as Json | undefined)?.neuralese as StandardLibrary | undefined;
  if (!library?.bodies?.read) throw new NeuraleseReadoutCapabilityError('services.neuralese is missing or has no read body');
  const store = task?.runtime.options.neuralese?.store;
  if (!store || !await store.has(library.bodies.read))
    throw new NeuraleseReadoutCapabilityError(`read body ${library.bodies.read} is unavailable in the task block store`);
  // The read body reads its library's dialect only. As at coercion (native/values.ts), a block whose dialect the
  // task's store cannot tell without I/O is not checked here.
  const written = store.peek?.(value.$neuralese.id)?.dialect;
  if (library.dialect && written !== undefined && written !== library.dialect)
    throw new NeuraleseDialectError(`read(value) reads dialect ${JSON.stringify(library.dialect)}, but block ` +
      `${value.$neuralese.id} was written in ${JSON.stringify(written)}; convert it first with ` +
      `\`convert(value, ${JSON.stringify(library.dialect)})\`, or read it with a standard library of its dialect`);
  return createNeuraleseLibrary(library).read(value);
}

/** The content ID of the zero-length block of a library's dialect and width (the identity of `combine`). */
function emptyId(library: StandardLibrary): string {
  return emptyBlock(library.dialect, library.width).meta.id;
}

/** The `natlang:neuralese` module: bound to the current task's `neuralese` service (a `StandardLibrary`). */
export const neuraleseModule = new Proxy({ __esModule: true } as Record<string, unknown>, {
  get(target, name) {
    if (name === '__esModule') return true;
    if (typeof name !== 'string' || name === 'then') return undefined;
    const library = (currentFrame()?.task.services as Json | undefined)?.neuralese as StandardLibrary | undefined;
    if (!library?.bodies) throw new Error('natlang:neuralese needs the task\'s neuralese service (a loaded standard library)');
    return (createNeuraleseLibrary(library) as Record<string, unknown>)[name];
  },
});
