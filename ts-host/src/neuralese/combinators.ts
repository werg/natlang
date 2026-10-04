/**
 * `natlang:neuralese`, the combinator library (spec/SPEC.md, Neuralese chapter; S0 §4): `map`, `zip`, `ap`, `combine`,
 * `empty`, `split`, `splitList`, `read`, `convert`, `gloss`.
 *
 * Each combinator except `empty` is a system natural-language function whose instructions are a soft body: a block
 * initialised from a text description (encoded in one pass through the port, or token embeddings on a server without
 * `encode`) and trained later like any other block (S5). The bodies live in a standard-library `.nz` file (`buildStandardLibrary` writes one for a server;
 * `loadStandardLibrary` reads it). Every combinator call and readout is recorded as an execution-graph node.
 *
 * Combinators answer by template readout, not free decoding: the call's opening is rendered as for any call, and its
 * first reply is forced to `return_result(status="success", value=…)`. A Neuralese result (`map`, `zip`, `ap`,
 * `combine`, `convert`) is written as a block at the value; any other result (`read`, `gloss`, and the record or list
 * of `split` and `splitList`, whose elements are blocks the model writes as it decodes) is decoded after the forced
 * call opening. The trajectory keeps the agentic format, so the same template trains the combinators.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { currentFrame } from '../runtime/context.js';
import { softFunction } from '../runtime/contexts.js';
import { graphNode, traceFor, valueInputs } from '../native/graph.js';
import { isNeuraleseRef, neuraleseRef, type NeuraleseRef } from '../native/neuralese.js';
import { emptyBlock, type NeuraleseBlock, type NeuraleseStore } from '../native/neuralese-store.js';
import { decodeNz, isSoftFunctionSpec, saveNz } from '../native/nz-file.js';
import { fetchModel } from '../model/chat-completion.js';
import { HttpNeuraleseStore } from '../model/neuralese-server.js';

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

export type StandardLibrary = { dialect: string; width: number; bodies: Record<CombinatorName, string> };

async function postJson(endpoint: string, path: string, body: unknown, headers: Record<string, string> = {}): Promise<Json> {
  const response = await fetchModel(endpoint.replace(/\/$/, '') + path, { method: 'POST',
    headers: { 'content-type': 'application/json', ...headers }, body: JSON.stringify(body) });
  if (!response.ok) throw new Error(`neuralese standard library: ${path} failed (${response.status}): ${(await response.text()).slice(0, 500)}`);
  return await response.json() as Json;
}

/** Initialise every combinator body from its description on a Neuralese server and write the library as a `.nz` file. */
export async function buildStandardLibrary(options: { endpoint: string; headers?: Record<string, string>; path?: string;
  store?: NeuraleseStore }): Promise<{ library: StandardLibrary; bytes: Uint8Array }> {
  const remote = new HttpNeuraleseStore(options.endpoint, options.headers);
  const info = await (await fetchModel(options.endpoint.replace(/\/$/, '') + '/v1/neuralese/info', { headers: options.headers })).json() as
    { dialects: string[]; width: number };
  const bodies = {} as Record<CombinatorName, string>;
  const exports: Record<string, { type: string; value: unknown; description: string }> = {};
  for (const [name, entry] of Object.entries(COMBINATORS) as [CombinatorName, (typeof COMBINATORS)[CombinatorName]][]) {
    // Encoded in one pass through the port (decision 42); servers without `encode` fall back to token embeddings.
    const meta = await postJson(options.endpoint, '/v1/neuralese/encode', { text: entry.text, type: `Neuralese<${entry.type}>` }, options.headers)
      .catch(() => postJson(options.endpoint, '/v1/neuralese/embed', { text: entry.text, type: `Neuralese<${entry.type}>` }, options.headers));
    bodies[name] = String(meta.id);
    exports[name] = { type: `Neuralese<${entry.type}>`, description: entry.text,
      value: { kind: 'soft-function', type: `Neuralese<${entry.type}>`, body: String(meta.id), captures: {} } };
  }
  const store = { async get(id: string): Promise<NeuraleseBlock | undefined> {
    return (await options.store?.get(id)) ?? (await remote.get(id)); } } as NeuraleseStore;
  const bytes = await saveNz(exports, { store, dialect: info.dialects[0]!, provenance: { kind: 'natlang-standard-library',
    initialisation: 'text-embeddings' } });
  if (options.path) writeFileSync(options.path, bytes);
  return { library: { dialect: info.dialects[0]!, width: info.width, bodies }, bytes };
}

/** Read a standard-library `.nz` file, registering its blocks in `store`. */
export async function loadStandardLibrary(source: string | Uint8Array, store?: NeuraleseStore): Promise<StandardLibrary> {
  const bytes = typeof source === 'string' ? new Uint8Array(readFileSync(source)) : source;
  const { header, blocks } = decodeNz(bytes);
  if (store) for (const block of blocks.values()) if (!(await store.has(block.meta.id))) {
    const { id: _, ...rest } = block.meta;
    await store.put({ ...rest, data: block.data });
  }
  const bodies = {} as Record<CombinatorName, string>;
  let width = 0, dialect = '';
  for (const name of Object.keys(COMBINATORS) as CombinatorName[]) {
    const value = (header.exports as Record<string, { value?: unknown }>)[name]?.value as Json | undefined;
    const spec = value?.['$neuralese-fn'] as { body?: string } | undefined ?? (isSoftFunctionSpec(value) ? value : undefined);
    if (!spec?.body) throw new Error(`the standard library has no soft body for ${name}`);
    bodies[name] = spec.body;
    const block = blocks.get(spec.body);
    if (block) { width = block.meta.width; dialect = block.meta.dialect; }
  }
  return { dialect, width, bodies };
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
    if (!found) functions.set(name, found = softFunction({ type: COMBINATORS[name].type, body: library.bodies[name], name: `natlang.${name}`,
      readout: 'template' }));
    return found;
  };
  const call = async (name: CombinatorName, args: unknown[]) => {
    record('combinator', { combinator: name, call_id: currentFrame()?.parentCallId ?? null }, args);
    return (fn(name) as unknown as (...a: unknown[]) => Promise<unknown>)(...args);
  };
  const elementType = (ref: NeuraleseRef) => /^Neuralese<(.*)>$/.exec(ref.$neuralese.type)?.[1] ?? 'unknown';
  return {
    map: (v: NeuraleseRef, f: unknown) => call('map', [v, f]),
    zip: (a: NeuraleseRef, b: NeuraleseRef) => call('zip', [a, b]),
    ap: (f: NeuraleseRef, a: unknown) => call('ap', [f, a]),
    combine: (...vs: NeuraleseRef[]) => vs.length === 1 ? Promise.resolve(vs[0]) : call('combine', [vs]),
    /** The zero-length value: the identity of `combine`. */
    empty: (): NeuraleseRef => neuraleseRef('Neuralese<unknown>', emptyId(library)),
    split: (v: NeuraleseRef) => call('split', [v]),
    splitList: (v: NeuraleseRef) => call('splitList', [v]),
    /** Typed readout: the result is checked against the value's own element type. */
    async read(v: NeuraleseRef): Promise<unknown> {
      if (!isNeuraleseRef(v)) throw new TypeError('read needs a Neuralese value');
      record('readout', { type: elementType(v), call_id: currentFrame()?.parentCallId ?? null }, [v]);
      const typed = softFunction({ type: `(v: ${v.$neuralese.type}) => ${elementType(v)}`, body: library.bodies.read, name: 'natlang.read',
        readout: 'template' });
      return (typed as unknown as (value: unknown) => Promise<unknown>)(v);
    },
    convert: (v: NeuraleseRef, dialect: string) => call('convert', [v, dialect]),
    gloss: (v: NeuraleseRef) => call('gloss', [v]) as Promise<string>,
  };
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
