/**
 * The Neuralese parity program (plans/neuralese/S4_RUNTIME_SERVERS.md §10: "a Natlang program that writes, stores,
 * imports, rebinds, combines and reads Neuralese values runs identically on Node and in the browser"). One natlang
 * project, compiled and run by whichever runtime namespace the host passes in: `dist/index.js` on Node against the
 * llama.cpp fork's native server, `dist/browser/natlang.js` in Chromium against the WebAssembly engine.
 *
 * It records what a client can observe: every model request as sent (the rendered prompt: messages, tools, template)
 * and its reply, every decision readout, each step's value, and the payload of every block it saw, so the driver
 * (scripts/browser-neuralese-parity.mjs) can compare the two hosts. Block IDs are content hashes of float payloads and
 * differ whenever the floats differ in their last bits, so the trace names blocks by order of first appearance and the
 * payloads are compared within the conformance tolerance instead.
 *
 * Host-neutral: no `node:*` import, no DOM.
 */

export const TEXT = 'The design review moved from Monday to Tuesday at 3 pm in room 4B; Dana will bring the budget sheet.';
export const QUESTION = 'When is the design review?';

/** The project: a template write, a decision over the written value, the generic builtin `view` at both
 * representations, and the standard library's `read` and `ask`. */
export function projectFiles(viewSource) {
  return {
    'note.nl': '---\nargs:\n  text: string\nreturns: Neuralese<string>\nreadout: template\n---\nNote what the text says, for later.\n',
    'meeting.nl': '---\nargs:\n  note: Neuralese<string>\nreturns: boolean\nreadout: decision\n---\nIs the note about a meeting?\n',
    'view.nl': viewSource,
    'main.ts': `import note from './note.nl';
import meeting from './meeting.nl';
import view from './view.nl';
import { read, ask } from 'natlang:neuralese';

export async function write(text: string): Promise<Neuralese<string>> {
  const n: Neuralese<string> = await note(text);
  return n;
}
export async function readBack(n: Neuralese<string>): Promise<string> {
  return String(await read(n));
}
export async function decide(n: Neuralese<string>): Promise<boolean> {
  return await meeting(n);
}
export async function softView(text: string): Promise<Neuralese<string>> {
  const v: Neuralese<string> = await view(text);
  return v;
}
export async function crispView(text: string): Promise<string> {
  const v: string = await view(text);
  return v;
}
export async function query(n: Neuralese<string>, question: string): Promise<string> {
  return await ask(n, question);
}
`,
  };
}

const BLOCK_ID = /nz1_[a-z2-7]{52}/g;

/** Block payload as plain floats (f32, f16 or bf16 storage). */
export function payloadFloats(block) {
  const { dtype } = block.meta, bytes = block.data;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = [];
  if (dtype === 'f32') for (let i = 0; i < bytes.length; i += 4) out.push(view.getFloat32(i, true));
  else if (dtype === 'bf16') for (let i = 0; i < bytes.length; i += 2) {
    const scratch = new DataView(new ArrayBuffer(4));
    scratch.setUint16(2, view.getUint16(i, true), false);
    out.push(scratch.getFloat32(0, false));
  } else if (dtype === 'f16') for (let i = 0; i < bytes.length; i += 2) {
    const h = view.getUint16(i, true), sign = h & 0x8000 ? -1 : 1, exp = (h >> 10) & 0x1f, frac = h & 0x3ff;
    out.push(exp === 0 ? sign * 2 ** -14 * (frac / 1024) : exp === 31 ? (frac ? NaN : sign * Infinity) : sign * 2 ** (exp - 15) * (1 + frac / 1024));
  } else throw new Error(`unknown block dtype ${dtype}`);
  return out;
}

/**
 * Names blocks by first appearance (`<nz:k>`) and drops what is per-run noise (completion IDs, timestamps, timings),
 * so two hosts' traces compare with deepEqual.
 */
export function blockNamer() {
  const order = new Map();
  const name = id => { if (!order.has(id)) order.set(id, order.size); return `<nz:${order.get(id)}>`; };
  const NOISE = new Set(['created', 'system_fingerprint', 'timings', 'usage', 'ms', 'elapsed_ms', 'duration_ms']);
  const normalize = value => {
    if (typeof value === 'string') return value.replace(BLOCK_ID, name);
    if (Array.isArray(value)) return value.map(normalize);
    if (value && typeof value === 'object') {
      const out = {};
      for (const [key, item] of Object.entries(value)) if (!NOISE.has(key)) out[key] = normalize(item);
      return out;
    }
    return value;
  };
  return { normalize, ids: () => [...order.keys()] };
}

/**
 * Run the program on one host. `natlang` is the runtime namespace, `endpoint` the Neuralese server (an HTTP URL, or the
 * in-page engine's endpoint), `store` the runtime's block store. Returns the raw trace; `blocks` maps every block ID
 * the trace mentions to its payload floats.
 */
export async function runParityProgram({ natlang, endpoint, store, viewSource, target, log = () => {} }) {
  const exchanges = [], decisions = [], steps = [];
  const info = await (await natlang.fetchModel(endpoint.replace(/\/$/, '') + '/v1/neuralese/info')).json();
  const base = natlang.neuraleseServerModelTurn({ endpoint, model: 'natlang-neuralese', store, request: { temperature: 0 },
    onExchange: exchange => {
      const { stream: _, ...request } = exchange.wireRequest ?? {};
      const response = exchange.wireResponse ?? {};
      const choice = response.choices?.[0] ?? {};
      exchanges.push({ request, reply: { message: choice.message ?? null, finish_reason: choice.finish_reason ?? null,
        blocks: (response.neuralese?.blocks ?? []).map(block => ({ id: block.id, length: block.length, truncated: Boolean(block.truncated),
          type: block.type ?? null, stop_logits: block.producer?.stop_logits ?? null })) } });
    } });
  const driver = Object.assign((request, signal, options) => base(request, signal, options), { neuralese: true, blocks: base.blocks,
    decide: async (question, signal) => {
      let scores;
      try { scores = await base.decide(question, signal); } catch (error) {
        decisions.push({ messages: question.messages, options: question.options, error: String(error?.message ?? error) });
        throw error;
      }
      decisions.push({ messages: question.messages, options: question.options, scores });
      return scores;
    } });
  // The standard library: each combinator body encoded on this host's server, then loaded into the runtime's store.
  let started = Date.now();
  const { bytes } = await natlang.buildStandardLibrary({ endpoint, store });
  const library = await natlang.loadStandardLibrary(bytes, store);
  steps.push({ step: 'standard-library', value: library.bodies, ms: Date.now() - started });
  const runtime = natlang.createNatlangRuntime({ model: { driver, maxTurns: 3, turnTokens: 96, temperature: 0 },
    neuralese: { store, dialect: info.dialects[0] }, services: { neuralese: library }, calls: false, specialization: 'off',
    seed: { mode: 'backend' } });
  const compiled = natlang.compileVirtualProject({ files: projectFiles(viewSource) }, natlang, { target });
  if (!compiled.ok) throw new Error(`the parity program does not compile: ${JSON.stringify(compiled.diagnostics).slice(0, 2000)}`);
  const app = compiled.require('main.ts');
  const step = async (name, fn) => {
    const first = exchanges.length, firstDecision = decisions.length;
    started = Date.now();
    let value, error;
    try { value = await runtime.run(fn); } catch (failure) { error = String(failure?.message ?? failure); }
    steps.push({ step: name, ...(error === undefined ? { value } : { error }), exchanges: [first, exchanges.length],
      decisions: [firstDecision, decisions.length], ms: Date.now() - started });
    log(`${name}: ${Date.now() - started} ms ${JSON.stringify(error ?? value).slice(0, 200)}`);
    return value;
  };
  const note = await step('write', () => app.write(TEXT));
  if (note?.$neuralese) {
    await step('read-back', () => app.readBack(note));
    await step('decide', () => app.decide(note));
  }
  await step('view-neuralese', () => app.softView(TEXT));
  await step('view-string', () => app.crispView(TEXT));
  if (note?.$neuralese) await step('ask', () => app.query(note, QUESTION));

  // Streaming: a plain turn and a turn that writes a block, each with its deltas assembled, and the plain reply
  // requested whole for comparison.
  const streaming = {};
  for (const [name, request] of [
    ['plain', { messages: [{ role: 'user', content: 'Name three colours, then stop.' }], tools: [], seed: null, max_tokens: 24 }],
    ['write', { messages: [{ role: 'user', content: 'Write a note about the review.' }], tools: [], seed: null, max_tokens: 16 }],
  ]) {
    const deltas = [];
    const turnDriver = name === 'write' ? natlang.neuraleseServerModelTurn({ endpoint, model: 'natlang-neuralese', store,
      request: { temperature: 0, x_natlang_forced: ['Note: ', { neuralese: 'write' }, ' done'] } }) : base;
    started = Date.now();
    const turn = await turnDriver(request, undefined, { onDelta: delta => deltas.push(delta) });
    const assembled = deltas.filter(delta => delta.type === 'text').map(delta => delta.text).join('');
    streaming[name] = { deltas: deltas.length, delta_types: [...new Set(deltas.map(delta => delta.type))].sort(), assembled,
      text: turn.text ?? null, neuralese_parts: deltas.filter(delta => delta.type === 'neuralese').map(delta => delta.part?.id ?? null),
      turn: JSON.parse(JSON.stringify(turn)), ms: Date.now() - started };
  }
  const whole = await natlang.fetchModel(endpoint.replace(/\/$/, '') + '/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ messages: [{ role: 'user', content: 'Name three colours, then stop.' }], max_tokens: 24, temperature: 0, stream: false }) });
  streaming.plain.whole = (await whole.json()).choices?.[0]?.message?.content ?? null;
  streaming.server_streams = info.stream === true;

  // Every block the trace names, with its payload from the runtime's store (or the server's, for one never copied).
  const ids = new Set(JSON.stringify({ steps, exchanges, decisions, streaming }).match(BLOCK_ID) ?? []);
  const blocks = {};
  for (const id of ids) {
    const block = (await store.get(id)) ?? (await base.blocks.get(id));
    blocks[id] = block ? { stored: await store.has(id), length: block.meta.length, width: block.meta.width, dtype: block.meta.dtype, type: block.meta.type ?? null,
      payload: payloadFloats(block) } : null;
  }
  return { info: { dialects: info.dialects, width: info.width, cutoff: info.cutoff, max_block_length: info.max_block_length, stream: info.stream ?? null },
    library: library.bodies, note: note ?? null, steps, exchanges, decisions, streaming, blocks };
}
