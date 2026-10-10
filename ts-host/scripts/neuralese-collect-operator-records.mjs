#!/usr/bin/env node
/**
 * Collect replay records for training the combinator bodies (plans/neuralese/S5_PROGRAM_TRAINING.md §4.3, §5.1).
 *
 * For every operator case (natlang_neuralese.data.operator_cases) this runs small natlang programs over the standard
 * library against a Neuralese server, each inside `valueAndGrad` with the library's bodies as the arguments, and keeps
 * every gradient session as a replay record (natlang:learning `replayRecords`):
 *
 * | Program | Loss (cross-entropy of the checked output) |
 * | --- | --- |
 * | `read(v)` | the exact span v was encoded from |
 * | `read(map(v, f))` | f(span), computed exactly by the case builder; f is a soft function from its instruction |
 * | `read(split(zip(a, b))[0])`, `[1]` | the first and second span |
 * | law `combineIdentity`, `splitZip` | the right side's readout (objectives.law) |
 *
 * Multi-call programs replay whole: `read` consumes the block `map` wrote, so its loss reaches the `map` body through
 * the producer chain. Records go to OUT (JSONL) with their blocks in OUT-without-.jsonl + `.blocks/`.
 *
 *   node scripts/neuralese-collect-operator-records.mjs --endpoint URL --library stdlib.nz --cases cases.jsonl \
 *     --out records.jsonl [--programs read,map,zipsplit,laws] [--limit N] [--gate-reply-tokens 8]
 */
import { readFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { createNatlangRuntime, createLearning, learningService, loadStandardLibrary, createNeuraleseLibrary, softFunction,
  replayRecordSink, initBodyInContext } from '../dist/index.js';
import { MemoryNeuraleseStore } from '../dist/native/neuralese-store.js';
import { neuraleseServerModelTurn } from '../dist/model/neuralese-server.js';

const { values: args } = parseArgs({ options: {
  endpoint: { type: 'string' }, library: { type: 'string' }, cases: { type: 'string' }, out: { type: 'string' },
  programs: { type: 'string', default: 'read,map,zipsplit,laws' }, limit: { type: 'string', default: '0' },
  split: { type: 'string', default: '' }, 'gate-reply-tokens': { type: 'string', default: '8' }, model: { type: 'string', default: 'natlang-neuralese' } } });
for (const key of ['endpoint', 'library', 'cases', 'out']) if (!args[key]) throw new Error(`--${key} is required`);

const programs = new Set(args.programs.split(',').filter(Boolean));
const store = new MemoryNeuraleseStore();
const library = await loadStandardLibrary(args.library, store);
const lib = createNeuraleseLibrary(library);
let current = {};
const sink = replayRecordSink({ path: args.out, endpoint: args.endpoint, annotate: () => current });
const runtime = createNatlangRuntime({ model: neuraleseServerModelTurn({ endpoint: args.endpoint, model: args.model, store }),
  neuralese: { store } });
const { valueAndGrad, objectives } = createLearning(learningService({ endpoint: args.endpoint, store, replayRecords: sink }),
  { library });

async function post(path, body) {
  const response = await fetch(args.endpoint.replace(/\/$/, '') + path, { method: 'POST',
    headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  if (!response.ok) throw new Error(`${path}: ${response.status} ${(await response.text()).slice(0, 300)}`);
  return response.json();
}
const encode = async (text, type = 'Neuralese<string>') => ({ $neuralese: { type, id: (await post('/v1/neuralese/encode', { text, type })).id } });
// A transform's soft body is initialised in context (src/neuralese/text-init.ts): in a call of the soft function on the
// case's text, so that at initialisation it is the text-instructed function. Its gate is kept in the case's records.
const FUNCTION_TYPE = 'Neuralese<(text: string) => string>';
let gate = null;
async function instructionBody(instruction, text) {
  const result = await initBodyInContext({ endpoint: args.endpoint, text: instruction, type: FUNCTION_TYPE,
    acceptFailedGate: true, replyTokens: Number(args['gate-reply-tokens']),
    render: (placeholder, driver) => createNatlangRuntime({ model: driver, neuralese: { store } })
      .run(() => softFunction({ type: '(text: string) => string', body: placeholder })(text)) });
  gate = { init: result.init, gate: result.gate };
  return result.id;
}

// The library bodies are the trained values: every program's gradient session takes them as its arguments.
const bodies = Object.fromEntries(Object.entries(library.bodies).map(([name, id]) =>
  [name, { $neuralese: { type: `Neuralese<${name}>`, id } }]));

const cases = readFileSync(args.cases, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line))
  .filter(item => !args.split || item.split === args.split);
const limit = Number(args.limit) || cases.length;
const counts = {};
let failures = 0;
async function collect(meta, program) {
  current = meta;
  try {
    await valueAndGrad(() => program(), bodies);
    counts[meta.operator] = (counts[meta.operator] ?? 0) + 1;
  } catch (error) {
    failures++;
    console.error(JSON.stringify({ case: meta.case, operator: meta.operator, error: String(error?.message ?? error).slice(0, 300) }));
  }
}

for (const item of cases.slice(0, limit)) {
  const base = { case: item.id, group: item.group, split: item.split, corpus: item.corpus };
  const v = await encode(item.text);
  if (programs.has('read'))
    await collect({ ...base, operator: 'read', expected: item.text },
      () => objectives.crossEntropy(() => runtime.run(() => lib.read(v)), item.text));
  if (programs.has('map')) {
    const f = softFunction({ type: '(text: string) => string', body: await instructionBody(item.map.instruction, item.text) });
    await collect({ ...base, operator: 'map', transform: item.map.transform, expected: item.map.expected, init: gate },
      () => objectives.crossEntropy(() => runtime.run(async () => lib.read(await lib.map(v, f))), item.map.expected));
  }
  if (programs.has('zipsplit')) {
    const b = await encode(item.partner);
    for (const [index, expected] of [[0, item.text], [1, item.partner]])
      await collect({ ...base, operator: 'zip-split', part: index, expected },
        () => objectives.crossEntropy(() => runtime.run(async () => {
          const parts = await lib.split(await lib.zip(v, b));
          const list = Array.isArray(parts) ? parts : Object.values(parts ?? {});
          if (list.length !== 2) throw new Error(`split returned ${list.length} parts`);
          return lib.read(list[index]);
        }), expected));
  }
  if (programs.has('laws')) {
    await collect({ ...base, operator: 'law', law: 'combineIdentity' },
      () => runtime.run(() => objectives.law('combineIdentity', v)));
    const b = await encode(item.partner);
    await collect({ ...base, operator: 'law', law: 'splitZip' },
      () => runtime.run(() => objectives.law('splitZip', v, b)));
  }
}
console.log(JSON.stringify({ records: sink.count, by_operator: counts, failures, out: args.out }));
