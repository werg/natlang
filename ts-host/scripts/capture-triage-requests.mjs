#!/usr/bin/env node
// The model requests of one examples/triage run, as the runtime sends them, for the prompt cache map
// (scripts/neuralese_data/prompt_cache_map.py --turns): one JSON line per episode, {key, turns: [{messages, tools,
// invocation_id}]}. A scripted model answers each call (classify by its rubric, urgency false, text calls their input),
// so the requests are the runtime's real prompts; only the model's replies are scripted.
//
//   node scripts/capture-triage-requests.mjs OUT.jsonl [RUNS]
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { buildProject, createNatlangRuntime } from '../dist/index.js';
import { scriptedModel } from '../test/support/natlang.mjs';

const [out, runs = '2'] = process.argv.slice(2);
if (!out) { console.error('usage: capture-triage-requests.mjs OUT.jsonl [RUNS]'); process.exit(2); }
const example = fileURLToPath(new URL('../../examples/triage', import.meta.url));
const runtimeModule = { url: new URL('../dist/index.js', import.meta.url).href, path: fileURLToPath(new URL('../dist/index.js', import.meta.url)),
  types: fileURLToPath(new URL('../dist/index.d.ts', import.meta.url)), specifiers: ['@natlang/node'] };
const outDir = mkdtempSync(join(tmpdir(), 'natlang-triage-capture-'));
const built = buildProject({ project: example, outDir, runtimeModule, writeDeclarations: false });
if (!built.ok) throw new Error(JSON.stringify(built.diagnostics));
const { triage } = await import(pathToFileURL(join(outDir, 'triage.js')).href);
const tickets = JSON.parse(readFileSync(join(example, 'tickets.json'), 'utf8'));
const lines = [];
for (let run = 0; run < Number(runs); run++) {
  const scripted = scriptedModel(opening => {
    if (opening.includes('Pick the label')) return /WATCHES/.test(opening) ? 'return "spam"' : 'return "billing"';
    if (opening.includes('attention within the hour') || opening.includes('A ticket is urgent')) return 'return true';
    if (opening.includes('Write one paragraph')) return 'return "Checkout is down and a customer was charged twice."';
    if (opening.includes('noticeably shorter')) return 'return "Checkout down; double charge."';
    return 'return true';   // property checks (holds) and any other judgment
  });
  const turns = [];
  const driver = async request => {
    turns.push({ messages: structuredClone(request.messages), tools: structuredClone(request.tools ?? []),
      invocation_id: null });
    if (!request.tools?.length) {   // a decision readout: the first offered value
      const offered = /these JSON values and nothing else: (.*)\.$/.exec(String(request.messages.at(-1)?.content ?? ''))?.[1];
      return { text: JSON.stringify(offered ? JSON.parse(`[${offered}]`)[0] : true), calls: [] };
    }
    return scripted.driver(request);
  };
  await createNatlangRuntime({ model: driver }).run(() => triage(run ? tickets.slice().reverse() : tickets, 'billing, technical, spam'));
  lines.push(JSON.stringify({ key: `triage-${run}`, turns }));
}
writeFileSync(out, lines.join('\n') + '\n');
console.log(JSON.stringify({ episodes: lines.length, out }));
