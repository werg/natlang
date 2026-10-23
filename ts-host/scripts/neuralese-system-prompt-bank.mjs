#!/usr/bin/env node
/** Build the natively initialized soft system-prompt bank for a native-state server's model.
 *
 * Usage: neuralese-system-prompt-bank.mjs --endpoint URL --out system-prompts.nz [--pieces id,id,...]
 *
 * Each prompt piece becomes a `Neuralese<SystemPrompt>` native state initialized in the actual sample call context.
 * The file has one export, `systemPrompts`, a record from piece ID to soft form; its provenance names each piece's
 * text digest, native source capture and runtime version. Training updates
 * the blocks (natlang_neuralese.train); the runtime uses a bank through `neuralese.systemPrompts`:
 *
 *   const { systemPrompts } = nzExports(await loadNz(bytes, { store }));
 *   createNatlangRuntime({ ..., neuralese: { store, systemPrompts: systemPromptBank(systemPrompts) } });
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { promptPieces, SYSTEM_PROMPT_TYPE, systemPromptBank, createNatlangRuntime, loadVirtualNatlang, initBodyInContext }
  from '../dist/index.js';
import { HttpNeuraleseStore } from '../dist/model/neuralese-server.js';
import { MemoryNeuraleseStore } from '../dist/native/neuralese-store.js';
import { saveNz } from '../dist/native/nz-file.js';
import { neuraleseRef } from '../dist/native/neuralese.js';

const arg = name => { const i = process.argv.indexOf(name); return i < 0 ? undefined : process.argv[i + 1]; };
const [endpoint, out, only] = [arg('--endpoint'), arg('--out'), arg('--pieces')];
if (arg('--init')) throw Error('--init is gone: bodies are initialised in context (src/neuralese/text-init.ts)');
if (!endpoint || !out) throw Error('Usage: neuralese-system-prompt-bank.mjs --endpoint URL --out FILE.nz [--pieces id,...]');
const pieces = promptPieces().filter(piece => !only || only.split(',').includes(piece.id));
const info = await (await fetch(`${endpoint}/v1/neuralese/info`)).json().catch(() => ({}));
if (!info.native_state?.enabled) throw new Error('neuralese-native-state-unavailable: prompt bank authoring requires native state producers');
const dialect = info.dialect ?? info.dialects?.[0];
if (!dialect) throw new Error('neuralese-native-state-unavailable: server did not declare a native state dialect');
// The sample call whose rendering holds the pieces: an ordinary text-instructed natlang function.
const sample = loadVirtualNatlang({ 'summarize.nl': '---\nargs: { text: string }\nreturns: string\n---\n' +
  'Summarize the text in one sentence.\n' }, 'summarize.nl');
const values = {}, digests = {}, inits = {};
const nativeArchive = new MemoryNeuraleseStore();
for (const piece of pieces) {
  const result = await initBodyInContext({ store: nativeArchive, endpoint, text: piece.text, type: SYSTEM_PROMPT_TYPE,
    render: (placeholder, driver) => createNatlangRuntime({ model: driver, neuralese: { store: nativeArchive,
      systemPrompts: systemPromptBank({ [piece.id]: neuraleseRef(SYSTEM_PROMPT_TYPE, placeholder) }) } })
      .run(() => sample('Anna meets Paul in Lyon on Tuesday to sign the contract.')) });
  values[piece.id] = neuraleseRef(SYSTEM_PROMPT_TYPE, result.id);
  digests[piece.id] = createHash('sha256').update(piece.text).digest('hex');
  inits[piece.id] = { context: result.context, init: result.init };
}
const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
const bytes = await saveNz({ systemPrompts: { type: `Record<string, ${SYSTEM_PROMPT_TYPE}>`, value: values,
  description: 'Soft forms of the runtime prompt pieces, initialised in context from their text.' } },
{ store: new HttpNeuraleseStore(endpoint), dialect, types: 'type SystemPrompt = string;', provenance: { kind: 'system-prompt-bank',
  init: 'native-text-in-context', runtime: `${pkg.name}@${pkg.version}`, pieces: digests, native_initializations: inits } });
writeFileSync(out, bytes, { flag: 'wx' });
console.log(JSON.stringify({ out, pieces: pieces.length, dialect,
  in_call: Object.values(inits).filter(entry => entry.context === 'call').length }));
