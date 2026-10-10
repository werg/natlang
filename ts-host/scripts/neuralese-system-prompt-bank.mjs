#!/usr/bin/env node
/** Build the text-initialised soft system-prompt bank (plans/neuralese/DECISIONS.md 40) for a Neuralese server's model.
 *
 * Usage: neuralese-system-prompt-bank.mjs --endpoint URL --out system-prompts.nz [--pieces id,id,...] [--accept-failed-gate]
 *
 * Each of this runtime's prompt pieces (src/native/system-prompts.ts) becomes a `Neuralese<SystemPrompt>` block
 * initialised in context (src/neuralese/text-init.ts, owner 2026-10-10): a sample natlang call is rendered with the
 * bank holding a placeholder for the piece, and the server returns the body under which that call is the
 * text-instructed call (`POST /v1/neuralese/init_body`, gated). A piece the sample call does not show is initialised as
 * the system message of a minimal call. The file has one export, `systemPrompts`, a record from piece ID to soft form;
 * its provenance names each piece's text digest, its init context and gate, and the runtime version. Training updates
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
const acceptFailedGate = process.argv.includes('--accept-failed-gate');
if (arg('--init')) throw Error('--init is gone: bodies are initialised in context (src/neuralese/text-init.ts)');
if (!endpoint || !out) throw Error('Usage: neuralese-system-prompt-bank.mjs --endpoint URL --out FILE.nz [--pieces id,...]');
const pieces = promptPieces().filter(piece => !only || only.split(',').includes(piece.id));
const info = await (await fetch(`${endpoint}/v1/neuralese/info`)).json().catch(() => ({}));
const dialect = info.dialect ?? info.dialects?.[0] ?? 'nd:natlang@1';
// The sample call whose rendering holds the pieces: an ordinary text-instructed natlang function.
const sample = loadVirtualNatlang({ 'summarize.nl': '---\nargs: { text: string }\nreturns: string\n---\n' +
  'Summarize the text in one sentence.\n' }, 'summarize.nl');
const values = {}, digests = {}, inits = {};
for (const piece of pieces) {
  const result = await initBodyInContext({ endpoint, text: piece.text, type: SYSTEM_PROMPT_TYPE, acceptFailedGate,
    fallbackToSystemMessage: true,
    render: (placeholder, driver) => createNatlangRuntime({ model: driver, neuralese: { store: new MemoryNeuraleseStore(),
      systemPrompts: systemPromptBank({ [piece.id]: neuraleseRef(SYSTEM_PROMPT_TYPE, placeholder) }) } })
      .run(() => sample('Anna meets Paul in Lyon on Tuesday to sign the contract.')) });
  values[piece.id] = neuraleseRef(SYSTEM_PROMPT_TYPE, result.id);
  digests[piece.id] = createHash('sha256').update(piece.text).digest('hex');
  inits[piece.id] = { context: result.context, init: result.init, gate: result.gate };
}
const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
const bytes = await saveNz({ systemPrompts: { type: `Record<string, ${SYSTEM_PROMPT_TYPE}>`, value: values,
  description: 'Soft forms of the runtime prompt pieces, initialised in context from their text.' } },
{ store: new HttpNeuraleseStore(endpoint), dialect, types: 'type SystemPrompt = string;', provenance: { kind: 'system-prompt-bank',
  init: 'text-in-context', runtime: `${pkg.name}@${pkg.version}`, pieces: digests, init_gates: inits } });
writeFileSync(out, bytes, { flag: 'wx' });
console.log(JSON.stringify({ out, pieces: pieces.length, dialect,
  gates_passed: Object.values(inits).filter(entry => entry.gate?.passed).length,
  in_call: Object.values(inits).filter(entry => entry.context === 'call').length }));
