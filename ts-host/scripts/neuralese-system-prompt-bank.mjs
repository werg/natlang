#!/usr/bin/env node
/** Build the text-initialised soft system-prompt bank (plans/neuralese/DECISIONS.md 40) for a Neuralese server's model.
 *
 * Usage: neuralese-system-prompt-bank.mjs --endpoint URL --out system-prompts.nz [--pieces id,id,...] [--init encode|embed]
 *
 * Each of this runtime's prompt pieces (src/native/system-prompts.ts) becomes a `Neuralese<SystemPrompt>` block made
 * by the server from its text: by default encoded in one forward pass through the port (`/v1/neuralese/encode`, one
 * vector per token, no summarising call), or with `--init embed` as raw token embeddings (`/v1/neuralese/embed`). The file has one export, `systemPrompts`, a
 * record from piece ID to soft form; its provenance names each piece's text digest and the runtime version. Training
 * updates the blocks (natlang_neuralese.train); the runtime uses a bank through `neuralese.systemPrompts`:
 *
 *   const { systemPrompts } = nzExports(await loadNz(bytes, { store }));
 *   createNatlangRuntime({ ..., neuralese: { store, systemPrompts: systemPromptBank(systemPrompts) } });
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { promptPieces, SYSTEM_PROMPT_TYPE } from '../dist/index.js';
import { HttpNeuraleseStore } from '../dist/model/neuralese-server.js';
import { saveNz } from '../dist/native/nz-file.js';
import { neuraleseRef } from '../dist/native/neuralese.js';

const arg = name => { const i = process.argv.indexOf(name); return i < 0 ? undefined : process.argv[i + 1]; };
const [endpoint, out, only, init = 'encode'] = [arg('--endpoint'), arg('--out'), arg('--pieces'), arg('--init')];
if (!['encode', 'embed'].includes(init)) throw Error('--init is encode or embed');
if (!endpoint || !out) throw Error('Usage: neuralese-system-prompt-bank.mjs --endpoint URL --out FILE.nz [--pieces id,...]');
const pieces = promptPieces().filter(piece => !only || only.split(',').includes(piece.id));
const info = await (await fetch(`${endpoint}/v1/neuralese/info`)).json().catch(() => ({}));
const dialect = info.dialect ?? 'nd:natlang@1';
const values = {}, digests = {};
for (const piece of pieces) {
  const response = await fetch(`${endpoint}/v1/neuralese/${init}`, { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ text: piece.text, type: SYSTEM_PROMPT_TYPE }) });
  if (!response.ok) throw Error(`${init} ${piece.id}: ${response.status} ${await response.text()}`);
  values[piece.id] = neuraleseRef(SYSTEM_PROMPT_TYPE, (await response.json()).id);
  digests[piece.id] = createHash('sha256').update(piece.text).digest('hex');
}
const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
const bytes = await saveNz({ systemPrompts: { type: `Record<string, ${SYSTEM_PROMPT_TYPE}>`, value: values,
  description: 'Soft forms of the runtime prompt pieces, initialised from their text.' } },
{ store: new HttpNeuraleseStore(endpoint), dialect, types: 'type SystemPrompt = string;', provenance: { kind: 'system-prompt-bank', init: init === 'encode' ? 'text-encode' : 'text-embedding',
  runtime: `${pkg.name}@${pkg.version}`, pieces: digests } });
writeFileSync(out, bytes, { flag: 'wx' });
console.log(JSON.stringify({ out, pieces: pieces.length, dialect }));
