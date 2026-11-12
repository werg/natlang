#!/usr/bin/env node
// ts-host's prompt composition for training-data rendering in Python (training/neuralese/natlang_neuralese/data/
// serving_prompt.py, prompt_rendering.py): one implementation, this runtime's, for serving and training rows.
//
//   node scripts/serving-request.mjs [DIST_DIR]
//
// stdin: JSON lines, one request each; stdout: one JSON line per request ({"error": ...} on failure).
//   {"op": "identity"}                                         -> {commit, dirty, pieces_sha256}
//   {"op": "serving", "messages", "tools", "target"?, "entry", "rendering"?} -> {messages, tools, report}
//   {"op": "render", "text", "mode", "pieces"?}                -> {text}   (renderSystemPrompt, with earlier piece texts;
//                                                                 `pieces`: an explicit piece list instead)
//   {"op": "pieces"}                                           -> {pieces} (promptPieces(): this runtime's pieces)
// DIST_DIR (default ../dist) is a tsc build of this checkout. Earlier texts of each piece come from
// training/prompt-pieces/pieces.json.
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createInterface } from 'node:readline';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const dist = resolve(process.argv[2] ?? join(HERE, '../dist'));
const { servingRequest, pieceBank, renderRecordedSystemPrompt } = await import(pathToFileURL(join(dist, 'teacher/serving-request.js')).href);
const { promptPieces } = await import(pathToFileURL(join(dist, 'native/system-prompts.js')).href);
const piecesFile = join(HERE, '../../training/prompt-pieces/pieces.json');
const raw = readFileSync(piecesFile);
const bank = pieceBank(JSON.parse(raw.toString('utf8')).pieces);
const git = (...args) => { try { return execFileSync('git', args, { cwd: join(HERE, '..'), encoding: 'utf8' }).trim(); } catch { return null; } };
const identity = { commit: git('log', '-1', '--format=%H', '--', '.'), dirty: !!git('status', '--porcelain', '--', 'src'),
  dist, pieces_sha256: createHash('sha256').update(raw).digest('hex') };

for await (const line of createInterface({ input: process.stdin })) {
  let out;
  try {
    const request = JSON.parse(line);
    if (request.op === 'identity') out = identity;
    else if (request.op === 'serving') out = servingRequest({ messages: request.messages, tools: request.tools ?? [], target: request.target ?? undefined },
      { entry: request.entry, bank, rendering: request.rendering ?? 'full' });
    else if (request.op === 'render') out = { text: renderRecordedSystemPrompt(request.text, request.mode, request.pieces ?? bank) };
    else if (request.op === 'pieces') out = { pieces: promptPieces() };
    else throw new Error(`unknown op ${JSON.stringify(request.op)}`);
  } catch (error) { out = { error: String(error?.message ?? error) }; }
  process.stdout.write(JSON.stringify(out) + '\n');
}
