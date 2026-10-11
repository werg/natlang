#!/usr/bin/env node
// Export ts-host's prompt piece table (promptPieces(): id, scope, text) for the Python training/serving renderer
// (natlang_neuralese/data/prompt_rendering.py), optionally with the historical bank, and optionally render recorded
// system prompts with the TS renderSystemPrompt as a parity fixture.
//
//   node scripts/export-prompt-pieces.mjs OUT.json [--history] [--fixture IN.jsonl FIXTURE_OUT.jsonl]
//
// --history (D9, plans/neuralese/D9_NATIVE_SFT_V13_REPAIR_AUDIT_20261011.md): recorded corpora carry system prompts
// of earlier runtime versions. Every revision of native/prompt.ts and model/text-neuralese-emulation.ts in git is
// loaded (as migrate-system-prompt.mjs promptHistory() does, also in the read_code-renamed form); each exported string
// constant and each derived piece of that revision becomes a historical piece with the scope its current counterpart
// has. Program declarations (depth-limit notice, directory-reducer declaration, Neuralese text-channel emulation,
// Neuralese string boundaries, the edit_code capability line) are 'program'; everything else is 'generic'.
// IN.jsonl rows: {"id": ..., "text": <system prompt>}; FIXTURE_OUT rows add "program_header": renderSystemPrompt(text).
import { readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promptPieces, renderSystemPrompt } from '../dist/native/system-prompts.js';

const args = process.argv.slice(2);
const out = args[0];
const history = args.includes('--history');
const fixtureAt = args.indexOf('--fixture');
if (!out) { console.error('usage: export-prompt-pieces.mjs OUT.json [--history] [--fixture IN.jsonl OUT.jsonl]'); process.exit(2); }
const REPO = join(dirname(fileURLToPath(import.meta.url)), '../..');
const git = (...a) => execFileSync('git', a, { cwd: REPO, encoding: 'utf8', maxBuffer: 1 << 28 });
const commit = git('log', '-1', '--format=%H', '--', 'ts-host/src/native').trim();
const pieces = promptPieces().map(p => ({ id: p.id, scope: p.scope, text: p.text, historical: false }));

const PROGRAM = new Set(['NL_DEPTH_LIMIT_NOTICE', 'DIRECTORY_REDUCER_DECLARATION', 'TEXT_NEURALESE_EMULATION_PROMPT']);
async function historicalPieces() {
  const ts = (await import('typescript')).default;
  const { migrateIrLine: renameCodeTools } = await import('./inline-curriculum/migrate-code-tools.mjs');
  const found = new Map(); // text -> piece
  const add = (id, scope, text, revision) => {
    if (typeof text !== 'string' || !text.trim()) return;
    for (const variant of [text, JSON.parse(renameCodeTools(JSON.stringify(text)))])
      if (!found.has(variant)) found.set(variant, { id: `${id}@${revision.slice(0, 8)}`, scope, text: variant, historical: true, commit: revision });
  };
  const failures = [];
  for (const file of ['ts-host/src/native/prompt.ts', 'ts-host/src/model/text-neuralese-emulation.ts']) {
    for (const revision of git('log', '--format=%H', '--', file).split('\n').filter(Boolean)) {
      let module;
      try {
        const source = git('show', `${revision}:${file}`).replace(/^import .*$/gm, '');
        const js = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
        module = await import(`data:text/javascript;base64,${Buffer.from(js).toString('base64')}`);
      } catch (error) { failures.push({ file, revision, error: String(error).slice(0, 200) }); continue; }
      for (const [name, value] of Object.entries(module)) {
        if (typeof value === 'string') add(name, PROGRAM.has(name) ? 'program' : 'generic', value, revision);
        if (typeof value === 'function' && name === 'directoryReducerPrompt')
          for (const surface of ['all', 'editor', 'files']) for (const adHoc of [true, false]) for (const writable of [true, false]) {
            try { add(`directoryReducerPrompt/${surface}/${adHoc}/${writable}`, 'generic', value(surface, adHoc, writable), revision); } catch {}
          }
        if (typeof value === 'function' && name === 'promptAtNlDepthLimit' && typeof module.TOOLS_PROMPT === 'string')
          try { add('interpreter/depth-limit', 'generic', value(module.TOOLS_PROMPT), revision); } catch {}
      }
      if (typeof module.FUNCTION_TOOLS_PROMPT === 'string' && module.FUNCTION_TOOLS_PROMPT.split('\n')[1])
        add('function-tools/edit-capability', 'program', module.FUNCTION_TOOLS_PROMPT.split('\n')[1], revision);
      if (typeof module.SEMANTIC_RESULT_PROMPT === 'string' && module.SEMANTIC_RESULT_PROMPT.split('\n\n')[1])
        add('neuralese/string-boundaries', 'program', module.SEMANTIC_RESULT_PROMPT.split('\n\n')[1], revision);
      const reducer = typeof module.DIRECTORY_REDUCER_PROMPT === 'string' &&
        module.DIRECTORY_REDUCER_PROMPT.split('\n').find(line => line.startsWith('This call is a directory reducer:'));
      if (reducer) add('directory-reducer/declaration', 'program', reducer, revision);
    }
  }
  const current = new Set(pieces.map(p => p.text));
  return { added: [...found.values()].filter(p => !current.has(p.text)), failures };
}

let report = {};
if (history) {
  const { added, failures } = await historicalPieces();
  pieces.push(...added);
  report = { historical: added.length, load_failures: failures };
}
writeFileSync(out, JSON.stringify({ schema: 'natlang.prompt-pieces/1', generated_from_commit: commit, history, pieces, ...report }, null, 1) + '\n');
if (fixtureAt >= 0) {
  const [fixtureIn, fixtureOut] = args.slice(fixtureAt + 1, fixtureAt + 3);
  const rows = readFileSync(fixtureIn, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line));
  writeFileSync(fixtureOut, rows.map(row => JSON.stringify({ ...row, program_header: renderSystemPrompt(row.text, 'program-header', pieces) })).join('\n') + '\n');
}
console.log(JSON.stringify({ pieces: pieces.length, commit, ...report, load_failures: report.load_failures?.length }));
