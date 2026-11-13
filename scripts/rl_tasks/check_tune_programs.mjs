#!/usr/bin/env node
// usage: node scripts/rl_tasks/check_tune_programs.mjs RUNTIME_TS_HOST PROGRAM_DIR... (tune programs from composed_tune_programs.py)
// Load every root function of each tune program with the runtime's file loader and score the expected values with
// the program's own verifier (expected must score 1; a perturbed value must score 0).
import fs from 'node:fs';
import path from 'node:path';
const [runtime, ...dirs] = process.argv.slice(2);
const { loadNamedFunction } = await import(runtime + '/dist/runtime/loader.js');
const { nodeSourceFiles } = await import(runtime + '/dist/runtime/node-files.js');
const { scoreRoot } = await import(runtime + '/dist/tune/reward.js');
for (const dir of dirs) {
  const rows = JSON.parse(fs.readFileSync(path.join(dir, 'examples.json'), 'utf8'));
  const fns = fs.readdirSync(dir, { recursive: true }).filter(f => f.endsWith('.nl') && !f.startsWith('.natlang'));
  const loaded = {};
  for (const fn of fns) { try { const f = loadNamedFunction(path.join(dir, fn), nodeSourceFiles(dir)); loaded[fn] = f ? 'ok' : 'none'; } catch (e) { loaded[fn] = String(e).slice(0, 300); } }
  let ok = 0, wrongAccepted = 0;
  for (const r of rows) {
    if ((await scoreRoot(dir, r, r.expected, false))?.quality === 1) ok++;
    const bad = typeof r.expected === 'number' ? r.expected * 1.5 + 7 : r.expected === 'A' ? 'B' : String(r.expected) + ' and something else entirely';
    if ((await scoreRoot(dir, r, bad, false))?.quality === 1) wrongAccepted++;
  }
  console.log(JSON.stringify({ program: path.basename(dir), loaded, roots: rows.length, expected_scores_1: ok, perturbed_accepted: wrongAccepted }));
}
