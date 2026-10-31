#!/usr/bin/env node
// Export the serving prompt surface of this ts-host build for training-data rendering
// (training/neuralese/natlang_neuralese/data/serving_prompt.py): the current prompt pieces (promptPieces(): id, scope,
// text) and the native tool schemas exactly as NativeToolAgent.toolsScope offers them, per tool name, in the order the
// runtime offers them, with and without ad hoc nl depth remaining (read_code's description differs).
//
//   node scripts/export-serving-surface.mjs OUT.json [DIST_DIR]
//
// DIST_DIR (default ../dist) is a tsc build of this checkout; the export records the git commit of ts-host/src.
// return_result's `value` schema depends on the call's declared result type: the export carries the type-free form and
// the training renderer keeps each recorded call's own type fields (it never invents a result type). `bases` holds the
// base system prompt of each real entry point (runtime default, teacher/eval/RL collector).
import { writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const [out, distArg] = process.argv.slice(2);
if (!out) { console.error('usage: export-serving-surface.mjs OUT.json [DIST_DIR]'); process.exit(2); }
const HERE = dirname(fileURLToPath(import.meta.url));
const dist = resolve(distArg ?? join(HERE, '../dist'));
const load = path => import(pathToFileURL(join(dist, path)).href);
const { promptPieces } = await load('native/system-prompts.js');
const { NativeToolAgent } = await load('native/agent.js');
const { FILE_TOOL_SURFACES, TOOLS_PROMPT, promptAtNlDepthLimit } = await load('native/prompt.js');
const { defaultSystemPrompt } = await load('teacher/collector.js');
const REPO = join(HERE, '../..');
const git = (...a) => execFileSync('git', a, { cwd: REPO, encoding: 'utf8' }).trim();

// toolsScope reads only these session fields; a non-lambda type gives return_result's type-free value schema.
const session = ({ depthLimited = false, ownCode = false, folder = null } = {}) => ({
  lam: { type: { kind: 'opaque' }, codebase: ownCode ? { f: {} } : {},
    projectTransaction: folder ? { folder: { access: folder } } : undefined },
  runtime: { frame: { adHocDepth: depthLimited ? 1e9 : 0 } } });
const offer = (opts, fileTools = 'all') =>
  NativeToolAgent.prototype.toolsScope.call({ options: { fileTools } }, session(opts));

const surfaces = {};
for (const depthLimited of [false, true])
  for (const ownCode of [false, true])
    for (const folder of [null, 'write', 'read'])
      for (const fileTools of folder ? FILE_TOOL_SURFACES : ['all']) {
        const key = JSON.stringify({ depthLimited, ownCode, folder, fileTools });
        surfaces[key] = offer({ depthLimited, ownCode, folder }, fileTools);
      }
// One definition per (name, depth): every surface must agree on it.
const tools = { default: {}, depth_limited: {} };
for (const [key, list] of Object.entries(surfaces)) {
  const bucket = JSON.parse(key).depthLimited ? tools.depth_limited : tools.default;
  for (const definition of list) {
    const name = definition.function.name;
    const text = JSON.stringify(definition);
    if (bucket[name] !== undefined && JSON.stringify(bucket[name]) !== text)
      throw new Error(`tool ${name} differs between surfaces`);
    bucket[name] = definition;
  }
}
const order = [];
for (const list of Object.values(surfaces))
  for (const definition of list) if (!order.includes(definition.function.name)) order.push(definition.function.name);
// Recorded corpora carry earlier runtime texts: every git revision's pieces (as export-prompt-pieces.mjs --history),
// so the renderer can recognise them and substitute the current text of the same logical piece.
const { historicalPromptPieces } = await import('./inline-curriculum/migrate-system-prompt.mjs');
const currentTexts = new Set(promptPieces().map(p => p.text));
const history = (await historicalPromptPieces()).filter(p => !currentTexts.has(p.text))
  .map(({ id, scope, text, commit }) => ({ id, scope, text, commit }));
const value = {
  schema: 'natlang.serving-prompt-surface/1',
  generated_from_commit: git('log', '-1', '--format=%H', '--', 'ts-host/src'),
  pieces: promptPieces().map(p => ({ id: p.id, scope: p.scope, text: p.text })),
  history,
  // The base system prompt (agent.ts systemPrompt(): options.systemPrompt ?? TOOLS_PROMPT, depth-limited by
  // promptAtNlDepthLimit) of the two real tool-loop entry points: `runtime` (natlang run / NativeToolAgent default)
  // and `collector` (teacher/collector.ts defaultSystemPrompt: eval probes, RL and teacher collection).
  bases: {
    runtime: { default: TOOLS_PROMPT, depth_limited: promptAtNlDepthLimit(TOOLS_PROMPT) },
    collector: { default: defaultSystemPrompt, depth_limited: promptAtNlDepthLimit(defaultSystemPrompt) },
  },
  tools, tool_order: order,
  surfaces: Object.fromEntries(Object.entries(surfaces).map(([k, list]) => [k, list.map(d => d.function.name)])),
};
writeFileSync(out, JSON.stringify(value, null, 1) + '\n');
console.log(JSON.stringify({ out, pieces: value.pieces.length, history: history.length, tools: order, commit: value.generated_from_commit }));
