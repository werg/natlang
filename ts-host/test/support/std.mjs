/** Test helpers for @natlang/std: a scripted local model with decisions, and examples built against this build. */
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { buildProject } from '../../dist/index.js';
import { scriptedModel } from './natlang.mjs';

const repo = fileURLToPath(new URL('../../../', import.meta.url));
const text = request => request.messages.flatMap(message => [typeof message.content === 'string' ? message.content : JSON.stringify(message.content),
  ...(message.tool_calls ?? []).map(call => call.function?.arguments ?? '')]).join('\n');
const scores = probabilities => ({ log_probs: probabilities.map(p => Math.log(p)) });

/** A scripted local model: eval stages answer from `respond`, decisions from `score(options, prompt)`. */
export function localModel(respond, score) {
  const model = scriptedModel(respond);
  const decisions = [];
  const driver = async request => request.tools.length ? model.driver(request) : { text: 'true', calls: [] };
  driver.decide = async request => {
    const prompt = text(request);
    decisions.push({ options: request.options, prompt });
    // The refinement judge asks true/false about a value; the scripted values hold.
    if (request.options.length === 2 && /Property: the value is/.test(prompt)) return scores([0.97, 0.03]);
    return scores(score(request.options, prompt));
  };
  return { driver, decisions, openings: model.openings };
}

/** Build an example with `@natlang/node` bound to this build; `@natlang/node/std` follows it. */
export async function buildExample(example) {
  const runtimeModule = { url: new URL('../../dist/index.js', import.meta.url).href, path: fileURLToPath(new URL('../../dist/index.js', import.meta.url)),
    types: fileURLToPath(new URL('../../dist/index.d.ts', import.meta.url)), specifiers: ['@natlang/node'] };
  const outDir = mkdtempSync(join(tmpdir(), `natlang-${example}-`));
  const built = buildProject({ project: join(repo, 'examples', example), outDir, runtimeModule, writeDeclarations: false });
  assert.equal(built.ok, true, JSON.stringify(built.diagnostics));
  assert.match(readFileSync(join(outDir, 'notes.js'), 'utf8'), /dist\/std\/index\.js/, '@natlang/node/std is bound beside the runtime');
  return async module => import(pathToFileURL(join(outDir, module)).href);
}

