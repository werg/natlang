#!/usr/bin/env node
/**
 * Preference pairs (build-preference-pairs.mjs) rendered by a model's chat template, as the SFT export renders turns:
 * one prompt, and the chosen and rejected completions. `chosen_masked` / `rejected_masked` count a completion's leading
 * characters that carry no loss (reasoning no model wrote, or shared by both sides as context).
 *
 *   node scripts/export-preference-pairs.mjs PAIRS.jsonl OUT.jsonl --server URL --end-token TOKEN [--reasoning-end TOKEN]
 */
import { readFile, writeFile } from 'node:fs/promises';
import { parseArgs } from 'node:util';
import { renderSftTurn, templateRenderer } from './export-native-sft.mjs';

const { values, positionals } = parseArgs({ allowPositionals: true, options: { server: { type: 'string', default: 'http://127.0.0.1:8083' },
  'end-token': { type: 'string', default: '<|im_end|>' }, 'reasoning-end': { type: 'string' }, workers: { type: 'string', default: '4' } } });
const [input, output] = positionals;
if (!output) throw new Error('usage: export-preference-pairs.mjs PAIRS.jsonl OUT.jsonl --server URL --end-token TOKEN [--reasoning-end TOKEN]');
const pairs = (await readFile(input, 'utf8')).split('\n').filter(Boolean).map(line => JSON.parse(line));
for (const pair of pairs) {
  if (pair.version !== 'natlang.preference_pair/2' || !pair.chosen?.target || !pair.rejected?.target ||
      pair.training_admission === false || pair.status === 'held')
    throw new Error(`${pair.id ?? 'unknown pair'}: held or unsupported preference records cannot be exported as trainable DPO pairs`);
}
const { render } = await templateRenderer(values.server);

async function renderPair(pair) {
  const side = async (name, { target, reasoning, reasoning_trained }) => renderSftTurn({ id: `${pair.id}:${name}`,
    messages: pair.messages, tools: pair.tools, target, teacher_reasoning: reasoning,
    ...(reasoning_trained === false ? { teacher_reasoning_trained: false } : {}) },
  render, values['end-token'], values['reasoning-end']);
  const [chosen, rejected] = [await side('chosen', pair.chosen), await side('rejected', pair.rejected)];
  if (chosen.prompt !== rejected.prompt) throw new Error(`${pair.id}: the two sides render different prompts`);
  return { id: pair.id, kind: pair.kind, program_id: pair.program_id, source_groups: pair.source_groups ?? [],
    evidence: pair.evidence, prompt: chosen.prompt, chosen: chosen.completion, rejected: rejected.completion,
    ...(chosen.completion_masked ? { chosen_masked: chosen.completion_masked } : {}),
    ...(rejected.completion_masked ? { rejected_masked: rejected.completion_masked } : {}) };
}

const rendered = [];
let next = 0;
await Promise.all(Array.from({ length: Number(values.workers) }, async () => {
  for (let at = next++; at < pairs.length; at = next++) rendered[at] = await renderPair(pairs[at]);
}));
const same = rendered.filter(pair => pair.chosen === pair.rejected).length;
const kept = rendered.filter(pair => pair.chosen !== pair.rejected);
await writeFile(output, kept.map(pair => JSON.stringify(pair)).join('\n') + (kept.length ? '\n' : ''));
console.log(`${kept.length} preference pairs -> ${output}${same ? ` (${same} with identical sides left out)` : ''}`);
