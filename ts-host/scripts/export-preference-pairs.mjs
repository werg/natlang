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
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { renderSftTurn, templateRenderer } from './export-native-sft.mjs';

/** Render both preference targets through the exact SFT serving boundary with an injected template renderer. */
export async function renderPreferencePair(pair, render, endToken = '<|im_end|>', reasoningEnd) {
  if (!pair?.chosen?.target || !pair?.rejected?.target || !Array.isArray(pair.messages) || !Array.isArray(pair.tools))
    throw new Error(`${pair?.id ?? 'unknown pair'}: unsupported preference pair shape`);
  const side = async (name, { target, reasoning, reasoning_trained }) => renderSftTurn({ id: `${pair.id}:${name}`,
    program_id: pair.program_id, split: pair.split, source_groups: pair.source_groups,
    messages: pair.messages, tools: pair.tools, target, teacher_reasoning: reasoning,
    ...(reasoning_trained === false ? { teacher_reasoning_trained: false } : {}) },
  render, endToken, reasoningEnd);
  const [chosen, rejected] = [await side('chosen', pair.chosen), await side('rejected', pair.rejected)];
  if (chosen.prompt !== rejected.prompt) throw new Error(`${pair.id}: the two sides render different prompts`);
  return { id: pair.id, kind: pair.kind, program_id: pair.program_id,
    ...(pair.split ? { split: pair.split } : {}), source_groups: pair.source_groups ?? [],
    evidence: pair.evidence, prompt: chosen.prompt, chosen: chosen.completion, rejected: rejected.completion,
    ...(chosen.completion_masked ? { chosen_masked: chosen.completion_masked } : {}),
    ...(rejected.completion_masked ? { rejected_masked: rejected.completion_masked } : {}) };
}

async function main() {
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
  const rendered = [];
  let next = 0;
  await Promise.all(Array.from({ length: Number(values.workers) }, async () => {
    for (let at = next++; at < pairs.length; at = next++)
      rendered[at] = await renderPreferencePair(pairs[at], render, values['end-token'], values['reasoning-end']);
  }));
  const same = rendered.filter(pair => pair.chosen === pair.rejected).length;
  const kept = rendered.filter(pair => pair.chosen !== pair.rejected);
  await writeFile(output, kept.map(pair => JSON.stringify(pair)).join('\n') + (kept.length ? '\n' : ''));
  console.log(`${kept.length} preference pairs -> ${output}${same ? ` (${same} with identical sides left out)` : ''}`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  main().catch(error => { console.error(error instanceof Error ? error.stack : String(error)); process.exitCode = 1; });
