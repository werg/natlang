#!/usr/bin/env node
/**
 * Corrected variants of admitted rows: where a call failed and its next successful turn fixed it, the same trajectory
 * with the fix made first (teacher/corrections.ts), checked by replay.
 *
 *   node scripts/inline-curriculum/corrections.mjs ADMITTED.jsonl TURNS.jsonl OUT.jsonl [--workers 4]
 *
 * TURNS.jsonl is the admitted rows materialized: it says which decisions failed ('decision contains a failed or
 * unexecuted proposal') and which were approved. A site is one or more failed turns of a call followed directly by an
 * approved turn of the same call. The variants are rows like the collector's; the materializer trains only the fix.
 */
import { createReadStream } from 'node:fs';
import { writeFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { parseArgs } from 'node:util';
import { correctedVariant } from '../../dist/teacher/corrections.js';
import { TOOLS_PROMPT } from '../../dist/native/prompt.js';

const FAILED = 'decision contains a failed or unexecuted proposal';

async function* lines(path) {
  for await (const line of createInterface({ input: createReadStream(path), crlfDelay: Infinity })) if (line.trim()) yield JSON.parse(line);
}

/** Sites per trajectory, from its materialized decisions. */
async function sitesOf(turnsPath) {
  const calls = new Map();
  for await (const turn of lines(turnsPath)) {
    const key = `${turn.teacher_trajectory_id}\u0000${JSON.stringify(turn.decision.durable_opening)}`;
    const entry = { index: turn.decision.index, approved: turn.training_admission.approved, reason: turn.training_admission.reason };
    calls.set(key, [...calls.get(key) ?? [], entry]);
  }
  const sites = new Map();
  for (const [key, decisions] of calls) {
    decisions.sort((a, b) => a.index - b.index);
    for (let i = 0; i < decisions.length; i++) {
      if (decisions[i].reason !== FAILED || (i > 0 && decisions[i - 1].reason === FAILED)) continue;
      let j = i;
      while (j < decisions.length && decisions[j].reason === FAILED) j++;
      if (decisions[j]?.approved) {
        const id = key.split('\u0000')[0];
        sites.set(id, [...sites.get(id) ?? [], { failed: decisions.slice(i, j).map(d => d.index), fixed: decisions[j].index }]);
      }
    }
  }
  return sites;
}

const { values, positionals } = parseArgs({ allowPositionals: true, options: { workers: { type: 'string', default: '4' } } });
const [admittedPath, turnsPath, outPath] = positionals;
if (!outPath) throw new Error('usage: corrections.mjs ADMITTED.jsonl TURNS.jsonl OUT.jsonl [--workers N]');
const sites = await sitesOf(turnsPath);
const jobs = [];
for await (const row of lines(admittedPath)) for (const site of sites.get(row.id) ?? []) jobs.push({ row, site });
const variants = [], rejected = {};
let next = 0;
await Promise.all(Array.from({ length: Number(values.workers) }, async () => {
  while (next < jobs.length) {
    const { row, site } = jobs[next++];
    const provenance = row.provenance;
    try {
      const result = await correctedVariant(row, site, { systemPrompt: TOOLS_PROMPT, contextTokens: provenance.context_tokens ?? 16384,
        maxTurns: provenance.max_turns, rootSeed: provenance.seed_policy?.root ?? 909 });
      if ('row' in result) variants.push(result.row); else rejected[result.rejected] = (rejected[result.rejected] ?? 0) + 1;
    } catch (error) {
      const reason = `replay failed: ${error instanceof Error ? error.message.split('\n')[0].slice(0, 80) : error}`;
      rejected[reason] = (rejected[reason] ?? 0) + 1;
    }
  }
}));
await writeFile(outPath, variants.map(row => JSON.stringify(row)).join('\n') + (variants.length ? '\n' : ''));
console.log(`${variants.length} corrected variants from ${jobs.length} sites -> ${outPath}; not made: ${JSON.stringify(rejected)}`);
