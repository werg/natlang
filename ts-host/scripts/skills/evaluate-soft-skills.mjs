#!/usr/bin/env node
/** Query quality of given soft skills on decision families, scored exactly as run-method-arms.mjs scores its arms
 * (same held-out sampling, same proper scoring rule), so skills produced elsewhere (a learned updater, S6 §5) compare
 * with the gradient arms at matched compute.
 *
 * Usage: evaluate-soft-skills.mjs --cases decision-cases.jsonl --endpoint URL --skills SKILLS.json --out ROWS.jsonl
 *          [--query 24 --sample stratified]
 * SKILLS.json: [{ "family": "decision:sst5", "arm": "learned-updater", "skill": "<block id>" }, ...]; the blocks must
 * already be on the server (PUT /v1/neuralese/blocks/{id}).
 */
import { readFileSync } from 'node:fs';
import { appendFile } from 'node:fs/promises';
import { caseTarget, casesByFamily, decisionSession, quality, sampleCases } from './decision-lib.mjs';

const options = { query: 24, sample: 'stratified' };
for (let i = 2; i < process.argv.length; i += 2) {
  const key = process.argv[i].replace(/^--/, ''), value = process.argv[i + 1];
  if (!['cases', 'endpoint', 'skills', 'out', 'query', 'sample'].includes(key) || value === undefined)
    throw Error('Usage: see the header of evaluate-soft-skills.mjs');
  options[key] = key === 'query' ? Number(value) : value;
}
for (const key of ['cases', 'endpoint', 'skills', 'out']) if (!options[key]) throw Error(`--${key} is required`);

const byFamily = casesByFamily(options.cases);
const session = decisionSession(options.endpoint);
for (const entry of JSON.parse(readFileSync(options.skills, 'utf8'))) {
  const cases = sampleCases(byFamily.get(entry.family).heldout, options.query, options.sample);
  const params = entry.skill ? { skill: { $neuralese: { type: 'Neuralese<string>', id: entry.skill } } } : {};
  let total = 0;
  const started = Date.now();
  for (const c of cases) total += quality(c, await session.readoutOf(c, params), caseTarget(c).gold);
  const row = { family: entry.family, arm: entry.arm, skill: entry.skill ?? null, query: total / cases.length,
    cases: cases.length, seconds: (Date.now() - started) / 1000 };
  await appendFile(options.out, JSON.stringify(row) + '\n');
  console.log(JSON.stringify(row));
}
