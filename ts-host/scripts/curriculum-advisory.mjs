#!/usr/bin/env node
/**
 * Advisory curriculum judgments (plans/NATLANG_NATIVE_REVIEW.md, P6). Output files are labelled advisory with the
 * advisor's hash; nothing the pipeline acts on reads them, and neither command admits a row or schedules a collection.
 *
 *   node scripts/curriculum-advisory.mjs equivalence --rows ROWS.jsonl [--out FILE] --server URL --model NAME
 *       ROWS.jsonl: one rejected row per line ({ task: { program_ir }, outcome }). Rows of other sources are skipped.
 *       Rows judged equivalent are released to human review (release: human_review); admit stays false.
 *   node scripts/curriculum-advisory.mjs next-batch --cases CASES.jsonl --batch-size N [--out FILE] --server URL --model NAME
 *       CASES.jsonl: one admitted program record per line (with curriculum.slice and curriculum.domain).
 *
 * `--dry-run` prints the inputs and calls no model. Nothing here loads a model; --server is an OpenAI-compatible endpoint.
 */
import { readFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import { advisoryFile, explainerIdentity, refuseExisting, writeAdvisory } from './advisory-file.mjs';

const root = fileURLToPath(new URL('../../applications/curriculum-advisor/', import.meta.url));
const dist = new URL('../../applications/dist/curriculum-advisor/index.js', import.meta.url).href;
const sources = () => Object.fromEntries(['judgeAnswerEquivalence', 'nextCollectionBatch'].map(name => [name, readFileSync(join(root, `${name}.nl`), 'utf8')]));
const lines = path => readFileSync(path, 'utf8').split('\n').filter(line => line.trim()).map(line => JSON.parse(line));

export async function main(argv, { createRuntime } = {}) {
  const [command, ...rest] = argv;
  const { values } = parseArgs({ args: rest, options: { rows: { type: 'string' }, cases: { type: 'string' }, 'batch-size': { type: 'string' },
    out: { type: 'string' }, server: { type: 'string' }, model: { type: 'string' }, 'dry-run': { type: 'boolean' } } });
  const app = await import(dist);
  const identity = explainerIdentity(sources(), values.model ?? 'unspecified');
  const runtime = async () => {
    if (createRuntime) return createRuntime();
    if (!values.server || !values.model) throw new Error('--server and --model are required (or --dry-run)');
    const { createNatlangRuntime, openAICompatibleModelTurn } = await import(new URL('../dist/index.js', import.meta.url).href);
    return createNatlangRuntime({ model: openAICompatibleModelTurn({ endpoint: values.server, model: values.model }), seed: { mode: 'backend' } });
  };
  if (command === 'equivalence') {
    const cards = lines(values.rows).map(app.equivalenceCard).filter(Boolean);
    if (values['dry-run']) { console.log(JSON.stringify({ cards: cards.length, explainer: identity.explainer })); return 0; }
    refuseExisting(values.out ?? `${values.rows}.equivalence-review.json`);
    const rt = await runtime();
    try {
      const releases = await app.judgeEquivalences(rt, cards);
      const out = values.out ?? `${values.rows}.equivalence-review.json`;
      writeAdvisory(out, advisoryFile('judgeAnswerEquivalence', identity, cards, releases));
      console.log(JSON.stringify({ advisory: out, released: releases.filter(item => item.release === 'human_review').length, held: releases.filter(item => item.release === 'keep_held').length }));
    } finally { rt.close?.(); }
    return 0;
  }
  if (command === 'next-batch') {
    const { SLICE_TARGETS, DOMAIN_TARGETS } = await import(new URL('../dist/teacher/curriculum.js', import.meta.url).href);
    const cases = lines(values.cases).map(record => ({ slice: record.curriculum?.slice, domain: record.curriculum?.domain })).filter(item => item.slice && item.domain);
    const coverage = app.coverageFacts(cases, { slices: SLICE_TARGETS, domains: DOMAIN_TARGETS }, Number(values['batch-size'] ?? 100));
    if (values['dry-run']) { console.log(JSON.stringify({ coverage, explainer: identity.explainer })); return 0; }
    refuseExisting(values.out ?? `${values.cases}.next-batch.json`);
    const rt = await runtime();
    try {
      const result = await app.proposeBatch(rt, coverage);
      const out = values.out ?? `${values.cases}.next-batch.json`;
      writeAdvisory(out, advisoryFile('nextCollectionBatch', identity, coverage, result));
      console.log(JSON.stringify({ advisory: out, problems: result.problems.length }));
    } finally { rt.close?.(); }
    return 0;
  }
  console.error('usage: curriculum-advisory.mjs equivalence|next-batch ... (see the file header)');
  return 2;
}

if (import.meta.url === `file://${process.argv[1]}`) process.exit(await main(process.argv.slice(2)));
