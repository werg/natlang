#!/usr/bin/env node
/** Teacher-window commands of the refine-judge family (plans/REFINEMENT_DATA.md). Run them through the memory ledger.
 *
 *   cli.mjs stage   --stage exemplify|nearMiss|verifyNearMiss --in rows.jsonl --out results.jsonl --endpoint URL --model NAME
 *   cli.mjs label   --in pairs.jsonl --out labels.jsonl --endpoint URL --model NAME [--teacher ID]
 *   cli.mjs messages --in pairs.jsonl            (no model: prints the judge messages of each pair, for parity checks)
 *
 * Options: --concurrency N (default 4, the shared teacher's polite limit), --request JSON (extra request fields, for
 * example '{"chat_template_kwargs":{"enable_thinking":false}}'). Outputs are appended and resumed by id; an existing
 * output is never rewritten.
 */
import { appendFileSync, existsSync, readFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { openAICompatibleModelTurn } from '../../dist/index.js';
import { judgeMessages, labelPairs, runStages } from './lib.mjs';

const readJsonl = path => existsSync(path) ? readFileSync(path, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line)) : [];
const { positionals, values } = parseArgs({ allowPositionals: true, options: {
  stage: { type: 'string' }, in: { type: 'string' }, out: { type: 'string' }, endpoint: { type: 'string' }, model: { type: 'string' },
  teacher: { type: 'string' }, concurrency: { type: 'string', default: '4' }, request: { type: 'string' } } });
const [command] = positionals;
if (!values.in) throw new Error('--in is required');
const input = readJsonl(values.in);

if (command === 'messages') {
  for (const pair of input) console.log(JSON.stringify({ id: pair.id, messages: await judgeMessages(pair.value, pair.predicate) }));
} else if (command === 'stage' || command === 'label') {
  if (!values.out || !values.endpoint || !values.model) throw new Error('--out, --endpoint and --model are required');
  const turn = openAICompatibleModelTurn({ endpoint: values.endpoint, model: values.model, stream: false, concurrency: Number(values.concurrency),
    ...(values.request ? { request: JSON.parse(values.request) } : {}) });
  const done = new Set(readJsonl(values.out).filter(row => !row.error && row.ok !== false).map(row => row.id));
  const onRow = row => appendFileSync(values.out, JSON.stringify(row) + '\n');
  const results = command === 'stage'
    ? await runStages({ stage: values.stage, rows: input, driver: turn, concurrency: Number(values.concurrency), done, onRow })
    : await labelPairs({ pairs: input, decide: turn.decide, teacher: values.teacher ?? `teacher:${values.model}`, concurrency: Number(values.concurrency), done, onRow });
  const failed = results.filter(row => row.error || row.ok === false).length;
  console.error(JSON.stringify({ command, input: input.length, skipped: done.size, ran: results.length, failed }));
} else {
  throw new Error('usage: cli.mjs stage|label|messages (see the header of this file)');
}
