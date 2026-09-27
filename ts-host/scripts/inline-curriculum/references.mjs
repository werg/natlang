#!/usr/bin/env node
/**
 * Training rows from the curriculum's own reference solutions: generated cases, their scripted actions replayed through
 * the collector's execution path, written as the collector writes a teacher's rows.
 *
 *   node scripts/inline-curriculum/references.mjs --seeds 2001-2004 --shapes 2 --out ../runs/inline-curriculum/ref.results.jsonl
 *     [--families a,b] [--workers 8]
 *
 * Only generated families of the interpreter track are used: a sourced family samples a fixed dataset whatever the
 * seed. Seeds must be ones no evaluation shard is built from (the probe is s102, the test pool s900).
 *
 * The reasoning is one line saying what the action does (actionNote), marked in provenance.synthetic_reasoning so that
 * export keeps it as context and trains the action only. A scripted result that no earlier output of its call shows,
 * such as a verdict worked out from facts it only read, is left untrained (assistant.untrained): the note does not
 * reason towards it, and training it would teach answering without reasoning. Every row must be admitted.
 */
import { writeFile } from 'node:fs/promises';
import { parseArgs } from 'node:util';
import { admitRow, referenceDriver } from '../../dist/teacher/curriculum.js';
import { defaultToolSurfaceHash, executeProgram, expectedProvenance, programRow, programRunId, trajectoryTurn }
  from '../../dist/teacher/collector.js';
import { TOOLS_PROMPT } from '../../dist/native/prompt.js';
import { FAMILIES, buildRecords } from './families.mjs';

const EVALUATION_SEEDS = new Set([102, 900]);
const SYNTHETIC_REASONING = 'action-notes/1';
const UNSUPPORTED = 'a scripted result that no earlier output of its call shows';

/**
 * Whether the last output before this turn shows the value (spacing and quotes aside): an eval's output shows its
 * value on the first line, and a staged result in "Staged … as the result".
 */
function shown(value, context) {
  const last = [...context].reverse().find(message => message.role === 'tool');
  if (last === undefined) return false;
  const squash = text => String(text).replace(/["'`\s]/g, ''), content = String(last.content), wanted = squash(JSON.stringify(value));
  const staged = /^Staged ([\s\S]*?) as the result\./m.exec(content)?.[1];
  return squash(content.split('\n')[0]) === wanted || (staged !== undefined && squash(staged) === wanted);
}

async function referenceRow(record, index, options) {
  const expected = { ...expectedProvenance(record, options), synthetic_reasoning: SYNTHETIC_REASONING };
  const reference = referenceDriver(record), trajectory = [];
  const driver = async request => {
    const response = await reference(request), turn = trajectoryTurn(request, response);
    const [tool, args] = response.calls[0];
    if (tool === 'return_result' && args.status === 'success' && !shown(args.value, request.messages))
      turn.assistant.untrained = UNSUPPORTED;
    trajectory.push(turn);
    return response;
  };
  const runId = programRunId(index, expected);
  const run = await executeProgram(record, driver, { ...options, runId });
  return programRow(record, options.modelId, runId, expected, run, trajectory);
}

const { values } = parseArgs({ options: { seeds: { type: 'string' }, shapes: { type: 'string', default: '1' },
  families: { type: 'string' }, out: { type: 'string' }, workers: { type: 'string', default: '8' } } });
if (!values.seeds || !values.out) throw new Error('usage: references.mjs --seeds A-B|A,B,... --shapes N --out OUT.jsonl');
const seeds = values.seeds.includes('-') ?
  (([a, b]) => Array.from({ length: b - a + 1 }, (_, i) => a + i))(values.seeds.split('-').map(Number)) :
  values.seeds.split(',').map(Number);
for (const seed of seeds) if (EVALUATION_SEEDS.has(seed)) throw new Error(`seed ${seed} builds evaluation cases`);
const generated = Object.keys(FAMILIES).filter(name => !FAMILIES[name].source && (FAMILIES[name].track ?? 'interpreter') === 'interpreter');
const families = values.families ? values.families.split(',') : generated;
for (const name of families) if (!generated.includes(name)) throw new Error(`${name} is not a generated interpreter family`);

const options = { modelId: 'curriculum-reference', rootSeed: 909, systemPrompt: TOOLS_PROMPT, contextTokens: 16384,
  maxTurns: 60, toolSurfaceSha256: await defaultToolSurfaceHash(), collectionRole: 'reference' };
const records = seeds.flatMap(seed => buildRecords({ seed, shapes: Number(values.shapes), families, hints: false }));
const rows = new Array(records.length);
let next = 0;
await Promise.all(Array.from({ length: Number(values.workers) }, async () => {
  while (next < records.length) {
    const index = next++;
    rows[index] = await referenceRow(records[index], index, options);
  }
}));
const refused = rows.map(row => [row, admitRow(row)]).filter(([, admission]) => !admission.admitted);
if (refused.length) throw new Error(`references not admitted:\n${refused.map(([row, admission]) =>
  `${row.task.program_ir.id}: ${admission.reasons.join(', ')}`).join('\n')}`);
await writeFile(values.out, rows.map(row => JSON.stringify(row)).join('\n') + '\n');
const turns = rows.flatMap(row => row.trajectory), untrained = turns.filter(turn => turn.assistant.untrained).length;
console.log(`${rows.length} reference rows from ${families.length} families, seeds ${seeds.join(',')}: ` +
  `${turns.length} turns, ${untrained} left untrained -> ${values.out}`);
