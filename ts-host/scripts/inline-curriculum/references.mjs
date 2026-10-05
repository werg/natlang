#!/usr/bin/env node
/**
 * Training rows from the curriculum's own reference solutions: generated cases, their scripted actions replayed through
 * the collector's execution path, written as the collector writes a teacher's rows.
 *
 *   node scripts/inline-curriculum/references.mjs --seeds 2001-2004 --shapes 2 --out ../runs/inline-curriculum/ref.results.jsonl
 *     [--families a,b] [--technique-weight 3] [--workers 8]
 *
 * --technique-weight N makes N times as many cases of the families whose references use an inline nl or iterateOn,
 * the techniques teacher runs show least.
 *
 * Only generated families of the interpreter track are used: a sourced family samples a fixed dataset whatever the
 * seed. Seeds must be ones no evaluation shard is built from (the probe is s102, the test pool s900).
 *
 * The reasoning is one line saying what the action does (actionNote), marked in provenance.synthetic_reasoning so that
 * export keeps it as context and trains the action only. A scripted result that no earlier output of its call shows,
 * such as a verdict worked out from facts it only read or a child's judgment, is a direct answer (assistant.direct_answer):
 * the note does not reason towards it, so it trains only a student that answers directly (materialize --direct-answers). Every row must be admitted.
 */
import { ACTION_PLAN_VERSION, actionPlan } from './action-plans.mjs';
import { writeFile } from 'node:fs/promises';
import { parseArgs } from 'node:util';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { admitRow, referenceDriver } from '../../dist/teacher/curriculum.js';
import { defaultToolSurfaceHash, executeProgram, expectedProvenance, programRow, programRunId, trajectoryTurn }
  from '../../dist/teacher/collector.js';
import { TOOLS_PROMPT } from '../../dist/native/prompt.js';
import { FAMILIES, buildRecords } from './families.mjs';

const EVALUATION_SEEDS = new Set([102, 900]);
const SYNTHETIC_REASONING = 'action-notes/1';

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

export async function referenceRow(record, index, options) {
  const expected = { ...expectedProvenance(record, options), synthetic_reasoning: options.authoredActionPlans ? ACTION_PLAN_VERSION : SYNTHETIC_REASONING,
    ...(options.authoredActionPlans ? { reasoning_supervision: 'authored-action-plan' } : {}) };
  const reference = referenceDriver(record), trajectory = [];
  const driver = async request => {
    const last = request.messages.at(-1);
    // Reference replay must actually expose complete evidence, including runtime cut-off pages.
    // Only follow the runtime's numeric cut-off/page note after an explicitly enabled read.
    const previous = trajectory.at(-1)?.assistant?.calls?.[0]?.tool;
    const nextPage = options.followCutoffPages && (['read_file','read_page'].includes(previous) ||
      options.followEvalCutoffPages && previous === 'eval') && last?.role === 'tool' ?
      /<<(?:cut off: \d+ of \d+ characters not shown|page \d+ of \d+ shown|full value: \d+ pages); (?:transcript\.entry\(\d+\)\.output holds all of it; )?read_page\("([a-z]+\d*)", (\d+)\) shows the (?:next part|first page)>>/.exec(String(last.content)) : null;
    const response = nextPage ? {calls:[['read_page',{id:nextPage[1],page:Number(nextPage[2])}]],reasoning:'I read the next page of that output.'} : await reference(request);
    if (options.authoredActionPlans) {
      response.execution_plan = actionPlan(response.calls ?? []);
      response.reasoning = response.execution_plan;
    }
    const turn = trajectoryTurn(request, response);
    const [tool, args] = response.calls?.[0] ?? [];
    if (tool === 'return_result' && args.status === 'success' && !shown(args.value, request.messages))
      turn.assistant.direct_answer = true;
    trajectory.push(turn);
    return response;
  };
  const runId = programRunId(index, expected);
  const run = await executeProgram(record, driver, { ...options, runId });
  return programRow(record, options.modelId, runId, expected, run, trajectory);
}

/** Whether a family's references use an inline nl or iterateOn, judged by its first case. */
function technique(name) {
  const [record] = FAMILIES[name].build(1, 0);
  return record.curriculum.reference.root.some(([tool, args]) => tool === 'eval' &&
    /\biterateOn\s*\(|\bnl\s*(?:<[^`]*>)?`/.test(String(args.code ?? '')));
}

async function main() {
const { values } = parseArgs({ options: { seeds: { type: 'string' }, shapes: { type: 'string', default: '1' },
  families: { type: 'string' }, out: { type: 'string' }, workers: { type: 'string', default: '8' },
  'technique-weight': { type: 'string', default: '1' } } });
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
const shapes = Number(values.shapes), weight = Number(values['technique-weight']);
const boosted = families.filter(technique);
// The extra shapes start past every family's base range (weights are at most 3), so no case repeats.
const records = seeds.flatMap(seed => [...buildRecords({ seed, shapes, families, hints: false }),
  ...(weight > 1 && boosted.length ? buildRecords({ seed, shapes: shapes * (weight - 1), start: Math.ceil(shapes * 3),
    families: boosted, hints: false }) : [])]);
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
const turns = rows.flatMap(row => row.trajectory), direct = turns.filter(turn => turn.assistant.direct_answer).length;
console.log(`${rows.length} reference rows from ${families.length} families (${boosted.length} weighted ${weight}x), ` +
  `seeds ${seeds.join(',')}: ` +
  `${turns.length} turns, ${direct} direct answers -> ${values.out}`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
