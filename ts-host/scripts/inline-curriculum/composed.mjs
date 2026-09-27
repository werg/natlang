// Composed families: tasks assembled from parts that each carry their code and their sentence, so the instructions,
// the reference solution and the expected result come from one source and agree by construction.
//
// composed_helpers: a task over a list of records whose steps match documented helper functions of the program (some
// written in TypeScript, some as natural-language functions that compute with eval); the reference calls the helpers by
// name and does the remaining exact work in code. No step is a judgment, so nothing is handed to an inline nl.
//
// composed_process: a process stated as a starting state, a step, a stopping rule and what to report; the reference
// runs the step with iterateOn until the rule holds.
import ts from 'typescript';
import { Random, curriculumCase, evalCall, returnCall } from './lib.mjs';

// ---- Record domains ----------------------------------------------------------------------------------------------

/** Domains of records: an entity with an id, numeric fields and a category field, each with a generator. */
const DOMAINS = [
  { type: 'Shipment', plural: 'shipments', singular: 'shipment', id: 'S',
    numbers: { days_late: [0, 9, 'days late'], weight_kg: [1, 60, 'weight in kg'] },
    category: ['carrier', 'carrier', ['Tollo', 'Brisk', 'Vantage', 'Northway']] },
  { type: 'Order', plural: 'orders', singular: 'order', id: 'O',
    numbers: { amount_cents: [500, 25000, 'amount in cents'], items: [1, 12, 'number of items'] },
    category: ['region', 'region', ['north', 'south', 'east', 'west']] },
  { type: 'Employee', plural: 'employees', singular: 'employee', id: 'E',
    numbers: { salary: [38000, 120000, 'salary'], years: [0, 25, 'years of service'] },
    category: ['team', 'team', ['platform', 'sales', 'support', 'research']] },
  { type: 'Reading', plural: 'readings', singular: 'reading', id: 'R',
    numbers: { value: [-20, 140, 'value'], minute: [0, 59, 'minute'] },
    category: ['sensor', 'sensor', ['intake', 'boiler', 'exhaust', 'vent']] },
  { type: 'Ticket', plural: 'tickets', singular: 'ticket', id: 'T',
    numbers: { priority: [1, 5, 'priority'], age_days: [0, 40, 'age in days'] },
    category: ['queue', 'queue', ['billing', 'access', 'hardware', 'returns']] },
];

function records(rng, domain, count) {
  const [category, , values] = domain.category;
  return Array.from({ length: count }, (_, i) => ({ id: `${domain.id}${i + 1}`,
    ...Object.fromEntries(Object.entries(domain.numbers).map(([field, [min, max]]) => [field, rng.int(min, max)])),
    [category]: rng.pick(values) }));
}

// ---- Operations over records ---------------------------------------------------------------------------------------
// Each operation has a signature, a body in TypeScript (run here too, for the expected result), a doc sentence for a
// helper, and a phrase for the task's instructions. `input` names what it takes: the records ('rows') or a
// per-category number table ('table').

const numeric = domain => Object.keys(domain.numbers);
const label = (domain, field) => domain.numbers[field][2];

const FILTERS = [
  (rng, d) => { const f = rng.pick(numeric(d)), [lo, hi] = d.numbers[f], min = rng.int(lo + Math.floor((hi - lo) / 4), hi - Math.floor((hi - lo) / 4));
    return { name: `${d.plural}_with_${f}_at_least`, params: `rows: ${d.type}[], min: number`, args: [min], returns: `${d.type}[]`,
      body: `return rows.filter(row => row.${f} >= min);`,
      doc: `Keep the ${d.plural} in rows whose ${label(d, f)} is at least min, in their order.`,
      condition: `whose ${label(d, f)} is at least ${min}` }; },
  (rng, d) => { const f = rng.pick(numeric(d)), [lo, hi] = d.numbers[f], max = rng.int(lo + Math.floor((hi - lo) / 4), hi - Math.floor((hi - lo) / 4));
    return { name: `${d.plural}_with_${f}_below`, params: `rows: ${d.type}[], max: number`, args: [max], returns: `${d.type}[]`,
      body: `return rows.filter(row => row.${f} < max);`,
      doc: `Keep the ${d.plural} in rows whose ${label(d, f)} is below max, in their order.`,
      condition: `whose ${label(d, f)} is below ${max}` }; },
  (rng, d) => { const [c, , values] = d.category, value = rng.pick(values);
    return { name: `${d.plural}_in_${c}`, params: `rows: ${d.type}[], ${c}: string`, args: [value], returns: `${d.type}[]`,
      body: `return rows.filter(row => row.${c} === ${c});`,
      doc: `Keep the ${d.plural} in rows whose ${c} is the given ${c}, in their order.`,
      condition: `whose ${c} is ${value}`, category: true }; },
];

const AGGREGATES = [
  (rng, d) => { const f = rng.pick(numeric(d));
    return { name: `total_${f}`, params: `rows: ${d.type}[]`, args: [], returns: 'number',
      body: `return rows.reduce((sum, row) => sum + row.${f}, 0);`,
      doc: `Add up the ${label(d, f)} of the ${d.plural} in rows.`, phrase: `the total ${label(d, f)}` }; },
  (rng, d) => { const f = rng.pick(numeric(d));
    return { name: `median_${f}`, params: `rows: ${d.type}[]`, args: [], returns: 'number | null',
      body: `const values = rows.map(row => row.${f}).sort((a, b) => a - b);\nif (!values.length) return null;\nconst middle = Math.floor(values.length / 2);\nreturn values.length % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;`,
      doc: `The median ${label(d, f)} of the ${d.plural} in rows (the mean of the middle two for an even count), or null for none.`,
      phrase: `the median ${label(d, f)} (the mean of the middle two for an even count, null when there are none)` }; },
  (rng, d) => { const f = rng.pick(numeric(d));
    return { name: `${d.singular}_with_most_${f}`, params: `rows: ${d.type}[]`, args: [], returns: 'string | null',
      body: `let best: ${d.type} | null = null;\nfor (const row of rows) if (best === null || row.${f} > best.${f}) best = row;\nreturn best === null ? null : best.id;`,
      doc: `The id of the ${d.singular} in rows with the largest ${label(d, f)} (the first on a tie), or null for none.`,
      phrase: `the id of the ${d.singular} with the largest ${label(d, f)} (the first on a tie, null when there are none)` }; },
  (rng, d) => { const f = rng.pick(numeric(d)), k = rng.int(2, 4);
    return { name: `top_${f}`, params: `rows: ${d.type}[], k: number`, args: [k], returns: 'string[]',
      body: `return [...rows].sort((a, b) => b.${f} - a.${f} || a.id.localeCompare(b.id)).slice(0, k).map(row => row.id);`,
      doc: `The ids of the k ${d.plural} in rows with the largest ${label(d, f)}, largest first, ties by id.`,
      phrase: `the ids of the ${k} ${d.plural} with the largest ${label(d, f)}, largest first and ties by id` }; },
  (rng, d) => { const [c] = d.category;
    return { name: `count_by_${c}`, params: `rows: ${d.type}[]`, args: [], returns: 'Record<string, number>',
      body: `const counts: Record<string, number> = {};\nfor (const row of rows) counts[row.${c}] = (counts[row.${c}] ?? 0) + 1;\nreturn counts;`,
      doc: `How many ${d.plural} in rows there are per ${c}, for each ${c} that has any.`,
      phrase: `how many there are per ${c} (only the ${c}s that have any)`, table: `number of ${d.plural}` }; },
  (rng, d) => { const [c] = d.category, f = rng.pick(numeric(d));
    return { name: `${f}_by_${c}`, params: `rows: ${d.type}[]`, args: [], returns: 'Record<string, number>',
      body: `const totals: Record<string, number> = {};\nfor (const row of rows) totals[row.${c}] = (totals[row.${c}] ?? 0) + row.${f};\nreturn totals;`,
      doc: `The total ${label(d, f)} of the ${d.plural} in rows per ${c}, for each ${c} that has any.`,
      phrase: `the total ${label(d, f)} per ${c} (only the ${c}s that have any)`, table: `total ${label(d, f)}` }; },
];

/** Steps on a per-category table: the final answer drawn from it. */
const TABLE_STEPS = [
  (d, measure) => ({ name: `largest_${d.category[0]}`, params: 'table: Record<string, number>', args: [], returns: 'string | null',
    body: `let best: string | null = null;\nfor (const key of Object.keys(table).sort()) if (best === null || table[key] > table[best]) best = key;\nreturn best;`,
    doc: `The key of table with the largest number (alphabetically first on a tie), or null for an empty table.`,
    phrase: `the ${d.category[0]} with the largest ${measure} (alphabetically first on a tie, null when there is none)` }),
  (d, measure) => ({ name: `ranked_${d.category[0]}s`, params: 'table: Record<string, number>', args: [], returns: 'string[]',
    body: `return Object.keys(table).sort((a, b) => table[b] - table[a] || a.localeCompare(b));`,
    doc: `The keys of table from the largest number to the smallest, ties alphabetically.`,
    phrase: `the ${d.category[0]}s ordered by ${measure}, largest first and ties alphabetically` }),
];

/** An operation's function in TypeScript. */
const source = op => `function ${op.name}(${op.params}): ${op.returns} {\n${op.body.split('\n').map(line => `  ${line}`).join('\n')}\n}`;

/** Run an operation's own TypeScript on an input, for the expected result. */
function run(op, input) {
  const js = ts.transpileModule(`${source(op)}\nreturn ${op.name}(input, ...args);`,
    { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText;
  return new Function('input', 'args', js)(input, op.args);
}

/** A natural-language helper: its doc, and a call that computes the result in code. */
function helperNlFile(op) {
  // Parameters split at top-level commas only: Record<string, number> is one type.
  const args = op.params.split(/, (?![^<]*>)/).map(part => part.split(': '));
  return `---\ndescription: ${JSON.stringify(op.doc)}\nargs:\n${args.map(([name, type]) => `  ${name}: ${JSON.stringify(type)}`).join('\n')}\n` +
    `returns: ${JSON.stringify(op.returns)}\n---\n${op.doc} Compute it exactly with eval.\n`;
}

/** The task's instructions, as one sentence or as numbered steps. */
function instructionsFor(rng, domain, filters, aggregate, tableStep) {
  const subject = `the ${domain.plural} ${filters.map(f => f.condition).join(' and ')}`;
  if (rng.next() < 0.5) {
    const lines = [`Start from ${domain.plural}.`, ...filters.map(f => `Keep only those ${f.condition}.`),
      tableStep ? `Work out ${aggregate.phrase}.` : `Return ${aggregate.phrase}.`, ...(tableStep ? [`Return ${tableStep.phrase}.`] : [])];
    return lines.map((line, i) => `${i + 1}. ${line}`).join('\n');
  }
  if (tableStep) return rng.pick([`Of ${subject}, work out ${aggregate.phrase}, and return ${tableStep.phrase}.`,
    `Among ${subject}, find ${tableStep.phrase}, going by ${aggregate.phrase}.`]);
  return rng.pick([`Of ${subject}, return ${aggregate.phrase}.`, `Find ${aggregate.phrase} of ${subject}.`,
    `Considering only ${subject}, work out ${aggregate.phrase}.`]);
}

/**
 * A task of one or two filters, an aggregate, and for a per-category aggregate one step over the table. Each step is a
 * helper of the program (TypeScript or natural language) or, for one step in some variants, work the task leaves to
 * the model's own code.
 */
export function composedHelpers(seed, index) {
  const rng = new Random(seed, `composed-helpers:${index}`);
  const domain = rng.pick(DOMAINS), rows = records(rng, domain, rng.int(8, 16));
  const aggregate = rng.pick(AGGREGATES)(rng, domain);
  // No category filter before counting per that category: every count would be one category's.
  const filters = rng.sample(FILTERS.filter((_, i) => !(aggregate.table && i === 2)), rng.int(1, 2)).map(make => make(rng, domain));
  const tableStep = aggregate.table ? rng.pick(TABLE_STEPS)(domain, aggregate.table) : undefined;
  const steps = [...filters, aggregate, ...(tableStep ? [tableStep] : [])];
  if (new Set(steps.map(step => step.name)).size !== steps.length) return composedHelpers(seed, `${index}'`);
  const root = rng.pick(['analyze', 'report', 'summarize', 'answer']) + `_${domain.plural}`;
  // A step the model writes itself: a filter, stated only in the instructions.
  const own = filters.length > 1 || rng.next() < 0.3 ? filters[filters.length - 1] : undefined;
  const helpers = steps.filter(step => step !== own).map(step => ({ ...step, kind: rng.next() < 0.5 ? 'ts' : 'nl' }));
  // TypeScript helpers are one module of the program's (a service it calls by module name); each natural-language
  // helper is a function of its own.
  const module = `${domain.singular}_tools`, typed = helpers.filter(h => h.kind === 'ts');
  const files = { 'types.ts': `export type ${domain.type} = { id: string, ${numeric(domain).map(f => `${f}: number`).join(', ')}, ${domain.category[0]}: string };\n` };
  if (typed.length) files[`${root}/${module}.ts`] = `import type { ${domain.type} } from '../types';\n\n` +
    typed.map(h => `/** ${h.doc} */\nexport ${source(h)}\n`).join('\n');
  for (const helper of helpers.filter(h => h.kind === 'nl')) files[`${root}/${helper.name}.nl`] = helperNlFile(helper);

  let value = rows, current = domain.plural;
  const lines = [], children = [];
  steps.forEach((step, i) => {
    const next = run(step, value), target = i === steps.length - 1 ? 'result' : `step${i + 1}`;
    const call = `${step.name}(${[current, ...step.args.map(arg => JSON.stringify(arg))].join(', ')})`;
    const helper = helpers.find(h => h.name === step.name);
    if (!helper) lines.push(`const ${target} = ${current}.filter(row => ${ownCondition(step)});`);
    else if (helper.kind === 'ts') lines.push(`const ${target} = ${module}.${call};`);
    else {
      lines.push(`const ${target} = await ${call};`);
      // Each helper is called once, so its doc identifies its call; the call computes in code, then returns.
      children.push({ match: helper.doc, calls: [evalCall(helper.body), returnCall(next)] });
    }
    value = next; current = target;
  });
  const instructions = instructionsFor(rng, domain, filters, aggregate, tableStep) +
    (rng.next() < 0.5 ? '\n\nUse the program\'s functions where one does a step.' : '');
  return [curriculumCase({ family: 'composed_helpers', shape: `h${index}`, variant: 'v0', splitGroup: `composed-helpers:h${index}`,
    slice: 'nested_scoped', domain: 'other', mode: 'single_call', inline: 'avoid',
    named: helpers.some(h => h.kind === 'nl') ? 'required' : undefined,
    evidence: { world: [`${rows.length} ${domain.plural}`], retrieved: [JSON.stringify(value)], background: [] },
    minimumSequence: ['read the function listing', 'call the helpers in order'],
    reference: { root: [evalCall(`${lines.join('\n')}\nreturn result;`), returnCall(value)], children },
    root: { name: root, args: { [domain.plural]: `${domain.type}[]` }, returns: steps.at(-1).returns, instructions },
    files, inputs: { [domain.plural]: rows }, expected: value })];
}

function ownCondition(step) {
  const match = /return rows\.filter\(row => (.*)\);/.exec(step.body);
  return match[1].replace(/\b(min|max)\b/, String(step.args[0])).replace(/=== (\w+)$/, `=== ${JSON.stringify(step.args[0])}`);
}


// ---- Processes for iterateOn -----------------------------------------------------------------------------------
// A process is a state type, a starting state built from the arguments, a step, ways to stop and things to report,
// each as code with its sentence. The reference runs the step with iterateOn; the expected result is the same
// TypeScript simulated here. Runs are kept to at most MAX_STEPS steps: iterateOn reviews a longer run's progress with a
// model call that a scripted solution cannot answer.

const MAX_STEPS = 9;

const PROCESSES = [
  rng => {
    const start = rng.int(100, 2000), rate = rng.int(2, 12), deposit = rng.pick([0, 25, 50, 100, 200]);
    return { args: { start: ['number', start], rate_percent: ['number', rate], deposit: ['number', deposit],
      target: ['number', null] }, target: 'target', state: '{ balance: number, months: number }', initial: '{ balance: start, months: 0 }',
      step: 's => ({ balance: Math.round(s.balance * (1 + rate_percent / 100)) + deposit, months: s.months + 1 })',
      stepText: 'add rate_percent percent interest to the balance, rounded to a whole number, and then add deposit',
      startText: 'a balance of start',
      stop: 's => s.balance >= target', stopText: 'the balance is at least target',
      measure: 's.balance',
      results: [['final.months', 'number', 'how many months that took'], ['final.balance', 'number', 'the balance at that point']] };
  },
  rng => {
    const jobs = Array.from({ length: rng.int(4, 9) }, (_, i) => ({ id: `J${i + 1}`, minutes: rng.int(5, 40) }));
    const total = jobs.reduce((sum, job) => sum + job.minutes, 0), budget = rng.int(Math.floor(total / 3), total);
    return { types: 'export type Job = { id: string, minutes: number };\n', args: { jobs: ['Job[]', jobs], budget: ['number', budget] },
      state: '{ queue: Job[], time: number, done: string[] }', initial: '{ queue: jobs, time: 0, done: [] }',
      step: 's => ({ queue: s.queue.slice(1), time: s.time + s.queue[0].minutes, done: [...s.done, s.queue[0].id] })',
      stepText: 'take the first job off the queue, add its minutes to the elapsed time and record its id as done',
      startText: 'jobs as the queue, no time elapsed and nothing done',
      stop: 's => s.queue.length === 0 || s.time >= budget', stopText: 'the queue is empty or the elapsed time has reached budget minutes',
      results: [['final.done', 'string[]', 'the ids of the jobs done, in order'], ['final.time', 'number', 'the elapsed minutes']] };
  },
  rng => {
    const level = rng.int(200, 5000), keep = rng.int(40, 85);
    return { args: { level: ['number', level], keep_percent: ['number', keep], threshold: ['number', null] }, target: 'threshold',
      state: '{ level: number, rounds: number }', initial: '{ level, rounds: 0 }',
      step: 's => ({ level: Math.floor(s.level * keep_percent / 100), rounds: s.rounds + 1 })',
      stepText: 'keep keep_percent percent of the level, rounded down', startText: 'level',
      stop: 's => s.level < threshold', stopText: 'the level is below threshold', measure: 's.level', falling: true,
      results: [['final.rounds', 'number', 'how many rounds that took'], ['final.level', 'number', 'the level at that point']] };
  },
  rng => {
    const n = rng.int(10 ** 5, 10 ** 12);
    return { args: { n: ['number', n] }, state: '{ n: number, replacements: number }', initial: '{ n, replacements: 0 }',
      step: 's => ({ n: String(s.n).split("").reduce((sum, digit) => sum + Number(digit), 0), replacements: s.replacements + 1 })',
      stepText: 'replace the number with the sum of its digits', startText: 'n',
      stop: 's => s.n < 10', stopText: 'a single digit remains',
      results: [['final.n', 'number', 'the digit that remains'], ['final.replacements', 'number', 'how many replacements it took']] };
  },
  rng => {
    const n = rng.int(3, 200);
    return { args: { n: ['number', n] }, state: '{ n: number, steps: number, peak: number }', initial: '{ n, steps: 0, peak: n }',
      step: 's => { const next = s.n % 2 === 0 ? s.n / 2 : 3 * s.n + 1; return { n: next, steps: s.steps + 1, peak: Math.max(s.peak, next) }; }',
      stepText: 'halve the number if it is even, and otherwise triple it and add one', startText: 'n',
      stop: 's => s.n === 1', stopText: 'the number is 1',
      results: [['final.steps', 'number', 'how many steps that took'], ['final.peak', 'number', 'the largest number reached, n included']] };
  },
  rng => {
    const moves = Array.from({ length: rng.int(5, 9) }, () => rng.pick(['N', 'E', 'S', 'W']));
    const path = moves.reduce((points, move) => { const [x, y] = points.at(-1);
      return [...points, [x + (move === 'E') - (move === 'W'), y + (move === 'N') - (move === 'S')]]; }, [[0, 0]]);
    const [gx, gy] = rng.next() < 0.7 ? rng.pick(path.slice(2)) : [rng.int(-3, 3), rng.int(-3, 3)];
    return { types: 'export type Point = { x: number, y: number };\n', args: { moves: ['string[]', moves], goal: ['Point', { x: gx, y: gy }] },
      state: '{ x: number, y: number, left: string[] }', initial: '{ x: 0, y: 0, left: moves }',
      step: 's => { const move = s.left[0]; return { x: s.x + (move === "E" ? 1 : move === "W" ? -1 : 0), y: s.y + (move === "N" ? 1 : move === "S" ? -1 : 0), left: s.left.slice(1) }; }',
      stepText: 'make the next move of the list (N adds 1 to y, S takes 1 from y, E adds 1 to x, W takes 1 from x)',
      startText: 'position (0, 0) with moves still to make',
      stop: 's => (s.x === goal.x && s.y === goal.y) || s.left.length === 0', stopText: 'the position is goal or no moves are left',
      results: [['final.left.length', 'number', 'how many moves were left unmade'], ['final.x === goal.x && final.y === goal.y', 'boolean', 'whether goal was reached']] };
  },
  rng => {
    const names = rng.sample(['Ada', 'Bo', 'Cy', 'Dee', 'Eli', 'Fay', 'Gus', 'Hal', 'Ivy', 'Jo'], rng.int(5, 9));
    const players = names.map(name => ({ name, score: rng.int(1, 30) }));
    return { types: 'export type Player = { name: string, score: number };\n',
      args: { players: ['Player[]', players], keep: ['number', rng.int(1, 3)] },
      state: '{ players: Player[], out: string[] }', initial: '{ players, out: [] }',
      step: 's => { const lowest = s.players.reduce((low, p) => p.score < low.score ? p : low); return { players: s.players.filter(p => p !== lowest), out: [...s.out, lowest.name] }; }',
      stepText: 'remove the player with the lowest score (the earliest in the list on a tie)', startText: 'players, nobody out',
      stop: 's => s.players.length <= keep', stopText: 'at most keep players remain',
      results: [['final.players.map(p => p.name)', 'string[]', 'the names of the remaining players, in list order'], ['final.out', 'string[]', 'the names of the removed players, in the order they went out']] };
  },
];

/**
 * Run a process's own TypeScript on argument values, stopping when `stop` holds: the number of steps and `report` of the
 * final state, or steps Infinity past MAX_STEPS. `trail` lists `report` after each of MAX_STEPS steps, never stopping.
 */
function simulate(process, values, { stop = process.stop, report = 'final', trail = false } = {}) {
  const code = `type State = ${process.state};\nconst step = (${process.step.replace(/^s =>/, '(s: State): State =>')});\n` +
    `const stop = ${stop.replace(/^s =>/, '(s: State) =>')};\nlet final: State = ${process.initial};\nlet steps = 0;\n` +
    (trail ? `const out = [];\nwhile (steps++ < ${MAX_STEPS}) { final = step(final); out.push(${report}); }\nreturn out;` :
      `while (!stop(final)) { if (steps === ${MAX_STEPS}) return { steps: Infinity }; final = step(final); steps++; }\n` +
      `return { steps, result: ${report} };`);
  const js = ts.transpileModule(code, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText;
  return new Function(...Object.keys(values), js)(...Object.values(values));
}

export function composedProcess(seed, index) {
  const rng = new Random(seed, `composed-process:${index}`);
  for (;;) {
    const process = rng.pick(PROCESSES)(rng);
    const values = Object.fromEntries(Object.entries(process.args).map(([name, [, value]]) => [name, value]));
    // A target the process passes within MAX_STEPS steps: the measure after one of steps 2 to MAX_STEPS (for a
    // falling measure, one above it, so that the step reaching it stops the run).
    if (process.target) {
      const trail = simulate(process, values, { report: process.measure.replace(/\bs\./g, 'final.'), trail: true });
      values[process.target] = trail[rng.int(1, MAX_STEPS - 1)] + (process.falling ? 1 : 0);
    }
    const [resultCode, returns, resultText] = rng.pick(process.results);
    const { steps, result: expected } = simulate(process, values, { report: resultCode });
    if (steps < 2 || steps > MAX_STEPS) continue;
    const args = Object.fromEntries(Object.entries(process.args).map(([name, [type]]) => [name, type]));
    const name = rng.pick(['simulate', 'run_process', 'play_out', 'work_through']);
    const instructions = rng.next() < 0.5 ?
      `Start from ${process.startText}. Repeat one step: ${process.stepText}. Stop as soon as ${process.stopText}, which may already hold at the start. Return ${resultText}.` :
      `1. Start from ${process.startText}.\n2. If ${process.stopText}, stop.\n3. Otherwise ${process.stepText}, and go back to 2.\n4. Return ${resultText}.`;
    const code = `type State = ${process.state};\nconst step = ${process.step.replace(/^s =>/, '(s: State): State =>')};\n` +
      `const final = await iterateOn(step, ${process.initial} as State).until(${process.stop.replace(/^s =>/, '(s: State) =>')});\nreturn ${resultCode};`;
    return [curriculumCase({ family: 'composed_process', shape: `p${index}`, variant: 'v0', splitGroup: `composed-process:p${index}`,
      slice: 'iterate', domain: 'other', mode: 'single_call', inline: 'avoid', iterate: 'required',
      evidence: { world: [], retrieved: [JSON.stringify(expected)], background: [] },
      minimumSequence: ['write the step', 'run it with iterateOn until the stopping rule holds'],
      reference: { root: [evalCall(code), returnCall(expected)] },
      root: { name, args, returns, instructions }, files: process.types ? { 'types.ts': process.types } : {},
      inputs: values, expected })];
  }
}
