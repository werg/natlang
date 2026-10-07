#!/usr/bin/env node
/**
 * Does a model write working tree code under a runtime's recursion rules? Eight tree tasks (counting, depth, sizes,
 * paths, costs, labels), each run `--trials` times through the collector against a served model, on the runtime in
 * `--runtime` (a ts-host checkout with a built dist). Run it once with the old runtime and once with the new one,
 * against the same model and seeds, and compare: correct answers, whether the eval code recursed, and recursion errors.
 *
 *   node scripts/recursion-eval.mjs --runtime ../ts-host --endpoint http://127.0.0.1:8082 --model student \
 *     --trials 3 --out runs/recursion-eval/new.jsonl
 *   --scripted: a scripted model that writes a recursive walk (checks the wiring, not a model).
 */
import { appendFile, mkdir } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { curriculumCase, evalCall, returnCall } from './inline-curriculum/lib.mjs';

const { values } = parseArgs({ options: {
  runtime: { type: 'string', default: resolve(import.meta.dirname, '..') }, endpoint: { type: 'string' },
  model: { type: 'string', default: 'student' }, trials: { type: 'string', default: '3' }, out: { type: 'string' },
  'context-tokens': { type: 'string', default: '8192' }, 'max-turns': { type: 'string', default: '12' },
  temperature: { type: 'string', default: '0.7' }, scripted: { type: 'boolean', default: false } } });
const runtime = resolve(values.runtime);
const module = relative => import(pathToFileURL(join(runtime, 'dist', relative)).href);
const [collector, prompts, transport] = await Promise.all([module('teacher/collector.js'), module('native/prompt.js'),
  module('model/openai-compatible.js')]);

/** Deterministic trees: a person with reports, a comment with replies, and so on, `depth` levels deep. */
function tree(depth, fanout, make, path = [0]) {
  const children = depth > 1 ? Array.from({ length: fanout(path) }, (_, i) => tree(depth - 1, fanout, make, [...path, i])) : [];
  return make(path, children);
}
const fan = path => 1 + (path.reduce((a, b) => a * 7 + b, 3) % 3);
const name = path => ['Ada', 'Bo', 'Cy', 'Di', 'Ed', 'Flo', 'Gus', 'Hal'][path.length - 1] + path.slice(1).join('');
const walk = (node, key, visit, depth = 1) => { visit(node, depth); for (const child of node[key]) walk(child, key, visit, depth + 1); };

const org = tree(5, fan, (path, reports) => ({ name: name(path), reports }));
const thread = tree(6, path => (path.length % 2 ? 2 : 1), (path, replies) => ({ text: `reply ${path.join('.')}`, replies }));
const files = tree(4, fan, (path, children) => children.length ? { name: `dir${path.join('')}`, children } :
  { name: `f${path.join('')}.txt`, size: 10 * path.reduce((a, b) => a + b + 1, 0), children: [] });
const parts = tree(4, fan, (path, components) => ({ part: `P${path.join('')}`, price: path.length * 3, qty: 1 + path.at(-1) % 2, components }));
const menu = tree(4, path => (path.length === 1 ? 3 : 2), (path, items) => ({ label: `M${path.join('')}`, items }));
const json = { a: 1, b: { c: [2, 3, { d: 4, e: 'x' }], f: { g: { h: 5, i: [6, { j: 7 }] } } }, k: 'y', l: [8, [9, [10]]] };

let count = 0, deepest = 0, size = 0, leaves = [], cost = 0, labels = [];
walk(org, 'reports', () => count++);
walk(thread, 'replies', (_, depth) => { deepest = Math.max(deepest, depth); });
walk(files, 'children', node => { size += node.size ?? 0; });
walk(org, 'reports', node => { if (!node.reports.length) leaves.push(node.name); });
const target = leaves.at(-1);
const pathTo = (node, goal) => node.name === goal ? [node.name] : node.reports.reduce((found, child) => found ?? (r => r && [node.name, ...r])(pathTo(child, goal)), null);
const costOf = node => node.price * node.qty + node.qty * node.components.reduce((sum, child) => sum + costOf(child), 0);
cost = costOf(parts);
const label = (node, prefix) => { const own = prefix ? `${prefix} > ${node.label}` : node.label; if (!node.items.length) labels.push(own); node.items.forEach(item => label(item, own)); };
label(menu, '');
const numbers = value => typeof value === 'number' ? 1 : value && typeof value === 'object' ? Object.values(value).reduce((s, v) => s + numbers(v), 0) : 0;

const TASKS = [
  { name: 'org_size', args: { org: 'Person' }, types: 'type Person = { name: string, reports: Person[] };', returns: 'number', inputs: { org },
    instructions: 'How many people are in the organisation chart org, counting org itself and everyone below?', expected: count },
  { name: 'thread_depth', args: { thread: 'Comment' }, types: 'type Comment = { text: string, replies: Comment[] };', returns: 'number',
    inputs: { thread }, instructions: 'How many levels deep is the comment thread? The top comment is level 1.', expected: deepest },
  { name: 'total_size', args: { root: 'Entry' }, types: 'type Entry = { name: string, size?: number, children: Entry[] };', returns: 'number',
    inputs: { root: files }, instructions: 'What is the total size of all files in the folder tree root? Folders have no size of their own.', expected: size },
  { name: 'leaf_names', args: { org: 'Person' }, types: 'type Person = { name: string, reports: Person[] };', returns: 'string[]', inputs: { org },
    instructions: 'List the names of everyone in org who has no reports, in the order they appear from top to bottom, left to right.', expected: leaves },
  { name: 'chain_to', args: { org: 'Person', person: 'string' }, types: 'type Person = { name: string, reports: Person[] };', returns: 'string[]',
    inputs: { org, person: target }, instructions: 'List the chain of names from the top of org down to person, both included.',
    expected: pathTo(org, target) },
  { name: 'bill_cost', args: { item: 'Part' }, types: 'type Part = { part: string, price: number, qty: number, components: Part[] };', returns: 'number',
    inputs: { item: parts }, instructions: 'What does item cost? A part costs price × qty, plus qty times the cost of each of its components.', expected: cost },
  { name: 'menu_labels', args: { menu: 'Menu' }, types: 'type Menu = { label: string, items: Menu[] };', returns: 'string[]', inputs: { menu },
    instructions: 'List every final menu entry (one with no items) as its full path of labels joined with " > ", starting at menu itself.',
    expected: labels },
  { name: 'json_numbers', args: { value: 'unknown' }, types: '', returns: 'number', inputs: { value: json },
    instructions: 'How many numbers does value contain anywhere inside it, including inside nested objects and arrays?', expected: numbers(json) },
];

const records = TASKS.map(task => curriculumCase({ family: 'recursion_eval', shape: task.name, variant: 'a', slice: 'single',
  domain: 'other', mode: 'single_call', inline: 'avoid', split: 'test', inputs: task.inputs, expected: task.expected,
  root: { name: task.name, args: task.args, returns: task.returns, instructions: task.instructions },
  files: task.types ? { 'types.ts': `export ${task.types}\n` } : {},
  reference: { root: [evalCall('return 0'), returnCall(task.expected)] } }));

const recursed = code => {
  for (const match of code.matchAll(/(?:function\s+([A-Za-z_$][\w$]*)|(?:const|let)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=>)/g)) {
    const fn = match[1] ?? match[2], body = code.slice(match.index + match[0].length);
    if (new RegExp(`\\b${fn}\\s*\\(`).test(body.slice(0, 2000))) return true;
  }
  return false;
};

/** A scripted model that walks the tree recursively: checks the evaluation's wiring, not a model. */
function scriptedDriver(record) {
  const task = TASKS.find(item => record.id.includes(`:${item.name}:`));
  const key = Object.keys(task.inputs)[0];
  let turn = 0;
  return async () => turn++ === 0 ? { calls: [['eval', { code:
    `const n = (v: any): number => typeof v === 'number' ? 1 : v && typeof v === 'object' ? Object.values(v).reduce((s: number, x: any) => s + n(x), 0) : 0;\n` +
    `const count = (p: any): number => 1 + (p.reports ?? []).reduce((s: number, c: any) => s + count(c), 0);\n` +
    `return ${task.name === 'json_numbers' ? `n(${key})` : task.name === 'org_size' ? `count(${key})` : 'null'};` }]] } : { text: 'done' };
}

const trials = Number(values.trials), out = values.out && resolve(values.out);
if (out) await mkdir(dirname(out), { recursive: true });
const summary = {};
for (const record of records) for (let trial = 0; trial < trials; trial++) {
  const codes = [];
  const send = values.scripted ? scriptedDriver(record) :
    transport.openAICompatibleModelTurn({ endpoint: values.endpoint, model: values.model, stream: false });
  const driver = async request => {
    const turn = await send({ ...request, temperature: Number(values.temperature), seed: 1000 * trial + 17 });
    for (const [name, args] of turn.calls ?? []) if (name === 'eval') codes.push(String(args.code ?? ''));
    return turn;
  };
  let run, error;
  try {
    run = await collector.executeProgram(record, driver, { systemPrompt: prompts.TOOLS_PROMPT,
      contextTokens: Number(values['context-tokens']), maxTurns: Number(values['max-turns']),
      temperature: Number(values.temperature), rootSeed: 1000 * trial + 17, runId: `recursion-eval-${trial}` });
  } catch (caught) { error = String(caught?.message ?? caught); }
  const trace = JSON.stringify(run?.trace ?? []);
  const row = { task: record.curriculum.shape, trial, correct: run?.outcome.accepted === true,
    recursed: codes.some(recursed), recursion_errors: (trace.match(/without a smaller argument|calls itself; recursion is not allowed|already running in its own call chain/g) ?? []).length,
    evals: codes.length, error };
  const entry = (summary[row.task] ??= { trials: 0, correct: 0, recursed: 0, recursion_errors: 0 });
  entry.trials++; entry.correct += Number(row.correct); entry.recursed += Number(row.recursed); entry.recursion_errors += row.recursion_errors;
  if (out) await appendFile(out, JSON.stringify({ runtime, ...row, codes }) + '\n');
}
const total = Object.values(summary).reduce((all, entry) => ({ trials: all.trials + entry.trials, correct: all.correct + entry.correct,
  recursed: all.recursed + entry.recursed, recursion_errors: all.recursion_errors + entry.recursion_errors }),
  { trials: 0, correct: 0, recursed: 0, recursion_errors: 0 });
console.log(JSON.stringify({ runtime, model: values.scripted ? 'scripted' : values.model, total, tasks: summary }, null, 1));
