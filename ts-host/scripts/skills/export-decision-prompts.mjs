#!/usr/bin/env node
/** Export the runtime's decision-readout prompts for decision cases: the exact messages and options a typed
 * `readout: decision` function sends to its model, with the gold distribution and (optionally) a teacher's.
 *
 * Usage: export-decision-prompts.mjs --cases decision-cases.jsonl --out prompts.jsonl [--labels labels.jsonl]...
 *          [--roles train,heldout] [--per-family N]
 * Each case becomes a function `decide(state: string)` whose instructions are the case's question and whose result
 * type is its options (choice), `boolean` (noul) or its level names (score). Training on these rows
 * (scripts/train_decision_readout.py) trains the same readout the runtime uses, prompt for prompt.
 */
import { createReadStream } from 'node:fs';
import { writeFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { createNatlangRuntime, loadVirtualNatlang } from '../../dist/index.js';

const args = { roles: 'train,heldout', 'per-family': '0', labels: [] };
for (let i = 2; i < process.argv.length; i += 2) {
  const key = process.argv[i].replace(/^--/, ''), value = process.argv[i + 1];
  if (!['cases', 'out', 'labels', 'roles', 'per-family'].includes(key) || value === undefined)
    throw Error('Usage: export-decision-prompts.mjs --cases FILE --out FILE [--labels FILE]... [--roles R] [--per-family N]');
  if (key === 'labels') args.labels.push(value); else args[key] = value;
}
const lines = async function* (path) { for await (const line of createInterface({ input: createReadStream(path) })) if (line.trim()) yield JSON.parse(line); };
const roles = new Set(args.roles.split(',')), perFamily = Number(args['per-family']);

/** The case's result values (JSON-able) and its gold distribution over them. */
export function caseTarget(c) {
  if (c.kind === 'choice') {
    const values = c.options;
    const gold = typeof c.answer === 'string' ? values.map(v => v === c.answer ? 1 : 0)
      : values.map(v => Number(c.answer?.[v] ?? 0));
    return { values, gold };
  }
  if (c.kind === 'noul') {
    const p = typeof c.answer === 'boolean' ? Number(c.answer) : Number(c.answer);
    return { values: [true, false], gold: [p, 1 - p] };
  }
  const k = c.levels.length, x = typeof c.answer === 'string' && c.levels.includes(c.answer) ? c.levels.indexOf(c.answer) : Number(c.answer);
  const gold = new Array(k).fill(0), low = Math.floor(x), frac = x - low;
  gold[low] += 1 - frac; if (frac > 0) gold[low + 1] += frac;
  return { values: c.levels, gold };
}

/** A teacher label (Jev/SystemOne answer shape) as a distribution over the case's values, or null. */
export function teacherDistribution(c, answer) {
  if (!answer || answer.error) return null;
  const { values } = caseTarget(c);
  if (c.kind === 'noul') {
    const p = typeof answer.noul === 'number' ? answer.noul : typeof answer.probability === 'number' ? answer.probability
      : typeof answer.p_yes === 'number' ? answer.p_yes : null;
    return p === null ? null : [p, 1 - p];
  }
  const weights = answer.probabilities;
  if (!weights || typeof weights !== 'object') return null;
  const raw = values.map((v, i) => Number(weights[v] ?? weights[String(i)] ?? 0));
  const total = raw.reduce((a, b) => a + b, 0);
  return total > 0 ? raw.map(w => w / total) : null;
}

const teachers = new Map();
for (const path of args.labels) for await (const label of lines(path)) {
  const byCase = teachers.get(label.id) ?? {};
  byCase[label.teacher] = label.answer;
  teachers.set(label.id, byCase);
}

let captured;
const driver = Object.assign(async () => { throw new Error('the tool loop must not run'); }, {
  decide: async request => { captured = request; return { log_probs: request.options.map(() => 0) }; } });
const runtime = createNatlangRuntime({ model: { driver }, seed: { mode: 'backend' } });
const typeOf = (c, values) => c.kind === 'noul' ? 'boolean' : values.map(v => JSON.stringify(v)).join(' | ');
const counts = new Map(), out = [];
for await (const c of lines(args.cases)) {
  if (!roles.has(c.role)) continue;
  const key = `${c.family}|${c.role}`;
  if (perFamily && (counts.get(key) ?? 0) >= perFamily) continue;
  counts.set(key, (counts.get(key) ?? 0) + 1);
  const { values, gold } = caseTarget(c);
  const fn = loadVirtualNatlang({ 'decide.nl': `---\nargs: { state: string }\nreturns: ${typeOf(c, values)}\nreadout: decision\n---\n${c.question}\n` }, 'decide.nl');
  captured = undefined;
  await runtime.run(() => fn(c.state));
  if (!captured) throw Error('no decision readout for ' + c.id);
  const labelled = Object.fromEntries(Object.entries(teachers.get(c.id) ?? {})
    .map(([teacher, answer]) => [teacher, teacherDistribution(c, answer)]).filter(([, d]) => d));
  out.push(JSON.stringify({ schema: 'natlang.decision-prompt/1', id: c.id, family: c.family, role: c.role, kind: c.kind,
    group: c.group, messages: captured.messages, options: captured.options, gold, teachers: labelled }));
}
await writeFile(args.out, out.join('\n') + '\n', { flag: 'wx' });
console.error(JSON.stringify({ rows: out.length, families: counts.size }));
