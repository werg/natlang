#!/usr/bin/env node
/** Score a decision model's labels against gold with the episode scorers: the teacher baseline per family.
 *
 * Usage: score-decision-labels.mjs --cases decision-cases.jsonl --labels labels.jsonl --out baseline.json
 * Each label's answer (Jev/SystemOne shape) is scored as the episode targets would be: choice probabilities by
 * Brier, noul probability by Brier against the label or frequency, score probabilities by ranked probability score. */
import { createReadStream } from 'node:fs';
import { writeFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { scoreGraded } from '../../dist/skills/graded.js';

const arg = name => { const i = process.argv.indexOf(name); return i < 0 ? undefined : process.argv[i + 1]; };
const [casesPath, labelsPath, out] = [arg('--cases'), arg('--labels'), arg('--out')];
if (!casesPath || !labelsPath || !out) throw Error('Usage: score-decision-labels.mjs --cases FILE --labels FILE --out FILE');
const lines = async function* (path) { for await (const line of createInterface({ input: createReadStream(path) })) if (line.trim()) yield JSON.parse(line); };

const cases = new Map();
for await (const c of lines(casesPath)) cases.set(c.id, c);
const metric = kind => ({ schema: 'natlang.skill-graded/1', kind });
const families = {};
for await (const label of lines(labelsPath)) {
  const c = cases.get(label.id);
  if (!c) continue;
  const f = families[`${c.family}|${c.role}`] ??= { family: c.family, role: c.role, kind: c.kind, n: 0, errors: 0, quality: 0, top: 0 };
  f.n++;
  if (label.answer.error) { f.errors++; continue; }
  let score;
  if (c.kind === 'choice') score = scoreGraded(metric('choice-brier'), { probabilities: label.answer.probabilities },
    { kind: 'choice', answer: c.answer, options: c.options });
  else if (c.kind === 'noul') score = scoreGraded(metric('binary-brier'), label.answer.noul, { kind: 'binary', answer: c.answer });
  else score = scoreGraded(metric('ordinal-rps'), { probabilities: label.answer.probabilities },
    { kind: 'ordinal', levels: c.levels, answer: c.answer });
  f.quality += score.quality;
  f.top += Object.values(score.gates).every(Boolean) ? 1 : 0;
}
const rows = Object.values(families).map(f => ({ ...f, quality: f.n - f.errors ? f.quality / (f.n - f.errors) : null,
  top: f.n - f.errors ? f.top / (f.n - f.errors) : null })).sort((a, b) => a.family < b.family ? -1 : 1);
await writeFile(out, JSON.stringify({ schema: 'natlang.decision-teacher-baseline/1', labels: labelsPath, families: rows }, null, 2) + '\n');
for (const r of rows) console.log(r.family.padEnd(28), r.role.padEnd(8), r.kind.padEnd(7), String(r.n).padStart(5),
  'quality', r.quality?.toFixed(3), 'all-gates', r.top?.toFixed(3), r.errors ? `errors ${r.errors}` : '');
