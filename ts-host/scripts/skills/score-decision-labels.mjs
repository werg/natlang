#!/usr/bin/env node
/** Score a decision model's labels against gold with the episode scorers: the teacher baseline per family.
 *
 * Usage: score-decision-labels.mjs --cases decision-cases.jsonl --labels labels.jsonl --out baseline.json
 * Legacy label contracts use choice probabilities by Brier, noul probability by Brier, and score probabilities by
 * ranked probability score. `natlang.choice-label-confidence/1` reports exact class accuracy and confidence
 * calibration separately; confidence is never expanded into a synthetic probability distribution. */
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
let sawConfidenceContract = false;
for await (const label of lines(labelsPath)) {
  const c = cases.get(label.id);
  if (!c) continue;
  const contract = label.decision_contract ?? 'natlang.typed-decision-probabilities/1';
  const f = families[`${c.family}|${c.role}|${c.kind}|${contract}`] ??= {
    family: c.family, role: c.role, kind: c.kind, contract, n: 0, errors: 0, quality: 0, top: 0,
    correct: 0, confidenceSum: 0, confidenceBrier: 0,
    confidenceBins: Array.from({ length: 10 }, (_, bin) => ({ lower: bin / 10, upper: (bin + 1) / 10, n: 0, confidence: 0, correct: 0 })),
  };
  f.n++;
  if (contract === 'natlang.choice-label-confidence/1') sawConfidenceContract = true;
  if (label.answer.error) { f.errors++; continue; }
  if (contract === 'natlang.choice-label-confidence/1') {
    const a = label.answer;
    const keys = a && typeof a === 'object' && !Array.isArray(a) ? Object.keys(a).sort() : [];
    if (c.kind !== 'choice' || keys.length !== 2 || keys[0] !== 'choice' || keys[1] !== 'confidence' ||
        typeof a.choice !== 'string' || !c.options.includes(a.choice) ||
        typeof a.confidence !== 'number' || !Number.isFinite(a.confidence) || a.confidence < 0 || a.confidence > 1) {
      f.errors++;
      continue;
    }
    const correct = a.choice === c.answer ? 1 : 0;
    f.correct += correct;
    f.confidenceSum += a.confidence;
    f.confidenceBrier += (a.confidence - correct) ** 2;
    const binIndex = Math.min(9, Math.floor(a.confidence * 10));
    const bin = f.confidenceBins[binIndex];
    bin.n++; bin.confidence += a.confidence; bin.correct += correct;
    continue;
  }
  if (contract !== 'natlang.typed-decision-probabilities/1') {
    f.errors++;
    continue;
  }
  let score;
  if (c.kind === 'choice') score = scoreGraded(metric('choice-brier'), { probabilities: label.answer.probabilities },
    { kind: 'choice', answer: c.answer, options: c.options });
  else if (c.kind === 'noul') score = scoreGraded(metric('binary-brier'), label.answer.noul, { kind: 'binary', answer: c.answer });
  else score = scoreGraded(metric('ordinal-rps'), { probabilities: label.answer.probabilities },
    { kind: 'ordinal', levels: c.levels, answer: c.answer });
  f.quality += score.quality;
  f.top += Object.values(score.gates).every(Boolean) ? 1 : 0;
}
const rows = Object.values(families).map(f => {
  if (f.contract === 'natlang.choice-label-confidence/1') {
    const scored = f.n - f.errors;
    const bins = f.confidenceBins.filter(bin => bin.n).map(bin => ({ ...bin,
      mean_confidence: bin.confidence / bin.n, accuracy: bin.correct / bin.n }));
    const ece = scored ? bins.reduce((sum, bin) => sum + bin.n / scored * Math.abs(bin.mean_confidence - bin.accuracy), 0) : null;
    return { family: f.family, role: f.role, kind: f.kind, contract: f.contract, n: f.n, errors: f.errors,
      scored, correct: f.correct, accuracy: scored ? f.correct / scored : null,
      mean_confidence: scored ? f.confidenceSum / scored : null,
      confidence_brier: scored ? f.confidenceBrier / scored : null,
      confidence_ece_10_bin: ece, confidence_bins: bins };
  }
  return { family: f.family, role: f.role, kind: f.kind,
    ...(sawConfidenceContract ? { contract: f.contract } : {}), n: f.n, errors: f.errors,
    quality: f.n - f.errors ? f.quality / (f.n - f.errors) : null,
    top: f.n - f.errors ? f.top / (f.n - f.errors) : null };
}).sort((a, b) => a.family.localeCompare(b.family) || a.role.localeCompare(b.role) || (a.contract ?? 'natlang.typed-decision-probabilities/1').localeCompare(b.contract ?? 'natlang.typed-decision-probabilities/1'));
await writeFile(out, JSON.stringify({ schema: sawConfidenceContract ? 'natlang.decision-teacher-baseline/2' : 'natlang.decision-teacher-baseline/1',
  labels: labelsPath, families: rows }, null, 2) + '\n');
for (const r of rows) {
  if (r.contract === 'natlang.choice-label-confidence/1') {
    console.log(r.family.padEnd(28), r.role.padEnd(8), r.kind.padEnd(7), String(r.n).padStart(5),
      'accuracy', r.accuracy?.toFixed(3), 'confidence-brier', r.confidence_brier?.toFixed(3),
      'confidence-ece', r.confidence_ece_10_bin?.toFixed(3), r.errors ? `errors ${r.errors}` : '');
  } else {
    console.log(r.family.padEnd(28), r.role.padEnd(8), r.kind.padEnd(7), String(r.n).padStart(5),
      'quality', r.quality?.toFixed(3), 'all-gates', r.top?.toFixed(3), r.errors ? `errors ${r.errors}` : '');
  }
}
