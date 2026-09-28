// Offline source audit: no model calls. Evaluate only expressions produced by the finite operator compiler.
import { readFileSync, writeFileSync } from 'node:fs';
import { commaqaArithmetic } from './sources-ai2.mjs';

const same = (a, b) => typeof a === 'number' && typeof b === 'number' ?
  Math.abs(a - b) <= 1e-8 : Array.isArray(a) && Array.isArray(b) ?
    a.length === b.length && a.every((value, i) => same(value, b[i])) : a === b;

export function auditArithmetic(worlds, split) {
  const mismatches = [], operators = {};
  let numericSteps = 0;
  worlds.forEach((world, w) => world.qa_pairs.forEach((qa, q) => {
    const values = [], names = [];
    qa.decomposition.forEach((step, i) => {
      if (!['table', 'text'].includes(step.m)) {
        // The compiler accepts only recognized operators and numeric literal thresholds; raw source code is never run.
        const expression = commaqaArithmetic(step);
        const javascript = expression.replaceAll(' as [string, number]', '');
        const actual = Function(...names, `return (${javascript});`)(...values);
        numericSteps++;
        operators[step.op] = (operators[step.op] ?? 0) + 1;
        if (!same(actual, step.a)) mismatches.push({ source: `${split}:${w}:${q}`, step: i + 1,
          operator: step.op, actual, expected: step.a });
      }
      names.push(`step${i + 1}`); values.push(step.a);
    });
  }));
  return { split, numeric_steps: numericSteps, operators, mismatches };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const [directory, output] = process.argv.slice(2);
  if (!directory || !output) throw new Error('usage: audit-commaqa-arithmetic.mjs NUMERIC_COMMAQA_DIR REPORT.json');
  const reports = ['train', 'dev', 'test'].map(split => auditArithmetic(
    JSON.parse(readFileSync(`${directory}/${split}.json`, 'utf8')), split));
  const summary = { numeric_steps: reports.reduce((sum, r) => sum + r.numeric_steps, 0),
    mismatches: reports.flatMap(r => r.mismatches), splits: reports };
  writeFileSync(output, JSON.stringify(summary, null, 2) + '\n');
  console.log(JSON.stringify({ numeric_steps: summary.numeric_steps, mismatches: summary.mismatches.length }));
  if (summary.mismatches.length) throw new Error('numeric source audit failed');
}
