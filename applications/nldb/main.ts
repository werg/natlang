/**
 * `natlang run applications/nldb -- DATABASE [--engine pure|sqlite] [REQUEST...]`: talk to a database. With no
 * requests on the command line, each line of input is a request. A pure database is a folder (`shop.nldb`); a
 * sqlite one is a file (`shop.sqlite`). `-- DATABASE --scenario scenarios/shop.json` runs and judges a scenario.
 */
import { readFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { resolve } from 'node:path';
import type { TargetContext } from '@natlang/node';
import { FolderDatabase, SqliteDatabase, type Database, type Outcome } from './index.js';

export function render(outcome: Outcome): string {
  if ('error' in outcome) return `error (${outcome.kind}): ${outcome.error}`;
  if (outcome.kind === 'unclear') return `? ${outcome.clarification}`;
  if (outcome.kind === 'question') {
    const { columns, rows, explanation, assumptions } = outcome.answer;
    const cells = [columns, ...rows.map(row => row.map(value => value === null ? 'null' : String(value)))];
    const widths = columns.map((_, i) => Math.max(...cells.map(row => (row[i] ?? '').length)));
    const line = (row: string[]) => row.map((cell, i) => cell.padEnd(widths[i]!)).join('  ');
    return [line(cells[0]!), widths.map(width => '-'.repeat(width)).join('  '), ...cells.slice(1).map(line),
      `(${rows.length} row${rows.length === 1 ? '' : 's'}) ${explanation}`, ...assumptions.map(a => `assumed: ${a}`)].join('\n');
  }
  const { statements, changes, summary, assumptions } = outcome.report;
  const counts = Object.entries(changes).map(([table, c]) => `${table} +${c.inserted} ~${c.updated} -${c.deleted}`).join(', ');
  return [`${summary}${counts ? ` [${counts}]` : ''}`, ...statements.map(s => `  ${s}`), ...assumptions.map(a => `assumed: ${a}`)].join('\n');
}

type Step = { request: string, kind: string, answer?: unknown[][], ordered?: boolean, fails?: boolean };
const norm = (value: unknown) => typeof value === 'number' ? Number(value.toFixed(6)) : typeof value === 'string' && /^-?\d+(\.\d+)?$/.test(value) ? Number(value) : value;

/** Whether an outcome is what a scenario step expects; the answer's rows are compared by value (and order, if ordered). */
export function judgeStep(step: Step, outcome: Outcome): string | null {
  if (outcome.kind !== step.kind) return `read as ${outcome.kind}, expected ${step.kind}`;
  if (step.fails) return 'error' in outcome ? null : 'expected the transaction to fail, and it committed';
  if ('error' in outcome) return outcome.error;
  if (step.answer && outcome.kind === 'question') {
    // Extra columns are fine: each expected row's values must appear in its row, and the row counts must agree.
    const same = (a: unknown, b: unknown) => a === b || (typeof a === 'string' && typeof b === 'string' && a.toLowerCase() === b.toLowerCase());
    const fits = (row: unknown[], want: unknown[]) => want.every(value => row.some(cell => same(norm(cell), norm(value))));
    const got = outcome.answer.rows, unused = new Set(got.keys());
    const matched = got.length === step.answer.length && step.answer.every((want, i) => {
      const at = step.ordered ? (fits(got[i]!, want) ? i : -1) : [...unused].find(j => fits(got[j]!, want)) ?? -1;
      return at >= 0 && unused.delete(at);
    });
    if (!matched) return `answered ${JSON.stringify(got)}, expected ${JSON.stringify(step.answer)}`;
  }
  return null;
}

export async function main(context: TargetContext): Promise<number> {
  const engineIndex = context.args.indexOf('--engine');
  const engine = engineIndex >= 0 ? context.args[engineIndex + 1] : undefined;
  const rest = engineIndex < 0 ? context.args : context.args.filter((_, i) => i !== engineIndex && i !== engineIndex + 1);
  const [path, ...requests] = rest;
  if (!path) { context.io.error.write('usage: DATABASE [--engine pure|sqlite] [REQUEST...]\n'); return 2; }
  const target = resolve(context.workspace, path);
  const run = <T>(fn: () => Promise<T>) => context.runtime.run(fn);
  const database: Database = (engine ?? (/\.(sqlite|db)$/.test(path) ? 'sqlite' : 'pure')) === 'sqlite'
    ? new SqliteDatabase(target, run) : new FolderDatabase(target, run);
  const answer = async (request: string) => { context.io.output.write(`${render(await database.ask(request))}\n\n`); };
  try {
    if (requests[0] === '--scenario') {
      // A scenario: each step's request in order, judged against what it expects.
      const steps = (JSON.parse(readFileSync(resolve(context.workspace, requests[1]!), 'utf8')) as { steps: Step[] }).steps;
      let passed = 0;
      for (const step of steps) {
        const started = performance.now(), outcome = await database.ask(step.request), problem = judgeStep(step, outcome);
        passed += problem ? 0 : 1;
        context.io.output.write(`${problem ? 'FAIL' : 'ok  '} ${((performance.now() - started) / 1000).toFixed(0)} s  ${step.request}\n${problem ? `     ${problem}\n` : ''}${render(outcome).replace(/^/gm, '     ')}\n`);
      }
      context.io.output.write(`${passed}/${steps.length} steps as expected\n`);
      return passed === steps.length ? 0 : 1;
    }
    if (requests.length) for (const request of requests) await answer(request);
    else for await (const line of createInterface({ input: process.stdin })) if (line.trim()) await answer(line.trim());
  } finally { database.close(); }
  return 0;
}
