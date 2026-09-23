/** `natlang run examples/triage -- TICKETS.json [RUBRIC]`: print a triage report for a JSON list of tickets. */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { TargetContext } from '@natlang/node';
import { triage } from './triage.js';

const DEFAULT_RUBRIC = 'billing: charges, invoices and payments; technical: outages, errors and bugs; spam: advertising';

export async function main(context: TargetContext): Promise<number> {
  const [path, rubric = DEFAULT_RUBRIC] = context.args;
  if (!path) { context.io.error.write('usage: natlang run examples/triage -- TICKETS.json [RUBRIC]\n'); return 2; }
  const tickets = JSON.parse(readFileSync(resolve(context.workspace, path), 'utf8')) as string[];
  const report = await context.runtime.run(() => triage(tickets, rubric));
  context.io.output.write(JSON.stringify(report, null, 2) + '\n');
  return 0;
}
