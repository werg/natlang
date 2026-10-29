/** `natlang run examples/triage [-- TICKETS.json [RUBRIC]]`: print a triage report for a JSON list of tickets (default: tickets.json here). */
import { readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import type { TargetContext } from '@natlang/node';
import { triage } from './triage.js';

const DEFAULT_RUBRIC = 'billing: charges, invoices and payments; technical: outages, errors and bugs; spam: advertising';

export async function main(context: TargetContext): Promise<number> {
  const [path, rubric = DEFAULT_RUBRIC] = context.args;
  const file = path ? resolve(context.workspace, path) : join(context.package?.root ?? context.workspace, 'tickets.json');
  const tickets = JSON.parse(readFileSync(file, 'utf8')) as string[];
  const report = await context.runtime.run(() => triage(tickets, rubric));
  context.io.output.write(JSON.stringify(report, null, 2) + '\n');
  return 0;
}
