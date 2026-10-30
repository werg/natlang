/** `natlang run examples/triage [-- FILE.json [RUBRIC [TICKETS]]]`: print a triage report for a list of tickets: TICKETS
 * as a JSON list, else the list in FILE, else the bundled tickets.json. natlang.json declares the arguments' types. */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { TargetContext } from '@natlang/node';
import { triage } from './triage.js';

export async function main(context: TargetContext): Promise<number> {
  const { file, rubric, tickets: given } = context.arguments as { file?: string; rubric: string; tickets?: string[] };
  const tickets = given ?? JSON.parse(readFileSync(file ?? join(context.package?.root ?? context.workspace, 'tickets.json'), 'utf8')) as string[];
  const report = await context.runtime.run(() => triage(tickets, rubric));
  context.io.output.write(JSON.stringify(report, null, 2) + '\n');
  return 0;
}
