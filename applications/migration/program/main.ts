/**
 * `natlang run applications/migration --target program -- propose UNIT.json OUT_DIR` writes OUT_DIR/proposal.json and,
 * for a candidate, the migrated program in OUT_DIR/program. `... -- settle SETTLE.json` prints accept, retry or
 * give-up. UNIT.json is a host.ts `Unit`; SETTLE.json is `{ comparison, attemptsLeft, history }`. The program under
 * migration is never written.
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import type { TargetContext } from '@natlang/node';
import { propose, settle, writeCandidate, type Unit } from './host.js';

export async function main(context: TargetContext): Promise<number> {
  const [command, input, out] = context.args;
  const read = (path: string) => JSON.parse(readFileSync(resolve(context.workspace, path), 'utf8'));
  if (command === 'propose' && input && out) {
    const unit = read(input) as Unit;
    const proposal = await context.runtime.run(() => propose(unit));
    const directory = resolve(context.workspace, out);
    mkdirSync(directory, { recursive: true });
    if (proposal.status === 'candidate' || proposal.status === 'unchecked') writeCandidate(proposal, join(directory, 'program'));
    writeFileSync(join(directory, 'proposal.json'), `${JSON.stringify(proposal, null, 1)}\n`);
    context.io.output.write(`${JSON.stringify({ program: proposal.program, status: proposal.status, keep: proposal.keep, problems: proposal.problems })}\n`);
    return 0;
  }
  if (command === 'settle' && input) {
    const { comparison, attemptsLeft, history } = read(input);
    const next = await context.runtime.run(() => settle(comparison, attemptsLeft, history ?? []));
    context.io.output.write(`${JSON.stringify({ next })}\n`);
    return 0;
  }
  context.io.error.write('usage: propose UNIT.json OUT_DIR | settle SETTLE.json\n');
  return 2;
}
