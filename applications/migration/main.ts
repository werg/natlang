/**
 * `natlang run applications/migration -- MANIFEST.json [--attempts N] [--seed TEXT]... [--exact P] [--settled P]`:
 * migrate a repository. MANIFEST.json is `{ "root": ".", "request": "...", "files": ["a.mjs"], "checks": [{ "id",
 * "argv", "timeoutMs"? }] }` (root is relative to the workspace). The original checkout is never written; the
 * candidate and its report are printed. Each policy point (exact, settled) runs its `crisp` or `natural-language`
 * implementation. Exit status 0 when the candidate's checks pass.
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { TargetContext } from '@natlang/node';
import { RepositoryMigration, migrate, type CheckCommand, type Implementation, type PolicyPoint } from './index.js';

const POINTS: PolicyPoint[] = ['exact', 'settled'];

export async function main(context: TargetContext): Promise<number> {
  const args = context.args;
  const options = (name: string) => args.flatMap((arg, i) => arg === name && args[i + 1] !== undefined ? [args[i + 1]!] : []);
  const path = args.find((arg, i) => !arg.startsWith('--') && !args[i - 1]?.startsWith('--'));
  if (!path) { context.io.error.write('usage: MANIFEST.json [--attempts N] [--seed TEXT]... [--exact|--settled crisp|natural-language]\n'); return 2; }
  const manifest = JSON.parse(readFileSync(resolve(context.workspace, path), 'utf8')) as
    { root?: string, request: string, files: string[], checks?: CheckCommand[] };
  const policy: Partial<Record<PolicyPoint, Implementation>> = {};
  for (const point of POINTS) {
    const choice = options(`--${point}`)[0];
    if (choice === undefined) continue;
    if (!['crisp', 'nl', 'shadow', 'natural-language'].includes(choice)) { context.io.error.write(`--${point} takes crisp, nl or shadow (natural-language is the deprecated spelling of nl)\n`); return 2; }
    policy[point] = choice as Implementation;
  }
  const repository = new RepositoryMigration(resolve(context.workspace, manifest.root ?? '.'),
    { files: manifest.files, checks: manifest.checks ?? [], policy });
  await repository.open();
  const result = await migrate(context.runtime, repository, manifest.request,
    { attempts: Number(options('--attempts')[0] ?? 3), seeds: options('--seed'), checks: (manifest.checks ?? []).map(check => check.id) });
  for (const event of repository.drainEvents()) context.io.error.write(`${JSON.stringify(event)}\n`);
  context.io.output.write(`${result.report.status}: ${result.summary}\n`);
  for (const line of result.next) context.io.output.write(`  - ${line}\n`);
  context.io.output.write(`${JSON.stringify(result.report, null, 2)}\n`);
  return result.report.status === 'reviewable' ? 0 : 1;
}
