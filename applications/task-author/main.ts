/** `natlang run applications/task-author -- [SEED.json]` (or stdin JSON). */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { TargetContext } from '@natlang/node';
import { authorAndQualifyTask } from './author.js';
import type { TaskSeed } from './types.js';

type CliPayload = {
  seed?: TaskSeed;
  reservedGroups?: string[];
  attempts?: number;
  escalateTo?: string;
} & Partial<TaskSeed>;

export async function main(context: TargetContext): Promise<number> {
  const args = (context.arguments ?? {}) as { file?: string };
  const path = args.file ?? context.args[0];
  const raw = path ? readFileSync(resolve(context.workspace, path), 'utf8') : readFileSync(0, 'utf8');
  const payload = JSON.parse(raw) as CliPayload;
  const seed: TaskSeed = payload.seed ?? {
    id: payload.id ?? 'sfp-cli-001',
    domain: payload.domain ?? 'general',
    construct: payload.construct ?? 'readout-decision',
    oracleKind: payload.oracleKind ?? 'exact',
    summary: payload.summary ?? '',
    splitGroup: payload.splitGroup ?? 'sfp:cli:001',
  };
  const reserved = new Set(payload.reservedGroups ?? []);
  const result = await context.runtime.run(() =>
    authorAndQualifyTask(seed, {
      isReserved: group => reserved.has(group),
      replay: (task, answer) => answer.trim() === task.reference.trim(),
      ...(payload.attempts !== undefined ? { attempts: payload.attempts } : {}),
      ...(payload.escalateTo !== undefined ? { escalateTo: payload.escalateTo } : {}),
    }),
  );
  context.io.output.write(`${JSON.stringify(result, null, 2)}\n`);
  return result.admitted ? 0 : 1;
}
