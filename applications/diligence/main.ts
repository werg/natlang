/** `natlang run applications/diligence -- [INPUT.json]` (or stdin JSON). */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { TargetContext } from '@natlang/node';
import { evaluateDossiers, type DiligenceOptions } from './diligence.js';
import type { Dossier, RubricCriterion } from './types.js';

type CliPayload = {
  dossiers: Dossier[];
  rubric: RubricCriterion[];
  options?: Pick<DiligenceOptions, 'goal' | 'escalateTo' | 'floor' | 'dealbreakerRisk' | 'concurrency' | 'carry' | 'viewInstructions'>;
};

export async function main(context: TargetContext): Promise<number> {
  const args = (context.arguments ?? {}) as { file?: string };
  const path = args.file ?? context.args[0];
  const raw = path ? readFileSync(resolve(context.workspace, path), 'utf8') : readFileSync(0, 'utf8');
  const payload = JSON.parse(raw) as CliPayload;
  const report = await context.runtime.run(() =>
    evaluateDossiers(payload.dossiers ?? [], payload.rubric ?? [], payload.options ?? {}),
  );
  context.io.output.write(`${JSON.stringify(report, null, 2)}\n`);
  return 0;
}
