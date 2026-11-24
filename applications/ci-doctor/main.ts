/** `natlang run applications/ci-doctor -- [INPUT.json]` (or stdin JSON). */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { TargetContext } from '@natlang/node';
import { diagnoseAndFixCi, type DoctorOptions } from './doctor.js';
import type { FailureTrace } from './types.js';

type CliPayload = {
  traces: FailureTrace[];
  failingCommits?: string[];
  options?: Pick<DoctorOptions, 'flakeMode' | 'partitionFloor' | 'concurrency' | 'bisectStop' | 'clustering'>;
};

export async function main(context: TargetContext): Promise<number> {
  const args = (context.arguments ?? {}) as { file?: string };
  const path = args.file ?? context.args[0];
  const raw = path ? readFileSync(resolve(context.workspace, path), 'utf8') : readFileSync(0, 'utf8');
  const payload = JSON.parse(raw) as CliPayload;
  const failing = payload.failingCommits ? new Set(payload.failingCommits) : null;
  const report = await context.runtime.run(() =>
    diagnoseAndFixCi(payload.traces ?? [], {
      ...(payload.options ?? {}),
      ...(failing ? { failsAtCommit: commit => failing.has(commit) } : {}),
    }),
  );
  context.io.output.write(`${JSON.stringify(report, null, 2)}\n`);
  return 0;
}
