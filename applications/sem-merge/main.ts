/** `natlang run applications/sem-merge -- [INPUT.json]` (or stdin JSON). */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { TargetContext } from '@natlang/node';
import { mergeFile, mergeRepository, type MergeOptions, type ThreeWayFileInput } from './merge.js';

type CliPayload =
  | { file: ThreeWayFileInput; options?: Omit<MergeOptions, 'verify'> }
  | { files: ThreeWayFileInput[]; options?: Omit<MergeOptions, 'verify'> & { concurrency?: number } }
  | ThreeWayFileInput;

export async function main(context: TargetContext): Promise<number> {
  const args = (context.arguments ?? {}) as { file?: string };
  const path = args.file ?? context.args[0];
  const raw = path ? readFileSync(resolve(context.workspace, path), 'utf8') : readFileSync(0, 'utf8');
  const payload = JSON.parse(raw) as CliPayload;

  const outcome = await context.runtime.run(async () => {
    if ('files' in payload && Array.isArray(payload.files)) {
      return mergeRepository(payload.files, payload.options ?? {});
    }
    if ('file' in payload && payload.file) {
      return mergeFile(payload.file, payload.options ?? {});
    }
    return mergeFile(payload as ThreeWayFileInput);
  });

  context.io.output.write(`${JSON.stringify(outcome, null, 2)}\n`);
  return 0;
}
