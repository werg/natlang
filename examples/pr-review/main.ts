/** `natlang run examples/pr-review`: review diff.json against rules.json. */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { untrusted, type TargetContext } from '@natlang/node';
import { reviewPullRequest, type Rule } from './review.js';
import type { Hunk } from './types.js';

export async function main(context: TargetContext): Promise<number> {
  const root = context.package?.root ?? context.workspace;
  const rawHunks = JSON.parse(readFileSync(join(root, 'diff.json'), 'utf8')) as Hunk[];
  const hunks = rawHunks.map(hunk => ({ ...hunk, patch: untrusted(hunk.patch, `PR diff ${hunk.file}`) }));
  const rules = JSON.parse(readFileSync(join(root, 'rules.json'), 'utf8')) as Rule[];
  const report = await context.runtime.run(() => reviewPullRequest(hunks, rules));
  context.io.output.write(JSON.stringify(report, null, 2) + '\n');
  return 0;
}
