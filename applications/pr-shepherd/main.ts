/** `natlang run applications/pr-shepherd -- [INPUT.json]` (or `--hunks JSON --rules JSON`, or stdin JSON). */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { TargetContext } from '@natlang/node';
import { shepherdPullRequest, type ShepherdOptions } from './shepherd.js';
import type { Hunk, Rule } from './types.js';

type CliPayload = {
  hunks: Hunk[];
  rules: Rule[];
  options?: Pick<ShepherdOptions, 'trivialMode' | 'escalateTo' | 'floor' | 'blockRiskFloor' | 'concurrency'>;
};

function readJsonInput(context: TargetContext, filePath?: string): CliPayload {
  const path = filePath ?? context.args[0];
  const raw = path
    ? readFileSync(resolve(context.workspace, path), 'utf8')
    : readFileSync(0, 'utf8');
  return JSON.parse(raw) as CliPayload;
}

export async function main(context: TargetContext): Promise<number> {
  const args = (context.arguments ?? {}) as { file?: string; hunks?: Hunk[]; rules?: Rule[] };
  let hunks: Hunk[];
  let rules: Rule[];
  let options: CliPayload['options'] = {};

  if (args.hunks && args.rules) {
    hunks = args.hunks;
    rules = args.rules;
  } else {
    const payload = readJsonInput(context, args.file);
    hunks = payload.hunks;
    rules = payload.rules;
    options = payload.options ?? {};
  }

  const report = await context.runtime.run(() => shepherdPullRequest(hunks, rules, options));
  context.io.output.write(`${JSON.stringify(report, null, 2)}\n`);
  return 0;
}
