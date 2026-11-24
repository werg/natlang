/** `natlang run applications/migration --target codemod -- [INPUT.json]` (or stdin JSON). */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { TargetContext } from '@natlang/node';
import { createSourceFile, migrateRepository } from './studio.js';
import type { CodemodSpec, CrispRule } from './types.js';

type CliPayload = {
  spec: CodemodSpec;
  files: Array<{ path: string; content: string }>;
  crispRules?: CrispRule[];
  forbiddenPatterns?: string[];
  requiredPatterns?: string[];
  attempts?: number;
  minPairsToMine?: number;
  mineRules?: boolean;
};

export async function main(context: TargetContext): Promise<number> {
  const args = (context.arguments ?? {}) as { file?: string };
  const path = args.file ?? context.args[0];
  const raw = path ? readFileSync(resolve(context.workspace, path), 'utf8') : readFileSync(0, 'utf8');
  const payload = JSON.parse(raw) as CliPayload;

  const files = (payload.files ?? []).map(f => createSourceFile(f.path, f.content));
  const forbidden = payload.forbiddenPatterns ?? [];
  const required = payload.requiredPatterns ?? [];
  const verify = (_filePath: string, content: string): string[] => {
    const problems: string[] = [];
    for (const pattern of forbidden) {
      if (content.includes(pattern)) problems.push(`Legacy pattern still present: ${pattern}`);
    }
    for (const pattern of required) {
      if (!content.includes(pattern)) problems.push(`Required pattern missing: ${pattern}`);
    }
    return problems;
  };

  const report = await context.runtime.run(() =>
    migrateRepository(files, payload.spec, {
      runtime: context.runtime,
      crispRules: payload.crispRules ?? [],
      verify,
      ...(payload.attempts !== undefined ? { attempts: payload.attempts } : {}),
      ...(payload.minPairsToMine !== undefined ? { minPairsToMine: payload.minPairsToMine } : {}),
      ...(payload.mineRules !== undefined ? { mineRules: payload.mineRules } : {}),
    }),
  );
  context.io.output.write(`${JSON.stringify(report, null, 2)}\n`);
  return report.failedFiles === 0 ? 0 : 1;
}
