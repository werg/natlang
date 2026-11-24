/** `natlang run applications/nldb --target grep -- [INPUT.json]` (or stdin JSON). */
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import type { TargetContext } from '@natlang/node';
import { askRepository } from './ask.js';
import { createRepoFile, parseCsv, runQuery } from './query.js';
import type { QueryPlan, TableRow } from './types.js';

type CliPayload = {
  files: Array<{ path: string; content: string }>;
  tables?: Record<string, TableRow[] | string>;
  plan?: QueryPlan;
  question?: string;
  dbPath?: string;
  concurrency?: number;
};

export async function main(context: TargetContext): Promise<number> {
  const args = (context.arguments ?? {}) as { file?: string };
  const path = args.file ?? context.args[0];
  const raw = path ? readFileSync(resolve(context.workspace, path), 'utf8') : readFileSync(0, 'utf8');
  const payload = JSON.parse(raw) as CliPayload;

  const files = (payload.files ?? []).map(f => createRepoFile(f.path, f.content));
  const tables: Record<string, TableRow[]> = {};
  for (const [name, data] of Object.entries(payload.tables ?? {})) {
    tables[name] = typeof data === 'string' ? parseCsv(data) : data;
  }

  if (payload.question) {
    const tempDir = payload.dbPath ? null : mkdtempSync(join(tmpdir(), 'natlang-nldb-grep-cli-'));
    const dbPath = payload.dbPath ? resolve(context.workspace, payload.dbPath) : join(tempDir!, 'repo.sqlite');
    try {
      const outcome = await askRepository(payload.question, { files, tables }, {
        path: dbPath,
        run: fn => context.runtime.run(fn),
      });
      context.io.output.write(`${JSON.stringify(outcome, null, 2)}\n`);
      return 'error' in outcome ? 1 : 0;
    } finally {
      if (tempDir) rmSync(tempDir, { recursive: true, force: true });
    }
  }

  const result = await context.runtime.run(() =>
    runQuery({ files, tables }, payload.plan ?? {}, {
      ...(payload.concurrency !== undefined ? { concurrency: payload.concurrency } : {}),
    }),
  );
  context.io.output.write(`${JSON.stringify(result, null, 2)}\n`);
  return 0;
}
