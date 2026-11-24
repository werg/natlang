/**
 * `natlang run applications/sem-sheet -- [INPUT.json]` (or `--cells JSON` / `--merge JSON`, or stdin JSON).
 * Node CLI entrypoint; core `sheet.ts` remains 100% browser-compatible with zero `node:*` imports.
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { TargetContext } from '@natlang/node';
import { evaluateSheet, mergeSheetDiffs } from './sheet.js';
import type { CellCoord, FormulaCell, SheetDiff } from './types.js';

type CliPayload = {
  cells?: Record<CellCoord, FormulaCell>;
  merge?: { base: SheetDiff; ours: SheetDiff; theirs: SheetDiff };
  concurrency?: number;
};

function readJsonInput(context: TargetContext, filePath?: string): CliPayload {
  const path = filePath ?? context.args[0];
  const raw = path
    ? readFileSync(resolve(context.workspace, path), 'utf8')
    : readFileSync(0, 'utf8');
  return JSON.parse(raw) as CliPayload;
}

export async function main(context: TargetContext): Promise<number> {
  const args = (context.arguments ?? {}) as {
    file?: string;
    cells?: Record<CellCoord, FormulaCell>;
    merge?: { base: SheetDiff; ours: SheetDiff; theirs: SheetDiff };
  };

  const payload: CliPayload = args.cells || args.merge
    ? { ...(args.cells ? { cells: args.cells } : {}), ...(args.merge ? { merge: args.merge } : {}) }
    : readJsonInput(context, args.file);

  const output = await context.runtime.run(async () => {
    if (payload.merge && !payload.cells) {
      return mergeSheetDiffs(payload.merge.base, payload.merge.ours, payload.merge.theirs);
    }
    const sheet = payload.cells
      ? await evaluateSheet(payload.cells, payload.concurrency ? { concurrency: payload.concurrency } : {})
      : undefined;
    const merged = payload.merge
      ? await mergeSheetDiffs(payload.merge.base, payload.merge.ours, payload.merge.theirs)
      : undefined;
    return merged && sheet ? { sheet, merged } : (sheet ?? merged);
  });

  context.io.output.write(`${JSON.stringify(output, null, 2)}\n`);
  return 0;
}
