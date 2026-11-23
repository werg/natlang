/** `natlang run applications/idiom-reviewer -- DIR [--escalate-to MODEL]`: review the `.nl` functions under DIR and
 * print the report as JSON. Exit 1 when it has findings. */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import type { TargetContext } from '@natlang/node';
import { reviewProgramIdioms } from './review.js';
import type { ProgramFile } from './types.js';

const SKIPPED = new Set(['node_modules', 'dist', '.natlang', '.git']);

/** Every `.nl` source under `root`, with whether its callable folder holds TypeScript helpers. */
export function programFiles(root: string): ProgramFile[] {
  const out: ProgramFile[] = [];
  const walk = (dir: string): void => {
    for (const name of readdirSync(dir).sort()) {
      const path = join(dir, name);
      if (statSync(path).isDirectory()) { if (!SKIPPED.has(name)) walk(path); continue; }
      if (!name.endsWith('.nl')) continue;
      const folder = path.slice(0, -'.nl'.length);
      const hasHelpers = existsSync(folder) && statSync(folder).isDirectory() &&
        readdirSync(folder).some(file => file.endsWith('.ts') && !file.endsWith('.d.ts') && !file.endsWith('.d.nl.ts'));
      out.push({ path: relative(root, path), content: readFileSync(path, 'utf8'), hasHelpers });
    }
  };
  walk(root);
  return out;
}

export async function main(context: TargetContext): Promise<number> {
  const args = [...context.args];
  const at = args.indexOf('--escalate-to');
  const escalateTo = at >= 0 ? args.splice(at, 2)[1] : undefined;
  if (args.length !== 1) {
    context.io.error.write('usage: natlang run applications/idiom-reviewer -- DIR [--escalate-to MODEL]\n');
    return 2;
  }
  const root = resolve(context.workspace, args[0]!);
  const report = await context.runtime.run(() => reviewProgramIdioms(programFiles(root), escalateTo ? { escalateTo } : {}));
  context.io.output.write(JSON.stringify(report, null, 2) + '\n');
  return report.clean ? 0 : 1;
}
