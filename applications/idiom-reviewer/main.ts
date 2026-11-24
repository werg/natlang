/** `natlang run applications/idiom-reviewer -- DIR [--escalate-to MODEL]`: review the `.nl` functions under DIR and
 * print the report as JSON. Exit 1 when it has findings. */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import type { TargetContext } from '@natlang/node';
import { reviewProgramIdioms } from './review.js';
import type { ProgramFile } from './types.js';

const SKIPPED = new Set(['node_modules', 'dist', '.natlang', '.git']);

function readAncestorTypes(root: string, fileDir: string): string | undefined {
  const resolvedRoot = resolve(root);
  const chunks: string[] = [];
  let current = resolve(fileDir);
  for (let guard = 0; guard < 16; guard++) {
    const typesPath = join(current, 'types.ts');
    if (existsSync(typesPath) && statSync(typesPath).isFile()) {
      chunks.push(readFileSync(typesPath, 'utf8'));
    }
    if (current === resolvedRoot) break;
    const parent = dirname(current);
    if (parent === current) break;
    current = parent;
  }
  return chunks.length > 0 ? chunks.join('\n') : undefined;
}

/** Every `.nl` source under `root`, with whether its callable folder holds TypeScript helpers and sibling `types.ts` source. */
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
      const typesContent = readAncestorTypes(root, dir);
      out.push({
        path: relative(root, path),
        content: readFileSync(path, 'utf8'),
        hasHelpers,
        ...(typesContent ? { typesContent } : {}),
      });
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
  const report = await context.runtime.run(() =>
    reviewProgramIdioms(programFiles(root), { rootDir: root, ...(escalateTo ? { escalateTo } : {}) }),
  );
  context.io.output.write(JSON.stringify(report, null, 2) + '\n');
  return report.clean ? 0 : 1;
}
