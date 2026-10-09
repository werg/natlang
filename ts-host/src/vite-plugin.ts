/**
 * A Vite (and Vitest) plugin that loads `.nl` imports as natlang functions, as `natlang run` does: the module's
 * default export is the callable `loadNatlang` returns. Applications whose tests import code that imports `.nl`
 * files add it to their config: `plugins: [natlangVitePlugin()]`.
 */
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const runtimeEntry = fileURLToPath(new URL('./index.js', import.meta.url));

/** The nearest directory above `file` with a package.json: the natlang package the function belongs to. */
function packageRoot(file: string): string {
  for (let directory = dirname(file); ; directory = dirname(directory)) {
    if (existsSync(join(directory, 'package.json'))) return directory;
    if (dirname(directory) === directory) return dirname(file);
  }
}

/**
 * `root`: the package root for every `.nl` file (default: the nearest package.json above each file). `live` (default
 * true): each function follows its sources while the process runs (loadNatlang's live option), so a long test run
 * uses edited instructions from the next call on; false loads each function once.
 */
export function natlangVitePlugin(options: { root?: string; live?: boolean } = {}) {
  return {
    name: 'natlang-functions',
    enforce: 'pre' as const,
    load(id: string): string | null {
      const file = id.split('?')[0]!;
      if (!file.endsWith('.nl')) return null;
      const root = options.root ?? packageRoot(file);
      return `import { loadNatlang } from ${JSON.stringify(runtimeEntry)};\n` +
        `export default loadNatlang(${JSON.stringify(file)}, ${JSON.stringify(root.endsWith('/') ? root : `${root}/`)}, ` +
        `{ live: ${options.live ?? true} });\n`;
    },
  };
}
