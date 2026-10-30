/**
 * The dashboard's embeddable assets, as a vendoring target (scripts/vendor-build.mjs `dashboard`, staged by
 * scripts/stage-vibestudio.mjs): the page modules (`pages/`), the shared modules they import, `style.css` (scoped under
 * `.nl-dash` in the `natlang` cascade layer) and `types.d.ts` (DashboardApi, RunHost, the page mount contract). The
 * standalone shell (`app.js`, `index.html`) stays out: a host supplies its own api, run host and navigation.
 *
 * Same shape as the bundle targets' build: it returns the staged paths (relative to `outdir`) and the sources they were
 * copied from (relative to ts-host), so the pin hashes every file and the staging refuses sources that differ from HEAD.
 */
import { cpSync, mkdirSync, readdirSync, statSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const assets = join(root, 'dashboard');
/** Files of ts-host/dashboard that belong to the standalone dashboard only. */
export const STANDALONE = new Set(['app.js', 'index.html']);

export function buildDashboardAssets(outdir) {
  const files = [];
  (function walk(directory) {
    for (const name of readdirSync(directory)) {
      const path = join(directory, name);
      if (statSync(path).isDirectory()) { walk(path); continue; }
      const file = relative(assets, path).split('\\').join('/');
      if (STANDALONE.has(file)) continue;
      mkdirSync(dirname(join(outdir, file)), { recursive: true });
      cpSync(path, join(outdir, file));
      files.push(file);
    }
  })(assets);
  files.sort();
  return { files, sources: files.map(file => `dashboard/${file}`) };
}
