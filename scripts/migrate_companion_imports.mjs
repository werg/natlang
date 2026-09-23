/** One-time mechanical migration from application-file imports to companion functions. */
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, cpSync, writeFileSync } from 'node:fs';
import { basename, dirname, extname, join, resolve } from 'node:path';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';

const roots = process.argv.slice(2).length ? process.argv.slice(2) : [
  'codebases', 'ts-host/examples/browser-board/program', 'ts-host/studio/programs', 'ts-host/studio/research/programs',
  'skills/natlang-authoring/assets/review',
];
const importLine = /^import\s+(type\s+)?([A-Za-z_$][\w$]*)\s+from\s+["'](\.[^"']+)["'];?\s*$/;
const typeImportLine = /^import\s+type\s+.+\s+from\s+["']\.[^"']+["'];?\s*$/;
const legacyHostImport = /^import\s+\{\s*(?:host|effects\s+as\s+fx)\s*\}\s+from\s+["']natlang:runtime["'];?\s*$/;
const allFiles = root => readdirSync(root, { withFileTypes: true }).flatMap(entry => {
  const path = join(root, entry.name);
  return entry.isDirectory() ? allFiles(path) : ['.nl', '.ts'].includes(extname(path)) ? [path] : [];
});
let copied = 0, cleaned = 0;
const digest = path => createHash('sha256').update(readFileSync(path)).digest('hex');
let tracked = [];
try { tracked = execFileSync('git', ['ls-files', '-z', '--', ...roots],
  { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).split('\0').filter(Boolean); }
catch { /* A migration root may be outside the repository. */ }
const origins = new Map();
const byDigest = new Map();
for (const path of tracked) if (existsSync(path) && ['.nl', '.ts'].includes(extname(path))) {
  const absolute = resolve(path), key = digest(absolute);
  const candidates = byDigest.get(key) ?? [];
  candidates.push(absolute); byDigest.set(key, candidates);
  origins.set(absolute, absolute);
}
for (const root of roots.map(path => resolve(path))) {
  if (!existsSync(root)) continue;
  // Copies may themselves reference siblings; keep expanding until every call has a companion.
  for (let pass = 0; pass < 100; pass++) {
    let added = 0;
    for (const file of allFiles(root)) {
      if (basename(file) === 'types.ts') continue;
      const companion = join(dirname(file), basename(file, extname(file)));
      for (const line of readFileSync(file, 'utf8').split(/\r?\n/)) {
        const match = importLine.exec(line);
        if (!match || match[1]) continue;
        const [, , alias, specifier] = match;
        const origin = origins.get(resolve(file)) ?? byDigest.get(digest(file))?.[0] ?? file;
        const sourceBase = resolve(dirname(origin), specifier);
        const source = [sourceBase, `${sourceBase}.nl`, `${sourceBase}.ts`, sourceBase.replace(/\.js$/, '.ts')]
          .find(candidate => existsSync(candidate) && statSync(candidate).isFile());
        if (!source) throw new Error(`Cannot resolve ${specifier} from ${file}`);
        if (extname(source) === '.ts' && basename(source, extname(source)) !== alias)
          throw new Error(`Import alias ${alias} differs from function ${source}`);
        const destination = join(companion, `${alias}${extname(source)}`);
        if (resolve(source) === resolve(destination)) continue;
        if (existsSync(destination)) {
          if (readFileSync(destination, 'utf8') !== readFileSync(source, 'utf8'))
            throw new Error(`Conflicting companion function ${destination}`);
          continue;
        }
        mkdirSync(companion, { recursive: true });
        cpSync(source, destination, { errorOnExist: true, force: false });
        origins.set(resolve(destination), origins.get(resolve(source)) ?? source);
        const sourceChildren = join(dirname(source), basename(source, extname(source)));
        const destinationChildren = join(companion, alias);
        if (existsSync(sourceChildren)) {
          if (existsSync(destinationChildren)) throw new Error(`Conflicting companion folder ${destinationChildren}`);
          cpSync(sourceChildren, destinationChildren, { recursive: true, errorOnExist: true, force: false });
        }
        copied++; added++;
      }
    }
    if (!added) break;
    if (pass === 99) throw new Error(`Companion expansion did not converge in ${root}; check for recursion`);
  }
  for (const file of allFiles(root)) {
    const before = readFileSync(file, 'utf8');
    const lines = before.split(/\r?\n/);
    const firstBody = lines.findIndex(line => !importLine.test(line) && line.trim() !== '');
    if (firstBody < 0) continue;
    const after = lines.filter(line => !importLine.test(line) && !typeImportLine.test(line) && !legacyHostImport.test(line))
      .join('\n').replace(/^\n+/, '');
    if (after !== before) { writeFileSync(file, after); cleaned++; }
  }
}
process.stdout.write(`Copied ${copied} companion functions; removed local imports from ${cleaned} files.\n`);
