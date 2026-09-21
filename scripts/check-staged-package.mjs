import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, extname, join, resolve } from 'node:path';

const root = resolve(process.argv[2] ?? '');
if (!root || !existsSync(root)) throw new Error('usage: check-staged-package.mjs DIRECTORY');
function files(path) {
  return readdirSync(path).flatMap(name => {
    const child = join(path, name); return statSync(child).isDirectory() ? files(child) : [child];
  });
}
const missing = [];
for (const file of files(root).filter(path => ['.js', '.ts'].includes(extname(path)))) {
  const text = readFileSync(file, 'utf8');
  for (const match of text.matchAll(/from\s+['"](\.[^'"]+)['"]/g)) {
    const target = resolve(dirname(file), match[1]);
    const declaration = target.endsWith('.js') ? target.slice(0, -3) + '.d.ts' : '';
    if (!existsSync(target) && (!declaration || !existsSync(declaration))) missing.push(`${file}: ${match[1]}`);
  }
}
if (missing.length) throw new Error(`broken relative package imports:\n${missing.join('\n')}`);
process.stdout.write(`checked ${root}\n`);
