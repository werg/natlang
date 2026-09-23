#!/usr/bin/env node
import { readFileSync, writeFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { basename, dirname, extname, relative, resolve, sep } from 'node:path';
import YAML from 'yaml';

const roots = process.argv.slice(2).map(path => resolve(path));
if (!roots.length) throw new Error('usage: migrate-crisp-modules.mjs <root>...');
const front = /^((?:import[^\n]*\n)*)\s*\/\*---\r?\n([\s\S]*?)\r?\n---\*\/\r?\n?([\s\S]*)$/;
const namedImport = /^import\s*\{\s*([A-Za-z_$][\w$]*)(?:\s+as\s+([A-Za-z_$][\w$]*))?\s*\}\s*from\s*(["'][^"']+["']);?\s*$/;
const naturalSource = /^((?:import[^\n]*\n)*)\s*---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/;

function files(path) {
  if (statSync(path).isFile()) return [path];
  return readdirSync(path).flatMap(name => {
    const child = resolve(path, name);
    return statSync(child).isDirectory() ? files(child) : [child];
  });
}
function nearestTypes(file) {
  let at = dirname(file);
  while (roots.some(root => at === root || at.startsWith(root + sep))) {
    const candidate = resolve(at, 'types.ts');
    if (candidate !== file && existsSync(candidate)) return candidate;
    const parent = dirname(at); if (parent === at) break; at = parent;
  }
}
function typeNames(file) {
  if (!file) return [];
  return [...readFileSync(file, 'utf8').matchAll(/\bexport\s+type\s+([A-Za-z_$][\w$]*)\s*=/g)].map(match => match[1]);
}
function specifier(from, to) {
  let path = relative(dirname(from), to).replaceAll(sep, '/').replace(/\.ts$/, '.js');
  if (!path.startsWith('.')) path = './' + path;
  return path;
}
function importsOf(block) {
  return block.trim().split(/\r?\n/).filter(Boolean).map(line => {
    const match = namedImport.exec(line.trim());
    if (!match) throw new Error(`only one named import was expected during migration: ${line}`);
    return `import ${match[2] ?? match[1]} from ${match[3]};`;
  });
}
function migrateTs(file) {
  const source = readFileSync(file, 'utf8');
  const match = front.exec(source);
  if (!match) {
    if (/\bexport\s+default\s+(?:async\s+)?function\b/.test(source)) {
      const typesFile = nearestTypes(file), names = typeNames(typesFile);
      if (typesFile && names.length) {
        const statement = `import type { ${names.join(', ')} } from ${JSON.stringify(specifier(file, typesFile))};`;
        if (!source.includes(`from ${JSON.stringify(specifier(file, typesFile))}`)) writeFileSync(file, statement + '\n' + source);
      }
      return;
    }
    if (basename(file) === 'types.ts') return;
    throw new Error(`${file}: expected the old crisp frontmatter format`);
  }
  const meta = YAML.parse(match[2]) ?? {};
  const declared = Object.entries(meta.args ?? {});
  const body = match[3].replace(/^\n+|\s+$/g, '');
  const chosen = new Map(declared.map(([raw]) => {
    const name = raw.replace(/\?$/, '');
    const collision = new RegExp(`\\b(?:const|let|var|function|class)\\s+${name}\\b`).test(body);
    return [name, collision ? `_${name}` : name];
  }));
  let migrated = body;
  for (const [name, parameter] of chosen) migrated = migrated.replace(new RegExp(`\\bargs\\.${name}\\b`, 'g'), parameter);
  const imports = importsOf(match[1]);
  const typesFile = nearestTypes(file), names = typeNames(typesFile);
  if (typesFile && names.length) imports.push(`import type { ${names.join(', ')} } from ${JSON.stringify(specifier(file, typesFile))};`);
  if (/\bhost\./.test(migrated)) imports.push('import { host } from "natlang:runtime";');
  if (/\bfx\./.test(migrated)) imports.push('import { effects as fx } from "natlang:runtime";');
  const parameters = declared.map(([raw, type]) => {
    const name = raw.replace(/\?$/, ''), optional = raw.endsWith('?') ? '?' : '';
    return `${chosen.get(name)}${optional}: ${type}`;
  }).join(', ');
  const async = /\bawait\b/.test(migrated), result = String(meta.returns);
  const returnType = async ? `Promise<${result}>` : result;
  const functionName = basename(file, extname(file));
  writeFileSync(file, `${imports.length ? imports.join('\n') + '\n\n' : ''}` +
    `export default ${async ? 'async ' : ''}function ${functionName}(${parameters}): ${returnType} {\n` +
    `${migrated}\n}\n`);
}
function migrateNl(file) {
  const source = readFileSync(file, 'utf8'), lines = source.match(/[^\n]*\n|[^\n]+$/g) ?? [];
  let changed = false;
  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i].replace(/\r?\n$/, ''), match = namedImport.exec(raw.trim());
    if (!match) { if (raw.trim()) break; continue; }
    lines[i] = `import ${match[2] ?? match[1]} from ${match[3]};\n`; changed = true;
  }
  if (changed) writeFileSync(file, lines.join(''));
}

function extractNaturalTypes(file) {
  const source = readFileSync(file, 'utf8'), match = naturalSource.exec(source);
  if (!match) return;
  const meta = YAML.parse(match[2]) ?? {}, aliases = meta.types ?? {};
  if (!Object.keys(aliases).length) return;
  const target = resolve(dirname(file), 'types.ts');
  let existing = existsSync(target) ? readFileSync(target, 'utf8').replace(/\s*$/, '') + '\n' : '';
  const declared = new Set([...existing.matchAll(/\b(?:export\s+)?type\s+([A-Za-z_$][\w$]*)\s*=/g)].map(item => item[1]));
  for (const [name, type] of Object.entries(aliases)) if (!declared.has(name)) existing += `export type ${name} = ${type};\n`;
  writeFileSync(target, existing);
  delete meta.types;
  writeFileSync(file, `${match[1]}---\n${YAML.stringify(meta).trimEnd()}\n---\n${match[3].replace(/^\n+/, '')}`);
}

const all = roots.flatMap(root => files(root));
for (const file of all) if (file.endsWith('.nl')) extractNaturalTypes(file);
for (const file of all) {
  if (file.endsWith('.ts') && basename(file) !== 'types.ts') migrateTs(file);
  else if (file.endsWith('.nl')) migrateNl(file);
}
