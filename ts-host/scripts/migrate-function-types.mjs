#!/usr/bin/env node
import { readFileSync, writeFileSync, readdirSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseType, formatType } from '../dist/native/types.js';

function files(path) {
  if (statSync(path).isFile()) return [path];
  return readdirSync(path).flatMap(name => files(resolve(path, name)));
}
function migrate(source, file) {
  let at = 0;
  while ((at = source.indexOf('Lambda<', at)) >= 0) {
    let depth = 0, end = at + 'Lambda'.length;
    for (; end < source.length; end++) {
      if (source[end] === '<') depth++;
      else if (source[end] === '>' && --depth === 0) { end++; break; }
    }
    if (depth !== 0) throw new Error(`${file}: unterminated Lambda type`);
    const old = source.slice(at, end);
    let replacement;
    try { replacement = formatType(parseType(old)); }
    catch { at = end; continue; }
    source = source.slice(0, at) + replacement + source.slice(end);
    at += replacement.length;
  }
  return source;
}

for (const root of process.argv.slice(2)) for (const file of files(resolve(root))) {
  if (!/\.(?:ts|mjs|md|json|jsonl)$/.test(file)) continue;
  const source = readFileSync(file, 'utf8'), changed = migrate(source, file);
  if (changed !== source) writeFileSync(file, changed);
}
