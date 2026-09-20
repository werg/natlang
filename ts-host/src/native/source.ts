import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { basename, dirname, extname, join, resolve } from 'node:path';
import YAML from 'yaml';
import { Reject, buildPending, type LambdaNode } from './values.js';

type FileDefinition = { description: string; args: Record<string, string>; returns: string;
  instructions?: string; code?: string; engine?: string; types: Record<string, string>;
  effects?: string[]; codebase: Record<string, FileDefinition>; function: string };
function inline(def: FileDefinition): Record<string, unknown> {
  const kind = def.code === undefined ? 'instructions' : 'code';
  const doc: Record<string, unknown> = { description: def.description, args: def.args,
    returns: def.returns, [kind]: def[kind] };
  if (Object.keys(def.types).length) doc.types = def.types;
  if (def.effects?.length) doc.effects = def.effects;
  if (kind === 'code' && def.engine && def.engine !== 'quickjs-isolated') doc.engine = def.engine;
  if (Object.keys(def.codebase).length) doc.codebase = Object.fromEntries(Object.entries(def.codebase)
    .map(([name, child]) => [name, inline(child)]));
  return doc;
}
const frontNl = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/;
const frontTs = /^\s*\/\*---\r?\n([\s\S]*?)\r?\n---\*\/\r?\n?([\s\S]*)$/;
const id = /^[A-Za-z_][A-Za-z0-9_]*$/;

function fileFor(path: string): string {
  const absolute = resolve(path);
  for (const candidate of [absolute, `${absolute}.nl`, `${absolute}.ts`])
    if (existsSync(candidate) && statSync(candidate).isFile() && ['.nl', '.ts'].includes(extname(candidate))) return candidate;
  throw new Reject([{ path, code: 'no-such-path', expected: 'a function file (.nl or .ts)' }]);
}

/** Load frontmatter, companion functions, lexical types, and explicit uses without Python. */
export function loadFunctionFile(path: string): LambdaNode {
  const active = new Set<string>();
  function read(name: string, inherited: Record<string, string>): FileDefinition {
    const file = fileFor(name);
    if (active.has(file)) throw new Reject([{ path: file, code: 'recursion' }]);
    active.add(file);
    try {
      const ts = extname(file) === '.ts';
      const match = (ts ? frontTs : frontNl).exec(readFileSync(file, 'utf8'));
      if (!match) throw new Reject([{ path: file, code: 'type-mismatch', expected: 'frontmatter between --- lines' }]);
      const meta = YAML.parse(match[1]!) as Record<string, unknown> ?? {};
      if (!meta || typeof meta !== 'object' || Array.isArray(meta))
        throw new Reject([{ path: file, code: 'type-mismatch', expected: 'frontmatter mapping' }]);
      const allowed = new Set(['description', 'args', 'returns', 'types', 'uses', 'effects', 'engine']);
      for (const key of Object.keys(meta)) if (!allowed.has(key))
        throw new Reject([{ path: file, code: 'unknown-field', got: key }]);
      if (typeof meta.returns !== 'string') throw new Reject([{ path: file, code: 'type-mismatch', expected: 'returns' }]);
      const functionName = basename(file, extname(file));
      if (!id.test(functionName)) throw new Reject([{ path: file, code: 'type-mismatch', expected: 'function identifier' }]);
      const folderTypes = join(dirname(file), 'types.ts');
      const localTypes: Record<string, string> = {};
      if (existsSync(folderTypes)) for (const found of readFileSync(folderTypes, 'utf8').matchAll(/type\s+(\w+)\s*=\s*([^;]+);/g))
        localTypes[found[1]!] = found[2]!.trim();
      const types = { ...inherited, ...localTypes, ...meta.types as Record<string, string> ?? {} };
      const children: Record<string, FileDefinition> = {};
      const companion = join(dirname(file), functionName);
      if (existsSync(companion) && statSync(companion).isDirectory()) {
        for (const child of readdirSync(companion).sort()) {
          if (!['.nl', '.ts'].includes(extname(child)) || child === 'types.ts') continue;
          children[basename(child, extname(child))] = read(join(companion, child), types);
        }
      }
      for (const [alias, target] of Object.entries(meta.uses as Record<string, string> ?? {})) {
        if (!id.test(alias)) throw new Reject([{ path: `${file}/uses/${alias}`, code: 'type-mismatch' }]);
        children[alias] = read(join(dirname(file), String(target)), {});
      }
      if (Object.keys(children).length > 12) throw new Reject([{ path: file, code: 'codebase-too-large' }]);
      const body = match[2]!.replace(/^\n+|\n+$/g, '') + '\n';
      return { description: String(meta.description ?? ''), args: meta.args as Record<string, string> ?? {},
        returns: meta.returns, [ts ? 'code' : 'instructions']: body,
        engine: String(meta.engine ?? 'quickjs-isolated'), types,
        effects: meta.effects as string[] ?? [], codebase: children, function: functionName };
    } finally { active.delete(file); }
  }
  const definition = read(path, {});
  const params = Object.entries(definition.args).map(([key, type]) => `${key}: ${type}`).join(', ');
  const kind = definition.code === undefined ? 'instructions' : 'code';
  const node = buildPending({ $lambda: { type: `Lambda<{ ${params} }, ${definition.returns}>`,
    [kind]: definition[kind], engine: definition.engine, types: definition.types,
    effects: definition.effects, function: definition.function } });
  if (node.nodeKind !== 'lambda') throw new Error('internal file source error');
  node.codebase = Object.fromEntries(Object.entries(definition.codebase)
    .map(([name, child]) => [name, inline(child)]));
  return node;
}
