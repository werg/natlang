import { readTypeAliases } from './type-aliases.js';
import YAML from 'yaml';
import { Reject, buildPending, type LambdaNode } from './values.js';

export type SourceFiles = {
  resolve(path: string): string;
  join(...parts: string[]): string;
  dirname(path: string): string;
  basename(path: string, extension?: string): string;
  extname(path: string): string;
  exists(path: string): boolean;
  isFile(path: string): boolean;
  isDirectory(path: string): boolean;
  read(path: string): string;
  list(path: string): string[];
};

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

function fileFor(path: string, files: SourceFiles): string {
  const absolute = files.resolve(path);
  for (const candidate of [absolute, `${absolute}.nl`, `${absolute}.ts`])
    if (files.isFile(candidate) && ['.nl', '.ts'].includes(files.extname(candidate))) return candidate;
  throw new Reject([{ path, code: 'no-such-path', expected: 'a function file (.nl or .ts)' }]);
}

/** Load frontmatter, companion functions, lexical types, and explicit uses without Python. */
export function loadFunctionSource(path: string, files: SourceFiles): LambdaNode {
  const active = new Set<string>();
  function read(name: string, inherited: Record<string, string>): FileDefinition {
    const file = fileFor(name, files);
    if (active.has(file)) throw new Reject([{ path: file, code: 'recursion' }]);
    active.add(file);
    try {
      const ts = files.extname(file) === '.ts';
      const match = (ts ? frontTs : frontNl).exec(files.read(file));
      if (!match) throw new Reject([{ path: file, code: 'type-mismatch', expected: 'frontmatter between --- lines' }]);
      const meta = YAML.parse(match[1]!) as Record<string, unknown> ?? {};
      if (!meta || typeof meta !== 'object' || Array.isArray(meta))
        throw new Reject([{ path: file, code: 'type-mismatch', expected: 'frontmatter mapping' }]);
      const allowed = new Set(['description', 'args', 'returns', 'types', 'uses', 'effects', 'engine']);
      for (const key of Object.keys(meta)) if (!allowed.has(key))
        throw new Reject([{ path: file, code: 'unknown-field', got: key }]);
      if (typeof meta.returns !== 'string') throw new Reject([{ path: file, code: 'type-mismatch', expected: 'returns' }]);
      const functionName = files.basename(file, files.extname(file));
      if (!id.test(functionName)) throw new Reject([{ path: file, code: 'type-mismatch', expected: 'function identifier' }]);
      const folderTypes = files.join(files.dirname(file), 'types.ts');
      const localTypes: Record<string, string> = {};
      if (files.isFile(folderTypes)) Object.assign(localTypes, readTypeAliases(files.read(folderTypes)));
      const types = { ...inherited, ...localTypes, ...meta.types as Record<string, string> ?? {} };
      const children: Record<string, FileDefinition> = {};
      const companion = files.join(files.dirname(file), functionName);
      if (files.isDirectory(companion)) {
        for (const child of files.list(companion).sort()) {
          if (!['.nl', '.ts'].includes(files.extname(child)) || child === 'types.ts') continue;
          children[files.basename(child, files.extname(child))] = read(files.join(companion, child), types);
        }
      }
      for (const [alias, target] of Object.entries(meta.uses as Record<string, string> ?? {})) {
        if (!id.test(alias)) throw new Reject([{ path: `${file}/uses/${alias}`, code: 'type-mismatch' }]);
        children[alias] = read(files.join(files.dirname(file), String(target)), {});
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
