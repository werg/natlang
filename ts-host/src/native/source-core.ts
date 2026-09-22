import { readTypeAliases } from './type-aliases.js';
import YAML from 'yaml';
import { Reject, buildPending, type LambdaNode } from './values.js';
import { formatType } from './types.js';

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
  effects?: string[]; codebase: Record<string, FileDefinition>; function: string;
  subtype: 'function' | 'directory-reducer' };
function inline(def: FileDefinition): Record<string, unknown> {
  const kind = def.code === undefined ? 'instructions' : 'code';
  const doc: Record<string, unknown> = { description: def.description, args: def.args,
    returns: def.returns, [kind]: def[kind] };
  if (Object.keys(def.types).length) doc.types = def.types;
  if (def.effects?.length) doc.effects = def.effects;
  if (def.subtype !== 'function') doc.subtype = def.subtype;
  if (kind === 'code' && def.engine && def.engine !== 'quickjs-isolated') doc.engine = def.engine;
  if (Object.keys(def.codebase).length) doc.codebase = Object.fromEntries(Object.entries(def.codebase)
    .map(([name, child]) => [name, inline(child)]));
  return doc;
}
const frontNl = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/;
const frontTs = /^\s*\/\*---\r?\n([\s\S]*?)\r?\n---\*\/\r?\n?([\s\S]*)$/;
const id = /^[A-Za-z_][A-Za-z0-9_]*$/;
const staticImport = /^import\s*\{\s*([A-Za-z_$][\w$]*)(?:\s+as\s+([A-Za-z_$][\w$]*))?\s*\}\s*from\s*["']([^"']+)["'];?\s*$/;
function splitImports(source: string): { imports: [string, string, string][]; source: string } {
  const lines = source.match(/[^\n]*\n|[^\n]+$/g) ?? [], imports: [string, string, string][] = [];
  let at = 0;
  while (at < lines.length) {
    const match = staticImport.exec(lines[at]!.replace(/\r?\n$/, ''));
    if (!match) break;
    imports.push([match[2] ?? match[1]!, match[1]!, match[3]!]); at++;
  }
  if (at && at < lines.length && !lines[at]!.trim()) at++;
  return { imports, source: lines.slice(at).join('') };
}

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
      const parsed = splitImports(files.read(file));
      const match = (ts ? frontTs : frontNl).exec(parsed.source);
      if (!match) throw new Reject([{ path: file, code: 'type-mismatch', expected: 'frontmatter between --- lines' }]);
      const meta = YAML.parse(match[1]!) as Record<string, unknown> ?? {};
      if (!meta || typeof meta !== 'object' || Array.isArray(meta))
        throw new Reject([{ path: file, code: 'type-mismatch', expected: 'frontmatter mapping' }]);
      const allowed = new Set(['description', 'args', 'returns', 'types', 'uses', 'effects', 'engine', 'kind']);
      for (const key of Object.keys(meta)) if (!allowed.has(key))
        throw new Reject([{ path: file, code: 'unknown-field', got: key }]);
      if (typeof meta.returns !== 'string') throw new Reject([{ path: file, code: 'type-mismatch', expected: 'returns' }]);
      const subtype = String(meta.kind ?? 'function');
      if (!['function', 'directory-reducer'].includes(subtype))
        throw new Reject([{ path: `${file}/kind`, code: 'type-mismatch', expected: 'function or directory-reducer', got: subtype }]);
      const functionName = files.basename(file, files.extname(file));
      if (!id.test(functionName)) throw new Reject([{ path: file, code: 'type-mismatch', expected: 'function identifier' }]);
      const folderTypes = files.join(files.dirname(file), 'types.ts');
      const localTypes: Record<string, string> = {};
      if (files.isFile(folderTypes)) Object.assign(localTypes, readTypeAliases(files.read(folderTypes)));
      const types = { ...inherited, ...localTypes, ...meta.types as Record<string, string> ?? {} };
      const children: Record<string, FileDefinition> = {};
      const companion = files.join(files.dirname(file), functionName);
      if (!parsed.imports.length && files.isDirectory(companion)) {
        for (const child of files.list(companion).sort()) {
          if (!['.nl', '.ts'].includes(files.extname(child)) || child === 'types.ts') continue;
          children[files.basename(child, files.extname(child))] = read(files.join(companion, child), types);
        }
      }
      for (const [alias, target] of Object.entries(meta.uses as Record<string, string> ?? {})) {
        if (!id.test(alias)) throw new Reject([{ path: `${file}/uses/${alias}`, code: 'type-mismatch' }]);
        if (!parsed.imports.some(([binding]) => binding === alias))
          children[alias] = read(files.join(files.dirname(file), String(target)), {});
      }
      for (const [binding, exported, specifier] of parsed.imports) {
        if (!specifier.startsWith('.')) throw new Reject([{ path: file, code: 'no-such-path',
          expected: 'a relative natlang module import', got: specifier }]);
        const child = read(files.join(files.dirname(file), specifier), types);
        if (child.function !== exported) throw new Reject([{ path: file, code: 'bad-import',
          expected: `export {${child.function}} from ${specifier}`, got: exported }]);
        if (Object.hasOwn(children, binding)) throw new Reject([{ path: file, code: 'duplicate-path', got: binding }]);
        children[binding] = child;
      }
      const body = match[2]!.replace(/^\n+|\n+$/g, '') + '\n';
      return { description: String(meta.description ?? ''), args: meta.args as Record<string, string> ?? {},
        returns: meta.returns, [ts ? 'code' : 'instructions']: body,
        engine: String(meta.engine ?? 'quickjs-isolated'), types,
        effects: meta.effects as string[] ?? [], codebase: children, function: functionName,
        subtype: subtype as FileDefinition['subtype'] };
    } finally { active.delete(file); }
  }
  const definition = read(path, {});
  const params = Object.entries(definition.args).map(([key, type]) => `${key}: ${type}`).join(', ');
  const kind = definition.code === undefined ? 'instructions' : 'code';
  const node = buildPending({ $lambda: { type: `Lambda<{ ${params} }, ${definition.returns}>`,
    [kind]: definition[kind], engine: definition.engine, types: definition.types,
    effects: definition.effects, function: definition.function,
    ...(definition.subtype !== 'function' ? { subtype: definition.subtype } : {}) } });
  if (node.nodeKind !== 'lambda') throw new Error('internal file source error');
  node.codebase = Object.fromEntries(Object.entries(definition.codebase)
    .map(([name, child]) => [name, inline(child)]));
  return node;
}

function sourceDefinition(node: LambdaNode): Record<string, unknown> {
  if (node.type.kind !== 'lambda') throw new Error('function source does not have a lambda type');
  const definition: Record<string, unknown> = {
    args: Object.fromEntries(node.type.params.fields.map(field =>
      [field.name + (field.optional ? '?' : ''), formatType(field.type)])),
    returns: formatType(node.type.returns),
    [node.kind]: node.body,
  };
  if (Object.keys(node.typesSrc).length) definition.types = node.typesSrc;
  if (node.effects.length) definition.effects = node.effects;
  if (node.subtype !== 'function') definition.kind = node.subtype;
  if (node.kind === 'code' && node.engine !== 'quickjs-isolated') definition.engine = node.engine;
  if (Object.keys(node.codebase).length) definition.codebase = node.codebase;
  return definition;
}

/** Load every top-level function in a directory as an anonymous program's codebase. */
export function loadCodebaseSource(path: string, files: SourceFiles): Record<string, unknown> {
  const root = files.resolve(path);
  if (!files.isDirectory(root))
    throw new Reject([{ path: root, code: 'no-such-path', expected: 'a codebase directory' }]);
  const entries: Record<string, unknown> = {};
  for (const name of files.list(root).sort()) {
    const extension = files.extname(name);
    if (!['.nl', '.ts'].includes(extension) || name === 'types.ts') continue;
    const file = files.join(root, name);
    // A natlang TypeScript function is explicitly marked by frontmatter. Ordinary
    // host TypeScript may coexist at an application or repository root.
    if (extension === '.ts' && !frontTs.test(splitImports(files.read(file)).source)) continue;
    const functionName = files.basename(name, extension);
    if (Object.hasOwn(entries, functionName))
      throw new Reject([{ path: file, code: 'duplicate-path', got: functionName }]);
    entries[functionName] = sourceDefinition(loadFunctionSource(file, files));
  }
  return entries;
}

/** Construct a zero-argument, Text-returning instruction over a directory codebase. */
export function loadAnonymousInstructionSource(path: string, instructions: string,
  files: SourceFiles): Record<string, unknown> {
  const body = instructions.trim();
  if (!body) throw new TypeError('anonymous instructions cannot be empty');
  return { $lambda: { type: 'Lambda<{}, Text>', instructions: body,
    function: 'anonymous', codebase: loadCodebaseSource(path, files) } };
}
