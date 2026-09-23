import { readTypeAliases } from './type-aliases.js';
import YAML from 'yaml';
import ts from 'typescript';
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
  instructions?: string; code?: string; async?: boolean; engine?: string; types: Record<string, string>;
  effects?: string[]; codebase: Record<string, FileDefinition>; function: string;
  subtype: 'function' | 'directory-reducer' };
function inline(def: FileDefinition): Record<string, unknown> {
  const kind = def.code === undefined ? 'instructions' : 'code';
  const doc: Record<string, unknown> = { description: def.description, args: def.args,
    returns: def.returns, [kind]: def[kind] };
  if (Object.keys(def.types).length) doc.types = def.types;
  if (def.effects?.length) doc.effects = def.effects;
  if (def.subtype !== 'function') doc.subtype = def.subtype;
  if (kind === 'code' && def.engine && def.engine !== 'typescript-host') doc.engine = def.engine;
  if (kind === 'code') doc.async = def.async === true;
  if (Object.keys(def.codebase).length) doc.codebase = Object.fromEntries(Object.entries(def.codebase)
    .map(([name, child]) => [name, inline(child)]));
  return doc;
}
const frontNl = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/;
const id = /^[A-Za-z_][A-Za-z0-9_]*$/;
const staticImport = /^import\s+([A-Za-z_$][\w$]*)\s+from\s*["']([^"']+)["'];?\s*$/;
function splitImports(source: string): { imports: [string, string, string][]; source: string } {
  const lines = source.match(/[^\n]*\n|[^\n]+$/g) ?? [], imports: [string, string, string][] = [];
  let at = 0;
  while (at < lines.length) {
    const match = staticImport.exec(lines[at]!.replace(/\r?\n$/, ''));
    if (!match) break;
    imports.push([match[1]!, '', match[2]!]); at++;
  }
  if (at && at < lines.length && !lines[at]!.trim()) at++;
  return { imports, source: lines.slice(at).join('') };
}

export function parseCrispModule(source: string, file: string, options: { packageImports?: boolean;
  resolveImport?: (specifier: string) => string } = {}): { imports: [string, string, string][];
  args: Record<string, string>; returns: string; code: string; effects: string[];
  types: Record<string, string>; async: boolean } {
  const parsed = ts.createSourceFile(file, source, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
  const errors = (parsed as unknown as { parseDiagnostics?: readonly ts.DiagnosticWithLocation[] }).parseDiagnostics ?? [];
  if (errors.length) throw new Reject([{ path: file, code: 'typescript-syntax',
    got: errors.map(item => ts.flattenDiagnosticMessageText(item.messageText, '\n')).join('; ') }]);
  const imports: [string, string, string][] = [];
  const runtimeImports: string[] = [];
  for (const statement of parsed.statements) if (ts.isImportDeclaration(statement)) {
    const specifier = ts.isStringLiteral(statement.moduleSpecifier) ? statement.moduleSpecifier.text : '';
    if (statement.importClause?.isTypeOnly) continue;
    if (specifier === 'natlang:runtime') continue;
    if (options.packageImports && (!specifier.startsWith('.') || statement.importClause?.namedBindings || /\.[cm]?js$/.test(specifier))) {
      const rewritten = options.resolveImport?.(specifier) ?? specifier;
      runtimeImports.push(ts.createPrinter().printNode(ts.EmitHint.Unspecified,
        ts.factory.updateImportDeclaration(statement, statement.modifiers, statement.importClause,
          ts.factory.createStringLiteral(rewritten), statement.attributes), parsed));
      continue;
    }
    if (!specifier.startsWith('.') || !statement.importClause?.name || statement.importClause.namedBindings)
      throw new Reject([{ path: file, code: 'bad-import', expected: 'a default import from a relative .ts or .nl module' }]);
    imports.push([statement.importClause.name.text, '', specifier]);
  }
  const functions = parsed.statements.filter(ts.isFunctionDeclaration).filter(statement =>
    statement.modifiers?.some(modifier => modifier.kind === ts.SyntaxKind.DefaultKeyword));
  if (functions.length !== 1)
    throw new Reject([{ path: file, code: 'type-mismatch', expected: 'exactly one default exported function' }]);
  const fn = functions[0]!;
  const expectedName = file.slice(file.lastIndexOf('/') + 1).replace(/\.ts$/, '');
  if (!fn.name || fn.name.text !== expectedName)
    throw new Reject([{ path: file, code: 'type-mismatch', expected: `default function ${expectedName}` }]);
  if (!fn.body || !fn.type)
    throw new Reject([{ path: file, code: 'type-mismatch', expected: 'an implementation and explicit return type' }]);
  const args: Record<string, string> = {};
  for (const parameter of fn.parameters) {
    if (!ts.isIdentifier(parameter.name) || !parameter.type || parameter.dotDotDotToken || parameter.initializer)
      throw new Reject([{ path: file, code: 'type-mismatch', expected: 'identifier parameters with explicit TypeScript types' }]);
    args[parameter.name.text + (parameter.questionToken ? '?' : '')] = parameter.type.getText(parsed);
  }
  let returns = fn.type.getText(parsed);
  if (ts.isTypeReferenceNode(fn.type) && ts.isIdentifier(fn.type.typeName) && fn.type.typeName.text === 'Promise' &&
      fn.type.typeArguments?.length === 1) returns = fn.type.typeArguments[0]!.getText(parsed);
  const code = source.slice(fn.body.getStart(parsed) + 1, fn.body.getEnd() - 1).replace(/^\s*\n|\s+$/g, '') + '\n';
  const effects = [...code.matchAll(/\bfx\.([A-Za-z_$][\w$]*)\.([A-Za-z_$][\w$]*)/g)]
    .map(match => `${match[1]}.${match[2]}`);
  return { imports, args, returns,
    code: runtimeImports.length ? runtimeImports.join('\n') + '\n' + code : code,
    effects: [...new Set(effects)], types: readTypeAliases(source),
    async: !!fn.modifiers?.some(modifier => modifier.kind === ts.SyntaxKind.AsyncKeyword) };
}

function fileFor(path: string, files: SourceFiles): string {
  const absolute = files.resolve(path);
  for (const candidate of [absolute, `${absolute}.nl`, `${absolute}.ts`])
    if (files.isFile(candidate) && ['.nl', '.ts'].includes(files.extname(candidate))) return candidate;
  throw new Reject([{ path, code: 'no-such-path', expected: 'a function file (.nl or .ts)' }]);
}

/** Load frontmatter, companion functions, lexical types, and explicit uses without Python. */
export function loadFunctionSource(path: string, files: SourceFiles, options: { packageImports?: boolean } = {}): LambdaNode {
  const active = new Set<string>();
  function read(name: string, inherited: Record<string, string>): FileDefinition {
    const file = fileFor(name, files);
    if (active.has(file)) throw new Reject([{ path: file, code: 'recursion' }]);
    active.add(file);
    try {
      const isTs = files.extname(file) === '.ts';
      const raw = files.read(file);
      const module = isTs ? parseCrispModule(raw, file, { ...options,
        resolveImport: specifier => specifier.startsWith('.') ? files.resolve(files.join(files.dirname(file), specifier)) : specifier }) : undefined;
      const parsed = isTs ? { imports: module!.imports, source: raw } : splitImports(raw);
      const match = isTs ? undefined : frontNl.exec(parsed.source);
      if (!isTs && !match) throw new Reject([{ path: file, code: 'type-mismatch', expected: 'frontmatter between --- lines' }]);
      const meta = isTs ? { args: module!.args, returns: module!.returns, effects: module!.effects,
        types: module!.types,
        engine: 'typescript-host' } : YAML.parse(match![1]!) as Record<string, unknown> ?? {};
      if (!meta || typeof meta !== 'object' || Array.isArray(meta))
        throw new Reject([{ path: file, code: 'type-mismatch', expected: 'frontmatter mapping' }]);
      const allowed = new Set(['description', 'args', 'returns', 'types', 'effects', 'engine', 'kind']);
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
      for (const [binding, exported, specifier] of parsed.imports) {
        if (!specifier.startsWith('.')) throw new Reject([{ path: file, code: 'no-such-path',
          expected: 'a relative natlang module import', got: specifier }]);
        const child = read(files.join(files.dirname(file), specifier), types);
        if (exported && child.function !== exported) throw new Reject([{ path: file, code: 'bad-import',
          expected: `export {${child.function}} from ${specifier}`, got: exported }]);
        if (Object.hasOwn(children, binding)) throw new Reject([{ path: file, code: 'duplicate-path', got: binding }]);
        children[binding] = child;
      }
      const body = isTs ? module!.code : match![2]!.replace(/^\n+|\n+$/g, '') + '\n';
      return { description: String(meta.description ?? ''), args: meta.args as Record<string, string> ?? {},
        returns: meta.returns, [isTs ? 'code' : 'instructions']: body,
        ...(isTs ? { async: module!.async } : {}),
        engine: String(meta.engine ?? 'typescript-host'), types,
        effects: meta.effects as string[] ?? [], codebase: children, function: functionName,
        subtype: subtype as FileDefinition['subtype'] };
    } finally { active.delete(file); }
  }
  const definition = read(path, {});
  const params = Object.entries(definition.args).map(([key, type]) => `${key}: ${type}`).join(', ');
  const kind = definition.code === undefined ? 'instructions' : 'code';
  const node = buildPending({ $lambda: { type: `(${params}) => ${definition.returns}`,
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
  if (node.kind === 'code' && node.engine !== 'typescript-host') definition.engine = node.engine;
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
    // Ordinary host TypeScript may coexist at an application or repository root.
    // A natlang crisp module is identified by its default exported function.
    if (extension === '.ts') {
      const parsed = ts.createSourceFile(file, files.read(file), ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
      if (!parsed.statements.some(statement => ts.isFunctionDeclaration(statement) &&
          statement.modifiers?.some(modifier => modifier.kind === ts.SyntaxKind.DefaultKeyword))) continue;
    }
    const functionName = files.basename(name, extension);
    if (Object.hasOwn(entries, functionName))
      throw new Reject([{ path: file, code: 'duplicate-path', got: functionName }]);
    const definition = sourceDefinition(loadFunctionSource(file, files));
    if (extension === '.ts') definition.async = parseCrispModule(files.read(file), file).async;
    entries[functionName] = definition;
  }
  return entries;
}

/** Construct a zero-argument, string-returning instruction over a directory codebase. */
export function loadAnonymousInstructionSource(path: string, instructions: string,
  files: SourceFiles): Record<string, unknown> {
  const body = instructions.trim();
  if (!body) throw new TypeError('anonymous instructions cannot be empty');
  return { $lambda: { type: '() => string', instructions: body,
    function: 'anonymous', codebase: loadCodebaseSource(path, files) } };
}
