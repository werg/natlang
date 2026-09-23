/**
 * Loading named natlang functions and callable folders.
 *
 * A callable folder (`foo/` beside `foo.nl`, or `natlang.d/`) contains `.nl` functions, TypeScript
 * modules, and nested folders. Each item becomes a JSON-serializable record; the tree of records is the
 * callable context a named lambda (and its inline descendants) may use. TypeScript modules execute as
 * module instances, one per source revision, through the shared lowering.
 */
import ts from 'typescript';
import YAML from 'yaml';
import { hexDigest } from '../native/hash.js';
import { readTypeAliases } from '../native/type-aliases.js';
import { parseType, TypeEnv } from '../native/types.js';
import { RESERVED_CALLABLE_PROPERTIES } from '../compiler/intrinsics.js';
import { checkConstrainedSource } from '../compiler/policy.js';

export type SourceFiles = {
  join(...parts: string[]): string;
  dirname(path: string): string;
  basename(path: string, extension?: string): string;
  extname(path: string): string;
  isFile(path: string): boolean;
  isDirectory(path: string): boolean;
  read(path: string): string;
  list(path: string): string[];
  /** Display path relative to the loaded root. */
  relative?(path: string): string;
};

export type ExportRecord =
  | { kind: 'function'; args: Record<string, string>; returns: string; async: boolean }
  | { kind: 'value'; type?: string };

export type NatlangRecord = { kind: 'natlang'; id: string; name: string; source: string; revision: string; text: string;
  description: string; args: Record<string, string>; returns: string; instructions: string;
  types: Record<string, string>; subtype: 'function' | 'directory-reducer'; codebase: Record<string, ItemRecord> };
export type ModuleRecord = { kind: 'module'; id: string; name: string; source: string; revision: string; text: string;
  types: Record<string, string>; exports: Record<string, ExportRecord>; imports: string[];
  codebase: Record<string, ItemRecord> };
export type NamespaceRecord = { kind: 'namespace'; name: string; source: string; codebase: Record<string, ItemRecord> };
export type ItemRecord = NatlangRecord | ModuleRecord | NamespaceRecord;

export class NatlangSourceError extends Error {
  constructor(readonly path: string, message: string) { super(`${path}: ${message}`); this.name = 'NatlangSourceError'; }
}

const IDENTIFIER = /^[A-Za-z_$][A-Za-z0-9_$]*$/;
const FRONTMATTER = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/;
const NL_KEYS = new Set(['description', 'args', 'returns', 'types', 'kind']);
const revisionOf = (text: string) => hexDigest(text).slice(0, 16);

function checkName(path: string, name: string): void {
  if (!IDENTIFIER.test(name)) throw new NatlangSourceError(path, `${JSON.stringify(name)} is not a valid callable name`);
  if (RESERVED_CALLABLE_PROPERTIES.has(name))
    throw new NatlangSourceError(path, `${JSON.stringify(name)} collides with a built-in function property; rename it`);
}

function checkSignature(path: string, args: Record<string, string>, returns: string, types: Record<string, string>): void {
  const env = new TypeEnv(Object.fromEntries(Object.entries(types).map(([name, text]) => [name, parseType(text)])));
  try {
    for (const type of Object.values(args)) env.checkNames(parseType(type));
    env.checkNames(parseType(returns));
  } catch (error) { throw new NatlangSourceError(path, (error as Error).message); }
}

/** Parse a `.nl` file. */
export function parseNatlang(path: string, text: string, inherited: Record<string, string>, files: SourceFiles): NatlangRecord {
  const match = FRONTMATTER.exec(text);
  if (!match) throw new NatlangSourceError(path, 'a natural-language function needs frontmatter between --- lines');
  const meta = (YAML.parse(match[1]!) ?? {}) as Record<string, unknown>;
  if (!meta || typeof meta !== 'object' || Array.isArray(meta)) throw new NatlangSourceError(path, 'frontmatter must be a mapping');
  for (const key of Object.keys(meta)) if (!NL_KEYS.has(key))
    throw new NatlangSourceError(path, `unknown frontmatter field ${JSON.stringify(key)}; allowed: ${[...NL_KEYS].join(', ')}`);
  if (typeof meta.returns !== 'string') throw new NatlangSourceError(path, 'frontmatter needs a returns type');
  const subtype = String(meta.kind ?? 'function');
  if (subtype !== 'function' && subtype !== 'directory-reducer')
    throw new NatlangSourceError(path, 'kind must be function or directory-reducer');
  const name = files.basename(path, '.nl');
  checkName(path, name);
  const types = { ...inherited, ...(meta.types as Record<string, string> ?? {}) };
  const args = (meta.args ?? {}) as Record<string, string>;
  checkSignature(path, args, meta.returns, types);
  const source = files.relative?.(path) ?? path;
  return { kind: 'natlang', id: `nl:${source}`, name, source, revision: revisionOf(text), text,
    description: String(meta.description ?? ''), args, returns: meta.returns,
    instructions: match[2]!.replace(/^\n+|\n+$/g, '') + '\n', types, subtype, codebase: {} };
}

function functionRecord(path: string, node: ts.SignatureDeclaration, file: ts.SourceFile, label: string): ExportRecord {
  const args: Record<string, string> = {};
  for (const parameter of node.parameters) {
    if (!ts.isIdentifier(parameter.name) || !parameter.type || parameter.dotDotDotToken)
      throw new NatlangSourceError(path, `${label}: parameters need names and explicit types`);
    args[parameter.name.text + (parameter.questionToken || parameter.initializer ? '?' : '')] = parameter.type.getText(file);
  }
  if (!node.type) throw new NatlangSourceError(path, `${label}: an exported function needs an explicit return type`);
  let returns = node.type.getText(file);
  const promised = ts.isTypeReferenceNode(node.type) && ts.isIdentifier(node.type.typeName) &&
    node.type.typeName.text === 'Promise' && node.type.typeArguments?.length === 1;
  if (promised) returns = node.type.typeArguments![0]!.getText(file);
  const isAsync = promised || !!ts.getModifiers(node as ts.FunctionDeclaration)?.some(modifier => modifier.kind === ts.SyntaxKind.AsyncKeyword);
  return { kind: 'function', args, returns, async: isAsync };
}

/** Parse a TypeScript module in a callable folder: its exports, imports, and type aliases. */
export function parseModule(path: string, text: string, inherited: Record<string, string>, files: SourceFiles): ModuleRecord {
  const file = ts.createSourceFile(path, text, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
  const errors = (file as unknown as { parseDiagnostics?: readonly ts.Diagnostic[] }).parseDiagnostics ?? [];
  if (errors.length) throw new NatlangSourceError(path, errors.map(item => ts.flattenDiagnosticMessageText(item.messageText, '\n')).join('; '));
  const policy = checkConstrainedSource(file);
  if (policy.length) throw new NatlangSourceError(path, policy.map(item => `${item.line}:${item.column} ${item.message}`).join('\n'));
  const exports: Record<string, ExportRecord> = {};
  const imports: string[] = [];
  const locals = new Map<string, ts.Node>();
  for (const statement of file.statements) {
    if (ts.isFunctionDeclaration(statement) && statement.name) locals.set(statement.name.text, statement);
    if (ts.isVariableStatement(statement)) for (const declaration of statement.declarationList.declarations)
      if (ts.isIdentifier(declaration.name)) locals.set(declaration.name.text, declaration);
  }
  /** Type of a simple literal initializer, for exported values without an annotation. */
  const literalType = (expression: ts.Expression): string | undefined => {
    let node = expression;
    while (ts.isAsExpression(node) || ts.isSatisfiesExpression(node) || ts.isParenthesizedExpression(node)) node = node.expression;
    if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node) || ts.isTemplateExpression(node)) return 'string';
    if (ts.isNumericLiteral(node) || (ts.isPrefixUnaryExpression(node) && ts.isNumericLiteral(node.operand))) return 'number';
    if (node.kind === ts.SyntaxKind.TrueKeyword || node.kind === ts.SyntaxKind.FalseKeyword) return 'boolean';
    if (node.kind === ts.SyntaxKind.NullKeyword) return 'null';
    if (ts.isArrayLiteralExpression(node)) {
      const kinds = [...new Set(node.elements.map(element => literalType(element)))];
      return kinds.length === 1 && kinds[0] ? `${kinds[0].includes(' ') ? `(${kinds[0]})` : kinds[0]}[]` : undefined;
    }
    if (ts.isObjectLiteralExpression(node)) {
      const fields: string[] = [];
      for (const property of node.properties) {
        if (!ts.isPropertyAssignment(property) || !(ts.isIdentifier(property.name) || ts.isStringLiteral(property.name))) return;
        const type = literalType(property.initializer);
        if (!type) return;
        fields.push(`${property.name.text}: ${type}`);
      }
      return `{ ${fields.join(', ')} }`;
    }
    return;
  };
  const describe = (name: string, node: ts.Node): ExportRecord => {
    if (ts.isFunctionDeclaration(node)) return functionRecord(path, node, file, name);
    if (ts.isVariableDeclaration(node)) {
      const initializer = node.initializer;
      if (initializer && (ts.isArrowFunction(initializer) || ts.isFunctionExpression(initializer)))
        return functionRecord(path, initializer, file, name);
      const type = node.type ? node.type.getText(file) : initializer ? literalType(initializer) : undefined;
      return { kind: 'value', ...(type ? { type } : {}) };
    }
    return { kind: 'value' };
  };
  const isExported = (node: ts.Node) => !!ts.getModifiers(node as ts.HasModifiers)?.some(modifier => modifier.kind === ts.SyntaxKind.ExportKeyword);
  const isDefault = (node: ts.Node) => !!ts.getModifiers(node as ts.HasModifiers)?.some(modifier => modifier.kind === ts.SyntaxKind.DefaultKeyword);
  for (const statement of file.statements) {
    if (ts.isImportDeclaration(statement) && ts.isStringLiteral(statement.moduleSpecifier)) {
      if (!statement.importClause?.isTypeOnly) imports.push(statement.moduleSpecifier.text);
    } else if (ts.isFunctionDeclaration(statement) && isExported(statement)) {
      const name = isDefault(statement) ? 'default' : statement.name?.text;
      if (!name) throw new NatlangSourceError(path, 'exported functions need names');
      exports[name] = describe(name, statement);
    } else if (ts.isVariableStatement(statement) && isExported(statement)) {
      for (const declaration of statement.declarationList.declarations)
        if (ts.isIdentifier(declaration.name)) exports[declaration.name.text] = describe(declaration.name.text, declaration);
    } else if (ts.isExportAssignment(statement) && !statement.isExportEquals) {
      const target = ts.isIdentifier(statement.expression) ? locals.get(statement.expression.text) : undefined;
      exports.default = target ? describe('default', target) : { kind: 'value' };
    } else if (ts.isExportDeclaration(statement) && !statement.moduleSpecifier && statement.exportClause &&
        ts.isNamedExports(statement.exportClause) && !statement.isTypeOnly) {
      for (const element of statement.exportClause.elements) {
        if (element.isTypeOnly) continue;
        const local = locals.get((element.propertyName ?? element.name).text);
        exports[element.name.text] = local ? describe(element.name.text, local) : { kind: 'value' };
      }
    } else if (ts.isExportDeclaration(statement) && statement.moduleSpecifier)
      throw new NatlangSourceError(path, 're-exports from other modules are not supported; import and export explicitly');
  }
  for (const name of Object.keys(exports)) if (name !== 'default') checkName(path, name);
  const name = files.basename(path, '.ts');
  checkName(path, name);
  const types = { ...inherited, ...readTypeAliases(text) };
  for (const record of Object.values(exports)) if (record.kind === 'function') {
    try { checkSignature(path, record.args, record.returns, types); }
    catch { /* TypeScript types outside the portable grammar are checked by the compiler, not here. */ }
  }
  const source = files.relative?.(path) ?? path;
  return { kind: 'module', id: `ts:${source}`, name, source, revision: revisionOf(text), text, types, exports, imports, codebase: {} };
}

const isSource = (name: string) => (name.endsWith('.nl') || (name.endsWith('.ts') && !name.endsWith('.d.ts'))) &&
  name !== 'types.ts';

/** Load a callable folder into a record tree. `inherited` holds type aliases from enclosing folders. */
export function loadCallableFolder(dir: string, files: SourceFiles, inherited: Record<string, string> = {}): Record<string, ItemRecord> {
  if (!files.isDirectory(dir)) return {};
  const typesFile = files.join(dir, 'types.ts');
  const types = { ...inherited, ...(files.isFile(typesFile) ? readTypeAliases(files.read(typesFile)) : {}) };
  const entries = files.list(dir).filter(name => !name.startsWith('.')).sort();
  const items: Record<string, ItemRecord> = {};
  const add = (name: string, record: ItemRecord, path: string) => {
    if (Object.hasOwn(items, name)) throw new NatlangSourceError(path, `two items in ${dir} are named ${JSON.stringify(name)}`);
    items[name] = record;
  };
  for (const entry of entries) {
    const path = files.join(dir, entry);
    if (!files.isFile(path) || !isSource(entry)) continue;
    const text = files.read(path);
    const record = entry.endsWith('.nl') ? parseNatlang(path, text, types, files) : parseModule(path, text, types, files);
    add(record.name, record, path);
  }
  for (const entry of entries) {
    const path = files.join(dir, entry);
    if (!files.isDirectory(path) || entry === 'node_modules') continue;
    const children = loadCallableFolder(path, files, items[entry] && 'types' in items[entry]! ? { ...types, ...(items[entry] as NatlangRecord).types } : types);
    const owner = items[entry];
    if (owner && owner.kind !== 'namespace') {
      if (owner.kind === 'module') for (const child of Object.keys(children)) if (Object.hasOwn(owner.exports, child))
        throw new NatlangSourceError(path, `${JSON.stringify(child)} is both an export of ${entry}.ts and an item in ${entry}/`);
      owner.codebase = children;
    } else if (Object.keys(children).length)
      add(entry, { kind: 'namespace', name: entry, source: files.relative?.(path) ?? path, codebase: children }, path);
  }
  return items;
}

/** Load a named `.nl` function with its companion folder as its callable context. */
export function loadNamedFunction(path: string, files: SourceFiles): NatlangRecord {
  if (!files.isFile(path) || files.extname(path) !== '.nl') throw new NatlangSourceError(path, 'expected a .nl file');
  const dir = files.dirname(path);
  const typesFile = files.join(dir, 'types.ts');
  const types = files.isFile(typesFile) ? readTypeAliases(files.read(typesFile)) : {};
  const record = parseNatlang(path, files.read(path), types, files);
  record.codebase = loadCallableFolder(files.join(dir, record.name), files, record.types);
  return record;
}

/** Find the nearest ancestor directory (including `start`) that contains `natlang.d/`. */
export function findApplicationContext(start: string, files: SourceFiles): string | undefined {
  let dir = start;
  while (true) {
    const candidate = files.join(dir, 'natlang.d');
    if (files.isDirectory(candidate)) return candidate;
    const parent = files.dirname(dir);
    if (parent === dir) return;
    dir = parent;
  }
}

/** Path operations with no file access, for reparsing an edited source by its path. */
export const PATH_ONLY: SourceFiles = {
  join: (...parts) => parts.filter(Boolean).join('/').replace(/\/+/g, '/'),
  dirname: path => path.includes('/') ? path.slice(0, path.lastIndexOf('/')) || '/' : '.',
  basename: (path, extension) => { const base = path.slice(path.lastIndexOf('/') + 1);
    return extension && base.endsWith(extension) ? base.slice(0, -extension.length) : base; },
  extname: path => { const base = path.slice(path.lastIndexOf('/') + 1); const at = base.lastIndexOf('.'); return at > 0 ? base.slice(at) : ''; },
  isFile: () => false, isDirectory: () => false,
  read: path => { throw new Error(`no file access for ${path}`); }, list: () => [],
  relative: path => path,
};
