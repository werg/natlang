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
import { finiteValues } from '../native/decision.js';
import { RESERVED_CALLABLE_PROPERTIES, invalidNameMessage, reservedNameMessage } from '../compiler/intrinsics.js';
import { checkConstrainedSource } from '../compiler/policy.js';
import { loadNzSync, registerImportedBlocks } from '../native/nz-file.js';

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
  /** Binary contents, for `.nz` files. Without it, `.nz` files in callable folders are not loaded. */
  readBytes?(path: string): Uint8Array;
};

export type ExportRecord =
  | { kind: 'function'; args: Record<string, string>; returns: string; async: boolean; doc?: string }
  | { kind: 'value'; type?: string };

export type NatlangRecord = { programId?: string; kind: 'natlang'; id: string; name: string; source: string; revision: string; text: string;
  description: string; args: Record<string, string>; returns: string; instructions: string;
  types: Record<string, string>; subtype: 'function' | 'directory-reducer'; codebase: Record<string, ItemRecord>;
  /** `readout: decision` in the frontmatter: answer by scoring the finite result values (native/decision.ts). */
  readout?: 'decision' | 'template';
  /** `model: NAME` in the frontmatter: the call runs on the runtime's model of that name (`models`), else the default. */
  model?: string;
  /**
   * `uses: [dir/item]` in the frontmatter: package items this function may call besides its companion folder, by path
   * from the package root. Each is in `codebase` under its base name.
   */
  uses?: string[];
  /**
   * Data entries of the companion folder bound by default (S0 §7): each `.nz` file's exports under the file's name,
   * as loaded (Neuralese references, data, soft-function specs). Its blocks are registered as imported blocks.
   */
  contextData?: Record<string, unknown> };
export type ModuleRecord = { programId?: string; kind: 'module'; id: string; name: string; source: string; revision: string; text: string;
  types: Record<string, string>; exports: Record<string, ExportRecord>; imports: string[];
  /** TypeScript declarations of the module's classes (and method-bearing interfaces), for function listings. */
  declarations?: Record<string, string>;
  codebase: Record<string, ItemRecord> };
export type NamespaceRecord = { kind: 'namespace'; name: string; source: string; codebase: Record<string, ItemRecord> };
export type ItemRecord = NatlangRecord | ModuleRecord | NamespaceRecord;

export class NatlangSourceError extends Error {
  constructor(readonly path: string, message: string) { super(`${path}: ${message}`); this.name = 'NatlangSourceError'; }
}

/**
 * Executable nodes built from source files (spec "Contexts": executable nodes come from files). The loader registers
 * every record it parses from a file; compiled modules register the records embedded at build time. Records parsed from
 * text at run time (`defineNatlang`) are not registered, so a context refuses them as new executable nodes.
 */
const FILE_NODES = new Set<string>();
export const nodeKey = (record: NatlangRecord | ModuleRecord): string => `${record.kind}:${record.id}@${record.revision}`;
export function registerFileRecords(codebase: Record<string, ItemRecord> | ItemRecord): void {
  const visit = (item: ItemRecord) => {
    if (item.kind !== 'namespace') FILE_NODES.add(nodeKey(item));
    for (const child of Object.values(item.codebase)) visit(child);
  };
  const single = typeof (codebase as ItemRecord).kind === 'string' && 'codebase' in codebase && typeof (codebase as ItemRecord).codebase === 'object';
  if (single) visit(codebase as ItemRecord);
  else for (const item of Object.values(codebase as Record<string, ItemRecord>)) visit(item);
}
/** Whether an executable node and all its children come from files. */
export function isFileRecord(record: ItemRecord): boolean {
  if (record.kind !== 'namespace' && !FILE_NODES.has(nodeKey(record))) return false;
  return Object.values(record.codebase).every(isFileRecord);
}

const IDENTIFIER = /^[A-Za-z_$][A-Za-z0-9_$]*$/;
const FRONTMATTER = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/;
const NL_KEYS = new Set(['description', 'args', 'returns', 'types', 'kind', 'readout', 'model', 'uses']);
const revisionOf = (text: string) => hexDigest(text).slice(0, 16);
const TYPE_KEYS = /^(args|types|returns):(.*)$/;

/** A type written as a YAML-quoted string is its contents, so `"string[]"` and `string[]` are the same type. */
function unquoteType(text: string): string {
  const trimmed = text.trim();
  if (/^(".*"|'.*')$/s.test(trimmed)) {
    try { const value = YAML.parse(trimmed); if (typeof value === 'string') return value.trim(); } catch { /* not one scalar */ }
  }
  return trimmed;
}

/** `{ a: T, b?: U }` read as a TypeScript object type: member names to type text. */
function typeLiteralMap(key: string, text: string): Record<string, string> {
  const file = ts.createSourceFile('frontmatter.ts', `type __ = ${text};`, ts.ScriptTarget.Latest, true);
  const alias = file.statements[0];
  const diagnostics = (file as unknown as { parseDiagnostics?: ts.Diagnostic[] }).parseDiagnostics ?? [];
  if (diagnostics.length || file.statements.length !== 1 || !alias || !ts.isTypeAliasDeclaration(alias) || !ts.isTypeLiteralNode(alias.type))
    throw new Error(`${key} must map names to types, as \`${key}: { name: string }\` or one \`name: type\` per indented line`);
  const map: Record<string, string> = {};
  for (const member of alias.type.members) {
    if (!ts.isPropertySignature(member) || !member.type || !(ts.isIdentifier(member.name) || ts.isStringLiteral(member.name)))
      throw new Error(`${key}: each entry needs a name and a type`);
    const type = ts.isLiteralTypeNode(member.type) && ts.isStringLiteral(member.type.literal) ? member.type.literal.text : member.type.getText(file);
    map[member.name.text + (member.questionToken ? '?' : '')] = type;
  }
  return map;
}

/**
 * `.nl` frontmatter. `args`, `types` and `returns` hold TypeScript type text, read verbatim rather than as YAML, so types
 * need no quoting: `names: string[]`, `rows: { title: string }[]`, `f: (x: string) => number`. A value wrapped entirely
 * in quotes is the quoted text, as in YAML. The other keys are YAML.
 */
export function readNatlangFrontmatter(source: string): Record<string, unknown> {
  const lines = source.split(/\r?\n/), yaml: string[] = [], typed: Record<string, unknown> = {};
  for (let index = 0; index < lines.length; index++) {
    const match = TYPE_KEYS.exec(lines[index]!);
    if (!match) { yaml.push(lines[index]!); continue; }
    const key = match[1]!, inline = match[2]!.trim(), block: string[] = [];
    while (index + 1 < lines.length && /^(\s|$)/.test(lines[index + 1]!)) block.push(lines[++index]!);
    const body = block.filter(line => line.trim() && !line.trim().startsWith('#'));
    if (key in typed) throw new Error(`frontmatter repeats ${key}`);
    if (key === 'returns') {
      const text = unquoteType([inline, ...body.map(line => line.trim())].join(' '));
      if (!text) throw new Error('returns needs a type');
      typed.returns = text;
    } else if (inline || body[0]?.trim().startsWith('{')) {
      if (/^["']/.test(inline)) throw new Error(`${key} must map names to types, not be one string`);
      typed[key] = typeLiteralMap(key, [inline, ...body.map(line => line.trim())].join(' ').trim());
    } else {
      const map: Record<string, string> = {};
      let entry: string | undefined;
      const indent = body[0]?.match(/^\s*/)![0].length ?? 0;
      for (const line of body) {
        const own = line.match(/^\s*/)![0].length;
        const named = own === indent ? /^\s*([A-Za-z_$][\w$]*\??)\s*:(.*)$/.exec(line) : null;
        if (named) { entry = named[1]!; if (entry in map) throw new Error(`${key} repeats ${entry}`); map[entry] = named[2]!.trim(); }
        else if (entry && own > indent) map[entry] += ' ' + line.trim();
        else throw new Error(`${key}: expected \`name: type\` on each indented line, got ${JSON.stringify(line.trim())}`);
      }
      for (const name of Object.keys(map)) {
        map[name] = unquoteType(map[name]!);
        if (!map[name]) throw new Error(`${key}: ${name} needs a type`);
      }
      typed[key] = map;
    }
  }
  const meta = YAML.parse(yaml.join('\n')) ?? {};
  if (!meta || typeof meta !== 'object' || Array.isArray(meta)) throw new Error('frontmatter must be a mapping');
  return { ...meta, ...typed };
}

function checkName(path: string, name: string): void {
  if (!IDENTIFIER.test(name)) throw new NatlangSourceError(path, invalidNameMessage(name));
  if (RESERVED_CALLABLE_PROPERTIES.has(name)) throw new NatlangSourceError(path, reservedNameMessage(name));
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
  let meta: Record<string, unknown>;
  try { meta = readNatlangFrontmatter(match[1]!); } catch (error) { throw new NatlangSourceError(path, (error as Error).message); }
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
  if (meta.readout !== undefined) {
    if (meta.readout !== 'decision' && meta.readout !== 'template') throw new NatlangSourceError(path, 'readout must be decision or template');
  }
  if (meta.model !== undefined && (typeof meta.model !== 'string' || !/^[A-Za-z_][\w-]*$/.test(meta.model)))
    throw new NatlangSourceError(path, 'model must name one of the runtime\'s models, such as small');
  if (meta.uses !== undefined && (!Array.isArray(meta.uses) || meta.uses.some(item => typeof item !== 'string' ||
      !/^[A-Za-z_$][\w$]*(\/[A-Za-z_$][\w$]*)*$/.test(item))))
    throw new NatlangSourceError(path, 'uses must list package items by path from the package root, such as [harness/cut]');
  if (meta.readout === 'decision') {
    const env = new TypeEnv(Object.fromEntries(Object.entries(types).map(([name, text]) => [name, parseType(text)])));
    if ((finiteValues(parseType(meta.returns), env)?.length ?? 0) < 2)
      throw new NatlangSourceError(path, 'readout: decision needs a finite returns type with at least two values, such as "yes" | "no"');
  }
  const source = files.relative?.(path) ?? path;
  return { kind: 'natlang', id: `nl:${source}`, name, source, revision: revisionOf(text), text,
    description: String(meta.description ?? ''), args, returns: meta.returns,
    instructions: match[2]!.replace(/^\n+|\n+$/g, '') + '\n', types, subtype, codebase: {},
    ...(meta.readout === 'decision' || meta.readout === 'template' ? { readout: meta.readout as 'decision' | 'template' } : {}),
    ...(typeof meta.model === 'string' ? { model: meta.model } : {}),
    ...(Array.isArray(meta.uses) && meta.uses.length ? { uses: meta.uses as string[] } : {}) };
}

/** The package root above `dir`: the nearest directory with `natlang.json` or `package.json`. */
function packageRoot(dir: string, files: SourceFiles): string {
  for (let current = dir; ; current = files.dirname(current)) {
    if (files.isFile(files.join(current, 'natlang.json')) || files.isFile(files.join(current, 'package.json'))) return current;
    if (files.dirname(current) === current) return current;
  }
}

/** Items being attached through `uses`, outermost first: a function cannot use itself or one of its users. */
const ATTACHING: string[] = [];
/** Functions of callable folders whose load is still running, by file path: a `uses` naming one shares that record. */
const LOADING = new Map<string, NatlangRecord>();
/** The `uses` edges between records of folders being loaded, by file path, to find a cycle among siblings. */
const LOADING_EDGES = new Map<string, Set<string>>();
const reaches = (from: string, to: string, seen = new Set<string>()): boolean => {
  if (from === to) return true;
  if (seen.has(from)) return false;
  seen.add(from);
  return [...LOADING_EDGES.get(from) ?? []].some(next => reaches(next, to, seen));
};
const cycleMessage = (item: string, path: string[]) =>
  `uses ${item}, which leads back to this function (${path.join(' -> ')}); remove one of these uses so that a function cannot reach itself`;

/**
 * Add the items `record.uses` names to its codebase under their base names: `harness/cut` is `<root>/harness/cut.nl`,
 * or `cut.ts`, loaded with its own companion folder. A used item counts as part of the user's callable context.
 */
function attachUses(record: NatlangRecord, path: string, files: SourceFiles): void {
  if (!record.uses?.length || files === PATH_ONLY) return;
  const root = packageRoot(files.dirname(path), files);
  for (const item of record.uses) {
    const base = files.join(root, ...item.split('/'));
    const name = item.slice(item.lastIndexOf('/') + 1);
    if (Object.hasOwn(record.codebase, name))
      throw new NatlangSourceError(path, `uses ${item}, but its folder already has an item named ${name}; rename one of them`);
    const target = files.isFile(`${base}.nl`) ? `${base}.nl` : files.isFile(`${base}.ts`) ? `${base}.ts` : undefined;
    if (!target) throw new NatlangSourceError(path, `uses ${item}, but the package has no ${item}.nl or ${item}.ts`);
    const base0 = (file: string) => files.relative?.(file) ?? file;
    if (ATTACHING.includes(target) || target === path)
      throw new NatlangSourceError(path, cycleMessage(item, [...ATTACHING.slice(Math.max(0, ATTACHING.indexOf(target))), target].map(base0)));
    // A sibling or relative in a folder this load is still reading is that very record, not a second copy of its owner.
    const live = LOADING.get(target);
    if (live) {
      if (reaches(target, path)) throw new NatlangSourceError(path, cycleMessage(item, [path, target, path].map(base0)));
      LOADING_EDGES.set(path, (LOADING_EDGES.get(path) ?? new Set()).add(target));
      record.codebase[name] = live;
      continue;
    }
    ATTACHING.push(target);
    try {
      if (target.endsWith('.nl')) record.codebase[name] = loadNamedFunction(target, files);
      else {
        const dir = files.dirname(target);
        const module = parseModule(target, files.read(target), packageTypes(dir, files), files);
        module.codebase = loadCallableFolder(files.join(dir, name), files, module.types);
        record.codebase[name] = module;
      }
    } finally { ATTACHING.pop(); }
  }
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
  const doc = docComment(node);
  return { kind: 'function', args, returns, async: isAsync, ...(doc ? { doc } : {}) };
}

/** The text of a function's JSDoc comment, on one line; for a function expression, its variable statement's. */
function docComment(node: ts.Node): string | undefined {
  const owner = ts.isFunctionDeclaration(node) ? node : node.parent && ts.isVariableDeclaration(node.parent) ?
    node.parent.parent?.parent : undefined;
  const docs = owner ? ((owner as { jsDoc?: ts.JSDoc[] }).jsDoc ?? []) : [];
  const text = docs.map(doc => ts.getTextOfJSDocComment(doc.comment) ?? '').join(' ').replace(/\s+/g, ' ').trim();
  return text || undefined;
}

/**
 * Types a module declares other than aliases. `types` gives them in natlang type text: a class is a live
 * class value; an interface of properties is a record, and one with methods is a live value checked by
 * its members. `declarations` gives the TypeScript a function listing shows for the live ones: the public
 * members of a class, or the interface itself.
 */
function declaredTypes(file: ts.SourceFile): { types: Record<string, string>; declarations: Record<string, string> } {
  const types: Record<string, string> = {}, declarations: Record<string, string> = {};
  const doc = (node: ts.Node, indent: string) => {
    const text = ((node as { jsDoc?: ts.JSDoc[] }).jsDoc ?? []).map(item => ts.getTextOfJSDocComment(item.comment) ?? '').join(' ').replace(/\s+/g, ' ').trim();
    return text ? [`${indent}/** ${text} */`] : [];
  };
  const hidden = (member: ts.ClassElement) => ts.getModifiers(member as ts.HasModifiers)?.some(modifier =>
    [ts.SyntaxKind.PrivateKeyword, ts.SyntaxKind.ProtectedKeyword, ts.SyntaxKind.StaticKeyword].includes(modifier.kind)) ||
    (member.name !== undefined && ts.isPrivateIdentifier(member.name));
  for (const statement of file.statements) {
    if (ts.isClassDeclaration(statement) && statement.name) {
      const name = statement.name.text, lines = [`declare class ${name} {`];
      for (const member of statement.members) {
        if (hidden(member) || !member.name || !ts.isIdentifier(member.name)) continue;
        const readonly = ts.getModifiers(member as ts.HasModifiers)?.some(modifier => modifier.kind === ts.SyntaxKind.ReadonlyKeyword);
        if (ts.isMethodDeclaration(member)) {
          const params = member.parameters.map(parameter => `${parameter.name.getText(file)}${parameter.questionToken || parameter.initializer ? '?' : ''}: ${parameter.type?.getText(file) ?? 'unknown'}`);
          lines.push(...doc(member, '  '), `  ${member.name.text}(${params.join(', ')}): ${member.type?.getText(file) ?? 'unknown'};`);
        } else if (ts.isPropertyDeclaration(member) && member.type)
          lines.push(...doc(member, '  '), `  ${readonly ? 'readonly ' : ''}${member.name.text}: ${member.type.getText(file)};`);
      }
      lines.push('}');
      types[name] = `Live<${JSON.stringify(name)}, "class", ${JSON.stringify(name)}>`;
      declarations[name] = lines.join('\n');
    } else if (ts.isInterfaceDeclaration(statement) && !statement.typeParameters?.length && !statement.heritageClauses?.length) {
      const name = statement.name.text, members = statement.members;
      const fields = members.map(member => ts.isPropertySignature(member) && member.type && (ts.isIdentifier(member.name) || ts.isStringLiteral(member.name)) ?
        `${member.name.text}${member.questionToken ? '?' : ''}: ${member.type.getText(file)}` : undefined);
      if (fields.every(field => field !== undefined)) { types[name] = `{ ${fields.join(', ')} }`; continue; }
      types[name] = `Live<${JSON.stringify(name)}, "shape", ${JSON.stringify(members.flatMap(member =>
        member.name && (ts.isIdentifier(member.name) || ts.isStringLiteral(member.name)) ? [member.name.text] : []).join(','))}>`;
      declarations[name] = statement.getText(file).replace(/^export\s+/, '');
    }
  }
  return { types, declarations };
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
  const declared = declaredTypes(file);
  const types = { ...inherited, ...declared.types, ...readTypeAliases(text) };
  for (const record of Object.values(exports)) if (record.kind === 'function') {
    try { checkSignature(path, record.args, record.returns, types); }
    catch { /* TypeScript types outside the portable grammar are checked by the compiler, not here. */ }
  }
  const source = files.relative?.(path) ?? path;
  return { kind: 'module', id: `ts:${source}`, name, source, revision: revisionOf(text), text, types, exports, imports, codebase: {},
    ...(Object.keys(declared.declarations).length ? { declarations: declared.declarations } : {}) };
}

const isSource = (name: string) => (name.endsWith('.nl') || (name.endsWith('.ts') && !name.endsWith('.d.ts'))) &&
  name !== 'types.ts';

/**
 * The type aliases of a `types.ts`, including those it re-exports from another types module with
 * `export type { A, B } from "../other/types.js"` or `export * from "./more.js"`. Two applications share one definition
 * that way instead of declaring it twice. Only relative specifiers that name a readable file are followed.
 */
export function readTypesFile(path: string, files: SourceFiles, seen: string[] = []): Record<string, string> {
  const text = files.read(path);
  const own = readTypeAliases(text);
  if (!/\bexport\b[^;]*\bfrom\b/.test(text)) return own;
  const source = ts.createSourceFile('types.ts', text, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
  const result: Record<string, string> = {};
  for (const statement of source.statements) {
    if (!ts.isExportDeclaration(statement) || !statement.moduleSpecifier || !ts.isStringLiteral(statement.moduleSpecifier)) continue;
    const specifier = statement.moduleSpecifier.text;
    if (!specifier.startsWith('.')) continue;
    const base = files.join(files.dirname(path), specifier.replace(/\.(?:[cm]?js|ts)$/, ''));
    const target = [`${base}.ts`, files.join(base, 'index.ts')].find(candidate => files.isFile(candidate));
    if (!target || seen.includes(target) || target === path) continue;
    const there = readTypesFile(target, files, [...seen, path]);
    const clause = statement.exportClause;
    if (!clause) Object.assign(result, there);
    else if (ts.isNamedExports(clause)) for (const element of clause.elements) {
      const from = (element.propertyName ?? element.name).text;
      if (there[from] !== undefined) result[element.name.text] = there[from]!;
    }
  }
  return { ...result, ...own };
}

/** Load a callable folder into a record tree. `inherited` holds type aliases from enclosing folders. */
export function loadCallableFolder(dir: string, files: SourceFiles, inherited: Record<string, string> = {}): Record<string, ItemRecord> {
  if (!files.isDirectory(dir)) return {};
  const typesFile = files.join(dir, 'types.ts');
  const types = { ...inherited, ...(files.isFile(typesFile) ? readTypesFile(typesFile, files) : {}) };
  const entries = files.list(dir).filter(name => !name.startsWith('.')).sort();
  const items: Record<string, ItemRecord> = {};
  const mine: string[] = [];
  const add = (name: string, record: ItemRecord, path: string) => {
    if (Object.hasOwn(items, name)) throw new NatlangSourceError(path, `two items in ${dir} are named ${JSON.stringify(name)}`);
    items[name] = record;
    if (record.kind === 'natlang' && files !== PATH_ONLY) { LOADING.set(path, record); mine.push(path); }
  };
  try {
  for (const entry of entries) {
    const path = files.join(dir, entry);
    if (!files.isFile(path) || !isSource(entry) || entry.endsWith('.d.ts') || entry.endsWith('.d.nl.ts')) continue;
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
      if (owner.kind === 'natlang') {
        const data = folderNzData(path, files, children);
        if (Object.keys(data).length) owner.contextData = data;
      }
    } else if (Object.keys(children).length)
      add(entry, { kind: 'namespace', name: entry, source: files.relative?.(path) ?? path, codebase: children }, path);
  }
  for (const [name, item] of Object.entries(items))
    if (item.kind === 'natlang' && item.uses) attachUses(item, files.join(dir, `${name}.nl`), files);
  if (files !== PATH_ONLY) registerFileRecords(items);
  return items;
  } finally { for (const path of mine) { LOADING.delete(path); LOADING_EDGES.delete(path); } }
}

/**
 * The `.nz` files directly in a companion folder, as data entries named by file (S0 §7: a context's data entries).
 * A name that is also an item of the folder is an error. Blocks are registered for runtimes to adopt.
 */
function folderNzData(dir: string, files: SourceFiles, items: Record<string, ItemRecord>): Record<string, unknown> {
  const data: Record<string, unknown> = {};
  for (const entry of files.readBytes ? files.list(dir).filter(name => name.endsWith('.nz') && !name.startsWith('.')).sort() : []) {
    const path = files.join(dir, entry);
    if (!files.isFile(path)) continue;
    const name = entry.slice(0, -'.nz'.length);
    if (!/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(name))
      throw new NatlangSourceError(path, `a .nz file in a callable folder needs an identifier name to bind as ${JSON.stringify(name)}`);
    if (Object.hasOwn(items, name)) throw new NatlangSourceError(path, `${JSON.stringify(name)} is both an item and a .nz file in ${dir}`);
    const loaded = loadNzSync(files.readBytes!(path));
    registerImportedBlocks(loaded.blocks.values());
    data[name] = loaded.exports;
  }
  // The folder's own skills (S2 §2.3): `skills/**` files, bound by default as data entries under their paths.
  const skills = files.join(dir, 'skills');
  const walk = (folder: string, prefix: string) => {
    for (const entry of files.list(folder).filter(name => !name.startsWith('.')).sort()) {
      const path = files.join(folder, entry), key = `${prefix}/${entry}`;
      if (files.isDirectory(path)) walk(path, key);
      else if (files.isFile(path)) data[key] = files.read(path);
    }
  };
  if (files.isDirectory(skills)) walk(skills, 'skills');
  return data;
}

/** Load a named `.nl` function with its companion folder as its callable context. */
export function loadNamedFunction(path: string, files: SourceFiles): NatlangRecord {
  if (!files.isFile(path) || files.extname(path) !== '.nl') throw new NatlangSourceError(path, 'expected a .nl file');
  const dir = files.dirname(path);
  // A child loaded by its own path has the same lexical types as when reached through its owner.
  const names = [files.basename(path, '.nl')];
  let ancestor = dir;
  while (true) {
    const parent = files.dirname(ancestor);
    if (parent === ancestor) break;
    const owner = files.join(parent, files.basename(ancestor) + '.nl');
    if (files.isFile(owner)) {
      let record: ItemRecord = loadNamedFunction(owner, files);
      for (const name of names) {
        const child: ItemRecord | undefined = record.codebase[name];
        if (!child) throw new NatlangSourceError(path, 'function is absent from its owning callable folder');
        record = child;
      }
      if (record.kind !== 'natlang') throw new NatlangSourceError(path, 'expected a natural-language function');
      return record;
    }
    names.unshift(files.basename(ancestor));
    ancestor = parent;
  }
  const record = parseNatlang(path, files.read(path), packageTypes(dir, files), files);
  const companion = files.join(dir, record.name);
  record.codebase = loadCallableFolder(companion, files, record.types);
  if (files.isDirectory(companion)) {
    const data = folderNzData(companion, files, record.codebase);
    if (Object.keys(data).length) record.contextData = data;
  }
  attachUses(record, path, files);
  registerFileRecords(record);
  return record;
}

/**
 * Type aliases for a named function outside any callable folder: the `types.ts` of its directory and of each
 * enclosing directory up to the package root (the nearest with `natlang.json` or `package.json`); nearer wins.
 */
function packageTypes(dir: string, files: SourceFiles): Record<string, string> {
  const layers: Record<string, string>[] = [];
  for (let current = dir; ; current = files.dirname(current)) {
    const typesFile = files.join(current, 'types.ts');
    if (files.isFile(typesFile)) layers.push(readTypesFile(typesFile, files));
    const root = files.isFile(files.join(current, 'natlang.json')) || files.isFile(files.join(current, 'package.json'));
    if (root || files.dirname(current) === current) break;
  }
  return Object.assign({}, ...layers.reverse());
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
