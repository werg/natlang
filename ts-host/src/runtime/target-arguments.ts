/**
 * A target's declared arguments (natlang.json `targets.<name>.arguments`). Each argument's `type` is natlang type text,
 * the grammar `.nl` signatures use, with the program's `types.ts` aliases in scope. `natlang run` reads the command line
 * into checked values for `context.arguments` (the same `coerce` the runtime applies at every typed slot); the dashboard
 * renders its run form from the expanded types and checks drafts through `readTargetArguments`.
 *
 * On the command line an argument whose type is text (string, string literals, refinements of those) is written as
 * is; any other type is written as JSON. A `path` argument is text naming a file or folder in the workspace, which the
 * launcher checks and passes on as an absolute path.
 */
import { existsSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import ts from 'typescript';
import { coerce, Reject } from '../native/values.js';
import { formatType, parseType, TypeEnv, TypeSyntaxError, type Type } from '../native/types.js';
import { readTypesFile } from './loader.js';
import { nodeSourceFiles } from './node-files.js';

export type TargetArgument = { name: string; type: string; description?: string; required?: boolean; default?: unknown;
  path?: 'file' | 'folder' };

/** A type expanded for an editor: aliases inlined (named by `alias`), record fields with their `types.ts` doc comments,
 * and a recursive or unknown name left as `{ kind: 'name' }` for a JSON editor. */
export type ExpandedType = Type & { alias?: string };
export type ArgumentDescription = TargetArgument & { required: boolean; text: string; tree: ExpandedType;
  /** How the command line spells it: `text` as written, `json` as JSON. */
  spelling: 'text' | 'json'; problem?: string };
export type ArgumentProblem = { argument: string; path: string; message: string };

const NAME = /^[a-zA-Z][a-zA-Z0-9_-]*$/;
const UNSUPPORTED = new Set(['lambda', 'host', 'neuralese']);

/** Structural checks of a manifest's `arguments` array; types are parsed, names resolve later against `types.ts`. */
export function parseTargetArguments(value: unknown, target: string): TargetArgument[] {
  if (!Array.isArray(value)) throw new TypeError(`target ${target} arguments must be an array`);
  const names = new Set<string>();
  return value.map((item: unknown, index) => {
    const where = `target ${target} argument ${index + 1}`;
    if (!item || typeof item !== 'object' || Array.isArray(item)) throw new TypeError(`${where} must be an object`);
    const arg = item as Record<string, unknown>;
    for (const key of Object.keys(arg)) if (!['name', 'type', 'description', 'required', 'default', 'path'].includes(key))
      throw new TypeError(`${where}: unknown field ${key}; use name, type, description, required, default or path`);
    if (typeof arg.name !== 'string' || !NAME.test(arg.name)) throw new TypeError(`${where}: name must be an identifier`);
    if (names.has(arg.name)) throw new TypeError(`${where}: duplicate argument name ${arg.name}`);
    names.add(arg.name);
    if (typeof arg.type !== 'string') throw new TypeError(`argument ${arg.name}: type must be natlang type text, for example "string" or "{ title: string }[]"`);
    if (arg.type === 'json') throw new TypeError(`argument ${arg.name}: json is not a natlang type; use unknown, or a precise type such as "Record<string, number>"`);
    let type: Type;
    try { type = parseType(arg.type); } catch (error) { throw new TypeError(`argument ${arg.name}: ${(error as Error).message}`); }
    const unsupported = findKind(type, UNSUPPORTED);
    if (unsupported) throw new TypeError(`argument ${arg.name}: a command-line argument is data; ${unsupported} types cannot be passed`);
    if (arg.description !== undefined && typeof arg.description !== 'string') throw new TypeError(`argument ${arg.name}: description must be text`);
    if (arg.required !== undefined && typeof arg.required !== 'boolean') throw new TypeError(`argument ${arg.name}: required must be true or false`);
    if (arg.required === true && arg.default !== undefined) throw new TypeError(`argument ${arg.name}: an argument with a default is optional; remove required`);
    if (arg.path !== undefined && arg.path !== 'file' && arg.path !== 'folder') throw new TypeError(`argument ${arg.name}: path must be "file" or "folder"`);
    if (arg.path !== undefined && formatType(type) !== 'string') throw new TypeError(`argument ${arg.name}: a path argument has type string`);
    return { name: arg.name, type: arg.type, ...(arg.description !== undefined ? { description: arg.description as string } : {}),
      ...(arg.required !== undefined ? { required: arg.required as boolean } : {}), ...(arg.default !== undefined ? { default: arg.default } : {}),
      ...(arg.path !== undefined ? { path: arg.path as 'file' | 'folder' } : {}) };
  });
}

function findKind(type: Type, kinds: Set<string>): string | undefined {
  if (kinds.has(type.kind)) return type.kind === 'host' ? (type as { name: string }).name : type.kind;
  const inner: Type[] = type.kind === 'record' ? type.fields.map(field => field.type) : type.kind === 'list' || type.kind === 'dict' ? [type.element] :
    type.kind === 'union' || type.kind === 'intersection' ? type.members : type.kind === 'refined' || type.kind === 'untrusted' || type.kind === 'index' ? [type.base] : [];
  for (const child of inner) { const found = findKind(child, kinds); if (found) return found; }
  return undefined;
}

/** The program's type environment: the aliases of its `types.ts`, with their field doc comments. */
export function programTypes(root: string): { env: TypeEnv; docs: Record<string, Record<string, string>> } {
  const path = join(root, 'types.ts');
  if (!existsSync(path)) return { env: new TypeEnv(), docs: {} };
  const aliases = readTypesFile(path, nodeSourceFiles(root));
  const names: Record<string, Type> = {};
  for (const [name, text] of Object.entries(aliases)) { try { names[name] = parseType(text); } catch { names[name] = parseType('unknown'); } }
  const docs: Record<string, Record<string, string>> = {};
  for (const [name, text] of Object.entries(aliases)) {
    const file = ts.createSourceFile('alias.ts', `type T = ${text};`, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
    const literal = (file.statements[0] as ts.TypeAliasDeclaration | undefined)?.type;
    if (!literal || !ts.isTypeLiteralNode(literal)) continue;
    for (const member of literal.members) {
      const comment = ts.getLeadingCommentRanges(file.text, member.pos)?.map(range => file.text.slice(range.pos, range.end)).join('\n');
      if (!comment || !member.name) continue;
      const doc = comment.replace(/^\/\*\*?|\*\/$/g, '').split('\n').map(line => line.replace(/^\s*(?:\*|\/\/)\s?/, '').trim()).filter(Boolean).join(' ');
      if (doc) (docs[name] ??= {})[member.name.getText(file).replace(/^["']|["']$/g, '')] = doc;
    }
  }
  return { env: new TypeEnv(names), docs };
}

/** `type` with aliases inlined for an editor; a name seen again on its own path (recursion) stays a name. */
export function expandType(type: Type, env: TypeEnv, docs: Record<string, Record<string, string>> = {}, path: string[] = []): ExpandedType {
  if (type.kind === 'name') {
    const found = env.lookup(type.name);
    if (!found || path.includes(type.name) || path.length > 12) return type;
    const inner = expandType(found, env, docs, [...path, type.name]);
    if (inner.kind === 'record' && docs[type.name])
      return { ...inner, alias: type.name, fields: inner.fields.map(field => docs[type.name]![field.name] ? { ...field, doc: docs[type.name]![field.name] } : field) } as ExpandedType;
    return { ...inner, alias: inner.alias ?? type.name };
  }
  const child = (inner: Type) => expandType(inner, env, docs, path);
  switch (type.kind) {
    case 'record': return { ...type, fields: type.fields.map(field => ({ ...field, type: child(field.type) })) };
    case 'list': case 'dict': return { ...type, element: child(type.element) };
    case 'union': return { ...type, members: type.members.map(child) };
    case 'refined': case 'untrusted': return { ...type, base: child(type.base) };
    case 'intersection': case 'index': { try { return child(env.resolve(type)); } catch { return type; } }
    default: return type;
  }
}

/** Whether values of `type` are strings only, so the command line writes them as they are. */
export function spelling(type: Type, env: TypeEnv): 'text' | 'json' {
  let resolved: Type;
  try { resolved = env.resolve(type); } catch { return 'json'; }
  if (resolved.kind === 'refined' || resolved.kind === 'untrusted') return spelling(resolved.base, env);
  if (resolved.kind === 'prim') return resolved.name === 'string' || resolved.name === 'Blob' ? 'text' : 'json';
  if (resolved.kind === 'lit') return typeof resolved.value === 'string' ? 'text' : 'json';
  if (resolved.kind === 'union') return resolved.members.every(member => spelling(member, env) === 'text') ? 'text' : 'json';
  return 'json';
}

/** The arguments of a target as the dashboard's form needs them. A type that does not resolve carries its `problem`. */
export function describeArguments(declared: TargetArgument[] | undefined, root: string): ArgumentDescription[] {
  if (!declared?.length) return [];
  let types: ReturnType<typeof programTypes>, typesProblem: string | undefined;
  try { types = programTypes(root); } catch (error) { types = { env: new TypeEnv(), docs: {} }; typesProblem = `types.ts: ${(error as Error).message}`; }
  return declared.map(arg => {
    const type = parseType(arg.type);
    let problem = typesProblem;
    try { types.env.checkNames(type); } catch (error) { problem ??= (error as Error).message; }
    if (!problem && arg.default !== undefined) problem = checkValue(arg.default, type, types.env, `${arg.name} default`).map(item => item.message)[0];
    return { ...arg, required: arg.required === true, text: formatType(type), tree: expandType(type, types.env, types.docs),
      spelling: spelling(type, types.env), ...(problem ? { problem } : {}) };
  });
}

/** `input: string, count?: number = 3`, the target's signature for usage lines and form headings. */
export function argumentSignature(declared: TargetArgument[]): string {
  return declared.map(arg => `${arg.name}${arg.required ? '' : '?'}: ${formatType(parseType(arg.type))}` +
    (arg.default !== undefined ? ` = ${JSON.stringify(arg.default)}` : '')).join(', ');
}

/** `<input: string> [count: number]`: the command-line usage of the arguments, required ones in angle brackets. */
export function argumentUsage(declared: TargetArgument[]): string {
  return declared.map(arg => { const text = `${arg.name}: ${formatType(parseType(arg.type))}`;
    return arg.required ? `<${text}>` : `[${text}]`; }).join(' ');
}

/**
 * Read command-line `raw` values into checked values by name. An empty or absent optional argument takes its default
 * or is left out (the dashboard writes an empty word for an argument it skips before a later one); `problems` lists
 * every argument that is missing, malformed or the wrong type.
 */
export function readTargetArguments(declared: TargetArgument[] | undefined, raw: string[], root: string, workspace: string):
  { values: Record<string, unknown>; problems: ArgumentProblem[] } {
  const values: Record<string, unknown> = {}, problems: ArgumentProblem[] = [];
  if (!declared) return { values, problems };
  if (raw.length > declared.length)
    problems.push({ argument: '', path: '', message: `takes ${declared.length} argument${declared.length === 1 ? '' : 's'} (${declared.map(arg => arg.name).join(', ') || 'none'}); got ${raw.length}` });
  let env: TypeEnv;
  try { env = programTypes(root).env; } catch (error) { problems.push({ argument: '', path: '', message: `types.ts: ${(error as Error).message}` }); return { values, problems }; }
  declared.forEach((arg, index) => {
    const type = parseType(arg.type), text = raw[index];
    let value: unknown;
    if (text === undefined || text === '') {
      if (arg.default !== undefined) value = arg.default;
      else if (arg.required) { problems.push({ argument: arg.name, path: arg.name, message: `${arg.name} is required (${formatType(type)})` }); return; }
      else return;
    } else if (spelling(type, env) === 'text') value = text;
    else {
      try { value = JSON.parse(text); }
      catch (error) {
        // A string where the type also takes strings (string | number) is the string. Broken JSON structure gets the
        // parser's message; a bare word gets the type's (`count: expected number, got "x"`).
        const asText = checkValue(text, type, env, arg.name);
        if (asText.length) {
          problems.push(...(/^\s*[[{"]/.test(text) ? [{ argument: arg.name, path: arg.name,
            message: `${arg.name} is written as JSON (${formatType(type)}): ${(error as Error).message}` }] : asText));
          return;
        }
        value = text;
      }
    }
    const found = checkValue(value, type, env, arg.name);
    if (found.length) { problems.push(...found); return; }
    if (arg.path && typeof value === 'string' && value !== '') {
      const absolute = resolve(workspace, value);
      const kind = existsSync(absolute) ? statSync(absolute).isDirectory() ? 'folder' : 'file' : undefined;
      if (kind !== arg.path) { problems.push({ argument: arg.name, path: arg.name, message: kind ? `${arg.name}: ${value} is a ${kind}, not a ${arg.path}` : `${arg.name}: no ${arg.path} ${value} in ${workspace}` }); return; }
      value = absolute;
    }
    values[arg.name] = coerceQuiet(value, type, env);
  });
  return { values, problems };
}

/** Problems of `value` as a `type`: the runtime's own check, plus record fields that are required and absent. */
export function checkValue(value: unknown, type: Type, env: TypeEnv, path: string): ArgumentProblem[] {
  const argument = path.split(/[ /]/)[0]!;
  try { coerce(value, type, env, path); }
  catch (error) {
    if (!(error instanceof Reject)) return [{ argument, path, message: `${path}: ${(error as Error).message}` }];
    return error.diagnostics.map(item => ({ argument, path: item.path, message: item.text ?? `${item.path}: ${item.code === 'type-mismatch' ? '' : item.code + ', '}` +
      [item.expected ? `expected ${item.expected}` : '', item.got ? `got ${item.got}` : ''].filter(Boolean).join(', ') }));
  }
  const missing: ArgumentProblem[] = [];
  const walk = (current: unknown, wanted: Type, at: string) => {
    let resolved: Type;
    try { resolved = env.resolve(wanted); } catch { return; }
    if (resolved.kind === 'refined' || resolved.kind === 'untrusted') return walk(current, resolved.base, at);
    if (resolved.kind === 'list' && Array.isArray(current)) current.forEach((item, index) => walk(item, resolved.element, `${at}/${index}`));
    if (resolved.kind === 'dict' && current && typeof current === 'object') for (const [key, item] of Object.entries(current)) walk(item, (resolved as { element: Type }).element, `${at}/${key}`);
    if (resolved.kind === 'record' && current && typeof current === 'object' && !Array.isArray(current))
      for (const field of resolved.fields) {
        if (Object.hasOwn(current, field.name)) walk((current as Record<string, unknown>)[field.name], field.type, `${at}/${field.name}`);
        else if (!field.optional) missing.push({ argument, path: `${at}/${field.name}`, message: `${at}/${field.name}: required field missing (${formatType(field.type)})` });
      }
  };
  walk(value, type, path);
  return missing;
}

function coerceQuiet(value: unknown, type: Type, env: TypeEnv): unknown {
  try { return coerce(value, type, env); } catch (error) { if (error instanceof Reject || error instanceof TypeSyntaxError) return value; throw error; }
}
