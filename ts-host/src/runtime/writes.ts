/**
 * `writes:` of a directory reducer: the files its calls write, as one type (spec/ext/directory-reducers.md "Declared
 * files"). The type is a record whose field names are folder paths, or globs (`*` is any name within one folder level,
 * `**` any number of levels), and whose field types type each file's text: `string`, a string literal, or a union of
 * them. Doc comments on the fields and on the alias say what each file holds; they reach the model with the alias, as
 * every type alias of a call's signature does.
 *
 * A call is held to it when it returns success: every file it created or changed must match a field, and that file's
 * text must fit the field's type. Files it removed (a move's source) and files it left alone are not checked.
 */
import { containsRefinement, formatType, parseTypeCached, TypeEnv, type Type } from '../native/types.js';

export type WritesField = { pattern: string; type: Type; match: (path: string) => boolean };

/** `*` matches within one path segment, `**` any number of segments (`**\/x.md` includes a top-level `x.md`). */
export function pathPattern(pattern: string): (path: string) => boolean {
  let source = '';
  for (let index = 0; index < pattern.length; index++) {
    const char = pattern.charAt(index);
    if (char !== '*') { source += char.replace(/[.+?^${}()|[\]\\]/g, '\\$&'); continue; }
    if (pattern[index + 1] !== '*') { source += '[^/]*'; continue; }
    index++;
    if (pattern[index + 1] === '/') { source += '(?:[^/]+/)*'; index++; } else source += '.*';
  }
  const expression = new RegExp(`^${source}$`);
  return path => expression.test(path);
}

/** Specificity rank of a path pattern: exact paths first, then single-segment globs, then globstars. */
function patternSpecificity(pattern: string): number {
  if (!pattern.includes('*')) return 0;
  if (!pattern.includes('**')) return 1;
  return 2;
}

/** Whether `type` types a file's text: string, a string literal, unknown, a refined/untrusted text type, or a union or alias of these. */
function textType(type: Type, env: TypeEnv, depth = 0): boolean {
  if (depth > 32) return false;
  const resolved = env.resolve(type);
  if (resolved.kind === 'refined' || resolved.kind === 'untrusted') return textType(resolved.base, env, depth + 1);
  if (resolved.kind === 'prim') return resolved.name === 'string' || resolved.name === 'unknown';
  if (resolved.kind === 'lit') return typeof resolved.value === 'string';
  if (resolved.kind === 'union') return resolved.members.every(member => textType(member, env, depth + 1));
  return false;
}

/** Whether `text` is a value of the text type `type` (structural check; refinements are checked asynchronously). */
function textFits(text: string, type: Type, env: TypeEnv, depth = 0): boolean {
  if (depth > 32) return false;
  const resolved = env.resolve(type);
  if (resolved.kind === 'refined' || resolved.kind === 'untrusted') return textFits(text, resolved.base, env, depth + 1);
  if (resolved.kind === 'prim') return true;
  if (resolved.kind === 'lit') return resolved.value === text;
  if (resolved.kind === 'union') return resolved.members.some(member => textFits(text, member, env, depth + 1));
  return false;
}

const BAD_PATH = /(^\/)|(^|\/)\.\.?(\/|$)|\/\/|\/$|\\/;

/**
 * The fields of a `writes:` type, checked: a record (or an alias of one) whose field names are relative paths or globs
 * and whose field types are text types. Throws an Error whose message is the load error.
 */
export function writesFields(text: string, env: TypeEnv): WritesField[] {
  const shape = 'writes is one record type whose fields are the files the call writes, each a path or glob from the ' +
    'folder with the type of that file\'s text, such as writes: { "report.md": string, "done/*.md": string }';
  let type: Type;
  try { type = parseTypeCached(text); } catch (error) { throw new Error(`${shape}; ${(error as Error).message}`); }
  let resolved: Type;
  try { resolved = env.resolve(type); } catch (error) { throw new Error(`${shape}; ${(error as Error).message}`); }
  if (resolved.kind !== 'record' || !resolved.fields.length) throw new Error(shape);
  return resolved.fields.map(field => {
    if (!field.name.trim() || BAD_PATH.test(field.name))
      throw new Error(`writes: ${JSON.stringify(field.name)} is not a path in the folder; write it relative to the folder, as "notes/summary.md" or "done/*.md"`);
    if (!textType(field.type, env))
      throw new Error(`writes: ${JSON.stringify(field.name)} has type ${formatType(field.type)}, but a file holds text; type it string, a string literal, Is<string, "...">, or a union of them, and say what the file holds in the field's doc comment`);
    return { pattern: field.name, type: field.type, match: pathPattern(field.name) };
  });
}

/** Whether a `writes:` declaration uses any `Is<string, P>` refined file type. */
export function writesContainsRefinement(text: string, env: TypeEnv): boolean {
  try { return writesFields(text, env).some(field => containsRefinement(field.type, env)); }
  catch { return false; }
}

/** All fields matching `path`, ordered most-specific first so exact rules are checked before broad globs. */
function matchingWritesFields(fields: readonly WritesField[], path: string): WritesField[] {
  return fields.filter(item => item.match(path))
    .sort((a, b) => patternSpecificity(a.pattern) - patternSpecificity(b.pattern) || b.pattern.length - a.pattern.length);
}

/** One sentence of what is wrong with the files a call wrote, or undefined when they fit its `writes:` type. */
export function writesProblem(fields: readonly WritesField[], env: TypeEnv, name: string,
  changes: readonly { path: string; kind: string; after?: Uint8Array }[]): string | undefined {
  const declared = fields.map(field => field.pattern).join(', ');
  const decoder = new TextDecoder();
  for (const change of changes) {
    if (change.kind === 'deleted') continue;
    const matched = matchingWritesFields(fields, change.path);
    if (!matched.length) return `${change.path} is not among the files ${name} writes (${declared}); ` +
      `undo that change (remove the file, or restore its text), and put what the result needs in the files it declares.`;
    const text = decoder.decode(change.after ?? new Uint8Array());
    for (const field of matched) {
      if (!textFits(text, field.type, env))
        return `${change.path} holds ${JSON.stringify(text.length > 80 ? text.slice(0, 77) + '...' : text)}, which is not a ${formatType(field.type)} ` +
          `(the type ${name} declares for ${field.pattern}); rewrite it as one.`;
    }
  }
  return undefined;
}

/** Written files whose matching `writes:` field carries an `Is<string, P>` refinement to check asynchronously. */
export function writesRefinementChecks(fields: readonly WritesField[], env: TypeEnv,
  changes: readonly { path: string; kind: string; after?: Uint8Array }[]): { path: string; pattern: string; type: Type; text: string }[] {
  const decoder = new TextDecoder();
  const out: { path: string; pattern: string; type: Type; text: string }[] = [];
  for (const change of changes) {
    if (change.kind === 'deleted') continue;
    const matched = matchingWritesFields(fields, change.path).filter(field => containsRefinement(field.type, env));
    if (!matched.length) continue;
    const text = decoder.decode(change.after ?? new Uint8Array());
    for (const field of matched) out.push({ path: change.path, pattern: field.pattern, type: field.type, text });
  }
  return out;
}
