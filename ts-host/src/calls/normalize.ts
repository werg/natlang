/**
 * Normalized eval programs (§5.1): the same approach written for different inputs should normalize to the same text.
 * A literal equal to an input value becomes that input's path (`$in.request`); a literal, or a word inside a string or
 * template, that is a part of an input string (the `144` of "status of order 144") becomes `$part.request`. Declared
 * locals are renamed in order (`v0`, `v1`, ...). Logging, `await` and `return` keywords, whitespace and comments go.
 * Token-based, so it is cheap enough to run when a call is recorded.
 */
import ts from 'typescript';
import { hexDigest } from '../native/hash.js';

/** Leaf values of the inputs by value, with the path that reaches each (the shortest when a value repeats). */
export function inputLeaves(inputs: Record<string, unknown>): Map<string, string> {
  const leaves = new Map<string, string>();
  const put = (key: string, path: string) => { const known = leaves.get(key); if (!known || path.length < known.length) leaves.set(key, path); };
  const visit = (value: unknown, path: string, depth: number) => {
    if (depth > 6) return;
    if (typeof value === 'string' || typeof value === 'number') {
      put(`${typeof value}:${value}`, path);
      // Parts of a string input: numbers and words a program may have copied out of it.
      if (typeof value === 'string') for (const part of value.match(/[\p{L}\p{N}_-]+/gu) ?? [])
        if (/\d/.test(part) || part.length >= 4) put(`part:${part.toLowerCase()}`, path);
      return;
    }
    if (Array.isArray(value)) { value.slice(0, 50).forEach((item, index) => visit(item, `${path}[${index}]`, depth + 1)); return; }
    if (value && typeof value === 'object' && (Object.getPrototypeOf(value) === Object.prototype || Object.getPrototypeOf(value) === null))
      for (const [key, item] of Object.entries(value).slice(0, 100))
        visit(item, /^[A-Za-z_$][\w$]*$/.test(key) ? `${path}.${key}` : `${path}[${JSON.stringify(key)}]`, depth + 1);
  };
  for (const [name, value] of Object.entries(inputs)) visit(value, name, 0);
  return leaves;
}

const DECLARING = new Set([ts.SyntaxKind.ConstKeyword, ts.SyntaxKind.LetKeyword, ts.SyntaxKind.VarKeyword, ts.SyntaxKind.FunctionKeyword]);
const DROPPED = new Set([ts.SyntaxKind.AwaitKeyword, ts.SyntaxKind.ReturnKeyword]);
const TEMPLATE_TEXT = new Set([ts.SyntaxKind.TemplateHead, ts.SyntaxKind.TemplateMiddle, ts.SyntaxKind.TemplateTail,
  ts.SyntaxKind.NoSubstitutionTemplateLiteral]);

/** Replace the input parts inside a piece of literal text. */
function replaceParts(text: string, leaves: Map<string, string>): string {
  return text.replace(/[\p{L}\p{N}_-]+/gu, word => {
    const path = (/\d/.test(word) || word.length >= 4) ? leaves.get(`part:${word.toLowerCase()}`) : undefined;
    return path ? `\${$part.${path}}` : word;
  });
}

/** One eval program in normalized form. */
export function normalizeProgram(code: string, inputs: Record<string, unknown> = {}, leaves = inputLeaves(inputs)): string {
  const scanner = ts.createScanner(ts.ScriptTarget.ES2022, true, ts.LanguageVariant.Standard, code);
  const tokens: { kind: ts.SyntaxKind; text: string; value: string }[] = [];
  // Template substitutions: a `}` that closes one is rescanned as the template's continuation.
  const braces: number[] = [];
  for (let kind = scanner.scan(); kind !== ts.SyntaxKind.EndOfFileToken; kind = scanner.scan()) {
    if (kind === ts.SyntaxKind.OpenBraceToken && braces.length) braces[braces.length - 1]!++;
    else if (kind === ts.SyntaxKind.CloseBraceToken && braces.length) {
      if (braces.at(-1)! > 0) braces[braces.length - 1]!--;
      else { braces.pop(); kind = scanner.reScanTemplateToken(false); }
    }
    if (kind === ts.SyntaxKind.TemplateHead || kind === ts.SyntaxKind.TemplateMiddle) braces.push(0);
    tokens.push({ kind, text: scanner.getTokenText(), value: scanner.getTokenValue() });
  }
  const renamed = new Map<string, string>();
  const out: string[] = [];
  let previous: ts.SyntaxKind | undefined;
  for (let index = 0; index < tokens.length; index++) {
    const token = tokens[index]!;
    // console.log(...) and friends: logging is not part of an approach.
    if (token.kind === ts.SyntaxKind.Identifier && token.text === 'console' && tokens[index + 1]?.kind === ts.SyntaxKind.DotToken &&
        tokens[index + 3]?.kind === ts.SyntaxKind.OpenParenToken) {
      let depth = 0, end = index + 3;
      for (; end < tokens.length; end++) {
        if (tokens[end]!.kind === ts.SyntaxKind.OpenParenToken) depth++;
        if (tokens[end]!.kind === ts.SyntaxKind.CloseParenToken && --depth === 0) break;
      }
      index = tokens[end + 1]?.kind === ts.SyntaxKind.SemicolonToken ? end + 1 : end;
      continue;
    }
    if (DROPPED.has(token.kind)) continue;
    // A statement that is only a name shows a value; it does no work.
    const next = tokens[index + 1]?.kind;
    if (token.kind === ts.SyntaxKind.Identifier && (previous === undefined || previous === ts.SyntaxKind.SemicolonToken ||
        previous === ts.SyntaxKind.CloseBraceToken) && (next === undefined || next === ts.SyntaxKind.SemicolonToken)) { index++; continue; }
    let text = token.text;
    if (token.kind === ts.SyntaxKind.Identifier) {
      if (previous !== undefined && DECLARING.has(previous) && !renamed.has(text)) renamed.set(text, `v${renamed.size}`);
      const name = renamed.get(text);
      if (name && previous !== ts.SyntaxKind.DotToken) text = name;
    } else if (token.kind === ts.SyntaxKind.StringLiteral || token.kind === ts.SyntaxKind.NoSubstitutionTemplateLiteral) {
      const path = leaves.get(`string:${token.value}`) ?? leaves.get(`part:${token.value.toLowerCase()}`);
      text = path ? `$${leaves.get(`string:${token.value}`) ? 'in' : 'part'}.${path}` : JSON.stringify(replaceParts(token.value, leaves));
    } else if (TEMPLATE_TEXT.has(token.kind)) {
      text = replaceParts(text, leaves);
    } else if (token.kind === ts.SyntaxKind.NumericLiteral) {
      const number = Number(token.value);
      const path = leaves.get(`number:${number}`) ?? leaves.get(`part:${token.value}`);
      if (path && number !== 0 && number !== 1) text = `$${leaves.get(`number:${number}`) ? 'in' : 'part'}.${path}`;
    } else if (token.kind === ts.SyntaxKind.SemicolonToken && (previous === ts.SyntaxKind.SemicolonToken || previous === undefined)) continue;
    out.push(text);
    previous = token.kind;
  }
  while (out.at(-1) === ';') out.pop();
  return out.join(' ');
}

/** The approach a successful agent call took: its successful evals, normalized. `answer-only` when it ran none. */
export function approachHash(evals: readonly string[], inputs: Record<string, unknown>): string {
  const leaves = inputLeaves(inputs);
  const programs = evals.map(code => normalizeProgram(code, inputs, leaves)).filter(Boolean);
  if (!programs.length) return 'answer-only';
  return hexDigest(programs.join('\n;;\n')).slice(0, 24);
}
