/**
 * Normalized eval programs (§5.1): the same approach written for different inputs should normalize to the same text.
 * Literals equal to an input value become that input's path (`$in.request`), declared locals are renamed in order
 * (`v0`, `v1`, ...), and whitespace and comments go. Token-based, so it is cheap enough to run when a call is recorded.
 */
import ts from 'typescript';
import { hexDigest } from '../native/hash.js';

/** Leaf values of the inputs by value, with the path that reaches each (the shortest when a value repeats). */
export function inputLeaves(inputs: Record<string, unknown>): Map<string, string> {
  const leaves = new Map<string, string>();
  const visit = (value: unknown, path: string, depth: number) => {
    if (depth > 6) return;
    if (typeof value === 'string' || typeof value === 'number') {
      const key = `${typeof value}:${value}`;
      const known = leaves.get(key);
      if (!known || path.length < known.length) leaves.set(key, path);
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

/** One eval program in normalized form. */
export function normalizeProgram(code: string, inputs: Record<string, unknown> = {}, leaves = inputLeaves(inputs)): string {
  const scanner = ts.createScanner(ts.ScriptTarget.ES2022, true, ts.LanguageVariant.Standard, code);
  const renamed = new Map<string, string>();
  const out: string[] = [];
  let previous: ts.SyntaxKind | undefined;
  for (let kind = scanner.scan(); kind !== ts.SyntaxKind.EndOfFileToken; kind = scanner.scan()) {
    let text = scanner.getTokenText();
    if (kind === ts.SyntaxKind.Identifier) {
      if (previous !== undefined && DECLARING.has(previous) && !renamed.has(text)) renamed.set(text, `v${renamed.size}`);
      const name = renamed.get(text);
      if (name && previous !== ts.SyntaxKind.DotToken) text = name;
    } else if (kind === ts.SyntaxKind.StringLiteral || kind === ts.SyntaxKind.NoSubstitutionTemplateLiteral) {
      const path = leaves.get(`string:${scanner.getTokenValue()}`);
      if (path) text = `$in.${path}`;
    } else if (kind === ts.SyntaxKind.NumericLiteral) {
      const path = leaves.get(`number:${Number(scanner.getTokenValue())}`);
      if (path && Number(scanner.getTokenValue()) !== 0 && Number(scanner.getTokenValue()) !== 1) text = `$in.${path}`;
    } else if (kind === ts.SyntaxKind.SemicolonToken) {
      if (previous === ts.SyntaxKind.SemicolonToken) continue;
    }
    out.push(text);
    previous = kind;
  }
  while (out.at(-1) === ';') out.pop();
  return out.join(' ');
}

/** The approach a successful agent call took: its successful evals, normalized. `answer-only` when it ran none. */
export function approachHash(evals: readonly string[], inputs: Record<string, unknown>): string {
  if (!evals.length) return 'answer-only';
  const leaves = inputLeaves(inputs);
  return hexDigest(evals.map(code => normalizeProgram(code, inputs, leaves)).join('\n;;\n')).slice(0, 24);
}
