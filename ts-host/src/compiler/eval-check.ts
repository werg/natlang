/**
 * Type-checked analysis of one lambda `eval` snippet: the same plan machinery as project builds,
 * over a virtual program whose declarations describe the lambda's typed scope.
 */
import ts from 'typescript';
import { createVirtualProgram, EVAL_COMPILER_OPTIONS } from './host.js';
import { analyzeInlineLambdas, type InlineLambdaPlan, type NatlangDiagnostic } from './inline.js';

export type EvalImport = { name: string; params: { name: string; type: string; optional?: boolean }[];
  returns: string; async: boolean; kind: 'natural language' | 'TypeScript' | 'directory reducer' | 'module';
  children: EvalImport[] };

export type EvalScopeDeclarations = {
  types: Record<string, string>;
  inputs: { name: string; type: string }[];
  locals: { name: string; type: string; mutable: boolean }[];
  captures: { name: string; type: string; mutable: boolean }[];
  imports: EvalImport[];
  services?: string[];
  result?: string;
  opaque?: string[];
};

export const EVAL_WRAPPER_PREFIX = 'async function __natlang_scope() {\n';
const SCOPE_FILE = '/__natlang__/eval/scope.ts';
const SNIPPET_FILE = '/__natlang__/eval/snippet.ts';

/** Natlang type text to TypeScript declaration text. `Live<"T", ...>` keeps its TypeScript name when resolvable. */
export function typeScriptText(natlang: string, known: ReadonlySet<string>): string {
  return natlang.replace(/Live<\s*"((?:[^"\\]|\\.)*)"(?:\s*,\s*"(?:[^"\\]|\\.)*")*\s*>/g, (_, text: string) => {
    const name = /^[A-Za-z_$][\w$]*/.exec(text)?.[0];
    return name && (known.has(name) || GLOBAL_TYPES.has(name)) ? text : 'any';
  });
}
const GLOBAL_TYPES = new Set(['Date', 'Map', 'Set', 'WeakMap', 'WeakSet', 'RegExp', 'Error', 'Promise', 'Uint8Array',
  'ArrayBuffer', 'Folder', 'FileHandle', 'IterationTrajectory']);

function importType(item: EvalImport, known: ReadonlySet<string>): string {
  const params = item.params.map(parameter => `${parameter.name}${parameter.optional ? '?' : ''}: ${typeScriptText(parameter.type, known)}`);
  const returns = typeScriptText(item.returns, known);
  const children = item.children.map(child => `readonly ${child.name}: ${importType(child, known)}`);
  const members = children.length ? ` & { ${children.join('; ')} }` : '';
  if (item.kind === 'module') return `{ ${children.join('; ')} }`;
  if (item.kind === 'TypeScript' && !item.async) return `((${params.join(', ')}) => ${returns})${members}`;
  return `NatlangFunction<[${params.join(', ')}], ${returns}>${members}`;
}

/** Declarations for the virtual scope file. */
export function scopeDeclarations(scope: EvalScopeDeclarations): string {
  const known = new Set(Object.keys(scope.types));
  const lines: string[] = [];
  for (const [name, text] of Object.entries(scope.types)) lines.push(`type ${name} = ${typeScriptText(text, known)};`);
  for (const input of scope.inputs) lines.push(`declare let ${input.name}: ${typeScriptText(input.type, known)};`);
  for (const local of scope.locals) lines.push(`declare ${local.mutable ? 'let' : 'const'} ${local.name}: ${typeScriptText(local.type, known)};`);
  for (const capture of scope.captures) lines.push(`declare ${capture.mutable ? 'let' : 'const'} ${capture.name}: ${typeScriptText(capture.type, known)};`);
  for (const item of scope.imports) lines.push(`declare const ${item.name}: ${importType(item, known)};`);
  for (const name of scope.services ?? []) lines.push(`declare const ${name}: any;`);
  for (const name of scope.opaque ?? []) lines.push(`declare const ${name}: any;`);
  if (scope.result) lines.push(`declare let result: ${typeScriptText(scope.result, known)};`);
  return lines.join('\n') + '\n';
}

/** Cheap test for whether a snippet needs the checked pass. */
export const needsEvalCheck = (source: string) => /\bnl\s*(?:<[^`]*>)?\s*`|\biterateOn\b/.test(source);

/**
 * Analyze `nl` expressions in an eval snippet. Spans in the returned plans and diagnostics are
 * relative to the snippet text.
 */
export function analyzeEvalSnippet(source: string, scope: EvalScopeDeclarations): { plans: InlineLambdaPlan[];
  diagnostics: NatlangDiagnostic[] } {
  const program = createVirtualProgram({ [SCOPE_FILE]: scopeDeclarations(scope), [SNIPPET_FILE]: `${EVAL_WRAPPER_PREFIX}${source}\n}\n` },
    EVAL_COMPILER_OPTIONS);
  const snippet = program.getSourceFile(SNIPPET_FILE)!;
  const scopeFile = program.getSourceFile(SCOPE_FILE)!;
  const inputs = new Set(scope.inputs.map(input => input.name));
  const { plans, diagnostics } = analyzeInlineLambdas(program, [snippet], {
    scopeFiles: [scopeFile], displayPath: () => 'eval',
    classify: declaration => declaration.getSourceFile() === scopeFile ?
      (ts.isVariableDeclaration(declaration) && ts.isIdentifier(declaration.name) && inputs.has(declaration.name.text) ? 'input' : 'local') :
      isSnippetTopLevel(declaration) ? 'local' : 'block',
  });
  const offset = EVAL_WRAPPER_PREFIX.length;
  const shift = <T extends { start: number; end: number; line: number }>(item: T): T =>
    ({ ...item, start: item.start - offset, end: item.end - offset, line: Math.max(1, item.line - 1) });
  return { plans: plans.map(plan => ({ ...plan, sourceSpan: shift(plan.sourceSpan),
    captures: plan.captures.map(capture => ({ ...capture, mentionSpan: capture.mentionSpan - offset })) })),
    diagnostics: diagnostics.map(shift) };
}

function isSnippetTopLevel(declaration: ts.Declaration): boolean {
  let current: ts.Node = declaration;
  while (current.parent) {
    const parent: ts.Node = current.parent;
    if (ts.isBlock(parent) && parent.parent && ts.isFunctionDeclaration(parent.parent) && parent.parent.name?.text === '__natlang_scope')
      return true;
    if (ts.isBlock(parent) || ts.isFunctionLike(parent)) return false;
    current = parent;
  }
  return false;
}

export const EVAL_OPTIONS = EVAL_COMPILER_OPTIONS;
