/**
 * Contracts and termination measures of `.nl` functions (plans/semantic-fp/01_COMPILER_AND_TYPE_SYSTEM.md L7).
 *
 * `ensures:` lists postconditions that relate a call's result to its arguments: `check:` clauses are TypeScript
 * expressions over `result`, the arguments and the companion folder's items, run first on read-only copies; `predicate:`
 * clauses are natural-language properties judged by the refinement judge with the named arguments as context. A failure
 * is a refinement failure of the return value, so the executing call repairs it within the refinement repair budget.
 * `measure:` is a TypeScript expression over the arguments whose value (a non-negative whole number) must fall at each
 * re-entry of a recursive function.
 *
 * Both kinds of expression are compiled as one synthetic callable-folder module (runtime/modules.ts: the same lowering,
 * guards and finite iteration as any helper), with the folder's items passed in by name.
 */
import ts from 'typescript';
import type { ItemRecord, ModuleRecord } from './loader.js';

export type EnsuresClause =
  | { check: string; message?: string }
  | { predicate: string; against?: string[]; message?: string };

/** Names an expression may use without declaring them: the realm's standard globals. */
const GLOBALS = new Set(['Math', 'JSON', 'Array', 'Object', 'String', 'Number', 'Boolean', 'Set', 'Map', 'Date', 'RegExp',
  'Infinity', 'NaN', 'undefined', 'isNaN', 'isFinite', 'parseInt', 'parseFloat', 'Intl', 'BigInt', 'Symbol', 'Promise',
  'Error', 'encodeURIComponent', 'decodeURIComponent', 'structuredClone', 'globalThis']);

/** The syntax error of `expression` as a TypeScript expression, or undefined when it is one. */
export function expressionError(expression: string): string | undefined {
  const file = ts.createSourceFile('expression.ts', `(${expression}\n);`, ts.ScriptTarget.ES2022, true);
  const diagnostics = (file as unknown as { parseDiagnostics?: ts.Diagnostic[] }).parseDiagnostics ?? [];
  if (diagnostics.length) return ts.flattenDiagnosticMessageText(diagnostics[0]!.messageText, ' ');
  const statement = file.statements[0];
  if (file.statements.length !== 1 || !statement || !ts.isExpressionStatement(statement)) return 'expected one expression';
  return undefined;
}

/** The free identifiers of a TypeScript expression (names it reads that it does not bind itself). */
export function freeNames(expression: string): string[] {
  const file = ts.createSourceFile('expression.ts', `(${expression}\n);`, ts.ScriptTarget.ES2022, true);
  const bound = new Set<string>(), used = new Set<string>();
  const bind = (name: ts.BindingName) => {
    if (ts.isIdentifier(name)) bound.add(name.text);
    else for (const element of name.elements) if (!ts.isOmittedExpression(element)) bind(element.name);
  };
  const visit = (node: ts.Node): void => {
    if (ts.isParameter(node) || ts.isVariableDeclaration(node) || ts.isBindingElement(node)) bind(node.name);
    if (ts.isIdentifier(node)) {
      const parent = node.parent;
      const property = (ts.isPropertyAccessExpression(parent) && parent.name === node) ||
        ((ts.isPropertyAssignment(parent) || ts.isMethodDeclaration(parent) || ts.isPropertyDeclaration(parent)) && parent.name === node) ||
        (ts.isBindingElement(parent) && parent.propertyName === node) || ts.isTypeReferenceNode(parent) ||
        ts.isQualifiedName(parent) || ts.isTypeAliasDeclaration(parent);
      if (!property) used.add(node.text);
    }
    ts.forEachChild(node, visit);
  };
  visit(file);
  return [...used].filter(name => !bound.has(name));
}

/**
 * The problem with a contract or measure expression of `name` whose parameters are `params`: a name it uses that is not
 * `result` (contracts only), an argument, an item of the companion folder or a standard global. Undefined when fine.
 */
export function unknownName(expression: string, params: readonly string[], items: readonly string[], withResult: boolean): string | undefined {
  const known = new Set([...params, ...items, ...(withResult ? ['result'] : [])]);
  return freeNames(expression).find(name => !known.has(name) && !GLOBALS.has(name));
}

/** The companion-folder items a `check:` clause names: they judge the call, so its edits cannot change them. */
export function checkItems(clauses: readonly EnsuresClause[] | undefined, params: readonly string[], codebase: Record<string, unknown>): string[] {
  const found = new Set<string>();
  for (const clause of clauses ?? []) if ('check' in clause)
    for (const name of freeNames(clause.check)) if (!params.includes(name) && Object.hasOwn(codebase, name)) found.add(name);
  return [...found];
}

/** Validate an `ensures:` frontmatter value against the function's parameter names; returns the clauses or throws. */
export function parseEnsures(value: unknown, params: readonly string[]): EnsuresClause[] {
  const shape = 'ensures lists postconditions, each { check: "<TypeScript over result and the arguments>", message: "<what must hold>" } ' +
    'or { predicate: "<property of the result>", against: [argument names] }';
  if (!Array.isArray(value) || !value.length) throw new Error(shape);
  return value.map(item => {
    if (!item || typeof item !== 'object' || Array.isArray(item)) throw new Error(shape);
    const entry = item as Record<string, unknown>;
    const keys = Object.keys(entry);
    if (typeof entry.message !== 'undefined' && (typeof entry.message !== 'string' || !entry.message.trim())) throw new Error(shape);
    const message = typeof entry.message === 'string' ? { message: entry.message.trim() } : {};
    if (typeof entry.check === 'string' && entry.predicate === undefined && keys.every(key => key === 'check' || key === 'message')) {
      const problem = expressionError(entry.check);
      if (problem) throw new Error(`ensures check ${JSON.stringify(entry.check)} is not a TypeScript expression: ${problem}`);
      return { check: entry.check.trim(), ...message };
    }
    if (typeof entry.predicate === 'string' && entry.predicate.trim() && entry.check === undefined &&
        keys.every(key => key === 'predicate' || key === 'against' || key === 'message')) {
      if (entry.against !== undefined && (!Array.isArray(entry.against) || entry.against.some(name => typeof name !== 'string')))
        throw new Error(shape);
      const against = (entry.against as string[] | undefined) ?? [];
      const missing = against.find(name => !params.includes(name));
      if (missing) throw new Error(`ensures predicate names ${JSON.stringify(missing)} in against, which is not an argument; list argument names, from: ${params.join(', ') || '(none)'}`);
      return { predicate: entry.predicate.trim(), ...(against.length ? { against: [...new Set(against)] } : {}), ...message };
    }
    throw new Error(shape);
  });
}

/**
 * The synthetic module of a function's contract checks and measure: `check_<i>(result, ...args, items)` and
 * `measure(...args, items)`, each returning its expression's value. `items` holds the companion folder's values.
 */
export function expressionModule(owner: { id: string; name: string; source?: string; revision?: string; types: Record<string, string>;
  params: readonly string[] }, clauses: readonly EnsuresClause[] | undefined, measure: string | undefined, items: readonly string[]): ModuleRecord {
  const params = owner.params.join(', ');
  const bring = items.filter(name => !owner.params.includes(name) && name !== 'result');
  const scope = bring.length ? `const { ${bring.join(', ')} } = __items; ` : '';
  const lines: string[] = [];
  (clauses ?? []).forEach((clause, index) => {
    if ('check' in clause) lines.push(`export async function check_${index}(result: any${params ? ', ' : ''}${owner.params.map(name => `${name}: any`).join(', ')}, __items: any): Promise<unknown> { ${scope}return (${clause.check}\n); }`);
  });
  if (measure) lines.push(`export async function measure(${owner.params.map(name => `${name}: any`).join(', ')}${params ? ', ' : ''}__items: any): Promise<unknown> { ${scope}return (${measure}\n); }`);
  const text = lines.join('\n') + '\n';
  const source = `${owner.source ?? owner.id}#contracts`;
  return { kind: 'module', id: `${owner.id}#contracts`, name: `${owner.name}__contracts`, source, revision: owner.revision ?? '',
    text, types: owner.types, exports: {}, imports: [], codebase: {} } as ModuleRecord;
}

/** A read-only copy of plain data (arrays and plain objects frozen all the way down); anything else as it is. */
export function readOnly<T>(value: T, depth = 0): T {
  if (!value || typeof value !== 'object' || depth > 64) return value;
  const prototype = Object.getPrototypeOf(value);
  if (Array.isArray(value)) return Object.freeze(value.map(item => readOnly(item, depth + 1))) as T;
  if (prototype !== null && Object.getPrototypeOf(prototype) !== null) return value;
  return Object.freeze(Object.fromEntries(Object.entries(value).map(([key, item]) => [key, readOnly(item, depth + 1)]))) as T;
}

export type { ItemRecord };
