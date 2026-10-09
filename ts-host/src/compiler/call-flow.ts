/**
 * The data flow of calls to named natural-language functions in TypeScript code (plans/FUSED_PIPELINES.md). One analysis
 * serves three users: the fact service (crisp TypeScript orchestrators and the eval code the call store recorded) and the
 * compiler, which marks the calls that make up a hand-off so the runtime can recognize them. For each call it says how the
 * result travels: straight into another named function (`B(await A(x))`), into a variable whose every use is visible, or
 * somewhere the analysis cannot follow, and who reads the value at each use. Syntax only; nothing here runs code.
 */
import ts from 'typescript';
import type { ItemRecord, NatlangRecord } from '../runtime/loader.js';

/** Who reads a value besides (or as) the planned consumer. */
export type ReaderKind =
  /** The stage the value is passed to. */
  | 'consumer'
  /** Another stage or function the same value is passed to. */
  | 'other-call'
  /** Crisp callable-folder or host TypeScript. */
  | 'crisp-code'
  /** A host service (a verifier, an index, a store). */
  | 'service'
  /** The orchestrating model's own code or reasoning: a field access, a condition, a loop, a record it builds. */
  | 'eval'
  /** The value, or something built from it, is returned to the host. */
  | 'host-return'
  /** The value is logged, reported, displayed or shown to a user. */
  | 'trace-ui';

export type Reader = { kind: ReaderKind; name?: string; site: string; detail: string;
  /** False when the analysis could not see how the value is used and assumes the worst. */
  certain: boolean };


export type Binding = { stage: NatlangRecord; path: string };

/**
 * What the analysis knows about the code it reads. `crisp-typescript`: an orchestrator written in TypeScript, where any
 * other function that receives a value is crisp code. `eval`: code the orchestrating model wrote and ran (recorded in the
 * call store), where a call to a host service is a `service` reader and any other use is the model reading the value.
 */
export type CodeContext = { file: ts.SourceFile; fileName: string; imports: ReadonlyMap<string, ItemRecord>; mode: 'crisp-typescript' | 'eval';
  /** Line numbers are shifted by this many lines (a wrapper added above the code). */
  lineOffset: number };

function memberStage(record: ItemRecord, names: string[]): NatlangRecord | undefined {
  let current: ItemRecord = record;
  for (const name of names) {
    const child: ItemRecord | undefined = current.codebase[name];
    if (!child) return undefined;
    current = child;
  }
  return current.kind === 'natlang' ? current : undefined;
}

/** The named function a call expression invokes: `plan(...)`, `database.plan(...)`. */
export function calleeOf(call: ts.CallExpression, imports: ReadonlyMap<string, ItemRecord>): Binding | undefined {
  const names: string[] = [];
  let expression: ts.Expression = call.expression;
  while (ts.isPropertyAccessExpression(expression)) { names.unshift(expression.name.text); expression = expression.expression; }
  if (!ts.isIdentifier(expression)) return undefined;
  const root = imports.get(expression.text);
  if (!root) return undefined;
  const stage = names.length ? memberStage(root, names) : root.kind === 'natlang' ? root : undefined;
  return stage ? { stage, path: [expression.text, ...names].join('.') } : undefined;
}

export const transparent = (node: ts.Node): boolean =>
  ts.isParenthesizedExpression(node) || ts.isAwaitExpression(node) || ts.isAsExpression(node) || ts.isNonNullExpression(node);
export const unwrap = (node: ts.Node): ts.Node => {
  let current = node;
  while (transparent(current)) current = (current as ts.ParenthesizedExpression).expression;
  return current;
};

/** The call a value is an argument of, looking through await, parentheses and casts. */
export function nestedCall(node: ts.Node): ts.CallExpression | undefined {
  for (let current: ts.Node | undefined = node.parent; current; current = current.parent) {
    if (ts.isCallExpression(current)) return current;
    if (!(ts.isParenthesizedExpression(current) || ts.isAwaitExpression(current) || ts.isAsExpression(current))) return undefined;
  }
  return undefined;
}

export const argumentPosition = (call: ts.CallExpression, node: ts.Node): number =>
  call.arguments.findIndex(argument => argument === node || (node.pos >= argument.pos && node.end <= argument.end));

export const lineIn = (ctx: CodeContext, node: ts.Node): number => ctx.file.getLineAndCharacterOfPosition(node.getStart(ctx.file)).line + 1 - ctx.lineOffset;
export const siteIn = (ctx: CodeContext, node: ts.Node): string => `${ctx.fileName}:${lineIn(ctx, node)}`;

/** A use of a value in code: who reads it, and the function when it is one of the named functions. */
export type CodeUse = { reader: Reader; consumer?: Binding; position?: number;
  /** The call expression of the consumer. */
  call?: ts.CallExpression };

/** Who reads the value `node` (an identifier holding it, or the call that made it) in the place it is used. */
export function useOf(ctx: CodeContext, node: ts.Node): CodeUse {
  const site = siteIn(ctx, node);
  const call = nestedCall(node);
  const asArgument = !!call && call.arguments.some(argument => unwrap(argument) === unwrap(node));
  const callee = call && asArgument ? calleeOf(call, ctx.imports) : undefined;
  if (callee) return { reader: { kind: 'consumer', name: callee.path, site, detail: `argument of ${callee.path}`, certain: true },
    consumer: callee, position: argumentPosition(call!, node), call: call! };
  if (ts.isReturnStatement(node.parent) || (node.parent && ts.isReturnStatement(node.parent.parent ?? node.parent)))
    return { reader: { kind: 'host-return', site, detail: ctx.mode === 'eval' ? 'returned by the orchestrator' : 'returned to the caller', certain: true } };
  if (call && asArgument) {
    const text = call.expression.getText(ctx.file);
    const crisp = ctx.mode === 'crisp-typescript' || ctx.imports.get(text.split('.')[0]!)?.kind === 'module';
    return { reader: { kind: crisp ? 'crisp-code' : 'service', name: text, site, detail: `passed to ${text}`, certain: true } };
  }
  return { reader: { kind: ctx.mode === 'eval' ? 'eval' : 'crisp-code', site,
    detail: `${ctx.mode === 'eval' ? 'read by the orchestrating model' : 'read by crisp code'}: ${node.parent.getText(ctx.file).slice(0, 60)}`, certain: true } };
}

/** How the result of one call of a named function travels in the code around it. */
export type CallFlow = { form: 'nested' | 'bound' | 'loose'; variable: string | null; uses: CodeUse[]; anchor: ts.Node };

export function flowOfCall(ctx: CodeContext, node: ts.CallExpression): CallFlow {
  // B(await A(x)): the value never has a name.
  const outer = nestedCall(node);
  const parent = node.parent;
  const direct = outer && outer !== node && (ts.isAwaitExpression(parent) || ts.isParenthesizedExpression(parent) || parent === outer);
  const consumer = direct ? calleeOf(outer!, ctx.imports) : undefined;
  if (consumer && outer!.arguments.some(argument => unwrap(argument) === node))
    return { form: 'nested', variable: null, anchor: node, uses: [{ reader: { kind: 'consumer', name: consumer.path, site: `${ctx.fileName}:${lineIn(ctx, outer!)}`,
      detail: `argument of ${consumer.path}`, certain: true }, consumer, position: argumentPosition(outer!, node), call: outer! }] };
  // const v = await A(x): every reference to v in the enclosing function is a reader.
  let declaration: ts.Node | undefined = parent;
  while (declaration && transparent(declaration)) declaration = declaration.parent;
  if (declaration && ts.isVariableDeclaration(declaration) && ts.isIdentifier(declaration.name)) {
    const bound = declaration;
    const name = declaration.name.text;
    let region: ts.Node = ctx.file;
    for (let current: ts.Node | undefined = declaration.parent; current; current = current.parent)
      if (ts.isFunctionLike(current) || ts.isSourceFile(current)) { region = current; break; }
    const uses: CodeUse[] = [];
    const find = (item: ts.Node): void => {
      if (ts.isIdentifier(item) && item.text === name && item !== bound.name &&
          !(ts.isPropertyAccessExpression(item.parent) && item.parent.name === item)) uses.push(useOf(ctx, item));
      ts.forEachChild(item, find);
    };
    find(region);
    // An exported variable is read by every module that imports it.
    const statement = bound.parent?.parent;
    if (statement && ts.isVariableStatement(statement) && statement.modifiers?.some(modifier => modifier.kind === ts.SyntaxKind.ExportKeyword))
      uses.push({ reader: { kind: 'host-return', site: siteIn(ctx, bound), detail: `${name} is exported`, certain: true } });
    return { form: 'bound', variable: name, uses, anchor: declaration };
  }
  // Anything else (a statement of its own, a destructuring, part of a larger expression): the surrounding code reads it.
  return { form: 'loose', variable: null, uses: [useOf(ctx, node)], anchor: node };
}


/** Where a call is, for matching the compiler's marks to the facts: the file and the offset of the call expression. */
export const siteKey = (fileName: string, node: ts.Node, file: ts.SourceFile): string => `${fileName}@${node.getStart(file)}`;

/**
 * The calls of named functions that take part in a candidate hand-off in `file`: producers whose result goes into another
 * named function (nested, or through a variable) and the consumers it goes into. Keys are `start:end` of the call in the
 * source, values the call's site ID. Whether a hand-off is fused is decided later, at run time; marking the call changes
 * nothing while fusion is off.
 */
export function fusionCallSites(file: ts.SourceFile, modulePath: string, importedNames: ReadonlyMap<string, ItemRecord>): Map<string, string> {
  const marks = new Map<string, string>();
  if (!importedNames.size) return marks;
  const ctx: CodeContext = { file, fileName: modulePath, imports: importedNames, mode: 'crisp-typescript', lineOffset: 0 };
  const mark = (call: ts.CallExpression) => marks.set(`${call.getStart(file)}:${call.getEnd()}`, siteKey(modulePath, call, file));
  const walk = (node: ts.Node): void => {
    if (ts.isCallExpression(node) && calleeOf(node, importedNames)) {
      const flow = flowOfCall(ctx, node);
      const consumers = flow.form === 'loose' ? [] : flow.uses.filter(use => use.consumer && use.call);
      if (consumers.length) { mark(node); for (const use of consumers) mark(use.call!); }
    }
    ts.forEachChild(node, walk);
  };
  walk(file);
  return marks;
}
