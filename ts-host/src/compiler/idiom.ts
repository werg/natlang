/**
 * The idiom floor of `natlang check` (plans/semantic-fp/03_SEMANTIC_STDLIB_AND_PATTERNS.md §3.1): warnings for
 * non-idiomatic forms that can be recognised exactly, without a model. Whether prose in a body is judgment or
 * mechanism is a judgment, so it belongs to the natural-language reviewer behind `natlang check --idiom` (WS3 I2),
 * not here. Each warning is one sentence that says the idiomatic form. There is no suppression syntax: a warning that
 * is wrong for a site is a defect of this file.
 *
 * - `idiom-decision-readout`: a leaf `.nl` function (no companion folder, no `uses:`, no `Folder` parameter, not a
 *   directory reducer, no representation-generic result) whose `returns` type has at least two finite values
 *   (`finiteValues`) and that declares no `readout:`.
 * - `idiom-code-in-prose`: a `.nl` body that contains one of `JSON.parse(`, `.slice(`, `Math.`, `eval({`, `=>` outside
 *   a fenced block. One warning per file, at the first such token.
 * - `idiom-fixed-cursor-iterate`: an `iterateOn(step, initial)` or `step.iterateOn(initial)` whose step is a function
 *   in the same file that advances exactly one state field by `+ 1` (reading a fixed array only at that index) or by
 *   `.slice(1)` (reading that field only at `[0]` and `.length`), and whose other state fields are never read except
 *   to carry them unchanged or to append to them (`[...s.f, x]`, `s.f.concat(x)`, `{ ...s.f, k: x }`, `s.f + x`), with
 *   `x` independent of the state. The `until` predicate must read only the cursor, so no iteration stops early on a
 *   result. Then every step is independent of the previous one and the iteration is a map. Steps that write outer
 *   variables, pass the whole state on, or read their own results are left alone.
 * - `untrusted-instruction` (an `Untrusted<T>` value interpolated into `nl` text) completes the floor; it is the
 *   existing error of `compiler/inline.ts`, not repeated here.
 */
import ts from 'typescript';
import type { NatlangDiagnostic } from './inline.js';
import type { ItemRecord, NatlangRecord } from '../runtime/loader.js';
import { decisionDomain } from '../native/decision.js';
import { parseTypeCached, TypeEnv } from '../native/types.js';

export const IDIOM_DECISION_READOUT = 'idiom-decision-readout';
export const IDIOM_FIXED_CURSOR_ITERATE = 'idiom-fixed-cursor-iterate';
export const IDIOM_CODE_IN_PROSE = 'idiom-code-in-prose';
export const IDIOM_CODES = [IDIOM_DECISION_READOUT, IDIOM_FIXED_CURSOR_ITERATE, IDIOM_CODE_IN_PROSE] as const;

/** Tokens that only executable code contains (§3.1), in the order they are looked for on a line. */
export const CODE_IN_PROSE_TOKENS = ['JSON.parse(', '.slice(', 'Math.', 'eval({', '=>'] as const;

const warning = (file: string, line: number, column: number, code: string, message: string): NatlangDiagnostic =>
  ({ file, start: 0, end: 0, line, column, code: code as NatlangDiagnostic['code'], severity: 'warning', message });

const baseName = (path: string) => path.split(/[\\/]/).pop() ?? path;

/** Frontmatter end (index of the closing `---` line) of a `.nl` text, or -1 when it has none. */
function frontmatterEnd(lines: readonly string[]): number {
  return lines[0]?.trim() === '---' ? lines.findIndex((line, index) => index > 0 && line.trim() === '---') : -1;
}

/** The idiom warnings of one `.nl` function, reported at `file` (its display path). */
export function checkNlIdioms(record: NatlangRecord, file: string): NatlangDiagnostic[] {
  const out: NatlangDiagnostic[] = [];
  const lines = record.text.split(/\r?\n/);
  const end = frontmatterEnd(lines);
  const leaf = record.subtype === 'function' && !record.generic && Object.keys(record.codebase).length === 0 &&
    !record.uses?.length && !Object.values(record.args).some(type => /(?<![\w$])Folder(?![\w$])/.test(type));
  if (leaf && record.readout === undefined) {
    let size = 0;
    try {
      const env = new TypeEnv(Object.fromEntries(Object.entries(record.types).map(([name, text]) => [name, parseTypeCached(text)])));
      size = decisionDomain(parseTypeCached(record.returns), env)?.size ?? 0;
    } catch { size = 0; }
    if (size >= 2) {
      const at = lines.findIndex((line, index) => index > 0 && (end < 0 || index < end) && line.startsWith('returns:'));
      out.push(warning(file, at < 0 ? 1 : at + 1, 1, IDIOM_DECISION_READOUT,
        `\`${baseName(file)}\` returns one of ${size} values: add \`readout: decision\` so the model reasons and scores them, ` +
        'or keep the tool loop if it must call helpers first.'));
    }
  }
  let fence: string | undefined;
  for (let index = end < 0 ? 0 : end + 1; index < lines.length; index++) {
    const line = lines[index]!;
    const marker = /^\s*(`{3,}|~{3,})/.exec(line)?.[1];
    if (marker) {
      if (!fence) fence = marker[0]!.repeat(marker.length);
      else if (marker[0] === fence[0] && marker.length >= fence.length) fence = undefined;
      continue;
    }
    if (fence) continue;
    const found = CODE_IN_PROSE_TOKENS.map(token => ({ token, column: line.indexOf(token) }))
      .filter(item => item.column >= 0).sort((a, b) => a.column - b.column)[0];
    if (!found) continue;
    const folder = file.replace(/\.nl$/, '');
    out.push(warning(file, index + 1, found.column + 1, IDIOM_CODE_IN_PROSE,
      `\`${baseName(file)}\` describes code (\`${found.token}\`) in its instructions: write that step as a TypeScript helper in ` +
      `\`${folder}/\` and call it from the function.`));
    break;
  }
  return out;
}

/**
 * The idiom warnings of every `.nl` function in `levels` (named functions, callable folders and their companion
 * folders), each once. `display` maps a record's source to the path reported.
 */
export function checkNlIdiomsIn(levels: Iterable<Record<string, ItemRecord>>, display: (source: string) => string): NatlangDiagnostic[] {
  const seen = new Set<string>();
  const out: NatlangDiagnostic[] = [];
  const visit = (level: Record<string, ItemRecord>) => {
    for (const item of Object.values(level)) {
      if (item.kind === 'natlang' && !seen.has(item.source)) {
        seen.add(item.source);
        out.push(...checkNlIdioms(item, display(item.source)));
      }
      if (item.kind !== 'natlang' || !item.uses?.length) visit(item.codebase);
      else visit(Object.fromEntries(Object.entries(item.codebase).filter(([name]) => !item.uses!.some(path => baseName(path) === name))));
    }
  };
  for (const level of levels) visit(level);
  return out;
}

const unwrap = (node: ts.Expression): ts.Expression => {
  while (ts.isParenthesizedExpression(node) || ts.isAsExpression(node) || ts.isNonNullExpression(node) ||
    ts.isSatisfiesExpression(node) || ts.isTypeAssertionExpression(node)) node = node.expression;
  return node;
};

type StepFunction = ts.ArrowFunction | ts.FunctionExpression | ts.FunctionDeclaration;

/** The function a step expression names: itself, or the one `const name = …` / `function name` in the file. */
function stepFunction(file: ts.SourceFile, expression: ts.Expression): StepFunction | undefined {
  const node = unwrap(expression);
  if (ts.isArrowFunction(node) || ts.isFunctionExpression(node)) return node;
  if (!ts.isIdentifier(node)) return undefined;
  const found: StepFunction[] = [];
  let other = false;
  const visit = (child: ts.Node): void => {
    if (ts.isFunctionDeclaration(child) && child.name?.text === node.text) found.push(child);
    else if (ts.isVariableDeclaration(child) && ts.isIdentifier(child.name) && child.name.text === node.text) {
      const init = child.initializer && unwrap(child.initializer);
      if (init && (ts.isArrowFunction(init) || ts.isFunctionExpression(init)) &&
        ts.isVariableDeclarationList(child.parent) && child.parent.flags & ts.NodeFlags.Const) found.push(init);
      else other = true;
    } else if (ts.isParameter(child) && ts.isIdentifier(child.name) && child.name.text === node.text) other = true;
    ts.forEachChild(child, visit);
  };
  visit(file);
  return !other && found.length === 1 ? found[0] : undefined;
}

/**
 * How a function reads its state parameter: `field(node)` names the state field a node reads (`s.f`, or a name bound
 * by destructuring the parameter), and `escapes` says the state is used whole somewhere.
 */
function stateReader(fn: ts.ArrowFunction | ts.FunctionExpression | ts.FunctionDeclaration) {
  const parameter = fn.parameters[0];
  if (!parameter || parameter.dotDotDotToken) return undefined;
  const name = parameter.name;
  const bound = new Map<string, string>();
  let whole: string | undefined;
  if (ts.isIdentifier(name)) whole = name.text;
  else if (ts.isObjectBindingPattern(name)) {
    for (const element of name.elements) {
      if (element.dotDotDotToken || !ts.isIdentifier(element.name) || element.initializer) return undefined;
      const property = element.propertyName ? (ts.isIdentifier(element.propertyName) ? element.propertyName.text : undefined) : element.name.text;
      if (!property) return undefined;
      bound.set(element.name.text, property);
    }
  } else return undefined;
  const isValueIdentifier = (node: ts.Identifier) => {
    const parent = node.parent;
    if (ts.isPropertyAccessExpression(parent) && parent.name === node) return false;
    if ((ts.isPropertyAssignment(parent) || ts.isPropertySignature(parent) || ts.isBindingElement(parent)) && parent.name === node) return false;
    if (ts.isParameter(parent) && parent.name === node) return false;
    return true;
  };
  /** The field this node reads, or undefined. */
  const field = (node: ts.Node): string | undefined => {
    if (ts.isExpression(node)) node = unwrap(node);
    if (whole && ts.isPropertyAccessExpression(node) && ts.isIdentifier(unwrap(node.expression)) &&
      (unwrap(node.expression) as ts.Identifier).text === whole) return node.name.text;
    if (ts.isIdentifier(node) && bound.has(node.text) && isValueIdentifier(node)) return bound.get(node.text);
    return undefined;
  };
  /** A bare use of the whole state, other than `s.f` and a spread in a returned object. */
  const isWhole = (node: ts.Node) => !!whole && ts.isIdentifier(node) && node.text === whole && isValueIdentifier(node) &&
    !(ts.isPropertyAccessExpression(node.parent) && node.parent.expression === node);
  return { field, isWhole, whole };
}

/** The returned expressions of a function: its expression body or every `return` outside nested functions. */
function returnedExpressions(fn: StepFunction): ts.Expression[] | undefined {
  if (!fn.body) return undefined;
  if (!ts.isBlock(fn.body)) return [fn.body];
  const out: ts.Expression[] = [];
  let bare = false;
  const visit = (node: ts.Node): void => {
    if (ts.isFunctionLike(node)) return;
    if (ts.isReturnStatement(node)) { if (node.expression) out.push(node.expression); else bare = true; }
    ts.forEachChild(node, visit);
  };
  visit(fn.body);
  return bare || !out.length ? undefined : out;
}

const isOne = (node: ts.Expression) => { node = unwrap(node); return ts.isNumericLiteral(node) && node.text === '1'; };

/** Whether `iterateOn(step, …)` with this step and `until` predicate is a fixed-cursor walk over a list. */
function fixedCursorStep(file: ts.SourceFile, fn: StepFunction, until: ts.Expression | undefined): boolean {
  const reader = stateReader(fn);
  const returned = returnedExpressions(fn);
  if (!reader || !returned || !fn.body) return false;
  if (fn.asteriskToken) return false;
  // Each return is an object literal: the next state, made of `...s`, unchanged fields, the cursor and appends.
  const objects = returned.map(expression => unwrap(ts.isAwaitExpression(unwrap(expression)) ? (unwrap(expression) as ts.AwaitExpression).expression : expression));
  if (!objects.every(ts.isObjectLiteralExpression)) return false;
  let cursor: { name: string; kind: 'index' | 'queue' } | undefined;
  const cursorUpdates = new Set<ts.Node>();
  const accumulations = new Map<ts.Node, string>();
  const unchanged = new Set<ts.Node>();
  const declaredFields = new Map<string, number>();
  for (const object of objects as ts.ObjectLiteralExpression[]) {
    for (const property of object.properties) {
      if (ts.isSpreadAssignment(property)) {
        const spread = unwrap(property.expression);
        if (reader.whole && ts.isIdentifier(spread) && spread.text === reader.whole) { unchanged.add(spread); continue; }
        return false;
      }
      if (!ts.isPropertyAssignment(property) && !ts.isShorthandPropertyAssignment(property)) return false;
      if (!ts.isIdentifier(property.name) && !ts.isStringLiteral(property.name)) return false;
      const key = property.name.text;
      declaredFields.set(key, (declaredFields.get(key) ?? 0) + 1);
      const value = ts.isPropertyAssignment(property) ? unwrap(property.initializer) : property.name;
      // The cursor: `s.i + 1` or `s.queue.slice(1)`.
      if (ts.isBinaryExpression(value) && value.operatorToken.kind === ts.SyntaxKind.PlusToken &&
        ((reader.field(value.left) === key && isOne(value.right)) || (reader.field(value.right) === key && isOne(value.left)))) {
        if (cursor && (cursor.name !== key || cursor.kind !== 'index')) return false;
        cursor = { name: key, kind: 'index' }; cursorUpdates.add(value); continue;
      }
      if (ts.isCallExpression(value) && ts.isPropertyAccessExpression(value.expression) && value.expression.name.text === 'slice' &&
        value.arguments.length === 1 && isOne(value.arguments[0]!) && reader.field(value.expression.expression) === key) {
        if (cursor && (cursor.name !== key || cursor.kind !== 'queue')) return false;
        cursor = { name: key, kind: 'queue' }; cursorUpdates.add(value); continue;
      }
      if (reader.field(value) === key) { unchanged.add(value); continue; }
      // Appends: `[...s.f, x]`, `s.f.concat(x)`, `{ ...s.f, k: x }`, `s.f + x`.
      if (ts.isArrayLiteralExpression(value) && value.elements.length >= 1 && ts.isSpreadElement(value.elements[0]!) &&
        reader.field(value.elements[0]!.expression) === key) { accumulations.set(value.elements[0]!.expression, key); continue; }
      if (ts.isObjectLiteralExpression(value) && value.properties.length >= 1 && ts.isSpreadAssignment(value.properties[0]!) &&
        reader.field(value.properties[0]!.expression) === key) { accumulations.set(value.properties[0]!.expression, key); continue; }
      if (ts.isCallExpression(value) && ts.isPropertyAccessExpression(value.expression) && value.expression.name.text === 'concat' &&
        reader.field(value.expression.expression) === key) { accumulations.set(value.expression.expression, key); continue; }
      if (ts.isBinaryExpression(value) && value.operatorToken.kind === ts.SyntaxKind.PlusToken && reader.field(value.left) === key) {
        accumulations.set(value.left, key); continue;
      }
      // Any other value is a new state field computed from the item; whether it reads the state is checked below.
    }
  }
  if (!cursor) return false;
  // Every return sets the cursor (one advance per step).
  if ((declaredFields.get(cursor.name) ?? 0) !== objects.length || cursorUpdates.size !== objects.length) return false;
  // The list: an index cursor reads one fixed expression at the cursor; a queue cursor reads its own `[0]`.
  let list: string | undefined;
  const listRoots = new Set<string>();
  let ok = true;
  const declaredInside = new Set<string>();
  const collectDeclared = (node: ts.Node): void => {
    if ((ts.isVariableDeclaration(node) || ts.isParameter(node) || ts.isBindingElement(node)) && ts.isIdentifier(node.name)) declaredInside.add(node.name.text);
    if ((ts.isFunctionDeclaration(node) || ts.isFunctionExpression(node)) && node.name && node !== fn) declaredInside.add(node.name.text);
    ts.forEachChild(node, collectDeclared);
  };
  collectDeclared(fn);
  const listAt = new Set<ts.Node>();
  const visit = (node: ts.Node): void => {
    if (!ok) return;
    // Writes to names declared outside the step are effects whose order a map would change.
    if (ts.isBinaryExpression(node) && node.operatorToken.kind >= ts.SyntaxKind.FirstAssignment && node.operatorToken.kind <= ts.SyntaxKind.LastAssignment) {
      let target = unwrap(node.left);
      while (ts.isPropertyAccessExpression(target) || ts.isElementAccessExpression(target)) target = unwrap(target.expression);
      if (!ts.isIdentifier(target) || !declaredInside.has(target.text)) { ok = false; return; }
    }
    if ((ts.isPrefixUnaryExpression(node) || ts.isPostfixUnaryExpression(node)) &&
      (node.operator === ts.SyntaxKind.PlusPlusToken || node.operator === ts.SyntaxKind.MinusMinusToken)) {
      const target = unwrap(node.operand);
      if (!ts.isIdentifier(target) || !declaredInside.has(target.text)) { ok = false; return; }
    }
    if (reader.isWhole(node)) { if (!unchanged.has(node)) ok = false; return; }
    const read = reader.field(node);
    if (read !== undefined) {
      if (read === cursor!.name) {
        if (cursor!.kind === 'index') {
          const parent = node.parent;
          if (ts.isElementAccessExpression(parent) && unwrap(parent.argumentExpression) === unwrap(node as ts.Expression)) {
            const text = unwrap(parent.expression).getText(file);
            if (list !== undefined && list !== text) { ok = false; return; }
            list = text; listAt.add(parent.expression);
            const root = rootIdentifier(parent.expression);
            if (root) listRoots.add(root);
          }
          return;
        }
        // A queue: read only at [0], .length, and in its own `.slice(1)`.
        const parent = node.parent;
        if (ts.isElementAccessExpression(parent) && parent.expression === node && ts.isNumericLiteral(unwrap(parent.argumentExpression)) &&
          (unwrap(parent.argumentExpression) as ts.NumericLiteral).text === '0') return;
        if (ts.isPropertyAccessExpression(parent) && parent.expression === node && parent.name.text === 'length') return;
        if (ts.isPropertyAccessExpression(parent) && parent.expression === node && parent.name.text === 'slice' &&
          cursorUpdates.has(parent.parent)) return;
        ok = false; return;
      }
      if (unchanged.has(unwrap(node as ts.Expression)) || accumulations.get(node) === read) return;
      ok = false; return;
    }
    ts.forEachChild(node, visit);
  };
  visit(fn.body);
  if (!ok) return false;
  if (cursor.kind === 'index') {
    if (list === undefined) return false;
    // The list is fixed: the step reads it only at the cursor (or its length) and never changes it.
    const fixed = (node: ts.Node): boolean => {
      if (ts.isExpression(node) && !listAt.has(node) && unwrap(node).getText(file) === list) {
        const parent = node.parent;
        if (ts.isPropertyAccessExpression(parent) && parent.expression === node && parent.name.text === 'length') return true;
        return false;
      }
      return ts.forEachChild(node, child => fixed(child) ? undefined : true) === undefined;
    };
    if (!fixed(fn.body)) return false;
    // A fixed list is a name or a property path declared outside the step, read the same way every step.
    if (!/^[A-Za-z_$][\w$]*(?:\??\.[A-Za-z_$][\w$]*)*$/.test(list)) return false;
    if ([...listRoots].some(root => declaredInside.has(root))) return false;
  }
  // The stop predicate reads only the cursor, so the walk stops at the end of the list, not on a result.
  if (!until) return false;
  const predicate = unwrap(until);
  if (!ts.isArrowFunction(predicate) && !ts.isFunctionExpression(predicate)) return false;
  const stop = stateReader(predicate);
  if (!stop) return false;
  let readsOther = false;
  const check = (node: ts.Node): void => {
    if (readsOther) return;
    if (stop.isWhole(node)) { readsOther = true; return; }
    const read = stop.field(node);
    if (read !== undefined && read !== cursor!.name) { readsOther = true; return; }
    if (ts.isCallExpression(node)) { readsOther = true; return; }
    ts.forEachChild(node, check);
  };
  check(predicate.body);
  return !readsOther;
}

function rootIdentifier(node: ts.Expression): string | undefined {
  node = unwrap(node);
  while (ts.isPropertyAccessExpression(node) || ts.isElementAccessExpression(node)) node = unwrap(node.expression);
  return ts.isIdentifier(node) ? node.text : undefined;
}

/** `idiom-fixed-cursor-iterate` warnings of one TypeScript source file. */
export function checkFixedCursorIterations(file: ts.SourceFile, displayPath: (file: ts.SourceFile) => string = source => source.fileName): NatlangDiagnostic[] {
  const out: NatlangDiagnostic[] = [];
  const untilOf = (call: ts.CallExpression): ts.Expression | undefined => {
    // Walk the chain `iterateOn(…).withLimit(…).withMeasure(…).until(pred)`.
    let node: ts.Node = call;
    while (ts.isPropertyAccessExpression(node.parent) && node.parent.expression === node && ts.isCallExpression(node.parent.parent) &&
      node.parent.parent.expression === node.parent) {
      const member = node.parent.name.text;
      if (member === 'until') return node.parent.parent.arguments[0];
      if (member !== 'withLimit' && member !== 'withMeasure') return undefined;
      node = node.parent.parent;
    }
    return undefined;
  };
  const visit = (node: ts.Node): void => {
    if (ts.isCallExpression(node)) {
      const callee = unwrap(node.expression);
      let step: ts.Expression | undefined;
      if (ts.isIdentifier(callee) && callee.text === 'iterateOn' && node.arguments.length >= 2) step = node.arguments[0];
      else if (ts.isPropertyAccessExpression(callee) && callee.name.text === 'iterateOn' && node.arguments.length >= 1) step = callee.expression;
      const fn = step && stepFunction(file, step);
      if (fn && fixedCursorStep(file, fn, untilOf(node))) {
        const start = node.getStart(file), position = file.getLineAndCharacterOfPosition(start);
        out.push({ file: displayPath(file), start, end: node.end, line: position.line + 1, column: position.character + 1,
          code: IDIOM_FIXED_CURSOR_ITERATE as NatlangDiagnostic['code'], severity: 'warning',
          message: 'This `iterateOn` steps through a fixed list one item at a time: map the step over the list with `Promise.all`, ' +
            'and keep `iterateOn` for steps that depend on the previous state.' });
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(file);
  return out;
}
