import ts from 'typescript';
import { spanOf, type NatlangDiagnostic } from './inline.js';

/**
 * Source policy for callable-folder TypeScript, `natlang.d/`, and lambda `eval`.
 *
 * The checks are syntactic so they run without a type checker. Array-only `for ... of`
 * is enforced by a runtime guard that the lowering inserts; a checker, when present, only adds
 * earlier diagnostics. Ordinary application TypeScript is not subject to this policy.
 */
export type PolicyOptions = {
  displayPath?: (file: ts.SourceFile) => string;
  /** Optional checker for static array checks and cross-file call resolution. */
  checker?: ts.TypeChecker;
  /** Allow `import(...)` expressions (package imports in eval are tracked by the workspace). */
  allowDynamicImport?: boolean;
};

const COMPARATORS = new Set([ts.SyntaxKind.LessThanToken, ts.SyntaxKind.LessThanEqualsToken,
  ts.SyntaxKind.GreaterThanToken, ts.SyntaxKind.GreaterThanEqualsToken]);
const GROWING_METHODS = new Set(['push', 'unshift', 'splice', 'concat']);
const ITERATOR_SYMBOLS = new Set(['iterator', 'asyncIterator']);
/** Names of the global object, through which a global such as `setInterval` can also be reached. */
const GLOBALS = new Set(['globalThis', 'window', 'self', 'global']);
const ITERATOR_REFUSAL = 'Defining iterators is not available here; build an array, or use `iterateOn` for an open-ended sequence.';

/** An identifier that names a property or a declaration rather than referring to a binding. */
function isNameOnly(node: ts.Identifier): boolean {
  const parent = node.parent;
  return ((ts.isPropertyAccessExpression(parent) || ts.isPropertyAssignment(parent) || ts.isMethodDeclaration(parent) ||
    ts.isPropertyDeclaration(parent) || ts.isPropertySignature(parent) || ts.isMethodSignature(parent)) && parent.name === node) ||
    ((ts.isVariableDeclaration(parent) || ts.isParameter(parent) || ts.isFunctionDeclaration(parent)) && parent.name === node);
}

/**
 * What a function's recursion guard compares when the function runs again inside itself: `this` for methods and
 * function expressions, then each parameter, a destructured one as the names it binds (spec: Iteration and termination).
 */
export function guardArguments(node: ts.SignatureDeclaration): string[] {
  const names: string[] = [];
  const bind = (name: ts.BindingName): void => {
    if (ts.isIdentifier(name)) names.push(name.text);
    else for (const element of name.elements) if (!ts.isOmittedExpression(element)) bind(element.name);
  };
  if (ts.isMethodDeclaration(node) || ts.isGetAccessorDeclaration(node) || ts.isSetAccessorDeclaration(node) ||
      ts.isFunctionExpression(node)) names.push('this');
  for (const parameter of node.parameters) if (!(ts.isIdentifier(parameter.name) && parameter.name.text === 'this')) bind(parameter.name);
  return names;
}

/**
 * Whether a function's own body can call anything: a call, `new`, a tagged template, `await`, `yield` or `for await`
 * outside nested functions. A function that cannot call cannot re-enter itself through anything but implicit
 * synchronous invocations (getters, `valueOf`), whose cycles end at the engine's stack limit, so it needs no
 * recursion guard.
 */
export function makesCalls(node: ts.SignatureDeclaration): boolean {
  let found = false;
  const visit = (child: ts.Node): void => {
    if (found) return;
    if (ts.isCallExpression(child) || ts.isNewExpression(child) || ts.isTaggedTemplateExpression(child) ||
        ts.isAwaitExpression(child) || ts.isYieldExpression(child) || ts.isDecorator(child) ||
        (ts.isForOfStatement(child) && child.awaitModifier)) { found = true; return; }
    if (ts.isFunctionLike(child)) {
      // A nested function is only created here; its parameter defaults and computed names still run here.
      for (const parameter of child.parameters) if (parameter.initializer) visit(parameter.initializer);
      if (child.name && ts.isComputedPropertyName(child.name)) visit(child.name);
      return;
    }
    ts.forEachChild(child, visit);
  };
  for (const parameter of node.parameters) if (parameter.initializer) visit(parameter.initializer);
  const body = (node as ts.FunctionLikeDeclaration).body;
  if (body) visit(body);
  return found;
}

/** A stream from `iterateOn(...).streamUntil(...)`. */
export function isIterationStream(expression: ts.Expression, checker?: ts.TypeChecker): boolean {
  let current: ts.Expression = expression;
  while (ts.isParenthesizedExpression(current)) current = current.expression;
  if (checker) {
    const type = checker.getTypeAtLocation(current);
    if (type.getProperty('__natlangIterationStream')) return true;
  }
  return ts.isCallExpression(current) && ts.isPropertyAccessExpression(current.expression) &&
    current.expression.name.text === 'streamUntil';
}

function assignedIdentifiers(node: ts.Node): Set<string> {
  const names = new Set<string>();
  const visit = (child: ts.Node): void => {
    let target: ts.Expression | undefined;
    if (ts.isBinaryExpression(child) && child.operatorToken.kind >= ts.SyntaxKind.FirstAssignment &&
        child.operatorToken.kind <= ts.SyntaxKind.LastAssignment) target = child.left;
    if ((ts.isPrefixUnaryExpression(child) || ts.isPostfixUnaryExpression(child)) &&
        (child.operator === ts.SyntaxKind.PlusPlusToken || child.operator === ts.SyntaxKind.MinusMinusToken)) target = child.operand;
    if (target && ts.isIdentifier(target)) names.add(target.text);
    ts.forEachChild(child, visit);
  };
  visit(node);
  return names;
}

function grownReceivers(node: ts.Node): Set<string> {
  const names = new Set<string>();
  const visit = (child: ts.Node): void => {
    if (ts.isCallExpression(child) && ts.isPropertyAccessExpression(child.expression) &&
        GROWING_METHODS.has(child.expression.name.text)) {
      let root: ts.Expression = child.expression.expression;
      while (ts.isPropertyAccessExpression(root) || ts.isElementAccessExpression(root)) root = root.expression;
      if (ts.isIdentifier(root)) names.add(root.text);
    }
    ts.forEachChild(child, visit);
  };
  visit(node);
  return names;
}

function rootIdentifier(expression: ts.Expression): string | undefined {
  let root = expression;
  while (ts.isPropertyAccessExpression(root) || ts.isElementAccessExpression(root) || ts.isParenthesizedExpression(root))
    root = root.expression;
  return ts.isIdentifier(root) ? root.text : undefined;
}

/** Resolve a const numeric literal declared earlier in the same lexical statement list. */
function constNumericLiteralBefore(node: ts.ForStatement, name: string): number | undefined {
  const list = node.parent;
  if (!ts.isBlock(list) && !ts.isSourceFile(list)) return;
  const loopIndex = list.statements.indexOf(node);
  if (loopIndex < 0) return;
  for (let i = loopIndex - 1; i >= 0; i--) {
    const statement = list.statements[i]!;
    if (!ts.isVariableStatement(statement)) continue;
    for (const declaration of statement.declarationList.declarations) {
      if (!ts.isIdentifier(declaration.name) || declaration.name.text !== name) continue;
      if (!(statement.declarationList.flags & ts.NodeFlags.Const) || !declaration.initializer ||
          !ts.isNumericLiteral(declaration.initializer)) return;
      const value = Number(declaration.initializer.text.replaceAll('_', ''));
      return Number.isFinite(value) && value > 0 ? value : undefined;
    }
  }
  return;
}

/** Check whether a counter `for` loop has a canonical monotone finite form. */
function canonicalFor(node: ts.ForStatement): string | undefined {
  const initializer = node.initializer;
  if (!initializer || !ts.isVariableDeclarationList(initializer) || initializer.declarations.length !== 1 ||
      !(initializer.flags & ts.NodeFlags.Let))
    return 'declare exactly one `let` counter in the loop initializer';
  const declaration = initializer.declarations[0]!;
  if (!ts.isIdentifier(declaration.name) || !declaration.initializer) return 'initialize a single named counter';
  const counter = declaration.name.text;
  const condition = node.condition;
  if (!condition || !ts.isBinaryExpression(condition) || !COMPARATORS.has(condition.operatorToken.kind))
    return 'compare the counter with a bound using <, <=, > or >=';
  const counterLeft = ts.isIdentifier(condition.left) && condition.left.text === counter;
  const counterRight = ts.isIdentifier(condition.right) && condition.right.text === counter;
  if (counterLeft === counterRight) return 'compare the counter itself with a bound';
  const bound = counterLeft ? condition.right : condition.left;
  const kind = condition.operatorToken.kind;
  const upward = counterLeft ? kind === ts.SyntaxKind.LessThanToken || kind === ts.SyntaxKind.LessThanEqualsToken :
    kind === ts.SyntaxKind.GreaterThanToken || kind === ts.SyntaxKind.GreaterThanEqualsToken;
  const step = node.incrementor;
  let direction: 'up' | 'down' | undefined;
  if (step && (ts.isPostfixUnaryExpression(step) || ts.isPrefixUnaryExpression(step)) &&
      ts.isIdentifier(step.operand) && step.operand.text === counter)
    direction = step.operator === ts.SyntaxKind.PlusPlusToken ? 'up' : step.operator === ts.SyntaxKind.MinusMinusToken ? 'down' : undefined;
  else if (step && ts.isBinaryExpression(step) && ts.isIdentifier(step.left) && step.left.text === counter &&
      (step.operatorToken.kind === ts.SyntaxKind.PlusEqualsToken || step.operatorToken.kind === ts.SyntaxKind.MinusEqualsToken) &&
      ((ts.isNumericLiteral(step.right) && Number(step.right.text.replaceAll('_', '')) > 0) ||
        (ts.isIdentifier(step.right) && step.right.text !== counter &&
          constNumericLiteralBefore(node, step.right.text) !== undefined)))
    direction = step.operatorToken.kind === ts.SyntaxKind.PlusEqualsToken ? 'up' : 'down';
  if (!direction) return 'step the counter with ++, --, += n or -= n for a positive numeric literal or a same-scope const numeric literal';
  if ((direction === 'up') !== upward) return 'step the counter toward its bound';
  const assigned = assignedIdentifiers(node.statement);
  if (assigned.has(counter)) return 'do not assign the counter inside the loop body';
  const boundRoot = rootIdentifier(bound);
  if (boundRoot && assigned.has(boundRoot)) return 'do not reassign the loop bound inside the loop body';
  if (boundRoot && grownReceivers(node.statement).has(boundRoot))
    return 'do not grow the collection that bounds the loop inside its body';
  if (ts.isCallExpression(bound)) return 'compute the bound once before the loop, for example ' +
    '`const pages = store.pages(); for (let p = 1; p <= pages; p++) { ... }`';
  return;
}

/** How a runtime `for ... of` refusal names the loop's source: its text (or the expression's), on one line and short. */
export function loopLabel(source: ts.Expression | string): string {
  const file = typeof source === 'string' ? undefined : parsedFile(source);
  const text = (typeof source === 'string' ? source : file ? source.getText(file) : 'value').replace(/\s+/g, ' ');
  return text.length > 60 ? `${text.slice(0, 57)}...` : text;
}

const parsedFile = (node: ts.Node): ts.SourceFile | undefined => {
  if (node.pos < 0) return undefined;
  let at: ts.Node | undefined = node;
  while (at && !ts.isSourceFile(at)) at = at.parent;
  return at;
};

/** Diagnose forbidden loops and dynamic code in a constrained source file. */
export function checkConstrainedSource(file: ts.SourceFile, options: PolicyOptions = {}): NatlangDiagnostic[] {
  const diagnostics: NatlangDiagnostic[] = [];
  const displayPath = options.displayPath ?? (source => source.fileName);
  const report = (node: ts.Node, code: NatlangDiagnostic['code'], message: string) =>
    diagnostics.push({ ...spanOf(node, displayPath), code, message, severity: 'error' });
  const loopHint = ' Use `for (const item of array)`, a counter `for (let i = 0; i < n; i++)`, an array method, ' +
    'or `step.iterateOn(initial).until(done)` for open-ended iteration.';
  const visit = (node: ts.Node): void => {
    if (ts.isWhileStatement(node)) report(node, 'forbidden-loop', '`while` loops are not allowed here.' + loopHint);
    else if (ts.isDoStatement(node)) report(node, 'forbidden-loop', '`do ... while` loops are not allowed here.' + loopHint);
    else if (ts.isForInStatement(node))
      report(node, 'forbidden-loop', '`for ... in` is not allowed here; iterate `Object.keys(value)` or `Object.entries(value)`.');
    else if (ts.isForStatement(node)) {
      const problem = canonicalFor(node);
      if (problem) report(node, 'forbidden-loop', `This \`for\` loop is not a checked finite counter loop: ${problem}.` + loopHint);
    } else if ((ts.isFunctionDeclaration(node) || ts.isFunctionExpression(node) || ts.isMethodDeclaration(node)) && node.asteriskToken)
      report(node, 'forbidden-loop', 'Generator functions are not allowed here.' + loopHint);
    else if (ts.isYieldExpression(node)) report(node, 'forbidden-loop', '`yield` is not allowed here.');
    else if (ts.isIdentifier(node) && (node.text === 'eval' || node.text === 'Function') &&
        !(ts.isPropertyAccessExpression(node.parent) && node.parent.name === node) &&
        !((ts.isVariableDeclaration(node.parent) || ts.isParameter(node.parent) || ts.isFunctionDeclaration(node.parent)) &&
          node.parent.name === node))
      report(node, 'forbidden-dynamic-code', `\`${node.text}\` is not allowed here.`);
    else if (ts.isIdentifier(node) && node.text === 'setInterval' && (!isNameOnly(node) ||
        (ts.isPropertyAccessExpression(node.parent) && ts.isIdentifier(node.parent.expression) && GLOBALS.has(node.parent.expression.text))))
      report(node, 'forbidden-loop', '`setInterval` is not available here; repeat with ' +
        '`iterateOn(step, initial).withLimit({ maxSteps })` and wait inside the step with `await new Promise(r => setTimeout(r, ms))`.');
    else if (ts.isPropertyAccessExpression(node) && ts.isIdentifier(node.expression) &&
        ((node.expression.text === 'Symbol' && ITERATOR_SYMBOLS.has(node.name.text)) ||
          (node.expression.text === 'Iterator' && node.name.text === 'from')))
      report(node, 'forbidden-loop', ITERATOR_REFUSAL);
    else if (ts.isHeritageClause(node) && node.token === ts.SyntaxKind.ExtendsKeyword &&
        node.types.some(type => ts.isIdentifier(type.expression) && ['Iterator', 'AsyncIterator'].includes(type.expression.text)))
      report(node, 'forbidden-loop', ITERATOR_REFUSAL);
    else if (ts.isCallExpression(node) && node.expression.kind === ts.SyntaxKind.ImportKeyword && !options.allowDynamicImport)
      report(node, 'forbidden-dynamic-code', 'Dynamic `import()` is not allowed here; use a static import.');
    else if (ts.isForOfStatement(node) && options.checker) {
      const type = options.checker.getTypeAtLocation(node.expression);
      const acceptable = (candidate: ts.Type): boolean => candidate.isUnion() ? candidate.types.every(acceptable) :
        !!(candidate.flags & (ts.TypeFlags.Any | ts.TypeFlags.StringLike)) || options.checker!.isArrayType(candidate) ||
        options.checker!.isTupleType(candidate) || ['Map', 'Set', 'ReadonlyMap', 'ReadonlySet'].includes(candidate.getSymbol()?.name ?? '');
      if (!acceptable(type))
        report(node.expression, 'forbidden-loop', '`for ... of` here must iterate an array, string, Map or Set.' + loopHint);
    }
    ts.forEachChild(node, visit);
  };
  visit(file);
  return diagnostics;
}

export type AuthoredCallable = { id: string; name: string; node: ts.Node; file: ts.SourceFile };

/** Functions authored in a source file that receive entry guards and appear in the call graph. */
export function authoredCallables(file: ts.SourceFile, idPrefix: string): AuthoredCallable[] {
  const found: AuthoredCallable[] = [];
  const nameOf = (node: ts.Node): string => {
    if ((ts.isFunctionDeclaration(node) || ts.isFunctionExpression(node) || ts.isMethodDeclaration(node) ||
        ts.isGetAccessorDeclaration(node) || ts.isSetAccessorDeclaration(node)) && node.name) return node.name.getText();
    if (ts.isConstructorDeclaration(node)) return 'constructor';
    const parent = node.parent;
    if (parent && (ts.isVariableDeclaration(parent) || ts.isPropertyAssignment(parent) || ts.isPropertyDeclaration(parent)) &&
        parent.name) return parent.name.getText();
    return 'anonymous';
  };
  const visit = (node: ts.Node): void => {
    if ((ts.isFunctionDeclaration(node) || ts.isFunctionExpression(node) || ts.isArrowFunction(node) ||
        ts.isMethodDeclaration(node) || ts.isGetAccessorDeclaration(node) || ts.isSetAccessorDeclaration(node) ||
        ts.isConstructorDeclaration(node)) && node.body)
      found.push({ id: `${idPrefix}#${nameOf(node)}@${node.getStart()}`, name: nameOf(node), node, file });
    ts.forEachChild(node, visit);
  };
  visit(file);
  return found;
}

