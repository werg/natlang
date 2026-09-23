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

/** A stream from `iterateOn(...).streamUntil(...)` is the one permitted `for await` source. */
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
      ts.isNumericLiteral(step.right) && Number(step.right.text) > 0)
    direction = step.operatorToken.kind === ts.SyntaxKind.PlusEqualsToken ? 'up' : 'down';
  if (!direction) return 'step the counter with ++, --, += n or -= n for a positive literal n';
  if ((direction === 'up') !== upward) return 'step the counter toward its bound';
  const assigned = assignedIdentifiers(node.statement);
  if (assigned.has(counter)) return 'do not assign the counter inside the loop body';
  const boundRoot = rootIdentifier(bound);
  if (boundRoot && assigned.has(boundRoot)) return 'do not reassign the loop bound inside the loop body';
  if (boundRoot && grownReceivers(node.statement).has(boundRoot))
    return 'do not grow the collection that bounds the loop inside its body';
  if (ts.isCallExpression(bound)) return 'compute a call-valued bound before the loop';
  return;
}

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
    else if (ts.isForOfStatement(node) && node.awaitModifier && !isIterationStream(node.expression, options.checker))
      report(node, 'forbidden-loop', '`for await` is only allowed over `iterateOn(...).streamUntil(...)`.' + loopHint);
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

/**
 * Report direct and mutual recursion among authored callables. `resolve` maps a call's callee to the
 * authored callable it names, if statically known (via a checker or same-file lexical lookup).
 */
export function findRecursion(callables: readonly AuthoredCallable[],
  resolve: (callee: ts.Expression) => AuthoredCallable | undefined,
  displayPath: (file: ts.SourceFile) => string = file => file.fileName): NatlangDiagnostic[] {
  const edges = new Map<string, { target: AuthoredCallable; at: ts.Node }[]>();
  const byId = new Map(callables.map(callable => [callable.id, callable]));
  for (const callable of callables) {
    const out: { target: AuthoredCallable; at: ts.Node }[] = [];
    const visit = (node: ts.Node): void => {
      if (node !== callable.node && callables.some(other => other.node === node) && !ts.isArrowFunction(node) &&
          !ts.isFunctionExpression(node)) return;
      if (ts.isCallExpression(node) || ts.isNewExpression(node)) {
        const target = resolve(node.expression);
        if (target) out.push({ target, at: node });
      }
      ts.forEachChild(node, visit);
    };
    ts.forEachChild(callable.node, visit);
    edges.set(callable.id, out);
  }
  // Tarjan's strongly connected components.
  let index = 0;
  const indices = new Map<string, number>(), low = new Map<string, number>(), stack: string[] = [], onStack = new Set<string>();
  const components: string[][] = [];
  const connect = (id: string): void => {
    indices.set(id, index); low.set(id, index); index++; stack.push(id); onStack.add(id);
    for (const edge of edges.get(id) ?? []) {
      const next = edge.target.id;
      if (!indices.has(next)) { connect(next); low.set(id, Math.min(low.get(id)!, low.get(next)!)); }
      else if (onStack.has(next)) low.set(id, Math.min(low.get(id)!, indices.get(next)!));
    }
    if (low.get(id) === indices.get(id)) {
      const component: string[] = [];
      let member: string;
      do { member = stack.pop()!; onStack.delete(member); component.push(member); } while (member !== id);
      components.push(component);
    }
  };
  for (const callable of callables) if (!indices.has(callable.id)) connect(callable.id);
  const diagnostics: NatlangDiagnostic[] = [];
  for (const component of components) {
    const members = new Set(component);
    const selfLoop = component.length === 1 && (edges.get(component[0]!) ?? []).some(edge => edge.target.id === component[0]);
    if (component.length < 2 && !selfLoop) continue;
    const first = byId.get(component[component.length - 1]!)!;
    const edge = (edges.get(first.id) ?? []).find(item => members.has(item.target.id))!;
    const path = [...component].reverse().map(id => byId.get(id)!.name);
    diagnostics.push({ ...spanOf(edge.at, displayPath), code: 'recursion', severity: 'error',
      message: component.length === 1 ? `\`${first.name}\` calls itself; recursion is not allowed in natlang callable code.` :
        `Mutual recursion is not allowed in natlang callable code: ${[...path, path[0]].join(' → ')}.` });
  }
  return diagnostics;
}

/** Same-file lexical resolution used when no checker is available (eval snippets). */
export function lexicalResolver(callables: readonly AuthoredCallable[]): (callee: ts.Expression) => AuthoredCallable | undefined {
  const named = new Map<string, AuthoredCallable>();
  for (const callable of callables) {
    const parent = callable.node.parent;
    if (ts.isFunctionDeclaration(callable.node) && callable.node.name) named.set(callable.node.name.text, callable);
    else if (parent && ts.isVariableDeclaration(parent) && ts.isIdentifier(parent.name)) named.set(parent.name.text, callable);
  }
  return callee => ts.isIdentifier(callee) ? named.get(callee.text) : undefined;
}
