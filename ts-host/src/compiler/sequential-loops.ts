/**
 * `natlang check` hint for loops that look sequential but need not be (plans/BATCHED_EXECUTION.md §3.5).
 *
 *   for (const ticket of tickets) labels.push(await classify(ticket));
 *
 * waits for each model call before it starts the next, so the model sees one request at a time. When the body is one
 * natural-language call, nothing else happens in the loop, and the call's arguments do not depend on anything the
 * loop writes, the iterations are independent and `await Promise.all(tickets.map(...))` has the same result with the
 * calls batched. The runtime does not parallelize such loops by itself: order and effects are the program's meaning.
 * The diagnostic is a warning and says what to write.
 *
 * What counts as "one natural-language call": a call of a name imported from a `.nl` module, or an `nl` tagged
 * template applied to arguments. What counts as "no carried state": the loop body is exactly one statement that
 * awaits that call and either declares a constant from it, pushes it, or stores it at an index; the call's arguments
 * are made of the loop variable, constants, property reads and literals (no other call, so no effect), and do not
 * read a name the loop writes.
 */
import ts from 'typescript';
import type { NatlangDiagnostic } from './inline.js';

export const SEQUENTIAL_LOOP_CODE = 'nl-sequential-loop';

export function checkSequentialNlLoops(file: ts.SourceFile, displayPath: (file: ts.SourceFile) => string = source => source.fileName): NatlangDiagnostic[] {
  const nlFunctions = new Set<string>();
  for (const statement of file.statements) {
    if (!ts.isImportDeclaration(statement) || !ts.isStringLiteral(statement.moduleSpecifier)) continue;
    const clause = statement.importClause;
    if (!clause || clause.isTypeOnly) continue;
    if (statement.moduleSpecifier.text.endsWith('.nl') && clause.name) nlFunctions.add(clause.name.text);
    if (statement.moduleSpecifier.text.endsWith('.nl') && clause.namedBindings && ts.isNamedImports(clause.namedBindings))
      for (const item of clause.namedBindings.elements) nlFunctions.add(item.name.text);
  }
  const unwrap = (node: ts.Expression): ts.Expression => {
    while (ts.isParenthesizedExpression(node) || ts.isNonNullExpression(node) || ts.isAsExpression(node)) node = node.expression;
    return node;
  };
  const isNlCall = (node: ts.Expression): node is ts.CallExpression => {
    node = unwrap(node);
    if (!ts.isCallExpression(node)) return false;
    const callee = unwrap(node.expression);
    if (ts.isIdentifier(callee)) return nlFunctions.has(callee.text);
    return ts.isTaggedTemplateExpression(callee) && ts.isIdentifier(callee.tag) && callee.tag.text === 'nl';
  };
  const simpleArgument = (node: ts.Node, loopNames: Set<string>, written: Set<string>): boolean => {
    let ok = true;
    const visit = (child: ts.Node): void => {
      if (!ok) return;
      if (ts.isCallExpression(child) || ts.isNewExpression(child) || ts.isAwaitExpression(child) || ts.isFunctionLike(child) ||
          ts.isYieldExpression(child) || ts.isTaggedTemplateExpression(child) ||
          (ts.isBinaryExpression(child) && child.operatorToken.kind >= ts.SyntaxKind.FirstAssignment && child.operatorToken.kind <= ts.SyntaxKind.LastAssignment) ||
          ((ts.isPrefixUnaryExpression(child) || ts.isPostfixUnaryExpression(child)) &&
            (child.operator === ts.SyntaxKind.PlusPlusToken || child.operator === ts.SyntaxKind.MinusMinusToken))) { ok = false; return; }
      if (ts.isIdentifier(child) && written.has(child.text) && !(ts.isPropertyAccessExpression(child.parent) && child.parent.name === child)) { ok = false; return; }
      ts.forEachChild(child, visit);
    };
    visit(node);
    return ok;
  };
  const rootName = (node: ts.Expression): string | undefined => {
    node = unwrap(node);
    while (ts.isPropertyAccessExpression(node) || ts.isElementAccessExpression(node)) node = unwrap(node.expression);
    return ts.isIdentifier(node) ? node.text : undefined;
  };
  const diagnostics: NatlangDiagnostic[] = [];
  const visit = (node: ts.Node): void => {
    if (ts.isForOfStatement(node) && !node.awaitModifier && ts.isVariableDeclarationList(node.initializer)) {
      const loopNames = new Set<string>();
      const collect = (name: ts.BindingName): void => {
        if (ts.isIdentifier(name)) loopNames.add(name.text);
        else for (const element of name.elements) if (ts.isBindingElement(element)) collect(element.name);
      };
      node.initializer.declarations.forEach(declaration => collect(declaration.name));
      const statements = ts.isBlock(node.statement) ? [...node.statement.statements] : [node.statement];
      if (statements.length === 1) {
        const only = statements[0]!;
        let awaited: ts.Expression | undefined, written: string | undefined, ok = false;
        if (ts.isVariableStatement(only) && only.declarationList.flags & ts.NodeFlags.Const && only.declarationList.declarations.length === 1) {
          const init = only.declarationList.declarations[0]!.initializer;
          if (init && ts.isAwaitExpression(unwrap(init))) { awaited = (unwrap(init) as ts.AwaitExpression).expression; ok = true; }
        } else if (ts.isExpressionStatement(only)) {
          const expression = unwrap(only.expression);
          if (ts.isCallExpression(expression) && ts.isPropertyAccessExpression(expression.expression) && expression.expression.name.text === 'push' &&
              expression.arguments.length === 1 && ts.isAwaitExpression(unwrap(expression.arguments[0]!))) {
            awaited = (unwrap(expression.arguments[0]!) as ts.AwaitExpression).expression; written = rootName(expression.expression.expression); ok = true;
          } else if (ts.isBinaryExpression(expression) && expression.operatorToken.kind === ts.SyntaxKind.EqualsToken &&
              ts.isElementAccessExpression(unwrap(expression.left)) && ts.isAwaitExpression(unwrap(expression.right))) {
            awaited = (unwrap(expression.right) as ts.AwaitExpression).expression; written = rootName(expression.left); ok = true;
          }
        }
        if (ok && awaited && isNlCall(awaited)) {
          const call = unwrap(awaited) as ts.CallExpression;
          const writes = new Set(written ? [written] : []);
          const callee = unwrap(call.expression);
          const pieces: ts.Node[] = [...call.arguments, ...(ts.isTaggedTemplateExpression(callee) ? [callee.template] : [])];
          // Arguments mention only the loop variable and names the loop does not write; the template of an inline
          // `nl` call is prose and its captures are the call's own arguments.
          if (pieces.every(piece => simpleArgument(piece, loopNames, writes))) {
            const start = node.getStart(file), position = file.getLineAndCharacterOfPosition(start);
            diagnostics.push({ file: displayPath(file), start, end: node.end, line: position.line + 1, column: position.character + 1,
              code: SEQUENTIAL_LOOP_CODE as NatlangDiagnostic['code'], severity: 'warning',
              message: 'This loop waits for each natural-language call before starting the next, so the model sees one request at a time. ' +
                'The calls do not depend on each other, so `await Promise.all(items.map(item => f(item)))` gives the same results with the ' +
                'calls batched. Keep the loop if the order of the calls matters.' });
          }
        }
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(file);
  return diagnostics;
}
