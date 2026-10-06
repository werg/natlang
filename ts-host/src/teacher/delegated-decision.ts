/** Prove that a result only forwards awaited child values, without a parent choice.
 * Flat model-request order puts a pending parent eval before its children's
 * observations. Use the actual invocation/action ledgers only for this narrow
 * case; absent or incomplete evidence keeps the conservative rejection.
 */
import ts from 'typescript';
import { callName } from './curriculum.js';

type Turn = { invocation_id?: string; context?: Parameters<typeof callName>[0];
  assistant?: { calls?: { tool: string; arguments: unknown }[] } };

export function delegatedDecisionObserved(trajectory: Turn[], outcome: Record<string, unknown>, decision: number, observed: number): boolean {
  const turn = trajectory[decision], witness = trajectory[observed];
  if (!turn?.invocation_id || !witness?.invocation_id || observed <= decision) return false;
  const parents = new Map<string, string | null>();
  for (const item of Array.isArray(outcome.invocation_ledger) ? outcome.invocation_ledger : []) {
    if (typeof item?.invocation_id !== 'string' || parents.has(item.invocation_id)) return false;
    parents.set(item.invocation_id, item.parent_invocation_id ?? null);
  }
  let ancestor: string | null = witness.invocation_id;
  const visited = new Set<string>();
  while (ancestor && ancestor !== turn.invocation_id) {
    if (visited.has(ancestor) || !parents.has(ancestor)) return false;
    visited.add(ancestor);
    ancestor = parents.get(ancestor) ?? null;
  }
  if (ancestor !== turn.invocation_id || witness.invocation_id === turn.invocation_id) return false;
  const childNames = new Set(trajectory.filter(t => t.invocation_id && parents.get(t.invocation_id) === turn.invocation_id)
    .map(t => callName(t.context ?? [])));
  for (const call of turn.assistant?.calls ?? []) {
    const args = call.arguments as { code?: unknown; finish?: unknown } | null;
    if (call.tool !== 'eval' || typeof args?.code !== 'string' || !onlyForwardsChildren(args.code, childNames, args.finish === true)) continue;
    const ledger = Array.isArray(outcome.action_ledger) ? outcome.action_ledger : [];
    const completed = ledger.find(action => action?.call_id === turn.invocation_id && action.name === 'eval' &&
      action.outcome === 'completed' && action.arguments?.code === args.code &&
      (action.arguments?.finish === true) === (args.finish === true));
    if (!completed) continue;
    const end = Date.parse(completed.observed_at), elapsed = Number(completed.elapsed_ms);
    if (!Number.isFinite(end) || !Number.isFinite(elapsed) || elapsed < 0) continue;
    // The evidence-bearing child must have actually executed during this eval.
    const childActions = ledger.filter(action => visited.has(action?.call_id));
    if (childActions.some(action => {
      const at = Date.parse(action.observed_at);
      return Number.isFinite(at) && at >= end - elapsed && at <= end;
    })) return true;
  }
  return false;
}

function onlyForwardsChildren(code: string, childNames: Set<string>, finish: boolean): boolean {
  const file = ts.createSourceFile('eval.ts', code, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
  if ((file as ts.SourceFile & { parseDiagnostics: unknown[] }).parseDiagnostics.length) return false;
  const values = new Set<string>();
  let awaited = false, returned = false;
  function argument(expression: ts.Expression): boolean {
    if (ts.isParenthesizedExpression(expression) || ts.isAsExpression(expression) || ts.isTypeAssertionExpression(expression))
      return argument(expression.expression);
    if (ts.isIdentifier(expression)) return !values.has(expression.text);
    if (ts.isPropertyAccessExpression(expression)) return argument(expression.expression);
    if (ts.isElementAccessExpression(expression)) return argument(expression.expression) &&
      !!expression.argumentExpression && (ts.isStringLiteral(expression.argumentExpression) || ts.isNumericLiteral(expression.argumentExpression));
    return false;
  }
  function child(call: ts.CallExpression): boolean {
    return ts.isIdentifier(call.expression) && childNames.has(call.expression.text) && call.arguments.every(argument);
  }
  function value(expression: ts.Expression): boolean {
    if (ts.isParenthesizedExpression(expression)) return value(expression.expression);
    if (ts.isIdentifier(expression)) return values.has(expression.text);
    if (ts.isArrayLiteralExpression(expression)) return expression.elements.length > 0 && expression.elements.every(item =>
      !ts.isSpreadElement(item) && value(item as ts.Expression));
    if (!ts.isAwaitExpression(expression) || !ts.isCallExpression(expression.expression)) return false;
    const call = expression.expression, callee = call.expression;
    const parallel = ts.isPropertyAccessExpression(callee) && callee.expression.getText(file) === 'Promise' &&
      callee.name.text === 'all' && call.arguments.length === 1 && ts.isArrayLiteralExpression(call.arguments[0]!) &&
      call.arguments[0]!.elements.length > 0 && call.arguments[0]!.elements.every(item => ts.isCallExpression(item) && child(item));
    if (!child(call) && !parallel) return false;
    awaited = true;
    return true;
  }
  for (const [index, statement] of file.statements.entries()) {
    if (returned) return false;
    if (ts.isVariableStatement(statement) && (statement.declarationList.flags & ts.NodeFlags.Const)) {
      for (const declaration of statement.declarationList.declarations) {
        if (!ts.isIdentifier(declaration.name) || !declaration.initializer) return false;
        if (value(declaration.initializer)) values.add(declaration.name.text);
        else if (!argument(declaration.initializer)) return false;
      }
    } else if (ts.isExpressionStatement(statement) && ts.isCallExpression(statement.expression) &&
        ts.isPropertyAccessExpression(statement.expression.expression) &&
        statement.expression.expression.expression.getText(file) === 'console' &&
        statement.expression.expression.name.text === 'log' && statement.expression.arguments.every(argument =>
          value(argument) || (ts.isCallExpression(argument) && argument.expression.getText(file) === 'JSON.stringify' &&
            argument.arguments.length === 1 && value(argument.arguments[0]!)))) {
      // Logging a forwarded value cannot choose or change it.
    } else if (ts.isReturnStatement(statement) && statement.expression && value(statement.expression)) {
      returned = true;
    } else if (finish && index === file.statements.length - 1 && ts.isExpressionStatement(statement) && value(statement.expression)) {
      returned = true;
    } else return false;
  }
  return awaited && returned;
}
