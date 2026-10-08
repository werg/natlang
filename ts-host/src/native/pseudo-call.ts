/**
 * A reply that is only `return_result({ status, value | reason })` written as text: models sometimes print the tool's
 * request instead of making it. Read it as the request when its argument is a literal (no code runs).
 */
import ts from 'typescript';

const NO_VALUE = Symbol('not a literal');

/** The value of a literal expression: objects, arrays, strings, numbers, booleans, null. */
function literal(node: ts.Expression): unknown {
  if (ts.isParenthesizedExpression(node)) return literal(node.expression);
  if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) return node.text;
  if (ts.isNumericLiteral(node)) return Number(node.text);
  if (ts.isPrefixUnaryExpression(node) && node.operator === ts.SyntaxKind.MinusToken && ts.isNumericLiteral(node.operand)) return -Number(node.operand.text);
  if (node.kind === ts.SyntaxKind.TrueKeyword) return true;
  if (node.kind === ts.SyntaxKind.FalseKeyword) return false;
  if (node.kind === ts.SyntaxKind.NullKeyword) return null;
  if (ts.isArrayLiteralExpression(node)) {
    const items = node.elements.map(element => ts.isSpreadElement(element) ? NO_VALUE : literal(element));
    return items.includes(NO_VALUE) ? NO_VALUE : items;
  }
  if (ts.isObjectLiteralExpression(node)) {
    const out: Record<string, unknown> = {};
    for (const property of node.properties) {
      if (!ts.isPropertyAssignment(property)) return NO_VALUE;
      const name = property.name;
      const key = ts.isIdentifier(name) || ts.isStringLiteral(name) || ts.isNumericLiteral(name) ? name.text : undefined;
      if (key === undefined) return NO_VALUE;
      const value = literal(property.initializer);
      if (value === NO_VALUE) return NO_VALUE;
      out[key] = value;
    }
    return out;
  }
  return NO_VALUE;
}

/** The request a reply writes out as `return_result({...})`, or undefined when the reply is anything else. */
export function writtenReturnResult(text: string): { status: string; value?: unknown; reason?: string } | undefined {
  const trimmed = text.trim().replace(/^```(?:ts|typescript|js|javascript)?\s*\n([\s\S]*?)\n```$/, '$1').trim();
  if (!/^return_result\s*\(/.test(trimmed) || trimmed.length > 200_000) return undefined;
  const source = ts.createSourceFile('reply.ts', trimmed, ts.ScriptTarget.ES2022, false, ts.ScriptKind.TS);
  if (source.statements.length !== 1) return undefined;
  const statement = source.statements[0]!;
  if (!ts.isExpressionStatement(statement) || !ts.isCallExpression(statement.expression)) return undefined;
  const call = statement.expression;
  if (!ts.isIdentifier(call.expression) || call.expression.text !== 'return_result' || call.arguments.length !== 1) return undefined;
  const request = literal(call.arguments[0]!);
  if (!request || typeof request !== 'object' || Array.isArray(request)) return undefined;
  const fields = request as Record<string, unknown>;
  if (!['success', 'blocked', 'failed'].includes(String(fields.status)) ||
      !Object.keys(fields).every(key => ['status', 'value', 'reason'].includes(key))) return undefined;
  if (fields.reason !== undefined && typeof fields.reason !== 'string') return undefined;
  return fields as { status: string; value?: unknown; reason?: string };
}
