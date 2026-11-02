/** Structural evidence only; source family and task custody belong to the caller. */
import ts from '../../ts-host/node_modules/typescript/lib/typescript.js';
import { createInterface } from 'node:readline';
import { pathToFileURL } from 'node:url';

export function inspectSemanticTicketCode(code, semanticPattern) {
  if (typeof code !== 'string' || code.length > 4 * 1024 * 1024 || typeof semanticPattern !== 'string')
    throw new TypeError('expected bounded code and its declared semantic-word pattern');
  const words = new RegExp(semanticPattern, 'i');
  const file = ts.createSourceFile('teacher-eval.ts', code, ts.ScriptTarget.Latest, true);
  const evidence = [];
  const literal = node => node && (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) ? node.text : null;
  const visit = node => {
    if (ts.isRegularExpressionLiteral(node) && words.test(node.text))
      evidence.push({ kind: 'semantic-regex', code: node.getText(file), start: node.getStart(file) });
    if (ts.isNewExpression(node) && ts.isIdentifier(node.expression) && node.expression.text === 'RegExp') {
      const text = literal(node.arguments?.[0]);
      if (text !== null && words.test(text))
        evidence.push({ kind: 'semantic-regex-constructor', code: node.getText(file), start: node.getStart(file) });
    }
    if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression) &&
        ['includes', 'startsWith', 'endsWith', 'indexOf'].includes(node.expression.name.text)) {
      const text = literal(node.arguments[0]);
      if (text !== null && words.test(text))
        evidence.push({ kind: 'semantic-keyword-test', code: node.getText(file), start: node.getStart(file) });
    }
    ts.forEachChild(node, visit);
  };
  visit(file);
  return { evidence, parse_errors: file.parseDiagnostics.length };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  for await (const line of createInterface({ input: process.stdin, crlfDelay: Infinity })) {
    if (!line.trim()) continue;
    const request = JSON.parse(line);
    process.stdout.write(JSON.stringify(inspectSemanticTicketCode(request.code, request.semanticPattern)) + '\n');
  }
}
