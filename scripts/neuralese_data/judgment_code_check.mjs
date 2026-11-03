/** Structural evidence of crisp text tests inside supervised eval code.
 *
 * Reports every regular-expression literal, `new RegExp(...)` and string-literal
 * `includes`/`startsWith`/`endsWith`/`indexOf`/`search`/`match` call, with the
 * pattern text and how its result is used (boolean test versus extraction). Slashes in
 * templates, comments and ordinary string literals are not regexes. Policy (which
 * nodes are semantic judgments and which patterns are exact parsing) belongs to the caller.
 */
import ts from '../../ts-host/node_modules/typescript/lib/typescript.js';
import { createInterface } from 'node:readline';
import { pathToFileURL } from 'node:url';

const KEYWORD_METHODS = new Set(['includes', 'startsWith', 'endsWith', 'indexOf', 'search', 'match']);
const REGEX_TESTS = new Set(['test', 'search']);
const REGEX_EXTRACTION = new Set(['exec', 'match', 'matchAll', 'replace', 'replaceAll', 'split']);

export function inspectJudgmentCode(code) {
  if (typeof code !== 'string' || code.length > 4 * 1024 * 1024) throw new TypeError('expected bounded code');
  const file = ts.createSourceFile('teacher-eval.ts', code, ts.ScriptTarget.Latest, true);
  const evidence = [];
  const literal = node => node && (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) ? node.text : null;
  const usage = node => {
    const parent = node.parent;
    // regex.test(x) / regex.exec(x)
    if (parent && ts.isPropertyAccessExpression(parent) && parent.expression === node && ts.isCallExpression(parent.parent))
      return parent.name.text;
    // x.match(regex) / x.replace(regex, ...)
    if (parent && ts.isCallExpression(parent) && parent.arguments.includes(node) && ts.isPropertyAccessExpression(parent.expression))
      return parent.expression.name.text;
    return 'other';
  };
  const regexKind = method => REGEX_TESTS.has(method) ? 'test' : REGEX_EXTRACTION.has(method) ? 'extraction' : 'other';
  const visit = node => {
    if (ts.isRegularExpressionLiteral(node)) {
      const text = node.text;
      const end = text.lastIndexOf('/');
      const method = usage(node);
      evidence.push({ kind: 'regex', pattern: text.slice(1, end), flags: text.slice(end + 1), method,
                      use: regexKind(method), code: node.getText(file), start: node.getStart(file) });
    }
    if (ts.isNewExpression(node) && ts.isIdentifier(node.expression) && node.expression.text === 'RegExp') {
      const text = literal(node.arguments?.[0]);
      const method = usage(node);
      evidence.push({ kind: 'regex-constructor', pattern: text, flags: literal(node.arguments?.[1]) ?? '', method,
                      use: regexKind(method), code: node.getText(file), start: node.getStart(file) });
    }
    if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression) &&
        KEYWORD_METHODS.has(node.expression.name.text)) {
      const text = literal(node.arguments[0]);
      if (text !== null)
        evidence.push({ kind: 'keyword', pattern: text, flags: '', method: node.expression.name.text,
                        use: node.expression.name.text === 'match' ? 'extraction' : 'test',
                        code: node.getText(file), start: node.getStart(file) });
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
    process.stdout.write(JSON.stringify(inspectJudgmentCode(request.code)) + '\n');
  }
}
