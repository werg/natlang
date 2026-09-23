import ts from 'typescript';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const typeText = (node) => node ? node.getText() : undefined;
const docText = (node) => {
  const docs = node.jsDoc ?? [];
  return docs.map((doc) => String(doc.comment ?? '').trim()).filter(Boolean).join('\n\n');
};

/** Extract documented, named function declarations without rewriting their source. */
export function extractFunctions(source, { path, sourceName, revision, license, instruction, fallbackInstruction } = {}) {
  const file = ts.createSourceFile(path ?? 'source.ts', source, ts.ScriptTarget.Latest, true,
    /\.tsx?$/.test(path ?? '') ? ts.ScriptKind.TS : ts.ScriptKind.JS);
  const executable = /\.tsx?$/.test(path ?? '')
    ? ts.createSourceFile('executable.js', ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } }).outputText, ts.ScriptTarget.Latest, true, ts.ScriptKind.JS)
    : file;
  const result = [];
  for (const node of file.statements) {
    if (!ts.isFunctionDeclaration(node) || !node.name || !node.body) continue;
    const description = instruction ?? (docText(node) || fallbackInstruction);
    if (!description) continue;
    const executableNode = executable.statements.find((s) => ts.isFunctionDeclaration(s) && s.name?.text === node.name.text);
    const parameters = node.parameters.map((p) => ({
      name: p.name.getText(file),
      ...(p.type ? { type: p.type.getText(file) } : {}),
    }));
    const declarations = new Set(parameters.map((p) => p.name));
    const collectBindings = (n) => {
      if ((ts.isVariableDeclaration(n) || ts.isParameter(n) || ts.isFunctionDeclaration(n) || ts.isClassDeclaration(n)) && n.name && ts.isIdentifier(n.name)) declarations.add(n.name.text);
      ts.forEachChild(n, collectBindings);
    };
    collectBindings(node.body);
    const dependencies = new Set();
    const propertyNames = new Set(['undefined', 'NaN', 'Infinity', 'this', 'super']);
    const visit = (n) => {
      if (ts.isIdentifier(n) && !propertyNames.has(n.text)) {
        const parent = n.parent;
        const isProperty = (ts.isPropertyAccessExpression(parent) && parent.name === n)
          || (ts.isPropertyAssignment(parent) && parent.name === n)
          || (ts.isMethodDeclaration(parent) && parent.name === n)
          || (ts.isVariableDeclaration(parent) && parent.name === n)
          || (ts.isParameter(parent) && parent.name === n)
          || (ts.isFunctionDeclaration(parent) && parent.name === n)
          || (ts.isTypeReferenceNode(parent) && parent.typeName === n);
        if (!isProperty && !declarations.has(n.text)) dependencies.add(n.text);
      }
      ts.forEachChild(n, visit);
    };
    // Include body only: signature identifiers are types/defaults and assessed separately.
    visit(node.body);
    const reasons = [];
    if (dependencies.size) reasons.push(`free runtime bindings: ${[...dependencies].sort().join(', ')}`);
    let mutates = false, usesCallback = false;
    const assess = (n) => {
      if (ts.isBinaryExpression(n) && n.operatorToken.kind >= ts.SyntaxKind.FirstAssignment && n.operatorToken.kind <= ts.SyntaxKind.LastAssignment
        && (ts.isPropertyAccessExpression(n.left) || ts.isElementAccessExpression(n.left))) mutates = true;
      if (ts.isPrefixUnaryExpression(n) || ts.isPostfixUnaryExpression(n)) {
        if (ts.isPropertyAccessExpression(n.operand) || ts.isElementAccessExpression(n.operand)) mutates = true;
      }
      if (ts.isCallExpression(n) && n.arguments.some((a) => ts.isArrowFunction(a) || ts.isFunctionExpression(a))) usesCallback = true;
      ts.forEachChild(n, assess);
    };
    assess(node.body);
    if (mutates) reasons.push('mutation detected');
    if (usesCallback) reasons.push('callback arguments deferred');
    if (node.modifiers?.some((m) => m.kind === ts.SyntaxKind.AsyncKeyword)) reasons.push('async function deferred');
    if (node.asteriskToken) reasons.push('generator function deferred');
    if (node.parameters.some((p) => p.dotDotDotToken)) reasons.push('rest parameters deferred');
    // Preserve a conservative transitive closure of sibling function declarations.
    // Do not copy module initializers: they may have effects or depend on live state.
    const siblings = new Map(file.statements.filter(s => ts.isFunctionDeclaration(s) && s.name && s.body && s.name.text !== node.name.text).map(s => [s.name.text, s]));
    const selected = new Map();
    const includeHelpers = body => {
      const visit = child => {
        if (ts.isIdentifier(child) && siblings.has(child.text) && !selected.has(child.text)) {
          const helper = siblings.get(child.text);
          selected.set(child.text, helper);
          includeHelpers(helper.body);
        }
        ts.forEachChild(child, visit);
      };
      visit(body);
    };
    includeHelpers(executableNode?.body ?? node.body);
    const functionDeclarations = new Map([[node.name.text, node], ...siblings]);
    const graph = new Map([...functionDeclarations].map(([name, declaration]) => {
      const calls = new Set();
      const visit = child => {
        if (ts.isCallExpression(child) && ts.isIdentifier(child.expression) && functionDeclarations.has(child.expression.text))
          calls.add(child.expression.text);
        ts.forEachChild(child, visit);
      };
      visit(declaration.body);
      return [name, calls];
    }));
    const active = new Set(), visited = new Set();
    const cyclic = name => {
      if (active.has(name)) return true;
      if (visited.has(name)) return false;
      active.add(name);
      for (const target of graph.get(name) ?? []) if (cyclic(target)) return true;
      active.delete(name); visited.add(name);
      return false;
    };
    const recursive = cyclic(node.name.text);
    if (recursive) reasons.push('recursive function graph');
    const printer = ts.createPrinter();
    const helpers = [...selected.values()].map(helper => printer.printNode(ts.EmitHint.Unspecified,
      ts.factory.updateFunctionDeclaration(helper, helper.modifiers?.filter(m => ![ts.SyntaxKind.ExportKeyword, ts.SyntaxKind.DefaultKeyword].includes(m.kind)),
        helper.asteriskToken, helper.name, helper.typeParameters, helper.parameters, helper.type, helper.body), file));
    result.push({
      version: 'natlang.code_task/1', id: `${sourceName ?? 'source'}:${revision ?? 'unknown'}:${path ?? 'source.ts'}:${node.name.text}`,
      group_id: `${sourceName ?? 'source'}:${path ?? 'source.ts'}:${node.name.text}`,
      kind: 'function', language: /\.tsx?$/.test(path ?? '') ? 'typescript' : 'javascript', instruction: description,
      source: { name: sourceName ?? 'unknown', revision: revision ?? 'unknown', path: path ?? 'source.ts', license: license ?? 'unknown' },
      function: {
        name: node.name.text, parameters,
        helpers,
        recursive,
        imports: file.statements.filter(ts.isImportDeclaration).map(statement => ({
          specifier: ts.isStringLiteral(statement.moduleSpecifier) ? statement.moduleSpecifier.text : '',
          source: statement.getText(file),
        })),
        ...(node.type ? { return_type: typeText(node.type) } : {}),
        body: (executableNode?.body ?? node.body).getText(executable), source: node.getText(file),
      },
      cases: [], verification: { status: reasons.length ? 'inventory' : 'unverified', ...(reasons.length ? { reasons } : {}) },
    });
  }
  return result;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const [mode, ...args] = process.argv.slice(2);
  const options = Object.fromEntries(args.map((v, i) => v.startsWith('--') ? [v.slice(2), args[i + 1]] : null).filter(Boolean));
  if (!['extract', 'instrument'].includes(mode) || !options.input || !options.output) {
    console.error('Usage: node extract.mjs <extract|instrument> --input FILE --output FILE [--source NAME] [--revision REV] [--license LICENSE]');
    process.exitCode = 2;
  } else {
    const input = resolve(options.input), output = resolve(options.output);
    if (existsSync(output)) throw new Error(`Refusing to overwrite ${output}`);
    const source = readFileSync(input, 'utf8');
    const metadata = { path: input, sourceName: options.source, revision: options.revision, license: options.license };
    if (mode === 'extract') writeFileSync(output, `${extractFunctions(source, metadata).map((r) => JSON.stringify(r)).join('\n')}\n`);
    else {
      const { instrumentSource } = await import('./capture.mjs');
      writeFileSync(output, instrumentSource(source, metadata).code);
    }
  }
}
