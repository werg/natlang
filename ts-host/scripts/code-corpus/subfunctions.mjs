import { existsSync, readFileSync } from 'node:fs';
import { dirname, extname, relative, resolve, sep } from 'node:path';
import ts from 'typescript';
import { packageNameFromSpecifier } from '../../dist/package-specifier.js';

const identifier = /^[A-Za-z_$][\w$]*$/;
const sourceFile = (name, source) => ts.createSourceFile(name, source, ts.ScriptTarget.Latest, true,
  /\.tsx?$/.test(name) ? ts.ScriptKind.TS : ts.ScriptKind.JS);

function functionDefinition(name, declaration, file) {
  if (!identifier.test(name) || !declaration.body || !declaration.type || declaration.asteriskToken ||
      declaration.parameters.some(parameter => !ts.isIdentifier(parameter.name) || !parameter.type ||
        parameter.dotDotDotToken || parameter.initializer))
    throw new Error(`Subfunction ${name} needs explicit portable parameter and return types`);
  const returnsPromise = ts.isTypeReferenceNode(declaration.type) &&
    ts.isIdentifier(declaration.type.typeName) && declaration.type.typeName.text === 'Promise' &&
    declaration.type.typeArguments?.length === 1;
  return { name, args: Object.fromEntries(declaration.parameters.map(parameter => [parameter.name.text,
    parameter.type.getText(file)])),
    returns: (returnsPromise ? declaration.type.typeArguments[0] : declaration.type).getText(file),
    async: returnsPromise || !!declaration.modifiers?.some(modifier => modifier.kind === ts.SyntaxKind.AsyncKeyword),
    code: declaration.body.getText(file).slice(1, -1).trim() + '\n', node: declaration.body,
    packageImports: [] };
}

function calls(body) {
  const found = new Set();
  const visit = node => {
    if (ts.isCallExpression(node) && ts.isIdentifier(node.expression)) found.add(node.expression.text);
    ts.forEachChild(node, visit);
  };
  visit(body);
  return found;
}

function importBindings(statement) {
  const clause = statement.importClause;
  if (!clause || clause.isTypeOnly) return [];
  const bindings = [];
  if (clause.name) bindings.push([clause.name.text, 'default']);
  if (clause.namedBindings) {
    if (!ts.isNamedImports(clause.namedBindings)) throw new Error('Namespace application imports cannot become subfunctions');
    for (const item of clause.namedBindings.elements)
      if (!item.isTypeOnly) bindings.push([item.name.text, item.propertyName?.text ?? item.name.text]);
  }
  return bindings;
}

/** Project local calls into the same tree as foo.nl -> foo/child.ts; never inline helpers. */
export function projectSubfunctions(record, { workspace } = {}) {
  const rootName = record.function.name;
  if (!identifier.test(rootName)) throw new Error('Documented function needs a named identifier');
  const rootSource = sourceFile(record.source.path ?? `${rootName}.js`,
    `function ${rootName}() ${record.function.body}`);
  const root = rootSource.statements.find(ts.isFunctionDeclaration);
  if (!root?.body) throw new Error('Captured function body is not a function block');
  const definitions = new Map();
  for (const helperText of record.function.helpers ?? []) {
    const file = sourceFile('helper.ts', helperText);
    const declaration = file.statements.find(ts.isFunctionDeclaration);
    if (!declaration?.name) throw new Error('Captured sibling helper is not a named function');
    if (declaration.name.text === rootName || definitions.has(declaration.name.text))
      throw new Error(`Duplicate or recursive subfunction ${declaration.name.text}`);
    definitions.set(declaration.name.text, functionDefinition(declaration.name.text, declaration, file));
  }
  const packageImports = [];
  const rootCalls = calls(root.body);
  for (const imported of record.function.imports ?? []) {
    const file = sourceFile('import.ts', imported.source);
    const statement = file.statements.find(ts.isImportDeclaration);
    if (!statement || !ts.isStringLiteral(statement.moduleSpecifier)) throw new Error('Invalid captured import');
    const specifier = statement.moduleSpecifier.text;
    if (!specifier.startsWith('.')) {
      packageNameFromSpecifier(specifier);
      if (!statement.importClause?.isTypeOnly) packageImports.push(imported.source);
      continue;
    }
    const bindings = importBindings(statement).filter(([alias]) => rootCalls.has(alias));
    if (!bindings.length) continue;
    if (!workspace) throw new Error('Local subfunction conversion requires the source workspace');
    const base = resolve(workspace, dirname(record.source.path), specifier);
    if (relative(resolve(workspace), base).startsWith(`..${sep}`) || relative(resolve(workspace), base) === '..')
      throw new Error(`Local subfunction import escapes workspace: ${specifier}`);
    const path = [base, `${base}.ts`, `${base}.js`, `${base}.mjs`].find(candidate => existsSync(candidate));
    if (!path || !['.ts', '.js', '.mjs'].includes(extname(path)))
      throw new Error(`Local subfunction source is unavailable: ${specifier}`);
    const module = sourceFile(path, readFileSync(path, 'utf8'));
    for (const [alias, exported] of bindings) {
      const declaration = module.statements.find(statement => ts.isFunctionDeclaration(statement) &&
        (exported === 'default' ? statement.modifiers?.some(modifier => modifier.kind === ts.SyntaxKind.DefaultKeyword) :
          statement.name?.text === exported && statement.modifiers?.some(modifier => modifier.kind === ts.SyntaxKind.ExportKeyword)));
      if (!declaration) throw new Error(`Local import ${alias} is not an exported function`);
      if (alias === rootName || definitions.has(alias)) throw new Error(`Duplicate or recursive subfunction ${alias}`);
      definitions.set(alias, functionDefinition(alias, declaration, module));
    }
  }
  const layout = {};
  const build = (name, active, parent) => {
    if (active.has(name)) throw new Error(`Recursive subfunction call graph: ${[...active, name].join(' -> ')}`);
    const definition = definitions.get(name);
    if (!definition) throw new Error(`Unknown subfunction ${name}`);
    const next = new Set([...active, name]);
    layout[name] = `${parent}/${name}.ts`;
    const nested = Object.fromEntries([...calls(definition.node)].filter(child => child === rootName || definitions.has(child))
      .sort().map(child => {
        if (child === rootName) throw new Error(`Recursive subfunction call graph: ${name} -> ${rootName}`);
        return [child, build(child, next, `${parent}/${name}`)];
      }));
    return { args: definition.args, returns: definition.returns, code: definition.packageImports.join('\n') + definition.code,
      async: definition.async, ...(Object.keys(nested).length ? { codebase: nested } : {}) };
  };
  if (rootCalls.has(rootName)) throw new Error(`Recursive function call: ${rootName}`);
  const parent = rootName;
  const codebase = Object.fromEntries([...rootCalls].filter(name => definitions.has(name)).sort()
    .map(name => [name, build(name, new Set([rootName]), parent)]));
  return { codebase, packageImports, sourceLayout: { root: `${rootName}.nl`, subfunctions: layout } };
}
