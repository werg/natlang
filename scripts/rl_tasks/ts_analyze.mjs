#!/usr/bin/env node
// Crisp TypeScript analyzer for codebase tasks (scripts/rl_tasks/sources/codebase.py).
// usage: node ts_analyze.mjs TS_HOST_DIR IN.json OUT.json   (IN: {path: text}; OUT: per-function/class/file facts)
// Per function (declarations, methods, and arrow/function expressions bound to a const or property; own body only,
// nested functions excluded): called names, thrown class names, effect kinds. Per class: heritage names. Per file:
// relative imports resolved to folder paths, exported names, module-level const names.
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
const [tsHost, input, output] = process.argv.slice(2);
const ts = createRequire(tsHost + '/package.json')('typescript');
const files = JSON.parse(fs.readFileSync(input, 'utf8'));
const EFFECTS = {
  'file-write': ['writeFileSync', 'appendFileSync', 'writeFile', 'appendFile', 'mkdirSync', 'mkdir', 'rmSync', 'rm',
    'unlinkSync', 'unlink', 'renameSync', 'rename', 'copyFileSync', 'copyFile', 'cpSync', 'rmdirSync', 'truncateSync',
    'createWriteStream', 'symlinkSync', 'linkSync', 'chmodSync'],
  subprocess: ['spawn', 'spawnSync', 'exec', 'execSync', 'execFile', 'execFileSync', 'fork'],
  network: ['fetch', 'request', 'createServer', 'connect', 'WebSocket', 'listen'],
  environment: ['exit', 'chdir', 'kill'],
};
const EFFECT_OWNERS = { 'file-write': ['fs', 'fsp', 'promises', 'fs.promises', 'fsPromises', 'node:fs'],
  subprocess: ['child_process', 'cp', 'childProcess'], network: ['http', 'https', 'net', 'globalThis', ''],
  environment: ['process'] };
const result = { functions: {}, classes: {}, files: {} };
const isFn = n => ts.isFunctionDeclaration(n) || ts.isMethodDeclaration(n) || ts.isArrowFunction(n) ||
  ts.isFunctionExpression(n) || ts.isConstructorDeclaration(n) || ts.isGetAccessor(n) || ts.isSetAccessor(n);
function fnName(node) {
  if (ts.isConstructorDeclaration(node)) return 'constructor';
  if (node.name && (ts.isIdentifier(node.name) || ts.isPrivateIdentifier(node.name))) return node.name.text;
  const parent = node.parent;
  if (parent && ts.isVariableDeclaration(parent) && ts.isIdentifier(parent.name)) return parent.name.text;
  if (parent && ts.isPropertyAssignment(parent) && ts.isIdentifier(parent.name)) return parent.name.text;
  if (parent && ts.isPropertyDeclaration(parent) && ts.isIdentifier(parent.name)) return parent.name.text;
  return undefined;
}
function resolveImport(from, spec) {
  if (!spec.startsWith('.')) return undefined;
  const base = path.posix.normalize(path.posix.join(path.posix.dirname(from), spec)).replace(/\.js$/, '');
  for (const candidate of [base + '.ts', base + '.tsx', base + '/index.ts', base]) if (files[candidate] !== undefined) return candidate;
  return undefined;
}
for (const [file, text] of Object.entries(files)) {
  if (!/\.tsx?$/.test(file)) continue;
  const source = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true);
  const facts = { imports: [], exports: [], consts: [] };
  result.files[file] = facts;
  const stack = [];
  const duplicate = new Set();
  function ownBody(node, info) {
    ts.forEachChild(node, child => {
      if (isFn(child) || ts.isClassDeclaration(child) || ts.isClassExpression(child)) { visit(child); return; }
      if (ts.isCallExpression(child) || ts.isNewExpression(child)) {
        const callee = child.expression;
        let name, owner = '';
        if (ts.isIdentifier(callee)) name = callee.text;
        else if (ts.isPropertyAccessExpression(callee)) { name = callee.name.text; owner = callee.expression.getText(source); }
        if (name && ts.isCallExpression(child)) info.calls.add(name);
        if (name) for (const [kind, names] of Object.entries(EFFECTS))
          if (names.includes(name) && (EFFECT_OWNERS[kind].includes(owner) || (kind === 'file-write' && /(^|\.)(fs|fsp|promises)$/.test(owner)) ||
              (kind === 'subprocess' && owner === '' && ['spawn', 'spawnSync', 'execSync', 'execFileSync', 'execFile', 'fork'].includes(name)) ||
              (kind === 'network' && owner === '' && ['fetch', 'WebSocket'].includes(name))))
            info.effects.add(kind);
      }
      if (ts.isThrowStatement(child) && child.expression) {
        const e = child.expression;
        if (ts.isNewExpression(e)) info.throws.add(e.expression.getText(source).split('.').pop());
      }
      ownBody(child, info);
    });
  }
  function visit(node) {
    if (ts.isImportDeclaration(node) && ts.isStringLiteral(node.moduleSpecifier)) {
      const target = resolveImport(file, node.moduleSpecifier.text);
      if (target) facts.imports.push(target);
    }
    if (ts.isExportDeclaration(node) && node.moduleSpecifier && ts.isStringLiteral(node.moduleSpecifier)) {
      const target = resolveImport(file, node.moduleSpecifier.text);
      if (target) facts.imports.push(target);
    }
    const exported = ts.canHaveModifiers(node) && (ts.getModifiers(node) ?? []).some(m => m.kind === ts.SyntaxKind.ExportKeyword);
    if (ts.isVariableStatement(node) && node.parent === source) {
      for (const d of node.declarationList.declarations) if (ts.isIdentifier(d.name)) {
        if (exported) facts.exports.push(d.name.text);
        if (node.declarationList.flags & ts.NodeFlags.Const && /^[A-Z][A-Z0-9_]{3,}$/.test(d.name.text)) facts.consts.push(d.name.text);
      }
    }
    if ((ts.isFunctionDeclaration(node) || ts.isClassDeclaration(node)) && exported && node.name) facts.exports.push(node.name.text);
    if (ts.isClassDeclaration(node) || ts.isClassExpression(node)) {
      const name = node.name?.text ?? (node.parent && ts.isVariableDeclaration(node.parent) && ts.isIdentifier(node.parent.name) ? node.parent.name.text : undefined);
      if (name) {
        const key = `${file}:${[...stack, name].join('.')}`;
        if (result.classes[key]) duplicate.add(key);
        result.classes[key] = (node.heritageClauses ?? []).flatMap(h => h.types.map(t => t.expression.getText(source).split('.').pop()));
        stack.push(name);
        ts.forEachChild(node, visit);
        stack.pop();
        return;
      }
    }
    if (isFn(node)) {
      const name = fnName(node);
      if (name && node.body) {
        const key = `${file}:${[...stack, name].join('.')}`;
        if (result.functions[key]) duplicate.add(key);
        const info = { calls: new Set(), throws: new Set(), effects: new Set(), exported: !!exported || facts.exports.includes(name) };
        result.functions[key] = info;
        stack.push(name);
        ownBody(node.body, info);
        stack.pop();
        return;
      }
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
  for (const key of duplicate) { delete result.functions[key]; delete result.classes[key]; }
}
for (const info of Object.values(result.functions)) for (const k of ['calls', 'throws', 'effects']) info[k] = [...info[k]].sort();
for (const facts of Object.values(result.files)) facts.imports = [...new Set(facts.imports)].sort();
fs.writeFileSync(output, JSON.stringify(result));
