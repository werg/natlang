import assert from 'node:assert/strict';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { createRequire } from 'node:module';
import { resolve, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { test } from 'node:test';

const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const nodeRoot = join(root, 'npm-packages/node');
const browserRoot = join(root, 'npm-packages/browser');
const requireNodePackage = createRequire(join(nodeRoot, 'package.json'));
const requireBrowserPackage = createRequire(join(browserRoot, 'package.json'));
const isStaged = (directory, entry) => statSync(join(directory, entry), { throwIfNoEntry: false })?.isFile() ?? false;
const importNodeSubpath = specifier => import(pathToFileURL(requireNodePackage.resolve(specifier)));

test('staged Node package resolves adaptation SDK, optimization SDK, and CLI exports', { skip: !isStaged(nodeRoot, 'dist/adaptation/node.js') }, async () => {
  const pkg = JSON.parse(readFileSync(join(nodeRoot, 'package.json'), 'utf8'));
  for (const key of ['adaptation', 'evaluation', 'optimize', 'cli']) {
    const specifier = `@natlang/node/${key}`;
    const resolved = requireNodePackage.resolve(specifier);
    assert.equal(resolved, resolve(nodeRoot, pkg.exports[`./${key}`].default));
    const api = await import(pathToFileURL(resolved));
    assert.ok(Object.keys(api).length > 0, `${specifier} exports its API`);
  }
  const adaptation = await importNodeSubpath('@natlang/node/adaptation');
  const evaluation = await importNodeSubpath('@natlang/node/evaluation');
  const optimize = await importNodeSubpath('@natlang/node/optimize');
  const cli = await importNodeSubpath('@natlang/node/cli');
  for (const name of ['bindAdaptation', 'parseAdaptation', 'validateAdaptation']) assert.equal(typeof adaptation[name], 'function');
  for (const name of ['defineEvaluationSuite', 'prepareEvaluationSuite', 'evaluate']) assert.equal(typeof evaluation[name], 'function');
  for (const name of ['optimize', 'resumeOptimization', 'loadAdaptation', 'exportAdaptationPatch']) assert.equal(typeof optimize[name], 'function');
  assert.equal(typeof cli.main, 'function');
});

test('staged Node npm contents include Ax provenance and license notices', { skip: !isStaged(nodeRoot, 'vendor/ax-gepa/UPSTREAM.json') }, () => {
  const pkg = JSON.parse(readFileSync(join(nodeRoot, 'package.json'), 'utf8'));
  assert.ok(pkg.files.includes('vendor'));
  const staged = join(nodeRoot, 'vendor/ax-gepa');
  for (const name of ['LICENSE', 'CHANGES.md', 'UPSTREAM.json'])
    assert.equal(readFileSync(join(staged, name), 'utf8'), readFileSync(join(root, 'vendor/ax-gepa', name), 'utf8'), `${name} staged byte for byte`);
  const manifest = JSON.parse(readFileSync(join(staged, 'UPSTREAM.json'), 'utf8'));
  assert.equal(manifest.commit, 'b780a14a3cb94d5ac572db04038399aef655c76c');
  assert.ok(manifest.sourcePaths.some(entry => entry.path === 'LICENSE'));
});

test('staged browser package keeps portable adaptation types and omits optimizer assets', { skip: !isStaged(browserRoot, 'dist/types/browser/index.d.ts') }, () => {
  const typeRoot = join(browserRoot, 'dist/types');
  const files = directory => readdirSync(directory).flatMap(name => {
    const path = join(directory, name); return statSync(path).isDirectory() ? files(path) : [path];
  });
  const declarations = files(typeRoot).filter(path => path.endsWith('.d.ts'));
  assert.ok(declarations.some(path => path.endsWith('/adaptation/types.d.ts')));
  assert.ok(!declarations.some(path => /\/(?:evaluation|optimization)\//.test(path)), 'no evaluation or optimizer declarations staged');
  for (const declaration of declarations) {
    const text = readFileSync(declaration, 'utf8');
    assert.doesNotMatch(text, /from ['"][^'"]*\/(?:evaluation|optimization)\//, declaration);
  }
  const browserTypes = readFileSync(join(typeRoot, 'browser/index.d.ts'), 'utf8');
  assert.match(browserTypes, /adaptation\/index\.js/);
  const assets = files(join(browserRoot, 'dist')).filter(path => path.endsWith('.js'));
  const bundled = assets.map(path => readFileSync(path, 'utf8')).join('\n');
  assert.doesNotMatch(bundled, /ComponentSelector|UsageGateway|defineEvaluationSuite|resumeOptimization/);
  assert.deepEqual(Object.keys(JSON.parse(readFileSync(join(browserRoot, 'package.json'), 'utf8')).exports), ['.']);
  assert.ok(requireBrowserPackage.resolve('@natlang/browser'));
});
