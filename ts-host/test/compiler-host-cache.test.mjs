import assert from 'node:assert/strict';
import test from 'node:test';
import ts from 'typescript';
import { createNatlangCompilerHost, createVirtualProgram, EVAL_COMPILER_OPTIONS } from '../dist/compiler/host.js';
import { INTRINSICS_FILE } from '../dist/compiler/intrinsics.js';

const target = ts.ScriptTarget.ES2022;
const host = (virtual = [], extra = {}) => createNatlangCompilerHost({
  options: EVAL_COMPILER_OPTIONS, virtual: new Map(virtual), ...extra,
});

test('user and generated declarations remain owned by each compiler host', () => {
  for (const path of ['/__natlang__/pkg/user.ts', '/__natlang__/pkg-export/1/types.d.ts',
    '/__natlang__/folder/user.ts', '/__natlang__/services.d.ts', '/__natlang__/eval/scope.ts',
    '/__natlang__/rewrite/snippet.ts', '/__natlang__/lib/lib.es2022.d.ts']) {
    const text = path.endsWith('.d.ts') ? 'declare const userValue: number;' : 'const userValue = 1;';
    const first = host([[path, text]]).getSourceFile(path, target);
    const second = host([[path, text]]).getSourceFile(path, target);
    assert.notEqual(first, second, path);
    assert.equal(first.text, second.text);
    const changed = host([[path, text.replace('userValue', 'newValue')]]).getSourceFile(path, target);
    assert.match(changed.text, /newValue/);
    assert.match(first.text, /userValue/);
  }
});

test('synthetic program files retain forced module detection', () => {
  const program = createVirtualProgram({
    '/__natlang__/eval/first.ts': 'const sameName = 1;',
    '/__natlang__/eval/second.ts': 'const sameName = 2;',
  }, { ...EVAL_COMPILER_OPTIONS, noLib: true, moduleDetection: ts.ModuleDetectionKind.Force });
  assert.deepEqual(program.getSemanticDiagnostics().filter(item => item.code === 2451), []);
  assert.ok(ts.isExternalModule(program.getSourceFile('/__natlang__/eval/first.ts')));
  assert.ok(ts.isExternalModule(program.getSourceFile('/__natlang__/eval/second.ts')));
});

test('immutable sharing preserves full parse options and is scoped to that options owner', () => {
  const moduleOptions = { languageVersion: target, setExternalModuleIndicator(file) { file.externalModuleIndicator = file; } };
  const plainOptions = { languageVersion: target, setExternalModuleIndicator() {} };
  const first = host().getSourceFile(INTRINSICS_FILE, moduleOptions);
  assert.equal(ts.isExternalModule(first), true);
  assert.equal(host().getSourceFile(INTRINSICS_FILE, moduleOptions), first);
  const plain = host().getSourceFile(INTRINSICS_FILE, plainOptions);
  assert.notEqual(plain, first);
  assert.equal(ts.isExternalModule(plain), false);
  assert.notEqual(host().getSourceFile(INTRINSICS_FILE, { ...moduleOptions }), first);
});

test('immutable libraries can be shared, explicitly disabled, or requested fresh', () => {
  const library = '/__natlang__/lib/lib.es2022.d.ts';
  const libs = name => name === 'lib.es2022.d.ts' ? 'declare const libraryValue: number;' : undefined;
  const firstHost = host([], { libs });
  const first = firstHost.getSourceFile(library, target);
  assert.equal(host([], { libs }).getSourceFile(library, target), first);
  assert.notEqual(firstHost.getSourceFile(library, target, undefined, true), first);
  assert.notEqual(host([], { libs, shareLibrarySourceFiles: false }).getSourceFile(library, target), first);
  assert.notEqual(host([], { libs: () => 'declare const revisedLibrary: string;' }).getSourceFile(library, target), first);
});


test('mutating a parse-options owner invalidates its previous library ASTs', () => {
  const options = { languageVersion: target, setExternalModuleIndicator() {} };
  const compiler = host();
  let previous = compiler.getSourceFile(INTRINSICS_FILE, options);
  options.impliedNodeFormat = ts.ModuleKind.ESNext;
  let current = compiler.getSourceFile(INTRINSICS_FILE, options);
  assert.notEqual(current, previous);
  assert.equal(current.impliedNodeFormat, ts.ModuleKind.ESNext);
  previous = current;
  options.setExternalModuleIndicator = file => { file.externalModuleIndicator = file; };
  current = compiler.getSourceFile(INTRINSICS_FILE, options);
  assert.notEqual(current, previous);
  assert.equal(ts.isExternalModule(current), true);
  previous = current;
  options.jsDocParsingMode = ts.JSDocParsingMode.ParseNone;
  current = compiler.getSourceFile(INTRINSICS_FILE, options);
  assert.notEqual(current, previous);
  assert.equal(current.jsDocParsingMode, ts.JSDocParsingMode.ParseNone);
  assert.equal(compiler.getSourceFile(INTRINSICS_FILE, options), current);
});
