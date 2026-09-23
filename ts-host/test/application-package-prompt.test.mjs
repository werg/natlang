import assert from 'node:assert/strict';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { NativeNatlangHost, NativeSourceWorkspace, TypeScriptEnvironment } from '../dist/index.js';

test('model prompt lists only installed direct application dependencies', async t => {
  const workspace = await mkdtemp(join(tmpdir(), 'natlang-package-prompt-'));
  t.after(() => rm(workspace, { recursive: true, force: true }));
  const fixture = join(workspace, 'fixture');
  await mkdir(fixture);
  await writeFile(join(fixture, 'package.json'), JSON.stringify({
    name: '@fixture/math', version: '1.0.0', type: 'module', exports: './index.js',
  }));
  await writeFile(join(fixture, 'index.js'), 'export const answer = 42;\n');
  await writeFile(join(workspace, 'package.json'), JSON.stringify({
    name: 'natlang-package-prompt', private: true, type: 'module',
    dependencies: { '@fixture/math': 'file:./fixture' },
  }));
  const environment = new TypeScriptEnvironment({ workspace });
  t.after(() => environment.close());
  assert.deepEqual(environment.packages.listAvailableDependencies(), []);
  await environment.installPackages(['file:./fixture']);
  await writeFile(join(workspace, 'package.json'), JSON.stringify({
    name: 'natlang-package-prompt', private: true, type: 'module',
    dependencies: { '@fixture/math': 'file:./fixture', 'not-installed': '1.0.0' },
  }));
  assert.deepEqual(environment.packages.listAvailableDependencies(), ['@fixture/math']);
  const host = new NativeNatlangHost({ environment });
  t.after(() => host.close());
  const prompts = [];
  let turn = 0;
  const result = await host.run({ source: { kind: 'program', program: { $lambda: {
    type: '() => number', instructions: 'Import the available math dependency and return its answer.',
  } } }, modelTurn: request => {
    prompts.push(request.messages[0].content);
    return ++turn === 1
      ? { calls: [['eval', { code: "import { answer } from '@fixture/math'; return answer;" }]], completion_tokens: 1 }
      : { calls: [['mark_lines', { start: 1 }]], completion_tokens: 1 };
  }, options: { model: { max_turns: 3 } } });
  assert.equal(result.outcome.kind, 'done', JSON.stringify(result.outcome));
  assert.equal(result.value, 42);
  assert.equal(prompts.length, 2);
  for (const prompt of prompts) {
    assert.match(prompt, /Importable application dependencies from package\.json:/);
    assert.match(prompt, /@fixture\/math/);
    assert.doesNotMatch(prompt, /not-installed/);
    assert.match(prompt, /Static imports and await import/);
  }
  const child = new NativeSourceWorkspace({ answer: {
    returns: 'number', instructions: 'Import the available math dependency and return its answer.',
  } }, 'answer');
  const childPrompts = [];
  turn = 0;
  const childResult = await child.invoke('answer', {}, { environment, modelTurn: request => {
    childPrompts.push(request.messages[0].content);
    return ++turn === 1
      ? { calls: [['eval', { code: "import { answer } from '@fixture/math'; return answer;" }]], completion_tokens: 1 }
      : { calls: [['mark_lines', { start: 1 }]], completion_tokens: 1 };
  } });
  assert.equal(childResult.outcome, 'done');
  assert.equal(childResult.value, 42);
  assert.ok(childPrompts.every(prompt => prompt.includes('@fixture/math')));
});

test('model prompt refreshes after installing a declared package during the run', async t => {
  const workspace = await mkdtemp(join(tmpdir(), 'natlang-package-refresh-'));
  t.after(() => rm(workspace, { recursive: true, force: true }));
  await mkdir(join(workspace, 'fixture'));
  await writeFile(join(workspace, 'fixture', 'package.json'), JSON.stringify({
    name: 'refresh-fixture', version: '1.0.0', type: 'module', exports: './index.js',
  }));
  await writeFile(join(workspace, 'fixture', 'index.js'), 'export const answer = 42;\n');
  await writeFile(join(workspace, 'package.json'), JSON.stringify({
    name: 'natlang-package-refresh', private: true, type: 'module',
    dependencies: { 'refresh-fixture': 'file:./fixture' },
  }));
  const host = new NativeNatlangHost({ workspace });
  t.after(() => host.close());
  const prompts = [];
  let turn = 0;
  const result = await host.run({ source: { kind: 'program', program: { $lambda: {
    type: '() => number', instructions: 'Install the declared dependency and return its answer.',
  } } }, modelTurn: request => {
    prompts.push(request.messages[0].content);
    return ++turn === 1
      ? { calls: [['eval', { code: 'await installPackages([]);' }]], completion_tokens: 1 }
      : turn === 2
        ? { calls: [['eval', { code: "import { answer } from 'refresh-fixture'; return answer;" }]], completion_tokens: 1 }
        : { calls: [['mark_lines', { start: 1 }]], completion_tokens: 1 };
  }, options: { model: { max_turns: 4 } } });
  assert.equal(result.outcome.kind, 'done', JSON.stringify(result.outcome));
  assert.equal(result.value, 42);
  assert.equal(prompts.length, 3);
  assert.doesNotMatch(prompts[0], /Importable application dependencies from package\.json:/);
  assert.match(prompts[1], /Importable application dependencies from package\.json:\n- "refresh-fixture"/);
  assert.match(prompts[2], /Importable application dependencies from package\.json:\n- "refresh-fixture"/);
});
