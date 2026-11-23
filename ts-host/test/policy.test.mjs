import assert from 'node:assert/strict';
import { test } from 'node:test';
import ts from 'typescript';
import { checkConstrainedSource } from '../dist/compiler/policy.js';
import { createVirtualProgram } from '../dist/compiler/host.js';

const check = source =>
  checkConstrainedSource(ts.createSourceFile('policy-case.ts', source, ts.ScriptTarget.ES2022, true));

test('static iteration-unbounded flags iterateOn(...).until(tsPredicate) without measure or limit', () => {
  for (const source of [
    'step.iterateOn(initial).until(s => s.done);',
    'step.iterateOn(initial).until(function (s) { return s.done; });',
    'step.iterateOn(initial).until(isDone);',
    'iterateOn(step, initial).until(s => s.done);',
    'folder.iterateOn(reducer, initial).onStep(log).until((s, f) => s.done);',
    'step.iterateOn(initial).withSiteId("loop").checkProgress("off").until(s => s.done);',
  ]) {
    const diags = check(source);
    const unbounded = diags.filter(item => item.code === 'iteration-unbounded');
    assert.equal(unbounded.length, 1, `${source} -> ${JSON.stringify(diags)}`);
    assert.match(unbounded[0].message, /withMeasure|withLimit/);
  }
});

test('static iteration-unbounded admits bounded chains, nl predicates, and unrelated .until calls', () => {
  for (const source of [
    'step.iterateOn(initial).withMeasure(s => s.remaining).until(s => s.remaining === 0);',
    'step.iterateOn(initial).withLimit({ maxSteps: 10 }).until(s => s.done);',
    'iterateOn(step, initial).withLimit({ maxSteps: 5 }).onStep(log).until(isDone);',
    'step.iterateOn(initial).until(nl`The draft is ready.`);',
    'step.iterateOn(initial).until(nl<boolean>`The draft is ready.`);',
    'step.iterateOn(initial).until(nl.with({ rubric })`The draft satisfies rubric.`);',
    'step.iterateOn(initial).until(nl`The draft is ready.`.in(ctx));',
    'const done = nl<boolean>`The draft is ready.`; step.iterateOn(initial).until(done);',
    'timer.until(s => s.done);',
  ]) {
    const diags = check(source);
    assert.deepEqual(diags, [], `${source} -> ${JSON.stringify(diags)}`);
  }
});

test('checker-backed policy distinguishes typed natlang callables from TypeScript predicate identifiers', () => {
  const source = `
    declare const step: NatlangFunction<[state: { n: number }], { n: number }>;
    declare const nlDone: NatlangFunction<[state: { n: number }], boolean>;
    const tsDone = (s: { n: number }) => s.n >= 3;
    export async function ok() {
      return await step.iterateOn({ n: 0 }).until(nlDone);
    }
    export async function bad() {
      return await step.iterateOn({ n: 0 }).until(tsDone);
    }
  `;
  const program = createVirtualProgram({ '/scope/main.ts': source });
  const file = program.getSourceFile('/scope/main.ts');
  const diags = checkConstrainedSource(file, { checker: program.getTypeChecker() });
  const unbounded = diags.filter(item => item.code === 'iteration-unbounded');
  assert.equal(unbounded.length, 1, JSON.stringify(diags));
  assert.ok(unbounded[0].start > source.indexOf('export async function bad()'));
});
