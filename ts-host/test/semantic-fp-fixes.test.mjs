import assert from 'node:assert/strict';
import { test } from 'node:test';
import ts from 'typescript';
import { createVirtualProgram } from '../dist/compiler/host.js';
import { analyzeInlineLambdas } from '../dist/compiler/inline.js';
import { compileProject } from '../dist/compiler/project.js';
import { checkConstrainedSource } from '../dist/compiler/policy.js';
import { virtualProjectFiles } from '../dist/runtime/virtual-project.js';
import { parseType, TypeEnv } from '../dist/native/types.js';
import { APPROACH_PROMPT } from '../dist/native/prompt.js';
import { crispExamples } from '../playground/examples/crisp.mjs';

function analyze(source, declarations = '') {
  const files = { '/scope/decls.ts': declarations, '/scope/main.ts': source };
  const program = createVirtualProgram(files);
  return analyzeInlineLambdas(program, [program.getSourceFile('/scope/main.ts')], {
    scopeFiles: [program.getSourceFile('/scope/decls.ts')],
  });
}

test('inline nl supports Is<T, P>, nested Is, Untrusted<Is<T, P>>, and object intersections', () => {
  const decls = `
    type Base = { id: string };
    type Reply = Is<string, "polite and concise">;
    declare const input: Untrusted<string>;
  `;
  const source = `
    async function run() {
      const a: Is<string, "one line of at most 60 characters"> = await nl\`Summarize input.\`(input);
      const b = await nl<Is<Is<string, "polite">, "under two sentences">>\`Reply to input.\`(input);
      const c = await nl<Untrusted<Is<string, "factual">>>\`Extract fact from input.\`(input);
      const d = await nl<Is<Untrusted<string>, "factual">>\`Extract fact from input.\`(input);
      const e = await nl<Reply>\`Draft a reply to input.\`(input);
      const f = await nl<Base & { score: number }>\`Score input.\`(input);
      return { a, b, c, d, e, f };
    }
  `;
  const { plans, diagnostics } = analyze(source, decls);
  assert.deepEqual(diagnostics, []);
  assert.equal(plans.length, 6);
  assert.equal(plans[0].returns.natlang, 'Is<string, "one line of at most 60 characters">');
  assert.equal(plans[1].returns.natlang, 'Is<Is<string, "polite">, "under two sentences">');
  assert.equal(plans[2].returns.natlang, 'Untrusted<Is<string, "factual">>');
  assert.equal(plans[3].returns.natlang, 'Untrusted<Is<string, "factual">>');
  assert.equal(plans[4].returns.natlang, 'Is<string, "polite and concise">');
  assert.equal(plans[5].returns.natlang, 'Base & { score: number }');
  assert.deepEqual(plans[5].returns.aliases, { Base: '{ id: string }' });

  // Every emitted natlang type string parses and resolves cleanly in native/types.ts.
  for (const plan of plans) {
    const env = new TypeEnv(Object.fromEntries(Object.entries(plan.returns.aliases).map(([k, v]) => [k, parseType(v)])));
    const parsed = parseType(plan.returns.natlang);
    env.checkNames(parsed);
  }
});

test('inline nl still rejects non-object intersections', () => {
  const { diagnostics } = analyze('async function run(x: string) { return await nl<string & { extra: number }>`Do x.`(x); }');
  assert.equal(diagnostics.length, 1);
  assert.match(diagnostics[0].message, /intersection types are not supported/);
});

test('compileProject reports refinedSlots for direct and aliased inline Is<T, P> calls', () => {
  const files = virtualProjectFiles({
    'main.ts': `
      import { nl, type Is } from '@natlang/node';
      type Reply = Is<string, "polite and concise">;
      export async function respond(ticket: string): Promise<{ r1: Reply; r2: Is<string, "one line"> }> {
        const r1 = await nl<Reply>\`Answer ticket.\`(ticket);
        const r2 = await nl<Is<string, "one line">>\`Title ticket.\`(ticket);
        return { r1, r2 };
      }
    `,
  });
  const built = compileProject({ project: '/project', files, emit: false, write: false, surfaceSpecifiers: ['@natlang/node'] });
  assert.equal(built.ok, true, JSON.stringify(built.diagnostics));
  assert.equal(built.manifest.inline.length, 2);
  const predicates = (built.refinedSlots ?? []).map(slot => slot.predicate).sort();
  assert.deepEqual(predicates, ['one line', 'polite and concise']);
});

test('NatlangFunction.in(context), crisp Folder reducers, and Folder/FileHandle moveTo type-check cleanly', () => {
  const files = virtualProjectFiles({
    'greet.nl': '---\nargs:\n  name: string\nreturns: string\n---\nGreet name.\n',
    'main.ts': `
      import { type Context, type Folder } from '@natlang/node';
      import greet from './greet.nl';
      export async function run(ctx: Context, folder: Folder) {
        const bound = greet.in(ctx);
        const message: string = await bound('Ada');
        const crispStep = async (f: Folder, state: { remaining: number }) => {
          await f.file('a.txt').moveTo(f.dir('out'));
          return { remaining: state.remaining - 1 };
        };
        const out = await folder.apply(crispStep, { remaining: 1 });
        const prop = await folder.propose(crispStep, { remaining: 1 });
        const iter = await folder.iterateOn(crispStep, { remaining: 2 })
          .withMeasure(s => s.remaining)
          .until(s => s.remaining === 0);
        return { message, out, prop, iter };
      }
    `,
  });
  const built = compileProject({ project: '/project', files, emit: false, write: false, surfaceSpecifiers: ['@natlang/node'] });
  assert.equal(built.ok, true, JSON.stringify(built.diagnostics));
  assert.equal(built.manifest.named.length, 1);
});

test('all playground crisp examples satisfy natlang finite-iteration source policy', () => {
  for (const example of crispExamples) {
    for (const [path, text] of Object.entries(example.files)) {
      if (!path.endsWith('.ts')) continue;
      const sf = ts.createSourceFile(path, text, ts.ScriptTarget.Latest, true);
      const diags = checkConstrainedSource(sf);
      assert.deepEqual(diags, [], `${example.id} (${path}) failed constrained policy: ${JSON.stringify(diags)}`);
    }
  }
});

test('APPROACH_PROMPT accurately reflects persistent top-level eval helpers', () => {
  assert.match(APPROACH_PROMPT, /Top-level helper functions and supported local values carry forward/);
  assert.doesNotMatch(APPROACH_PROMPT, /Ordinary JavaScript function declarations and function-valued locals are scoped to the eval/);
});
