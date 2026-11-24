import assert from 'node:assert/strict';
import test from 'node:test';
import * as core from '../dist/index.js';
import { workerdPlatform } from '../dist/core/workerd.js';
import { currentFrame, runInFrame } from '../dist/runtime/context.js';

const nl = body => `---\nargs: {}\nreturns: string\n---\n${body}`;
function platformFor(mode) {
  return mode === 'node' ? core.nodePlatform() : workerdPlatform({
    realm: { call(body, params, args) { return Function(...params, body)(...args); } },
  });
}
const services = Object.fromEntries(['exact', 'tree', 'tsTree', 'sibling', 'wrongTs', 'global'].map(name =>
  [name, { read: () => name }]));
const declarations = Object.fromEntries(Object.keys(services).map(name => [name, 'export function read(): string;']));
const serviceScopes = {
  exact: ['judge.nl'], tree: ['judge.nl/**'], tsTree: ['owner.ts/**'],
  sibling: ['other.nl'], wrongTs: ['other.ts/**'],
};
const names = session => Object.keys(session.availableServices()).sort();

for (const mode of ['node', 'workerd']) {
  test(`${mode} service scopes use authored ancestry through program-owned TS and NL calls`, async () => {
    const platform = platformFor(mode), seen = {};
    const rt = core.createNatlangRuntime({ platform, calls: false, services, serviceDeclarations: declarations,
      serviceScopes, agent: async session => {
        seen[session.lam.functionName] = names(session);
        const result = await session.applyAsync('eval', { code: session.lam.functionName === 'judge'
          ? 'return exact.read() + ":" + await child();' : 'return tree.read();', finish: true });
        assert.equal(result.kind, 'completed', result.text);
      } });
    try {
      const compiled = core.compileVirtualProject({ root: '/scoped-program', files: {
        'owner.ts': 'import judge from "./judge.nl"; export default async function owner():Promise<string>{return await judge();}',
        'judge.nl': nl('Ask child and use only the granted services.'),
        'judge/child.nl': nl('Use the inherited service.'),
      } }, core, { platform, programId: 'program-with-explicit-owner', declarations });
      assert.equal(compiled.ok, true, JSON.stringify(compiled.diagnostics));
      assert.equal(await rt.run(() => compiled.require('owner.ts').default()), 'exact:tree');
      assert.deepEqual(seen.judge, ['exact', 'global', 'tree', 'tsTree']);
      assert.deepEqual(seen.child, ['global', 'tree', 'tsTree']);
    } finally { rt.close(); }
  });

  test(`${mode} bound source scopes belong to their session outside or inside another frame`, async () => {
    const platform = platformFor(mode), seen = [];
    const rt = core.createNatlangRuntime({ platform, calls: false, services, serviceDeclarations: declarations,
      serviceScopes, agent: async session => {
        const owned = session.runtime.frame;
        assert.ok(owned.chain.at(-1).includes('bound-program'), 'recursion identity retains program ownership');
        assert.match(owned.chain.at(-1), /#/, 'recursion identity retains bound context');
        assert.equal(owned.serviceChain.at(-1), 'nl:judge.nl');
        const expected = names(session);
        platform.contextStore.run(undefined, () => {
          assert.equal(currentFrame(), undefined);
          assert.deepEqual(names(session), expected, 'RPC callback without ambient frame keeps only owned grants');
        });
        runInFrame({ ...owned, serviceChain: ['ts:other.ts', 'nl:other.nl'] }, () => {
          assert.deepEqual(names(session), expected, 'unrelated ambient frame confers no sibling grants');
        });
        seen.push(expected);
        const result = await session.applyAsync('eval', { code: 'return exact.read();', finish: true });
        assert.equal(result.kind, 'completed', result.text);
      } });
    try {
      const body = 'import {Context} from "@natlang/node"; import judge from "./judge.nl"; ' +
        'export default async function run():Promise<string>{return await judge.in(Context.empty().with({policy:"bound"}))();}';
      const compiled = core.compileVirtualProject({ root: '/bound-program', files: {
        'owner.ts': body, 'other.ts': body, 'judge.nl': nl('Read the granted service under the bound context.'),
      } }, core, { platform, programId: 'bound-program', declarations });
      assert.equal(compiled.ok, true, JSON.stringify(compiled.diagnostics));
      assert.equal(await rt.run(() => compiled.require('owner.ts').default()), 'exact');
      assert.equal(await rt.run(() => compiled.require('other.ts').default()), 'exact');
      assert.deepEqual(seen, [
        ['exact', 'global', 'tree', 'tsTree'],
        ['exact', 'global', 'tree', 'wrongTs'],
      ]);
    } finally { rt.close(); }
  });
}
