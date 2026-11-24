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

// This mirrors the coordinator/reader/chooser authority split: declarations
// describe admitted values, while the chooser returns a step for its owner.
for (const mode of ['node', 'workerd']) {
  test(`${mode} service declaration, search, analysis and actual values agree on reader versus chooser authority`, async () => {
    const { evalDeclarations } = await import('../dist/runtime/hooks.js');
    const platform = platformFor(mode), observed = [];
    const rt = core.createNatlangRuntime({ platform, calls: false,
      services: { spectroliteGoal: { async follow(target, revision) { observed.push([target, revision]); return `${target}@${revision}`; } } },
      serviceDeclarations: { spectroliteGoal: 'export function follow(target:string,revision:string):Promise<string>;' },
      serviceScopes: { spectroliteGoal: ['owner.ts', 'owner/readKnowledgeNote.nl'] },
      agent: async session => {
        const allowed = session.lam.functionName === 'readKnowledgeNote';
        const scope = evalDeclarations(session);
        assert.equal(scope.services.includes('spectroliteGoal'), allowed);
        assert.equal(Object.hasOwn(scope.serviceDeclarations, 'spectroliteGoal'), allowed);
        assert.equal(session.searchCode('spectroliteGoal').some(row => row.name.startsWith('spectroliteGoal')), allowed);
        const declaration = await session.applyAsync('read_code', { name: 'spectroliteGoal' });
        assert.equal(declaration.kind, allowed ? 'ok' : 'rejected', declaration.text);
        if (!allowed) assert.doesNotMatch(declaration.text, /export function follow/);
        const result = await platform.contextStore.run(undefined, () => session.applyAsync('eval', {
          code: 'return await spectroliteGoal.follow("reader", "r1");', finish: true,
        }));
        if (allowed) assert.equal(result.kind, 'completed', result.text);
        else {
          assert.equal(result.kind, 'rejected', result.text);
          assert.ok(result.codes.includes('callable-scope'), result.text);
          assert.match(result.text, /unavailable in this call/);
          const finished = await session.applyAsync('eval', { code: 'const spectroliteGoal = {label:"step"}; return spectroliteGoal.label;', finish: true });
          assert.equal(finished.kind, 'completed', finished.text);
        }
      } });
    try {
      const compiled = core.compileVirtualProject({ root: '/service-values', files: {
        'owner.ts': 'import {spectroliteGoal} from "natlang:services"; import readKnowledgeNote from "./owner/readKnowledgeNote.nl"; import chooseKnowledgeStep from "./owner/chooseKnowledgeStep.nl"; export default async function owner():Promise<string>{return await spectroliteGoal.follow("coordinator","r1") + ":" + await readKnowledgeNote() + ":" + await chooseKnowledgeStep();}',
        'services.d.ts': 'declare module "natlang:services" {export const spectroliteGoal:{follow(target:string,revision:string):Promise<string>};}',
        'owner/readKnowledgeNote.nl': nl('Read the actual granted source.'),
        'owner/chooseKnowledgeStep.nl': nl('Return the next step for the coordinator.'),
      } }, core, { platform, programId: 'explicit-service-owner', services: {declarations: { spectroliteGoal: 'export function follow(target:string,revision:string):Promise<string>;' },scopes:{spectroliteGoal:['owner.ts','owner/readKnowledgeNote.nl']}} });
      assert.equal(compiled.ok, true, JSON.stringify(compiled.diagnostics));
      assert.equal(await rt.run(() => compiled.require('owner.ts').default()), 'coordinator@r1:reader@r1:step');
      assert.deepEqual(observed, [['coordinator','r1'],['reader','r1']], 'chooser never invokes the hidden native port');
    } finally { rt.close(); }
  });

  test(`${mode} a local or input sharing a hidden service name remains an ordinary value`, async () => {
    const platform = platformFor(mode);
    const rt = core.createNatlangRuntime({ platform, calls:false,
      services:{spectroliteGoal:{follow(){throw Error('Hidden port must never run');}}},
      serviceScopes:{spectroliteGoal:[]},
      serviceDeclarations:{spectroliteGoal:'export function follow():string;'},
      agent:async session=>{
        const result=await session.applyAsync('eval',{code:'return spectroliteGoal.label;',finish:true});
        assert.equal(result.kind,'completed',result.text);
      } });
    try {
      const compiled=core.compileVirtualProject({root:'/shadowed-service',files:{
        'root.nl':'---\nargs: {spectroliteGoal: {label:string}}\nreturns: string\n---\nReturn the input label.',
        'owner.ts':'import root from "./root.nl"; export default async function owner():Promise<string>{return await root({label:"input-value"});}',
      }},core,{platform,programId:'shadowed-service'});
      assert.equal(compiled.ok,true,JSON.stringify(compiled.diagnostics));
      assert.equal(await rt.run(()=>compiled.require('owner.ts').default()),'input-value');
    } finally {rt.close();}
  });
}
