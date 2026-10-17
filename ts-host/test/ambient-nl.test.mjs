import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createVirtualProgram } from '../dist/compiler/host.js';
import { analyzeInlineLambdas } from '../dist/compiler/inline.js';
import { checkAuthoring } from '../dist/teacher/authoring.js';

function analyze(source, extra = {}) {
  const program = createVirtualProgram({ '/scope/main.ts': source, ...extra });
  return analyzeInlineLambdas(program, [program.getSourceFile('/scope/main.ts')]);
}
const callable = '<T>(instructions: TemplateStringsArray, ...values: unknown[]) => (...args: unknown[]) => Promise<T>';
const call = 'export async function judge(note: string): Promise<boolean> { return await nl<boolean>`Judge note.`(note); }';

test('erased callable const/let/var and function declarations retain the builtin tag', () => {
  for (const declaration of [
    ...['const', 'let', 'var'].map(kind => `declare ${kind} nl: ${callable};`),
    'declare function nl<T>(instructions: TemplateStringsArray, ...values: unknown[]): (...args: unknown[]) => Promise<T>;',
  ]) {
    const result = analyze(declaration + call);
    assert.deepEqual(result.diagnostics, [], declaration);
    assert.equal(result.plans.length, 1, declaration);
  }
});

test('ambient callable with captures resolves through the builtin base', () => {
  const source = `declare const nl: { <T>(instructions: TemplateStringsArray, ...values: unknown[]): (...args: unknown[]) => Promise<T>; with<T>(captures: object): ${callable} };\n` +
    'export async function judge(note: string): Promise<boolean> { const policy="urgent"; return await nl.with<boolean>({policy})`Judge note against policy.`(note); }';
  const result = analyze(source);
  assert.deepEqual(result.diagnostics, []);
  assert.equal(result.plans.length, 1);
  assert.deepEqual(result.plans[0].captures.map(item => item.name), ['policy']);
});

test('runtime bindings and custom imports are never promoted to builtin tags', () => {
  for (const declaration of [
    `const nl: ${callable} = null as any;`,
    'function nl<T>(strings: TemplateStringsArray): (note:string)=>Promise<T> { return null as any; }',
    'class nl {}',
  ]) {
    const result = analyze(declaration + call);
    assert.equal(result.plans.length, 0);
    assert.ok(result.diagnostics.some(item => item.code === 'nl-shadowed'));
  }
  const parameter = analyze('export async function judge(nl: any, note: string): Promise<boolean> { return await nl`Judge note.`(note); }');
  assert.equal(parameter.plans.length, 0);
  assert.ok(parameter.diagnostics.some(item => item.code === 'nl-shadowed'));
  const imported = analyze('import {nl} from "./custom";' + call, {
    '/scope/custom.ts': `export declare const nl: ${callable};`,
  });
  assert.equal(imported.plans.length, 0);
});

test('ambient typing executes actual compiled authoring and retains all independent oracle checks', async () => {
  const files = { 'app.ts': `declare const nl: ${callable};\n` +
    'export async function urgent(tickets: {id:string,text:string}[]): Promise<string[]> { const flags=await Promise.all(tickets.map(ticket=>nl<boolean>`Does ticket report an outage?`(ticket))); return tickets.filter((ticket,i)=>flags[i]).map(ticket=>ticket.id); }' };
  const spec = { module:'app.ts', export:'urgent', requires:{nl:true}, runs:[
    {args:[[{id:'a',text:'site down'},{id:'b',text:'typo'}]],expected:['a']},
    {args:[[{id:'b',text:'typo'}]],expected:[]},
  ],oracle:[{match:['"a"'],value:true},{match:['"b"'],value:false}] };
  assert.deepEqual(await checkAuthoring(files,spec),{ok:true,problems:[],results:[['a'],[]]});
  assert.equal((await checkAuthoring({'app.ts':files['app.ts'].replace('flags[i]','!flags[i]')},spec)).ok,false);
  assert.equal((await checkAuthoring({'app.ts':files['app.ts'].replace('declare const nl:', 'const nl:').replace(';\n',' = null as any;\n')},spec)).ok,false);
});


test('callable-folder modules lower explicit capture tags without any bare nl tag', async () => {
  const files = { 'app.ts': 'const policy="An outage is urgent"; export async function urgent(ticket:{id:string,text:string}):Promise<boolean> { return await nl.with<boolean>({policy})`Judge ticket against policy.`(ticket); }' };
  const spec = {module:'app.ts',export:'urgent',requires:{nl:true},runs:[
    {args:[{id:'a',text:'site down'}],expected:true},
    {args:[{id:'b',text:'typo'}],expected:false},
  ],oracle:[{match:['"a"'],value:true},{match:['"b"'],value:false}]};
  assert.deepEqual(await checkAuthoring(files,spec),{ok:true,problems:[],results:[true,false]});
  const factory = {...files,'app.ts':files['app.ts'].replace('export async function urgent(ticket:{id:string,text:string}):Promise<boolean> { return await ', 'const classify:(ticket:{id:string,text:string})=>Promise<boolean> = ').replace('(ticket); }','; export async function urgent(ticket:{id:string,text:string}):Promise<boolean> { return classify(ticket); }')};
  assert.deepEqual(await checkAuthoring(factory,spec),{ok:true,problems:[],results:[true,false]});
});

test('custom tagged templates remain ordinary runtime code', async () => {
  const files={'app.ts':'const label=(parts:TemplateStringsArray)=>parts[0]; export async function value():Promise<string> { return label`plain`; }'};
  const spec={module:'app.ts',export:'value',runs:[{args:[],expected:'plain'}]};
  assert.deepEqual(await checkAuthoring(files,spec),{ok:true,problems:[],results:['plain']});
});


test('an erased declaration may describe only the builtin capture factory', async () => {
  const source='declare const nl: { with<T>(context:Record<string,unknown>): (parts:TemplateStringsArray,...subs:unknown[]) => (input:unknown) => Promise<T> }; const policy="urgent"; export async function judge(note:string):Promise<boolean> { return await nl.with<boolean>({policy})`Judge note against policy.`(note); }';
  const result=analyze(source);assert.deepEqual(result.diagnostics,[]);assert.equal(result.plans.length,1);
  assert.deepEqual(await checkAuthoring({'app.ts':source},{module:'app.ts',export:'judge',requires:{nl:true},runs:[{args:['a'],expected:true}],oracle:[{match:['"a"'],value:true}]}),{ok:true,problems:[],results:[true]});
  const noncallable=analyze('declare const nl: number;'+call);
  assert.equal(noncallable.plans.length,0);assert.ok(noncallable.diagnostics.some(item=>item.code==='nl-shadowed'));
});
