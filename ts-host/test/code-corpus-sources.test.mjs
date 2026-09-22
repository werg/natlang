import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve, join } from 'node:path';
import { enumerateExercism } from '../scripts/code-corpus/exercism.mjs';
import { SOURCES, fetchHuggingFaceRows, huggingFaceRequestHeaders, resolveGitCommit } from '../scripts/code-corpus/sources.mjs';
import { collect } from '../scripts/code-corpus/collect.mjs';

async function put(root, path, content) {
  const target = join(root, path);
  await mkdir(resolve(target, '..'), { recursive: true });
  await writeFile(target, content);
}

test('registry covers planned sources and marks deferred adapters explicitly', () => {
  const ids = new Set(SOURCES.map((source) => source.id));
  for (const id of ['es-toolkit', 'radashi', 'simple-statistics', 'exercism-typescript', 'exercism-javascript', 'problem-specifications', 'remeda', 'ramda', 'case2code', 'xlam', 'tiny-codes', 'stack-edu', 'spoc', 'codeact', 'code-feedback', 'commitpackft', 'bugsjs', 'menvdata', 'codesearchnet', 'magicoder', 'mceval', 'deno-std', 'javascript-algorithms', 'd3-array', '30-seconds-of-code']) assert.ok(ids.has(id), `missing ${id}`);
  assert.ok(SOURCES.every((source) => source.url && source.type && source.expected.length && source.adapterReadiness));
  assert.equal(SOURCES.find((source) => source.id === 'stack-edu').adapterReadiness, 'deferred');
  assert.equal(SOURCES.find((source) => source.id === 'codesearchnet').defaultConfig, 'javascript');
  assert.equal(SOURCES.find((source) => source.id === 'codesearchnet').defaultSplit, 'train');
  assert.equal(SOURCES.find((source) => source.id === 'xlam').defaultConfig, 'dataset');
  assert.equal(SOURCES.find((source) => source.id === 'xlam').license, 'CC-BY-4.0');
  assert.equal(SOURCES.find((source) => source.id === 'tiny-codes').license, 'MIT');
  assert.ok(SOURCES.filter((source) => ['codesearchnet','magicoder','mceval','deno-std','javascript-algorithms','d3-array','30-seconds-of-code'].includes(source.id)).every((source) => source.adapterReadiness === 'generic' || source.adapterReadiness === 'ready'));
});

test('Exercism inventory follows declared files and returns instruction, exemplar, and test paths', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'exercism-inventory-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await put(root, 'exercises/practice/two-fer/.meta/config.json', JSON.stringify({ slug: 'two-fer', files: { solution: 'two-fer.ts', exemplar: '.meta/example.ts', test: 'two-fer.test.ts' } }));
  await put(root, 'exercises/practice/two-fer/.docs/instructions.md', 'Return a greeting.');
  await put(root, 'exercises/practice/two-fer/.meta/example.ts', 'export const twoFer = () => "one for you";');
  await put(root, 'exercises/practice/two-fer/two-fer.ts', 'export const twoFer = () => "";');
  await put(root, 'exercises/practice/two-fer/two-fer.test.ts', 'test("works", () => {});');
  const result = await enumerateExercism(root);
  assert.equal(result.length, 1);
  assert.equal(result[0].slug, 'two-fer');
  assert.equal(result[0].instruction, 'Return a greeting.');
  assert.equal(result[0].sourcePath, 'exercises/practice/two-fer/.meta/example.ts');
  assert.deepEqual(result[0].testPaths, ['exercises/practice/two-fer/two-fer.test.ts']);
  assert.deepEqual(result[0].stubPaths, ['exercises/practice/two-fer/two-fer.ts']);
  assert.deepEqual(result[0].inventoryWarnings, []);
});

test('Exercism inventory supports external instruction overrides and conventional exemplar fallback', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'exercism-override-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await put(root, 'exercises/practice/hello-world/.meta/config.json', JSON.stringify({ slug: 'hello-world', files: { test: 'hello-world.spec.js' } }));
  await put(root, 'exercises/practice/hello-world/.meta/example.js', 'export const hello = () => "Hello, World!";');
  await put(root, 'exercises/practice/hello-world/hello-world.spec.js', '');
  const [entry] = await enumerateExercism(root, { descriptionOverrides: { 'hello-world': 'Canonical description.' } });
  assert.equal(entry.instruction, 'Canonical description.');
  assert.equal(entry.instructionPath, null);
  assert.equal(entry.sourcePath, 'exercises/practice/hello-world/.meta/example.js');
});

test('Exercism inventory accepts the track files.example key for canonical reference implementations', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'exercism-example-inventory-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await put(root, 'exercises/practice/strain/.meta/config.json', JSON.stringify({ slug: 'strain', files: { example: '.meta/proof.ci.ts', solution: 'strain.ts', test: 'strain.test.ts' } }));
  await put(root, 'exercises/practice/strain/.docs/instructions.md', 'Implement collection filtering.');
  await put(root, 'exercises/practice/strain/.meta/proof.ci.ts', 'export function keep(values, predicate) { return values.filter(predicate); }');
  await put(root, 'exercises/practice/strain/strain.ts', 'export function keep(values, predicate) { return values; }');
  await put(root, 'exercises/practice/strain/strain.test.ts', '');
  const [entry] = await enumerateExercism(root);
  assert.equal(entry.sourcePath, 'exercises/practice/strain/.meta/proof.ci.ts');
  assert.equal(entry.instruction, 'Implement collection filtering.');
  assert.deepEqual(entry.stubPaths, ['exercises/practice/strain/strain.ts']);
});

test('Hugging Face fetch rejects invalid bounds before making a request', async () => {
  await assert.rejects(fetchHuggingFaceRows('tiny-codes', { length: 1001 }), /between 1 and 1000/);
  await assert.rejects(fetchHuggingFaceRows('es-toolkit'), /Unknown Hugging Face source/);
});

test('Hugging Face requests use optional bearer auth only on the two approved HTTPS hosts', async () => {
  const token='test-token-value';
  assert.deepEqual(huggingFaceRequestHeaders('https://huggingface.co/api/datasets/example',{HF_TOKEN:token}),{Authorization:`Bearer ${token}`});
  assert.deepEqual(huggingFaceRequestHeaders('https://datasets-server.huggingface.co/rows',{HF_TOKEN:token}),{Authorization:`Bearer ${token}`});
  assert.deepEqual(huggingFaceRequestHeaders('http://huggingface.co/api/datasets/example',{HF_TOKEN:token}),{});
  assert.deepEqual(huggingFaceRequestHeaders('https://huggingface.co:444/api/datasets/example',{HF_TOKEN:token}),{});
  assert.deepEqual(huggingFaceRequestHeaders('https://evil.huggingface.co/api/datasets/example',{HF_TOKEN:token}),{});
  assert.deepEqual(huggingFaceRequestHeaders('https://example.test/',{HF_TOKEN:token}),{});
});

test('Hugging Face row fetch sends environment auth without exposing it in errors', async (t) => {
  const savedToken=process.env.HF_TOKEN, savedHubToken=process.env.HUGGINGFACE_HUB_TOKEN, savedFetch=globalThis.fetch;
  process.env.HF_TOKEN='mocked-secret'; delete process.env.HUGGINGFACE_HUB_TOKEN;
  let request;
  globalThis.fetch=async (url, options) => { request={url:String(url),options}; return {ok:true,json:async()=>({rows:[{row:{id:1}}]})}; };
  t.after(()=>{
    globalThis.fetch=savedFetch;
    if (savedToken===undefined) delete process.env.HF_TOKEN; else process.env.HF_TOKEN=savedToken;
    if (savedHubToken===undefined) delete process.env.HUGGINGFACE_HUB_TOKEN; else process.env.HUGGINGFACE_HUB_TOKEN=savedHubToken;
  });
  assert.deepEqual(await fetchHuggingFaceRows('tiny-codes',{length:1}),[{row:{id:1}}]);
  assert.equal(new URL(request.url).hostname,'datasets-server.huggingface.co');
  assert.equal(request.options.headers.Authorization,'Bearer mocked-secret');
  globalThis.fetch=async () => ({ok:false,status:401});
  await assert.rejects(fetchHuggingFaceRows('tiny-codes',{length:1}),error=>!error.message.includes('mocked-secret'));
});

test('bounded collector authenticates Hugging Face metadata and rows requests', async (t) => {
  const root=await mkdtemp(join(tmpdir(),'hf-auth-collect-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  const savedToken=process.env.HF_TOKEN, savedHubToken=process.env.HUGGINGFACE_HUB_TOKEN, savedFetch=globalThis.fetch;
  process.env.HF_TOKEN='mocked-collector-secret'; delete process.env.HUGGINGFACE_HUB_TOKEN;
  const requests=[];
  globalThis.fetch=async (input,options={})=>{
    const url=new URL(String(input)); requests.push({url,headers:options.headers??{}});
    if(url.hostname==='huggingface.co') return {ok:true,json:async()=>({sha:'fixed-dataset-revision'})};
    return {ok:true,json:async()=>({rows:[{row:{query:'Find a tool',tools:[],answers:[]}}]})};
  };
  t.after(()=>{
    globalThis.fetch=savedFetch;
    if(savedToken===undefined) delete process.env.HF_TOKEN; else process.env.HF_TOKEN=savedToken;
    if(savedHubToken===undefined) delete process.env.HUGGINGFACE_HUB_TOKEN; else process.env.HUGGINGFACE_HUB_TOKEN=savedHubToken;
  });
  await collect(join(root,'snapshot'),{ids:['xlam'],limit:1});
  assert.equal(requests.length,3);
  assert.ok(requests.every(({url,headers})=>['huggingface.co','datasets-server.huggingface.co'].includes(url.hostname)&&headers.Authorization==='Bearer mocked-collector-secret'));
});

test('git revision resolver accepts a full SHA directly and resolves named refs with git ls-remote', () => {
  const sha = 'a'.repeat(40);
  let calls = 0;
  assert.equal(resolveGitCommit({ url: 'https://example.test/repo.git' }, sha.toUpperCase(), () => { calls += 1; }), sha);
  assert.equal(calls, 0);
  const resolved = 'b'.repeat(40);
  assert.equal(resolveGitCommit({ url: 'https://example.test/repo.git' }, 'main', () => { calls += 1; return `${resolved}\trefs/heads/main`; }), resolved);
  assert.equal(calls, 1);
});
