import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { inventory, inventoryExports, inventoryMarkdown } from '../scripts/code-corpus/inventory.mjs';

async function put(root, path, content) {
  const target=join(root,path);
  await mkdir(join(target,'..'),{recursive:true});
  await writeFile(target,content);
}

test('algorithm inventory attaches nearest README prose to undocumented functions and preserves paths', async (t) => {
  const root=await mkdtemp(join(tmpdir(),'algorithm-inventory-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  await put(root,'README.md','# Algorithms\n\nImplement the documented algorithm faithfully.');
  await put(root,'src/search/binary.js','export function binarySearch(values, target) { return values.indexOf(target); }');
  await put(root,'src/search/__tests__/binary_test.ts','/** Should never enter the corpus. */ export function fixture() {}');
  await put(root,'src/search/binary_test.ts','/** Should never enter the corpus. */ export function fixture() {}');
  await put(root,'src/search/binary.test.js','/** Should never enter the corpus. */ export function fixture() {}');
  await put(root,'src/search/vite.config.js','/** Should never enter the corpus. */ export function fixture() {}');
  const [row]=await inventory(root,{sourceName:'javascript-algorithms',revision:'abc123',license:'MIT'});
  assert.equal(row.function.name,'binarySearch');
  assert.equal(row.source.path,'src/search/binary.js');
  assert.match(row.instruction,/Implement the documented algorithm faithfully/);
  assert.match(row.instruction,/README\.md/);
  assert.equal(row.raw.readmePath,'README.md');
  assert.match(row.verification.reasons.at(-1),/function alignment not verified/);
  assert.equal((await inventory(root,{sourceName:'javascript-algorithms',revision:'abc123',license:'MIT'})).length,1);
});

test('algorithm README fallback never escapes the pinned checkout', async (t) => {
  const parent=await mkdtemp(join(tmpdir(),'algorithm-parent-'));
  t.after(()=>rm(parent,{recursive:true,force:true}));
  const root=join(parent,'checkout');
  await put(parent,'README.md','Parent prose must not be used.');
  await put(root,'src/algorithm.js','export function algorithm(value) { return value; }');
  assert.deepEqual(await inventory(root,{sourceName:'deno-std',revision:'abc123',license:'MIT'}),[]);
});

test('D3 export inventory includes anonymous default functions and named exported arrows', async (t) => {
  const root=await mkdtemp(join(tmpdir(),'d3-inventory-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  await put(root,'src/anonymous.js','export default function(values) { return values.length; }');
  await put(root,'src/named.js','export const sum = values => values.reduce((a, b) => a + b, 0);');
  await put(root,'src/named.test.js','export function notCorpus() { return true; }');
  await put(root,'test/anonymous-test.js','it("anonymous(values) returns the number of values", () => {});');
  await put(root,'test/named.test.js','test("sum(values) returns the total", () => {});');
  const rows=await inventoryExports(root,{sourceName:'d3-array',revision:'abc123',license:'ISC'});
  assert.deepEqual(rows.map(row=>row.function.name).sort(),['anonymous','sum']);
  assert.deepEqual(rows.map(row=>row.source.path).sort(),['src/anonymous.js','src/named.js']);
  assert.match(rows.find(row=>row.function.name==='anonymous').instruction,/returns the number of values/);
  assert.match(rows.find(row=>row.function.name==='sum').instruction,/returns the total/);
  assert.equal(rows.find(row=>row.function.name==='anonymous').raw.testPath,'test/anonymous-test.js');
  assert.ok(rows.every(row=>!row.instruction_quality));
  assert.ok(rows.every(row=>row.verification.reasons.some(reason=>reason.includes('expected outputs not extracted'))));
  assert.ok(rows.every(row=>row.verification.status==='inventory'));
});

test('30 seconds markdown frontmatter and JavaScript fences become preserved, non-executed task rows', async (t) => {
  const root=await mkdtemp(join(tmpdir(),'30sec-inventory-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  const code='function chunk(array, size) { return Array.from({ length: Math.ceil(array.length / size) }, (_, i) => array.slice(i * size, i * size + size)); }';
  await put(root,'content/Array/chunk.md',`---\ntitle: Chunk an array\ndescription: Split an array into chunks of a given size.\ntags: [array]\n---\n\nExample usage.\n\n\`\`\`js\n${code}\n\`\`\``);
  const [row]=await inventoryMarkdown(root,{sourceName:'30-seconds-of-code',revision:'abc123',license:'CC-BY-4.0'});
  assert.match(row.instruction,/Chunk an array/);
  assert.match(row.instruction,/Split an array into chunks of a given size/);
  assert.match(row.instruction,/Example usage/);
  assert.equal(row.group_id,'30-seconds-of-code:content/Array/chunk.md');
  assert.equal(row.source.path,'content/Array/chunk.md');
  assert.equal(row.raw.code,code);
  assert.equal(row.raw.frontmatter.title,'Chunk an array');
  assert.match(row.raw.document,/Example usage/);
  assert.equal(row.function.source,code);
});
