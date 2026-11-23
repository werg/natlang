import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {createHash} from 'node:crypto';
import {inventoryNativePassthroughs,applyNativePassthroughs} from './native_passthroughs.mjs';
const reader=process.env.NATLANG_PASSTHROUGH_READER||resolve('.');
async function fixture(t,body,extra={}){
 const root=await mkdtemp(join(tmpdir(),'natlang-native-leaf-'));t.after(()=>rm(root,{recursive:true,force:true}));
 await writeFile(join(root,'natlang.json'),JSON.stringify({serviceBinding:'explicit',serviceScopes:{native:['read.nl']}}));
 await writeFile(join(root,'services.d.ts'),'declare module "natlang:services" { export const native:{read(id:string):Promise<string>}; }');
 await writeFile(join(root,'read.nl'),'---\nargs:\n  id: string\nreturns: string\n'+(extra.frontmatter||'')+'---\n'+body+'\n');
 return root;
}
test('exact passthrough keeps native authority and the typed signature, removes obsolete scope',async t=>{
 const root=await fixture(t,'Return native.read(id) unchanged.');
 const report=await inventoryNativePassthroughs(root,reader);assert.equal(report.rows[0].status,'convert');
 await applyNativePassthroughs(report);
 assert.match(await readFile(join(root,'read.ts'),'utf8'),/import \{native\} from "natlang:services"/);
 assert.match(await readFile(join(root,'read.ts'),'utf8'),/read\(id\)/);
 assert.deepEqual(JSON.parse(await readFile(join(root,'natlang.json'),'utf8')).serviceScopes.native,[]);
 await mkdir(join(root,'meta'));await writeFile(join(root,'meta/NATIVE_PASSTHROUGH_INVENTORY.json'),JSON.stringify(report));
 assert.equal((await inventoryNativePassthroughs(root,reader)).rows[0].status,'converted');
});
test('selection and per-source iteration remain semantic functions',async t=>{
 for(const body of ['Call native.read. Match the named item in the goal and return it.','Call native.read once per distinct source. Return the ordered results.','Return native.read(id) unchanged. Summarize the result.']){
  const root=await fixture(t,body);assert.equal((await inventoryNativePassthroughs(root,reader)).rows.filter(x=>x.status==='convert').length,0);
 }
});
test('prose arguments require a reviewed source-bound mapping',async t=>{
 const root=await fixture(t,'Call native.read with the exact observed ID. Return its result unchanged.');
 const report=await inventoryNativePassthroughs(root,reader);assert.equal(report.rows[0].status,'kept');
 const source=await readFile(join(root,'read.nl'),'utf8');const source_sha256=createHash('sha256').update(source).digest('hex');
 const reviewed=await inventoryNativePassthroughs(root,reader,{'read.nl':{source_sha256,arguments:'id',rationale:'Reviewed unchanged input'}});
 assert.equal(reviewed.rows[0].status,'convert');
 await assert.rejects(()=>inventoryNativePassthroughs(root,reader,{'read.nl':{source_sha256:'wrong',arguments:'id'}}),/source changed/);
});
test('callable owner subtrees and postconditions remain in NL',async t=>{
 const root=await fixture(t,'Return native.read(id) unchanged.');await mkdir(join(root,'read'));
 assert.match((await inventoryNativePassthroughs(root,reader)).rows[0].reason,/subtree/);
 const withChecks=await fixture(t,'Return native.read(id) unchanged.',{frontmatter:'ensures: actual result\n'});
 assert.match((await inventoryNativePassthroughs(withChecks,reader)).rows[0].reason,/semantics/);
});
test('application refuses stale source instead of overwriting it',async t=>{
 const root=await fixture(t,'Return native.read(id) unchanged.');const report=await inventoryNativePassthroughs(root,reader);
 await writeFile(join(root,'read.nl'),'updated semantic policy');await assert.rejects(()=>applyNativePassthroughs(report),/source changed/);
});
