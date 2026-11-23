import test from 'node:test';
import assert from 'node:assert/strict';
import * as core from '../dist/index.js';
import {loadNamedFunction} from '../dist/runtime/loader.js';
import {virtualSourceFiles} from '../dist/runtime/virtual-project.js';
const nl=(uses=[],returns='Shared')=>`---\nargs: {}\nreturns: ${returns}\n${uses.length?`uses: [${uses.join(', ')}]\n`:''}---\nUse the actual local context.\n`;
const source=()=>({'natlang.json':'{}','types.ts':'export type Shared={answer:string};','root.nl':nl(['./left','./right']),'left.nl':nl(['./leaf']),'right.nl':nl(['./leaf']),'leaf.nl':nl()});
test('one load shares a diamond by actual source, while a later revision gets fresh records and types',()=>{
 const files=source(),owner=virtualSourceFiles(files),reads=new Map();
 owner.read=path=>{reads.set(path,(reads.get(path)??0)+1);return files[path.slice("/project/".length)];};
 const root=loadNamedFunction('/project/root.nl',owner),left=root.codebase.left.codebase.leaf,right=root.codebase.right.codebase.leaf;
 assert.equal(left,right);assert.equal(reads.get('/project/leaf.nl'),1);assert.equal(reads.get('/project/types.ts'),1);
 files['leaf.nl']=nl()+'Changed source revision.';files['types.ts']='export type Shared={answer:number};';
 const revised=loadNamedFunction('/project/root.nl',owner).codebase.left.codebase.leaf;
 assert.notEqual(revised,left);assert.match(revised.instructions,/Changed source revision/);assert.match(revised.types.Shared,/number/);
});
test('source graph opt-out retains independent records, and recursive uses still reject',()=>{
 const files=source(),owner=virtualSourceFiles(files);owner.sourceGraphReuse=false;
 const root=loadNamedFunction('/project/root.nl',owner);assert.notEqual(root.codebase.left.codebase.leaf,root.codebase.right.codebase.leaf);
 files['leaf.nl']=nl(['./root']);assert.throws(()=>loadNamedFunction('/project/root.nl',virtualSourceFiles(files)),/leads back|recursive/);
});
test('reachable public dependencies have their own declarations while private children keep lexical ownership',()=>{
 const files=Object.fromEntries(Object.entries(source()).map(([path,text])=>['library/'+path,text]));
 Object.assign(files,{'natlang.json':'{}','entry.ts':'import root from "./library/root.nl"; export default async function main(){return await root();}',
   'library/root/types.ts':'export type Local={privateField:number};','library/root/helper.nl':nl([],'Local')});
 const compiled=core.compileVirtualProject({root:'/project',files},core,{platform:core.nodePlatform(),adaptation:false});
 assert.equal(compiled.ok,true,JSON.stringify(compiled.diagnostics));
 for(const name of ['root','left','right','leaf'])assert.ok(compiled.declarations['/project/library/'+name+'.d.nl.ts']);
 assert.match(compiled.declarations['/project/library/left.d.nl.ts'],/import type .*\.\/leaf\.nl/);
 assert.match(compiled.declarations['/project/library/root.d.nl.ts'],/readonly helper:/);
 assert.equal(compiled.declarations['/project/library/root/helper.d.nl.ts'],undefined);
});
for(const mode of ['node','workerd'])test(`compiled ${mode} shared public record graph executes both diamond branches`,async()=>{
 const platform=mode==='node'?core.nodePlatform():(await import('../dist/core/workerd.js')).workerdPlatform({realm:{call(body,params,args){return Function(...params,body)(...args);}}});
 const files={...source(),'entry.ts':'import root from "./root.nl"; export default async function main(){return await root();}'};
 let turn=0;const actions=[['eval',{code:'const a=await left();const b=await right();return {answer:a.answer+b.answer};',finish:true}],['eval',{code:'return await leaf();',finish:true}],['return_result',{status:'success',value:{answer:'A'}}],['eval',{code:'return await leaf();',finish:true}],['return_result',{status:'success',value:{answer:'B'}}]];
 const runtime=core.createNatlangRuntime({platform,model:async()=>({calls:[actions[turn++]]})});
 try{
  const compiled=core.compileVirtualProject({root:'/project',files},core,{platform,adaptation:false});assert.equal(compiled.ok,true,JSON.stringify(compiled.diagnostics));
  const graph=Object.keys(compiled.outputs).filter(path=>path.endsWith('/.natlang-records.cjs'));assert.equal(graph.length,1);
  for(const [path,text]of Object.entries(compiled.outputs))if(path.endsWith('.nl.js')){assert.match(text,/__records\[/);assert.doesNotMatch(text,/Use the actual local context/);}
  const actual=await runtime.run(()=>compiled.require('entry.ts').default());assert.deepEqual(actual,{answer:'AB'});assert.equal(turn,5);
 }finally{runtime.close();}
});
for(const mode of ['node','workerd'])test(`compiled ${mode} inline NL contexts use the same owner-local record graph`,async()=>{
 const platform=mode==='node'?core.nodePlatform():(await import('../dist/core/workerd.js')).workerdPlatform({realm:{call(body,params,args){return Function(...params,body)(...args);}}});
 const files={'entry.ts':'import {nl} from "@natlang/node";export default async function main(){return await nl<string>`Use leaf.`();}', 'natlang.d/leaf.nl':nl([],'string')};
 let turn=0;const runtime=core.createNatlangRuntime({platform,model:async()=>({calls:[++turn===1?['eval',{code:'return await leaf();',finish:true}]:['return_result',{status:'success',value:'checked'}]]})});
 try{const compiled=core.compileVirtualProject({root:'/project',files},core,{platform,adaptation:false});assert.equal(compiled.ok,true,JSON.stringify(compiled.diagnostics));assert.equal(await runtime.run(()=>compiled.require('entry.ts').default()),'checked');assert.equal(turn,2);}finally{runtime.close();}
});
test('inline-record opt-out remains executable without a shared module',async()=>{
 const platform=core.nodePlatform(),runtime=core.createNatlangRuntime({platform,model:async()=>({calls:[['return_result',{status:'success',value:'checked'}]]})});
 try{const compiled=core.compileVirtualProject({root:'/project',files:{'entry.ts':'import leaf from "./leaf.nl";export default async function main(){return await leaf();}','leaf.nl':nl([],'string')}},core,{platform,adaptation:false,inlineRecords:true});assert.equal(compiled.ok,true,JSON.stringify(compiled.diagnostics));assert.ok(!Object.keys(compiled.outputs).some(path=>path.includes('.natlang-records.')));assert.equal(await runtime.run(()=>compiled.require('entry.ts').default()),'checked');}finally{runtime.close();}
});
test('ESM output imports its shared context graph without leaking a CommonJS require',async()=>{
 const {mkdtempSync,writeFileSync,mkdirSync,rmSync}=await import('node:fs');const {tmpdir}=await import('node:os');const {join}=await import('node:path');const {pathToFileURL,fileURLToPath}=await import('node:url');
 const dir=mkdtempSync(join(tmpdir(),'natlang-esm-records-'));
 let turn=0;const runtime=core.createNatlangRuntime({model:async()=>({calls:[++turn===1?['eval',{code:'return await leaf();',finish:true}]:['return_result',{status:'success',value:'checked'}]]})});
 try{
  writeFileSync(join(dir,'package.json'),JSON.stringify({type:'module'}));mkdirSync(join(dir,'natlang.d'));
  writeFileSync(join(dir,'entry.ts'),'import {nl} from "@natlang/node";export async function main(){return await nl<string>`Use leaf.`();}');writeFileSync(join(dir,'natlang.d/leaf.nl'),nl([],'string'));
  const result=core.buildProject({project:dir,outDir:join(dir,'dist'),module:'esm',adaptation:false,runtimeModule:{specifiers:['@natlang/node'],url:new URL('../dist/index.js',import.meta.url).href,types:fileURLToPath(new URL('../dist/index.d.ts',import.meta.url))}});
  assert.equal(result.ok,true,JSON.stringify(result.diagnostics));
  const loaded=await import(pathToFileURL(join(dir,'dist/entry.js')).href);assert.equal(await runtime.run(()=>loaded.main()),'checked');
 }finally{runtime.close();rmSync(dir,{recursive:true,force:true});}
});
