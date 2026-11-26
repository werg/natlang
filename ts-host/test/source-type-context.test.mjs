import test from 'node:test';
import assert from 'node:assert/strict';
import * as core from '../dist/index.js';
import {virtualSourceFiles} from '../dist/runtime/virtual-project.js';
import {loadNamedFunction} from '../dist/runtime/loader.js';
const options={target:'node',emit:false,declarations:false,adaptation:false};
const nl=(returns='Shared')=>`---\nargs: {}\nreturns: ${returns}\n---\nReturn the requested owned value.\n`;
const source=()=>({'natlang.json':'{}','types.ts':'export type Shared={value:string};','root.nl':nl(),'root/types.ts':'export type Local={own:number};','root/helper.nl':nl('Local'),'other.nl':nl(),'other/types.ts':'export type Secret={secret:string};','other/secretView.nl':nl('Secret'),'root/SKILL.md':'Actual owned knowledge.'});
function reads(files,context){const owner=virtualSourceFiles(files,'/project',context),read=owner.read,seen=[];owner.read=path=>{seen.push(path);return read(path);};return{owner,seen};}
test('validated frontend linked aliases survive JSON and are reused without reading their type source again',()=>{
 const files=source(),product=core.compileVirtualProjectProduct({files,root:'/project'},options);assert.equal(product.build.ok,true,JSON.stringify(product.build.diagnostics));
 const wire=JSON.parse(JSON.stringify(product));assert.ok(wire.sourceTypes.modules['/project/types.ts']);assert.equal(wire.build.linkedTypes,undefined,'do not duplicate alias maps in product');
 assert.equal(core.sourceTypeContextMatches(wire.sourceTypes,files,'/project',options),true);
 const cached=reads(files,wire.sourceTypes),raw=reads(files);const actual=loadNamedFunction('/project/root.nl',cached.owner),expected=loadNamedFunction('/project/root.nl',raw.owner);
 assert.deepEqual(actual,expected);assert.equal(cached.seen.includes('/project/types.ts'),false);assert.equal(cached.seen.includes('/project/root/types.ts'),false);assert.equal(raw.seen.includes('/project/types.ts'),true);
 assert.equal(actual.codebase.secretView,undefined);assert.equal(actual.types.Secret,undefined);assert.equal(actual.codebase.helper.types.Secret,undefined);
});
test('owner records and aliases are fresh and cannot poison another owner or serialized frontend data',()=>{
 const files=source(),product=core.compileVirtualProjectProduct({files,root:'/project'},options),context=JSON.parse(JSON.stringify(product.sourceTypes));
 const a=loadNamedFunction('/project/root.nl',virtualSourceFiles(files,'/project',context));a.types.Shared='{value:number}';a.codebase.helper.instructions='Owner A edit';
 const b=loadNamedFunction('/project/root.nl',virtualSourceFiles(files,'/project',context));assert.match(b.types.Shared,/value:string/);assert.notEqual(b.codebase.helper.instructions,a.codebase.helper.instructions);assert.match(context.modules['/project/types.ts'].Shared,/value:string/);
 const owner=virtualSourceFiles(files,'/project',context);context.modules['/project/types.ts'].Shared='{value:boolean}';assert.match(loadNamedFunction('/project/root.nl',owner).types.Shared,/value:string/,'validated owner snapshots primitive maps');
});
test('changed source, imported types, mounts and SKILL bytes invalidate metadata and retain current aliases',()=>{
 const files=source(),context=core.compileVirtualProjectProduct({files,root:'/project'},options).sourceTypes;
 for(const[key,value]of [['types.ts','export type Shared={value:number};'],['root/SKILL.md','New owned knowledge.'],['mounted/types.ts','export type Mounted={value:boolean};']]){
  const changed={...files,[key]:value};assert.equal(core.sourceTypeContextMatches(context,changed,'/project',options),false);const {owner,seen}=reads(changed,context);const record=loadNamedFunction('/project/root.nl',owner);assert.equal(seen.includes('/project/types.ts'),true);if(key==='types.ts')assert.match(record.types.Shared,/value:number/);
 }
});
test('lexical root, compiler options and mutated metadata cannot silently reuse another product context',()=>{
 const files=source(),context=core.compileVirtualProjectProduct({files,root:'/project'},options).sourceTypes;
 assert.equal(core.sourceTypeContextMatches(context,files,'/other',options),false);assert.equal(core.sourceTypeContextMatches(context,files,'/project',{...options,target:'browser'}),false);assert.equal(core.sourceTypeContextMatches(context,files,'/project',{...options,services:{declarations:{probe:'unknown'},scopes:{probe:[]}}}),false);
 const changed=structuredClone(context);changed.modules['/project/types.ts'].Shared='{value:number}';const {owner,seen}=reads(files,changed);assert.match(loadNamedFunction('/project/root.nl',owner).types.Shared,/value:string/);assert.equal(seen.includes('/project/types.ts'),true);
});
test('explicit and environment opt-outs keep the fresh reference parser',()=>{
 const files=source(),product=core.compileVirtualProjectProduct({files,root:'/project'},options);assert.equal(core.compileVirtualProjectProduct({files,root:'/project'},{...options,reuseSourceTypes:false}).sourceTypes,undefined);
 const before=process.env.NATLANG_SOURCE_TYPE_REUSE;try{process.env.NATLANG_SOURCE_TYPE_REUSE='0';const {owner,seen}=reads(files,product.sourceTypes);loadNamedFunction('/project/root.nl',owner);assert.equal(seen.includes('/project/types.ts'),true);}finally{if(before===undefined)delete process.env.NATLANG_SOURCE_TYPE_REUSE;else process.env.NATLANG_SOURCE_TYPE_REUSE=before;}
});
for(const mode of ['node','workerd'])test(`${mode} product instantiation retains independent native services and module state`,async()=>{
 const platform=mode==='node'?core.nodePlatform():(await import('../dist/core/workerd.js')).workerdPlatform({realm:{call:(body,params,args)=>Function(...params,body)(...args)}});
 const files={'types.ts':'export type Shared={value:string};','root.nl':nl(),'root/count.ts':'let n=0;export function next():number{return ++n;}','root/read.ts':'import {probe} from "natlang:services";export default async function read():Promise<string>{return probe.read();}','services.d.ts':'declare module "natlang:services" {export const probe:{read():Promise<string>};}'};
 const product=core.compileVirtualProjectProduct({files,root:'/project'},{...options,services:{declarations:{probe:'{read():Promise<string>}'},scopes:{probe:[]}}});assert.equal(product.build.ok,true,JSON.stringify(product.build.diagnostics));
 for(const marker of ['A','B']){let turns=0;const runtime=core.createNatlangRuntime({platform,services:{probe:{read:async()=>marker}},model:async()=>{turns++;return{calls:[['eval',{code:'return {value:(await read())+":"+count.next()+":"+count.next()};',finish:true}]]};}});try{const fn=core.loadVirtualNatlang(files,'root.nl','/project',platform,product.sourceTypes);assert.deepEqual(await runtime.run(()=>fn()),{value:marker+':1:2'});assert.equal(turns,1);}finally{runtime.close();}}
});
test('browser public loader surface consumes the same immutable metadata without exposing sibling private types',async()=>{
 const browser=await import('../dist/browser/index.js'),files=source(),product=core.compileVirtualProjectProduct({files,root:'/project'},options);assert.equal(browser.sourceTypeContextMatches(product.sourceTypes,files,'/project',options),true);
 const platform=core.nodePlatform();const fn=browser.loadVirtualNatlang(files,'root.nl','/project',platform,product.sourceTypes);assert.equal(typeof fn,'function');
 const {owner}=reads(files,product.sourceTypes);const record=loadNamedFunction('/project/root.nl',owner);assert.equal(record.types.Secret,undefined);assert.deepEqual(Object.keys(record.codebase).sort(),['helper']);
});

test('type metadata keeps only pure frontend options, never the live platform or module grants',()=>{
 const files=source(),platform=core.nodePlatform();const compiled=core.compileVirtualProject({files,root:'/project'},core,{...options,platform,modules:{probe:{read:()=>"owner"}}});assert.equal(compiled.ok,true);assert.equal(Object.hasOwn(compiled.sourceTypes.compilerOptions,'platform'),false);assert.equal(Object.hasOwn(compiled.sourceTypes.compilerOptions,'modules'),false);
});

test('external type dependencies remain outside the validated root and publish no reusable aliases',()=>{
 const files={'types.ts':'import type {Outer} from "../external/types.js";export type Shared={outer:Outer};','root.nl':nl(),'/external/types.ts':'type Hidden={value:string};export type Outer={hidden:Hidden};'};
 const product=core.compileVirtualProjectProduct({files,root:'/project'},options);
 assert.equal(product.build.ok,false);assert.ok(product.build.diagnostics.some(item=>item.code==='TS6059'));
 assert.equal(product.sourceTypes,undefined,'a failed frontend cannot attest reusable linked types');
});
