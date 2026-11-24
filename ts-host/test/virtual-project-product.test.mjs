import test from 'node:test';
import assert from 'node:assert/strict';
import * as core from '../dist/index.js';
const project={files:{'main.ts':'let calls=0; export function next():number { return ++calls; }'},root:'/product'};
const options={target:'node',adaptation:false,constrained:false};
test('pure frontend product survives JSON transport and separately instantiates module state',()=>{
 const product=core.compileVirtualProjectProduct(project,options);
 assert.equal(product.build.ok,true,JSON.stringify(product.build.diagnostics));
 const wire=JSON.parse(JSON.stringify(product));
 const a=core.instantiateVirtualProjectProduct(wire,core,{platform:core.nodePlatform()});
 const b=core.instantiateVirtualProjectProduct(wire,core,{platform:core.nodePlatform()});
 assert.equal(a.require('main.ts').next(),1);assert.equal(a.require('main.ts').next(),2);
 assert.equal(b.require('main.ts').next(),1);
 a.outputs.poison='changed';assert.equal(wire.build.outputs.poison,undefined);
 project.files['main.ts']='invalid';assert.match(product.project.files['main.ts'],/let calls/);
});
test('external service modules are rebound per instantiation rather than captured in compiler product',()=>{
 const product=core.compileVirtualProjectProduct({files:{'main.ts':'import { marker } from "probe"; export const value=marker;','probe.d.ts':'declare module "probe" { export const marker:string; }'}},options);
 assert.equal(product.build.ok,true,JSON.stringify(product.build.diagnostics));
 const a=core.instantiateVirtualProjectProduct(product,core,{platform:core.nodePlatform(),modules:{probe:{marker:'a'}}});
 const b=core.instantiateVirtualProjectProduct(product,core,{platform:core.nodePlatform(),modules:{probe:{marker:'b'}}});
 assert.equal(a.require('main.ts').value,'a');assert.equal(b.require('main.ts').value,'b');
});
test('validation-only products preserve diagnostics/declarations without executable instances',()=>{
 const product=core.compileVirtualProjectProduct({files:{'entry.nl':'---\nargs: { value: number }\nreturns: number\n---\nDouble value.'}},{...options,emit:false});
 assert.equal(product.build.ok,true,JSON.stringify(product.build.diagnostics));
 assert.deepEqual(product.build.outputs,{});assert.ok(Object.keys(product.build.declarations).length);
 assert.throws(()=>core.instantiateVirtualProjectProduct({...product,schema:'wrong'},core,{platform:core.nodePlatform()}),/unsupported/);
});

test('an executable owner releases declaration documents after its public build result is dropped',async()=>{
 const {execFile}=await import('node:child_process');
 const {promisify}=await import('node:util');
 const script=`
 import assert from 'node:assert/strict';
 import * as core from ${JSON.stringify(new URL('../dist/index.js',import.meta.url).href)};
 const product=core.compileVirtualProjectProduct({files:{'main.ts':'let count=0; export function next():number {return ++count;}','answer.nl':'---\\nargs: {}\\nreturns: string\\n---\\nAnswer.'},root:'/lifetime'},{target:'node',adaptation:false,constrained:false});
 assert.equal(product.build.ok,true,JSON.stringify(product.build.diagnostics));
 function owner(){
  const compiled=core.instantiateVirtualProjectProduct(product,core,{platform:core.nodePlatform()});
  assert.ok(Object.keys(compiled.declarations).length);
  assert.equal(compiled.require('main.ts').next(),1);
  return {require:compiled.require,documents:new WeakRef(compiled.declarations)};
 }
 const live=owner();
 for(let attempt=0;attempt<30;attempt++){
  await new Promise(resolve=>setImmediate(resolve));globalThis.gc();
  await new Promise(resolve=>setImmediate(resolve));
  if(live.documents.deref()===undefined)break;
 }
 assert.equal(live.documents.deref(),undefined,'runtime closures retained unused declaration documents');
 assert.equal(live.require('main.ts').next(),2,'module state must remain owned and live');
 `;
 await promisify(execFile)(process.execPath,['--expose-gc','--input-type=module','--eval',script],{timeout:30000});
});
