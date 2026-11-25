import assert from 'node:assert/strict';
import test from 'node:test';
import {snapshot,inputFeatures,inputLeaves} from '../dist/calls/index.js';
import {CallCapture} from '../dist/calls/recorder.js';
import {createNatlangRuntime} from '../dist/index.js';
import {semanticMap} from '../dist/std/index.js';
import {DEFAULT_SETTINGS} from '../dist/calls/types.js';

const forbidden=()=>{throw Error('Application observer hook must not execute');};
test('recording refuses enumerable getters and never invokes constructor or conversion hooks',()=>{
 let reads=0;const hook=()=>{reads++;return forbidden();};
 for(const key of ['field','constructor','toJSON']){const value={};Object.defineProperty(value,key,{enumerable:true,get:hook});const result=snapshot(value,1024);assert.equal(result.ref.complete,false);assert.equal(result.ref.reason,'nonportable');}
 for(const symbol of [Symbol.toPrimitive,Symbol.toStringTag]){const value=Object.create({});Object.defineProperty(value,symbol,{get:hook});assert.equal(snapshot(value,1024).ref.complete,false);}
 const array=[1];Object.defineProperty(array,0,{get:hook});assert.equal(snapshot(array,1024).ref.complete,false);
 assert.equal(reads,0);
});
test('plain data serializes its actual properties without invoking hidden toJSON or inherited hooks',()=>{
 const value={x:1,nested:[true,'actual']};Object.defineProperty(value,'toJSON',{value:forbidden});
 assert.equal(snapshot(value,1024).text,'{"x":1,"nested":[true,"actual"]}');
 const data=Object.create(null);Object.defineProperty(data,'__proto__',{value:{actual:2},enumerable:true});assert.equal(snapshot(data,1024).text,'{"__proto__":{"actual":2}}');
});
test('feature and approach walks skip accessors including arrays, preserving real data',()=>{
 const nested={actual:'evidence 144'};Object.defineProperty(nested,'hidden',{enumerable:true,get:forbidden});const array=[nested];Object.defineProperty(array,1,{enumerable:true,get:forbidden});
 const input={nested,array};Object.defineProperty(input,'secret',{enumerable:true,get:forbidden});
 assert.equal(inputFeatures(input)['str:nested.actual'],'evidence 144');assert.equal(inputFeatures(input)['str:secret'],undefined);
 assert.equal(inputLeaves(input).get('string:evidence 144'),'nested.actual');
});
test('CallCapture keeps original data identity while recording accessor inputs as unknown',()=>{
 const records=[],store={record:(r)=>records.push(r),settings:()=>DEFAULT_SETTINGS};
 const capture=new CallCapture(store,DEFAULT_SETTINGS,{callId:'c',parentCallId:null,parentActionIndex:null,taskId:'t',programId:null,buildHash:null,programRoot:null,definition:{id:'f',name:'f',key:'k',interface:'i',source:null,site:'internal',subtype:'function',params:[],returns:'unknown',types:{},instructions:{complete:false,reason:'excluded'}},model:{id:null,revision:null}});
 const actual={x:1},values={actual};Object.defineProperty(values,'secret',{get:forbidden,enumerable:true});
 capture.setInputs(values,{secret:'stale captured value'});assert.equal(capture.hostInputs.secret,undefined);assert.equal(capture.features['str:secret'],undefined);assert.equal(capture.hostInputs.actual,actual);assert.deepEqual(capture.inputs.secret,{complete:false,reason:'nonportable'});
 capture.finish({outcome:'done',detail:'actual result',output:actual,hasOutput:true,events:[]});assert.equal(records.length,1);
});
test('revoked proxies are honestly nonportable rather than replacing an application result',()=>{const {proxy,revoke}=Proxy.revocable({},{});revoke();assert.equal(snapshot(proxy,1024).ref.reason,'nonportable');});

test('real native std execution returns the identical accessor-bearing result without observational effects',async()=>{
 let reads=0;const actual={};Object.defineProperty(actual,'applicationOnly',{enumerable:true,get(){reads++;throw Error('application getter');}});
 const records=[],store={record:r=>records.push(r),settings:()=>({...DEFAULT_SETTINGS,specialization:'off'}),version:()=>0,currentCompilation:()=>undefined,caseServed(){},enqueue(){}};
 const runtime=createNatlangRuntime({calls:store});
 try{const result=await runtime.run(()=>semanticMap([1],()=>actual));assert.equal(result[0],actual);assert.equal(reads,0);assert.equal(records[0].output.complete,false);assert.equal(records[0].outcome,'done');}finally{runtime.close();}
});
