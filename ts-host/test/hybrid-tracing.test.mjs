import assert from 'node:assert/strict';
import {test} from 'node:test';
import {
  hybridTracingPlatform,createNatlangRuntime,currentFrame
} from '../dist/runtime/node.js';
import {moduleInstance} from '../dist/runtime/modules.js';
import {parseModule,PATH_ONLY} from '../dist/runtime/loader.js';
const mockOwned = mode=>({
  authority:'owned-training-js/1',mode,
  scopeCapabilities:{allowModules:false,allowNetwork:false},
  execute(){return {result:41,events:[]};},
  async executeAsync(){return {result:42,events:[]};},
  evaluateModule(){throw new Error('owned module route reached');},
  fork(){return mockOwned('fresh');},
  close(){}
});

test('default hybrid route executes owned code without native provenance',async()=>{
  let runtime;
  try {
    runtime=createNatlangRuntime({platform:hybridTracingPlatform(mockOwned)});
    const result=await runtime.run(()=>currentFrame().task.environment().execute({
      code:'1 + 2',scope:{},body:false,path:'hybrid://owned-default'
    }));
    assert.equal(result.result,41);
    assert.deepEqual(result.events,[]);
  } finally {runtime?.close();}
});

test('explicit owned route retains owned result without native provenance',async()=>{
  let runtime;
  try {
    runtime=createNatlangRuntime({platform:hybridTracingPlatform(mockOwned,{route:()=> 'owned'})});
    const result=await runtime.run(()=>currentFrame().task.environment().execute({
      code:'irrelevant',scope:{},body:false,path:'hybrid://owned'
    }));
    assert.equal(result.result,41);
    assert.deepEqual(result.events,[]);
  } finally {runtime?.close();}
});

test('invalid route is rejected before either engine runs',async()=>{
  let runtime;
  try {
    runtime=createNatlangRuntime({platform:hybridTracingPlatform(mockOwned,{route:()=> 'maybe'})});
    await assert.rejects(runtime.run(()=>currentFrame().task.environment().execute({
      code:'1',scope:{},body:false,path:'hybrid://invalid'
    })),/route must return owned or native/);
  } finally {runtime?.close();}
});

test('native module route refuses execution before module bootstrap',()=>{
  const observed=[];
  const platform=hybridTracingPlatform(mockOwned,{
    route:()=> 'native',onNativeExecution:event=>observed.push(event)
  });
  const record=parseModule('/__natlang__/folder/hybrid.ts',
    'export const answer: number = 1;',{},PATH_ONLY);
  assert.throws(()=>moduleInstance(platform,record,{hybrid:record}),/Trace required/);
  assert.equal(observed.length,0);
});

test('explicit native per-call route is rejected before evaluation',async()=>{
  let runtime;
  try {
    runtime=createNatlangRuntime({platform:hybridTracingPlatform(mockOwned,{route:()=> 'native'})});
    await assert.rejects(runtime.run(()=>currentFrame().task.environment().execute({
      code:'1 + 2',scope:{},body:false,path:'hybrid://native-rejected'
    })),/Trace required/);
  } finally {runtime?.close();}
});
