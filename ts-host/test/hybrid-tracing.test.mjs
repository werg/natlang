import assert from 'node:assert/strict';
import {test} from 'node:test';
import {
  installHybridTracingEnvironment,createNatlangRuntime,
  setDefaultEnvironmentFactory,currentFrame
} from '../dist/runtime/node.js';
import {setModuleRealm,moduleInstance} from '../dist/runtime/modules.js';
import {parseModule,PATH_ONLY} from '../dist/runtime/loader.js';
import {TypeScriptEnvironment} from '../dist/environment.js';

function restore() {
  setDefaultEnvironmentFactory(options=>new TypeScriptEnvironment({
    workspace:options.workspace,network:options.network
  }));
  setModuleRealm(()=>new TypeScriptEnvironment({mode:'retained'}));
}
const mockOwned = mode=>({
  authority:'owned-training-js/1',mode,
  scopeCapabilities:{allowModules:false,allowNetwork:false},
  execute(){return {result:41,events:[]};},
  async executeAsync(){return {result:42,events:[]};},
  evaluateModule(){throw new Error('owned module route reached');},
  fork(){return mockOwned('fresh');},
  close(){}
});

test('compatibility-first route executes ordinary Node JS and tags result untraced',async()=>{
  const observed=[];
  let runtime;
  try {
    installHybridTracingEnvironment(mockOwned,{
      onNativeExecution:event=>observed.push(event)
    });
    runtime=createNatlangRuntime();
    const result=await runtime.run(()=>currentFrame().task.environment().execute({
      code:'1 + 2',scope:{},body:false,path:'hybrid://native'
    }));
    assert.equal(result.result,3);
    assert.equal(result.events[0].operation,'execution.provenance');
    assert.equal(result.events[0].authority,'native-untraced/1');
    assert.equal(result.events[0].path,'hybrid://native');
    assert.equal(observed.length,1);
  } finally {runtime?.close();restore();}
});

test('explicit owned route retains owned result without native provenance',async()=>{
  let runtime;
  try {
    installHybridTracingEnvironment(mockOwned,{route:()=> 'owned'});
    runtime=createNatlangRuntime();
    const result=await runtime.run(()=>currentFrame().task.environment().execute({
      code:'irrelevant',scope:{},body:false,path:'hybrid://owned'
    }));
    assert.equal(result.result,41);
    assert.deepEqual(result.events,[]);
  } finally {runtime?.close();restore();}
});

test('invalid route is rejected before either engine runs',async()=>{
  let runtime;
  try {
    installHybridTracingEnvironment(mockOwned,{route:()=> 'maybe'});
    runtime=createNatlangRuntime();
    await assert.rejects(runtime.run(()=>currentFrame().task.environment().execute({
      code:'1',scope:{},body:false,path:'hybrid://invalid'
    })),/route must return owned or native/);
  } finally {runtime?.close();restore();}
});

test('compatibility route installs a retained native callable module and reports provenance',()=>{
  const observed=[];
  try {
    installHybridTracingEnvironment(mockOwned,{onNativeExecution:event=>observed.push(event)});
    const record=parseModule('/__natlang__/folder/hybrid.ts',
      'export const answer: number = 1;',{},PATH_ONLY);
    const instance=moduleInstance(record,{hybrid:record});
    assert.equal(instance.answer,1);
    assert.ok(observed.some(event=>event.kind==='evaluateModule'&&event.mode==='retained'));
  } finally {restore();}
});
