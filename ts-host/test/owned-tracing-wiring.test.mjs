import assert from 'node:assert/strict';
import {test} from 'node:test';
import {
  ownedTracingPlatform,createNatlangRuntime,currentFrame
} from '../dist/runtime/node.js';
import {moduleInstance} from '../dist/runtime/modules.js';
import {parseModule,PATH_ONLY} from '../dist/runtime/loader.js';
import {TypeScriptEnvironment} from '../dist/environment.js';

test('opt-in owned evaluator platform replaces the per-call Node evaluator without implicit fallback',async()=>{
  const created=[];
  const make=mode=>{
    const env={
      authority:'owned-training-js/1',mode,
      scopeCapabilities:{allowModules:false,allowNetwork:false},
      execute(){return {result:7,events:[]};},
      executeAsync:async()=>({result:7,events:[]}),
      evaluateModule(){throw new Error('unsupported owned module');},
      fork(){return make('fresh');},
      close(){}
    };
    created.push(mode);
    return env;
  };
  const runtime=createNatlangRuntime({platform:ownedTracingPlatform(mode=>make(mode))});
  const authority=await runtime.run(()=>{
    const env=currentFrame().task.environment();
    assert.equal(env.mode,'fresh');
    assert.equal(env.execute({code:'7',body:false,scope:{},path:'test'}).result,7);
    return env.authority;
  });
  assert.equal(authority,'owned-training-js/1');
  assert.deepEqual(created,['fresh']);
  runtime.close();
  const rejected=createNatlangRuntime({platform:ownedTracingPlatform(mode=>new TypeScriptEnvironment({mode}))});
  await assert.rejects(rejected.run(()=>
    currentFrame().task.environment()),/native fallback forbidden/);
  rejected.close();
});

test('owned dual-seam factory rejects missing capabilities or invalid mode',async()=>{
  assert.throws(()=>ownedTracingPlatform(null),/factory required/);
  const bad=createNatlangRuntime({platform:ownedTracingPlatform(mode=>({
    authority:'owned-training-js/1',mode:'retained',
    execute(){},executeAsync(){},evaluateModule(){},close(){},fork(){}
  }))});
  try{
    await assert.rejects(bad.run(()=>currentFrame().task.environment()),/incompatible evaluator/);
  }finally{bad.close();}
});

test('each platform value owns its module realm and module exports',()=>{
  const record=parseModule('/__natlang__/folder/test.ts',
    'export const answer: number = 1;',{},PATH_ONLY);
  let closed=0;
  const authority=value=>({
    mode:'retained',authority:'owned-training-js/1',
    execute(){throw new Error('not used');},
    async executeAsync(){throw new Error('not used');},
    evaluateModule(_code,bindings){bindings.exports.answer=value;},
    fork(){throw new Error('not used');},
    close(){closed++;}
  });
  const two=ownedTracingPlatform(()=>authority(2)),three=ownedTracingPlatform(()=>authority(3));
  const first=moduleInstance(two,record,{test:record});
  assert.equal(first.answer,2);
  assert.equal(moduleInstance(two,record,{test:record}),first);
  const second=moduleInstance(three,record,{test:record});
  assert.equal(second.answer,3);
  assert.notEqual(first,second,'exports of one authority are never served by another');
  assert.equal(first.answer,2);
  assert.equal(moduleInstance(two,record,{test:record}),first,'the first platform is unaffected');
  assert.equal(closed,0);
});
