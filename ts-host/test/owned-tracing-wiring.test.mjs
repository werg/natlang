import assert from 'node:assert/strict';
import {test} from 'node:test';
import {
  installOwnedTracingEnvironment,createNatlangRuntime,
  setDefaultEnvironmentFactory,currentFrame
} from '../dist/runtime/node.js';
import {setModuleRealm} from '../dist/runtime/modules.js';
import {TypeScriptEnvironment} from '../dist/environment.js';

test('opt-in owned evaluator wiring replaces the per-call Node evaluator without implicit fallback',async()=>{
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
  try{
    installOwnedTracingEnvironment(mode=>make(mode));
    const runtime=createNatlangRuntime();
    const authority=await runtime.run(()=>{
      const env=currentFrame().task.environment();
      assert.equal(env.mode,'fresh');
      assert.equal(env.execute({code:'7',body:false,scope:{},path:'test'}).result,7);
      return env.authority;
    });
    assert.equal(authority,'owned-training-js/1');
    assert.deepEqual(created,['fresh']);
    runtime.close();
    installOwnedTracingEnvironment(mode=>
      new TypeScriptEnvironment({mode}));
    const rejected=createNatlangRuntime();
    await assert.rejects(rejected.run(()=>
      currentFrame().task.environment()),/native fallback forbidden/);
    rejected.close();
  }finally{
    // Restore the default Node environment for other tests in this process.
    setDefaultEnvironmentFactory(options=>new TypeScriptEnvironment({
      workspace:options.workspace,network:options.network
    }));
    setModuleRealm(()=>new TypeScriptEnvironment({mode:'retained'}));
  }
});

test('owned dual-seam factory rejects missing capabilities or invalid mode',async()=>{
  assert.throws(()=>installOwnedTracingEnvironment(null),/factory required/);
  const bad=createNatlangRuntime();
  try{
    installOwnedTracingEnvironment(mode=>({
      authority:'owned-training-js/1',mode:'retained',
      execute(){},executeAsync(){},evaluateModule(){},close(){},fork(){}
    }));
    await assert.rejects(bad.run(()=>currentFrame().task.environment()),/incompatible evaluator/);
  }finally{
    bad.close();
    setDefaultEnvironmentFactory(options=>new TypeScriptEnvironment({
      workspace:options.workspace,network:options.network
    }));
    setModuleRealm(()=>new TypeScriptEnvironment({mode:'retained'}));
  }
});
