import assert from 'node:assert/strict';
import {test} from 'node:test';
import {
  installSelectedSemanticEnvironment,createNatlangRuntime,currentFrame,
  setDefaultEnvironmentFactory
} from '../dist/runtime/node.js';
import {setModuleRealm,moduleInstance,moduleInstanceAsync} from '../dist/runtime/modules.js';
import {parseModule,PATH_ONLY} from '../dist/runtime/loader.js';
import {itemValue} from '../dist/runtime/callable.js';
import {TypeScriptEnvironment} from '../dist/environment.js';

function restoreNative(){
  setDefaultEnvironmentFactory(options=>new TypeScriptEnvironment({
    workspace:options.workspace,network:options.network
  }));
  setModuleRealm(()=>new TypeScriptEnvironment({mode:'retained'}));
}

function controlledSelected(mode,receipts){
  return {
    mode,authority:'selected-semantic/4',
    scopeCapabilities:{allowModules:true,allowNetwork:false},
    execute(){throw new Error('synchronous model path must never execute');},
    async executeAsync(request){
      receipts.push(['entry',mode,request.code]);
      return {result:73,events:[{operation:'selected.receipt',mode}]};
    },
    evaluateModule(){throw new Error('native synchronous VM module path forbidden');},
    async evaluateModuleAsync(code,bindings){
      receipts.push(['module',mode,code]);
      // A controlled action fixture, not real semantic inference. The mock
      // explicitly populates CommonJS module exports after async settlement.
      if(code.includes('require("./child")')){
        const child=bindings.require('./child');
        bindings.module.exports={answer:()=>child.value+1};
      } else if(code.includes('child') || code.includes('value')) {
        bindings.module.exports={value:12};
      } else {
        bindings.module.exports={default:()=>42,answer:()=>42};
      }
    },
    fork(){return controlledSelected('fresh',receipts);},
    close(){receipts.push(['closed',mode]);}
  };
}

test('selected evaluation uses async entry with explicit fail-closed synchronous route',async()=>{
  const receipts=[];
  let runtime;
  try{
    installSelectedSemanticEnvironment(mode=>controlledSelected(mode,receipts));
    runtime=createNatlangRuntime();
    const value=await runtime.run(async()=>{
      const env=currentFrame().task.environment();
      assert.equal(env.mode,'fresh');
      assert.equal(env.authority,'selected-semantic/4');
      assert.throws(()=>env.execute({code:'5',scope:{},body:false,path:'example'}),
        /asynchronous/);
      return env.executeAsync({code:'5',scope:{},body:false,path:'example'});
    });
    assert.equal(value.result,73);
    assert.deepEqual(receipts.filter(r=>r[0]==='entry'),[['entry','fresh','5']]);
  }finally{runtime?.close();restoreNative();}
});

test('async callable module initializes exactly once and uses selected evaluator',async()=>{
  const receipts=[];
  const record=parseModule('/__natlang__/folder/item.ts',
    'export default function answer(): number {return 42;}',{},PATH_ONLY);
  try{
    installSelectedSemanticEnvironment(mode=>controlledSelected(mode,receipts));
    assert.throws(()=>moduleInstance(record,{item:record}),/asynchronous/);
    const a=await moduleInstanceAsync(record,{item:record});
    const b=await moduleInstanceAsync(record,{item:record});
    assert.equal(a,b,'retained cached exports identity');
    assert.equal(a.answer(),42);
    const callable=itemValue(record,{item:record});
    assert.equal(await callable(),42);
    assert.equal(receipts.filter(r=>r[0]==='module').length,1);
    const first=a;
    setModuleRealm(()=>controlledSelected('retained',receipts));
    const next=await moduleInstanceAsync(record,{item:record});
    assert.notEqual(next,first,'authority cutover invalidates async module cache');
    assert.equal(receipts.filter(r=>r[0]==='module').length,2);
  }finally{restoreNative();}
});

test('async module init rejects missing selected authority or async module method',async()=>{
  try{
    installSelectedSemanticEnvironment(mode=>({
      ...controlledSelected(mode,[]),evaluateModuleAsync:undefined
    }));
    const record=parseModule('/__natlang__/folder/missing.ts',
      'export const answer: number = 1;',{},PATH_ONLY);
    await assert.rejects(moduleInstanceAsync(record,{missing:record}),
      /Selected asynchronous module evaluation|async module execution/);
  }finally{restoreNative();}
});
