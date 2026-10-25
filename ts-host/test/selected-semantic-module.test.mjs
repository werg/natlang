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
import {EvalFailure} from '../dist/native/evaluator.js';

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
      } else if(code.includes('exports.value')) {
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

test('selected evaluator preserves the actual EvalFailure class, events, close and fork ownership',async()=>{
  const receipts=[];let runtime;
  try{
    installSelectedSemanticEnvironment(mode=>({...controlledSelected(mode,receipts),
      async executeAsync(){throw Object.assign(new Error('service rejected'),{
        events:[{operation:'service.failed'}],debug:{logs:['before failure']}});}
    }));
    runtime=createNatlangRuntime();
    await runtime.run(async()=>{
      const env=currentFrame().task.environment();
      await assert.rejects(env.executeAsync({code:'0',scope:{},path:'task',body:false}),error=>{
        assert(error instanceof EvalFailure);assert.deepEqual(error.events,[{operation:'service.failed'}]);
        assert.deepEqual(error.debug.logs,['before failure']);return true;
      });
      const child=env.fork();assert.equal(child.mode,'fresh');child.close();
      env.close();env.close();
      assert.throws(()=>env.fork(),/closed/);
      await assert.rejects(env.executeAsync({code:'0',scope:{},path:'task',body:false}),/closed/);
    });
    assert.equal(receipts.filter(row=>row[0]==='closed').length,2);
  }finally{runtime?.close();restoreNative();}
});

test('concurrent roots in a cyclic module graph initialize once without deadlock',async()=>{
  const a=parseModule('/__natlang__/folder/a.ts','import {b} from "./b"; export const a = 1;',{},PATH_ONLY);
  const b=parseModule('/__natlang__/folder/b.ts','import {a} from "./a"; export const b = 2;',{},PATH_ONLY);
  const seen=[];
  try{
    installSelectedSemanticEnvironment(mode=>({...controlledSelected(mode,[]),
      async evaluateModuleAsync(code,bindings){
        const isA=bindings.__natlang_module.source.endsWith('/a.ts');
        bindings.require(isA?'./b':'./a');
        seen.push(isA?'a':'b');bindings.exports[isA?'a':'b']=isA?1:2;
      }
    }));
    const [first,second]=await Promise.all([moduleInstanceAsync(a,{a,b}),moduleInstanceAsync(b,{a,b})]);
    assert.equal(first.a,1);assert.equal(second.b,2);assert.deepEqual(seen,['b','a']);
  }finally{restoreNative();}
});

test('isolated task module scopes close their resources and never share captured state',async()=>{
  const record=parseModule('/__natlang__/folder/state.ts','export function next(): number {return 1;}',{},PATH_ONLY);
  const receipts=[];let runtime,stale;
  try{
    installSelectedSemanticEnvironment(mode=>({...controlledSelected(mode,receipts),
      async evaluateModuleAsync(_code,bindings){let n=0;bindings.exports.next=()=>++n;}
    }));
    runtime=createNatlangRuntime({isolateModules:true});
    const first=await runtime.run(async()=>{
      stale=await moduleInstanceAsync(record,{state:record});return [stale.next(),stale.next()];
    });
    const second=await runtime.run(async()=>{
      const fresh=await moduleInstanceAsync(record,{state:record});return fresh.next();
    });
    assert.deepEqual(first,[1,2]);assert.equal(second,1);
    assert.throws(()=>stale.next(),/authority changed or closed/);
    assert.equal(receipts.filter(row=>row[0]==='closed').length,2);
  }finally{runtime?.close();restoreNative();}
});

test('cutover invalidates handed-out callable references and pending initialization',async()=>{
  const record=parseModule('/__natlang__/folder/waiting.ts','export function next(): number {return 1;}',{},PATH_ONLY);
  let settle;const entered=new Promise(resolve=>{settle=resolve;});let finish;
  try{
    installSelectedSemanticEnvironment(mode=>({...controlledSelected(mode,[]),
      async evaluateModuleAsync(_code,bindings){settle();await new Promise(resolve=>{finish=resolve;});bindings.exports.next=()=>1;}
    }));
    const callable=itemValue(record,{waiting:record});
    const pending=moduleInstanceAsync(record,{waiting:record});await entered;
    setModuleRealm(()=>controlledSelected('retained',[]));finish();
    await assert.rejects(pending,/closed/);
    assert.throws(()=>callable.next(),/authority changed or closed/);
  }finally{restoreNative();}
});

test('failed initialization cannot repeat an uncertain native effect on cache retry',async()=>{
  const record=parseModule('/__natlang__/folder/fail.ts','export const value = 1;',{},PATH_ONLY);
  let writes=0;
  try{
    installSelectedSemanticEnvironment(mode=>({...controlledSelected(mode,[]),
      async evaluateModuleAsync(){writes++;throw new Error('uncertain initialization write');}
    }));
    await assert.rejects(moduleInstanceAsync(record,{fail:record}),/uncertain initialization write/);
    await assert.rejects(moduleInstanceAsync(record,{fail:record}),/uncertain initialization write/);
    assert.equal(writes,1);
  }finally{restoreNative();}
});
