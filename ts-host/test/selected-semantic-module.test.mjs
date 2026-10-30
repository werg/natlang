import assert from 'node:assert/strict';
import {test} from 'node:test';
import {
  selectedSemanticPlatform,createNatlangRuntime,currentFrame
} from '../dist/runtime/node.js';
import {moduleInstance,moduleInstanceAsync} from '../dist/runtime/modules.js';
import {parseModule,PATH_ONLY} from '../dist/runtime/loader.js';
import {itemValue} from '../dist/runtime/callable.js';
import {EvalFailure} from '../dist/native/evaluator.js';

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
    const platform=selectedSemanticPlatform(mode=>controlledSelected(mode,receipts));
    runtime=createNatlangRuntime({platform});
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
  }finally{runtime?.close();}
});

test('async callable module initializes exactly once and uses selected evaluator',async()=>{
  const receipts=[];
  const record=parseModule('/__natlang__/folder/item.ts',
    'export default function answer(): number {return 42;}',{},PATH_ONLY);
  {
    const platform=selectedSemanticPlatform(mode=>controlledSelected(mode,receipts));
    assert.throws(()=>moduleInstance(platform,record,{item:record}),/asynchronous/);
    const a=await moduleInstanceAsync(platform,record,{item:record});
    const b=await moduleInstanceAsync(platform,record,{item:record});
    assert.equal(a,b,'retained cached exports identity');
    assert.equal(a.answer(),42);
    const callable=itemValue(record,{item:record},undefined,platform);
    assert.equal(await callable(),42);
    assert.equal(receipts.filter(r=>r[0]==='module').length,1);
    // Module instances belong to the platform value: another platform runs its own copy.
    const other=selectedSemanticPlatform(mode=>controlledSelected(mode,receipts));
    const next=await moduleInstanceAsync(other,record,{item:record});
    assert.notEqual(next,a,'a different platform never serves this platform\'s module exports');
    assert.equal(receipts.filter(r=>r[0]==='module').length,2);
  }
});

test('async module init rejects missing selected authority or async module method',async()=>{
  {
    const platform=selectedSemanticPlatform(mode=>({
      ...controlledSelected(mode,[]),evaluateModuleAsync:undefined
    }));
    const record=parseModule('/__natlang__/folder/missing.ts',
      'export const answer: number = 1;',{},PATH_ONLY);
    await assert.rejects(moduleInstanceAsync(platform,record,{missing:record}),
      /Selected asynchronous module evaluation|async module execution/);
  }
});

test('selected evaluator preserves the actual EvalFailure class, events, close and fork ownership',async()=>{
  const receipts=[];let runtime;
  try{
    const platform=selectedSemanticPlatform(mode=>({...controlledSelected(mode,receipts),
      async executeAsync(){throw Object.assign(new Error('service rejected'),{
        events:[{operation:'service.failed'}],debug:{logs:['before failure']}});}
    }));
    runtime=createNatlangRuntime({platform});
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
  }finally{runtime?.close();}
});

test('concurrent roots in a cyclic module graph initialize once without deadlock',async()=>{
  const a=parseModule('/__natlang__/folder/a.ts','import {b} from "./b"; export const a = 1;',{},PATH_ONLY);
  const b=parseModule('/__natlang__/folder/b.ts','import {a} from "./a"; export const b = 2;',{},PATH_ONLY);
  const seen=[];
  {
    const platform=selectedSemanticPlatform(mode=>({...controlledSelected(mode,[]),
      async evaluateModuleAsync(code,bindings){
        const isA=bindings.__natlang_module.source.endsWith('/a.ts');
        bindings.require(isA?'./b':'./a');
        seen.push(isA?'a':'b');bindings.exports[isA?'a':'b']=isA?1:2;
      }
    }));
    const [first,second]=await Promise.all([moduleInstanceAsync(platform,a,{a,b}),moduleInstanceAsync(platform,b,{a,b})]);
    assert.equal(first.a,1);assert.equal(second.b,2);assert.deepEqual(seen,['b','a']);
  }
});

test('isolated task module scopes close their resources and never share captured state',async()=>{
  const record=parseModule('/__natlang__/folder/state.ts','export function next(): number {return 1;}',{},PATH_ONLY);
  const receipts=[];let runtime,stale;
  try{
    const platform=selectedSemanticPlatform(mode=>({...controlledSelected(mode,receipts),
      async evaluateModuleAsync(_code,bindings){let n=0;bindings.exports.next=()=>++n;}
    }));
    runtime=createNatlangRuntime({platform,isolateModules:true});
    const first=await runtime.run(async()=>{
      stale=await moduleInstanceAsync(platform,record,{state:record});return [stale.next(),stale.next()];
    });
    const second=await runtime.run(async()=>{
      const fresh=await moduleInstanceAsync(platform,record,{state:record});return fresh.next();
    });
    assert.deepEqual(first,[1,2]);assert.equal(second,1);
    assert.throws(()=>stale.next(),/authority closed/);
    assert.equal(receipts.filter(row=>row[0]==='closed').length,2);
  }finally{runtime?.close();}
});

test('failed initialization cannot repeat an uncertain native effect on cache retry',async()=>{
  const record=parseModule('/__natlang__/folder/fail.ts','export const value = 1;',{},PATH_ONLY);
  let writes=0;
  {
    const platform=selectedSemanticPlatform(mode=>({...controlledSelected(mode,[]),
      async evaluateModuleAsync(){writes++;throw new Error('uncertain initialization write');}
    }));
    await assert.rejects(moduleInstanceAsync(platform,record,{fail:record}),/uncertain initialization write/);
    await assert.rejects(moduleInstanceAsync(platform,record,{fail:record}),/uncertain initialization write/);
    assert.equal(writes,1);
  }
});
