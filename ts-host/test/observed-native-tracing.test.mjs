import assert from 'node:assert/strict';
import {test} from 'node:test';
import {
  installObservedNativeTracingEnvironment,createNatlangRuntime,
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

test('unsupported native JS emits start/completion observation records',async()=>{
  const records=[];
  let runtime;
  try {
    installObservedNativeTracingEnvironment(record=>records.push(record));
    runtime=createNatlangRuntime();
    const result=await runtime.run(()=>currentFrame().task.environment().execute({
      code:'[1,2,3].map(x=>x*2).join(",")',body:false,scope:{},path:'observe://native'
    }));
    assert.equal(result.result,'2,4,6');
    assert.deepEqual(records.map(record=>record.phase),['start','completed']);
    assert.equal(records[0].schema,'natlang-native-observation/1');
    assert.equal(result.events[0].authority,'native-observed/1');
  }finally {runtime?.close();restore();}
});

test('native eval failure records start and failure without a completion',async()=>{
  const records=[];
  let runtime;
  try {
    installObservedNativeTracingEnvironment(record=>records.push(record));
    runtime=createNatlangRuntime();
    await assert.rejects(runtime.run(()=>currentFrame().task.environment().execute({
      code:'throw new Error("boom")',body:true,scope:{},path:'observe://failure'
    })),/boom/);
    assert.deepEqual(records.map(record=>record.phase),['start','failed']);
  }finally {runtime?.close();restore();}
});

test('retained module bootstrap is observed, not claimed as owned instruction trace',()=>{
  const records=[];
  try {
    installObservedNativeTracingEnvironment(record=>records.push(record));
    const record=parseModule('/__natlang__/folder/observed.ts',
      'export const answer: number = 1;',{},PATH_ONLY);
    const exports=moduleInstance(record,{observed:record});
    assert.equal(exports.answer,1);
    assert.ok(records.some(item=>item.kind==='evaluateModule'&&item.phase==='start'));
    assert.ok(records.some(item=>item.kind==='evaluateModule'&&item.phase==='completed'));
  }finally {restore();}
});

test('retained callable invocations are observed with receiver and thrown errors',()=>{
  const records=[];
  try {
    installObservedNativeTracingEnvironment(record=>records.push(record));
    const record=parseModule('/__natlang__/folder/observed-call.ts',
      'export function twice(n: number): number { return n * 2; }\n' +
      'export function fail(): never { throw new Error("call failed"); }',{},PATH_ONLY);
    const exports=moduleInstance(record,{'observed-call':record});
    assert.equal(exports.twice(4),8);
    assert.throws(()=>exports.fail(),/call failed/);
    const calls=records.filter(item=>item.kind==='exportCall');
    assert.deepEqual(calls.map(item=>[item.exportName,item.phase]),[
      ['twice','start'],['twice','completed'],
      ['fail','start'],['fail','failed']
    ]);
    assert.equal(calls[0].invocationId,calls[1].invocationId);
    assert.notEqual(calls[0].invocationId,calls[2].invocationId);
  } finally {restore();}
});
