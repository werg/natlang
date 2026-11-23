import assert from 'node:assert/strict';
import {test} from 'node:test';
import {mkdtempSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {CallCapture} from '../dist/calls/recorder.js';
import {CallStore} from '../dist/calls/store.js';
import {recordingServices} from '../dist/native/effects.js';

function capture(store,id='pending',extra={}) {
  return new CallCapture(store,store.settings(),{callId:id,parentCallId:null,parentActionIndex:null,taskId:'task',programId:null,buildHash:null,programRoot:null,
    definition:{id:'function',name:'work',source:'work.nl',key:'key',interface:'interface',site:'named',instructions:{complete:false,reason:'excluded'},params:[],returns:'string',types:{}},model:{id:'teacher:luna',revision:null},...extra});
}
test('reopening a store retains unfinished inputs, exact native effect values and no invented completion',async()=>{
  const root=mkdtempSync(join(tmpdir(),'natlang-partial-'));
  let store=CallStore.open(root);store.writeSettings({minFreeBytes:0});
  try {
    const call=capture(store);call.setInputs({document:'owned'});call.announce();
    call.effect({phase:'requested',service:'documents',method:'read',seq:0,exact:{args:['owned']}});
    await call.effect({phase:'completed',service:'documents',method:'read',seq:0,exact:{result:{text:'actual native bytes',revision:7},async:true}});
    store.close();store=CallStore.open(root);
    assert.equal(store.call('pending'),undefined);
    const state=store.annotations('pending').find(row=>row.kind==='call-checkpoint').value;
    assert.equal(state.version,'natlang.call-checkpoint/1');
    assert.deepEqual(store.value(state.inputs.document),'owned');
    assert.deepEqual(store.value(state.effects[0].result),{text:'actual native bytes',revision:7});
    assert.equal(state.ended_at,undefined);assert.equal(state.output,undefined);
  } finally {store.close();rmSync(root,{recursive:true,force:true});}
});
test('async service results wait for acknowledgement and synchronous return types stay synchronous',async()=>{
  let release;const durable=new Promise(resolve=>{release=resolve;});let delivered=false;
  const services=recordingServices({native:{read:async()=>({receipt:'native'}),immediate:()=>42}},event=>event.phase==='completed'&&event.exact?.async?durable:undefined);
  const pending=services.native.read().then(value=>{delivered=true;return value;});
  await Promise.resolve();await Promise.resolve();assert.equal(delivered,false);
  assert.equal(services.native.immediate(),42);
  release();assert.deepEqual(await pending,{receipt:'native'});
});
test('incremental checkpoints can be explicitly disabled without disabling final call records',()=>{
  const rows=[];const store={settings:()=>({maxValueBytes:1024}),checkpoint:()=>{rows.push('checkpoint');},begin:()=>{},record:()=>{rows.push('record');}};
  const call=capture(store,'off',{checkpoints:false});call.setInputs({id:1});call.announce();
  call.finish({outcome:'done',detail:'',hasOutput:true,output:'answer',events:[]});
  assert.deepEqual(rows,['record']);
});

test('ordinary recording failures stay diagnostic while required capture failures prevent continuation',async()=>{
  const store={settings:()=>({maxValueBytes:1024}),begin:()=>{},checkpoint:()=>{throw new Error('storage unavailable');}};
  const ordinary=capture(store,'ordinary');ordinary.announce();
  ordinary.effect({phase:'requested',service:'native',method:'read',seq:0,exact:{args:[]}});
  await ordinary.effect({phase:'completed',service:'native',method:'read',seq:0,exact:{result:42,async:true}});
  const required=capture({...store,checkpointRequired:true},'required');required.announce();
  required.effect({phase:'requested',service:'native',method:'read',seq:0,exact:{args:[]}});
  await assert.rejects(async()=>required.effect({phase:'completed',service:'native',method:'read',seq:0,exact:{result:42,async:true}}),/storage unavailable/);
});

test('late progress cannot overwrite completed evidence or leave unknown-call blobs',()=>{
  const root=mkdtempSync(join(tmpdir(),'natlang-late-progress-'));const store=CallStore.open(root);store.writeSettings({minFreeBytes:0});
  try {
    const call=capture(store,'finished');call.setInputs({value:'input'});call.announce();
    call.finish({outcome:'done',detail:'',hasOutput:true,output:'actual',events:[{kind:'recorded',value:'original'}]});
    const events=store.events('finished');const bytes=store.totalBytes();
    store.progress('finished','{"kind":"forged-late"}\n');store.progress('never-started','{"kind":"orphan"}\n');
    assert.deepEqual(store.events('finished'),events);assert.equal(store.totalBytes(),bytes);
  } finally {store.close();rmSync(root,{recursive:true,force:true});}
});
