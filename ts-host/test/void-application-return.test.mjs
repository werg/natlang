import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {verifyApplicationReturn,verifyEntry} from '../scripts/observed-workspace-evidence.mjs';
test('void application transport binds only exact completed native TS void roots',()=>{
  const binding={kind:'void',root_call_id:'root',valueJson:'null',selectedResult:null};
  const entry={program:'skills/reference/review.ts',resultType:'void'};
  const record={outcome:'done',executor:{kind:'crisp'},definition:{returns:'void'},output:{complete:true,bytes:4,hash:createHash('sha256').update('null').digest('hex')}};
  const check=(b=binding,r=record,e=entry)=>verifyApplicationReturn(b,new Map([['root',r]]),['root'],e);
  check();
  for(const [section,key,value] of [['entry','resultType','string'],['entry','program','review.nl'],['record','outcome','error'],['binding','valueJson',' null'],['binding','selectedResult','invented'],['binding','kind','unknown']]){
    const changed=structuredClone({binding,entry,record});changed[section][key]=value;
    assert.throws(()=>check(changed.binding,changed.record,changed.entry));
  }
  for(const [field,value] of [['definition',{returns:'string'}],['executor',{kind:'agent'}],['output',{...record.output,hash:'0'.repeat(64)}]])
    assert.throws(()=>check(binding,{...record,[field]:value}));
  const missing={...binding};delete missing.selectedResult;assert.throws(()=>check(missing));
});

test('independent entry verification binds the actual TS default export',()=>{
 const text='"reference"',entry={program:'skills/reference/review.ts',callable:'review',arguments:{goal:'reference'},argumentJson:{goal:text}};
 const record={program_root:'skills/reference',outcome:'done',executor:{kind:'crisp'},definition:{name:'review',source:'review.ts',id:'ts:review.ts#default'},inputs:{goal:{complete:true,bytes:Buffer.byteLength(text),hash:createHash('sha256').update(text).digest('hex')}}};
 const interaction={program:entry.program,entry};verifyEntry(interaction,record,false);
 for(const id of ['ts:review.ts#other',undefined])assert.throws(()=>verifyEntry(interaction,{...record,definition:{...record.definition,id}},false),/default export/);
 assert.throws(()=>verifyEntry(interaction,{...record,executor:{kind:'agent'}},false),/default export/);
});
