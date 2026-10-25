import assert from 'node:assert/strict';
import test from 'node:test';
import {createNatlangSelectedEnvironment}
  from '../../tracing-js-self/src/natlang-selected.js';
import {encodeWitness}
  from '../../tracing-js-self/src/training/format.js';
import {installSelectedSemanticEnvironment,setDefaultEnvironmentFactory}
  from '../dist/runtime/node.js';
import {setModuleRealm,moduleInstanceAsync}
  from '../dist/runtime/modules.js';
import {parseModule,PATH_ONLY} from '../dist/runtime/loader.js';
import {itemValue} from '../dist/runtime/callable.js';
import {TypeScriptEnvironment} from '../dist/environment.js';

const valid=prefix=>{
  const semantic=prefix.committed.filter(item=>item.kind==='semantic').length;
  if(prefix.entry==='initialize'){
    if(semantic===0)return encodeWitness('allocate',
      {name:'o2',kind:'closure',code:'answer',properties:[],captures:[]});
    if(semantic===1)return encodeWitness('write',
      {object:'o0',key:'default',value:['ref','o2'],present:true});
    if(semantic===2)return encodeWitness('return',{value:['ref','o0']});
    if(semantic===3)return encodeWitness('end',{kind:'return'});
  }
  if(prefix.entry==='answer'){
    if(semantic===0)return encodeWitness('return',{value:['num',42]});
    if(semantic===1)return encodeWitness('end',{kind:'return'});
  }
  throw new Error('Selected Natlang test did not supply source context or entry');
};
function restore(){
  setDefaultEnvironmentFactory(options=>new TypeScriptEnvironment({
    workspace:options.workspace,network:options.network
  }));
  setModuleRealm(()=>new TypeScriptEnvironment({mode:'retained'}));
}

test('actual pinned tracer selected output runs Natlang callable module through async realm',async()=>{
  const record=parseModule('/__natlang__/folder/owned.ts',
    'export default function answer(): number {return 42;}',{},PATH_ONLY);
  const selected=[];
  try{
    installSelectedSemanticEnvironment(mode=>createNatlangSelectedEnvironment({
      mode,select:prefix=>{selected.push([prefix.entry,prefix.committed.length]);
        return valid(prefix);}
    }));
    const exports=await moduleInstanceAsync(record,{owned:record});
    assert.equal(typeof exports.default,'function');
    assert.equal(await exports.default(),42);
    const callable=itemValue(record,{owned:record});
    assert.equal(await callable(),42);
    assert.deepEqual(selected.filter(s=>s[0]==='initialize').length,4);
    assert.equal(selected.filter(s=>s[0]==='answer').length,4);
  }finally{restore();}
});
