import assert from 'node:assert/strict';
import test from 'node:test';
import {createNatlangSelectedEnvironment}
  from '../../tracing-js-self/src/natlang-selected.js';
import {encodeWitness,decodeWitness}
  from '../../tracing-js-self/src/training/format.js';
import {analyzeSemanticProgram} from '../../tracing-js-self/src/training/semantic-program.js';
import {prepareHostRequest} from '../../tracing-js-self/src/host/operations.js';
import {createNatlangRuntime,installSelectedSemanticEnvironment,setDefaultEnvironmentFactory}
  from '../dist/runtime/node.js';
import {setModuleRealm,moduleInstanceAsync}
  from '../dist/runtime/modules.js';
import {parseModule,PATH_ONLY} from '../dist/runtime/loader.js';
import {itemValue} from '../dist/runtime/callable.js';
import {TypeScriptEnvironment} from '../dist/environment.js';
import {session as nativeSession} from './support/natlang.mjs';
import {parseType} from '../dist/native/types.js';
import {selectedNativeSurface,selectedNativeIdentity} from '../dist/native/selected-surface.js';
import {recordingServices,selectedServiceSurface} from '../dist/native/effects.js';
import {isNatlangCallable} from '../dist/runtime/callable.js';
import {Folder} from '../dist/native/scoped-fs.js';
import {EvalFailure} from '../dist/native/evaluator.js';

const valid=prefix=>{
  const semantic=prefix.committed.filter(item=>item.kind==='semantic').length;
  if(prefix.entry==='initialize'){
    if(semantic===0)return encodeWitness('allocate',
      {name:'o99',kind:'closure',code:'answer',properties:[],captures:[]});
    if(semantic===1)return encodeWitness('write',
      {object:'o0',key:'default',value:['ref','o99'],present:true});
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

test('actual generated scope binds live locals, creates inline child and commits selected capture writes',{timeout:10000},async()=>{
  const nativeCalls=[],selectedEntries=[];let rootSteps;
  const returned=prefix=>{
    const observed=decodeWitness(prefix.committed.findLast(item=>item.kind==='result').line).fields;
    assert.equal(observed.status,'fulfilled',JSON.stringify(observed));return observed.result;
  };
  const heapNode=(prefix,id)=>new Map(prefix.selectedState.heap).get(id);
  const environment=createNatlangSelectedEnvironment({select:prefix=>{
    selectedEntries.push(prefix.entry);
    const count=prefix.committed.filter(item=>item.kind==='semantic').length;
    if(prefix.entry.startsWith('function:')){
      const start=Number(prefix.entry.slice('function:'.length));
      const fn=analyzeSemanticProgram(prefix.source).functions.find(item=>item.start===start);
      const locals=new Map(prefix.selectedState.locals);
      if(fn.params[0]==='__v'){
        const aliases=new Map(prefix.selectedState.captureAliases),value=locals.get(aliases.get('__v')??'__v');
        if(count===0){assert.deepEqual(value,['num',2]);return encodeWitness('compute',{expr:{op:'local',name:'__v'},result:value,bind:'count'});}
        if(count===1)return encodeWitness('return',{value:['undefined']});
      }else if(count===0)return encodeWitness('return',{value:locals.get('counterCell')});
      return encodeWitness('end',{kind:'return'});
    }
    if(!rootSteps){
      assert.match(prefix.source,/const __natlang_inline = async/);
      assert.match(prefix.source,/await __natlang_inline\(/,'the actual generated immediate caller is migrated');
      assert.match(prefix.source,/__live\.bindLocal\?\./);
      const functions=analyzeSemanticProgram(prefix.source).functions;
      const getter=functions.find(fn=>/^\(\) => count$/.test(prefix.source.slice(fn.start,fn.end)));
      const setter=functions.find(fn=>/^\(__v\) => \{ count = __v; \}$/.test(prefix.source.slice(fn.start,fn.end)));
      assert(getter&&setter,'compiler publishes the actual captured accessor source');
      const binding=name=>new RegExp('__live\\["'+name+'"\\] = (__selected_binding_\\d+)').exec(prefix.source)[1];
      const handle=name=>new RegExp('function '+binding(name)+'\\([^]*?host\\.natlangCall(?:Sync)?\\("([^"]+)"').exec(prefix.source)[1];
      const alloc=(name,kind,properties,length)=>()=>encodeWitness('allocate',{name,kind,properties,...(length===undefined?{}:{length})});
      const call=(child,callee,args)=>()=>encodeWitness('call',{child,callee,arguments:args,receiver:['undefined']});
      const request=(p,child,operation,effect,args,promise)=>{
        const heap=Object.fromEntries(p.selectedState.heap.map(([id,node])=>[id,{...node,
          properties:Object.fromEntries([...node.properties].map(([key,value])=>[key,{value}]))}]));
        const sites=analyzeSemanticProgram(p.source).sites.filter(site=>site.operation===operation);
        const site=sites.find(site=>p.source.slice(site.start,site.end).includes(JSON.stringify(args[0][1])))??sites[0];
        return encodeWitness('request',{invocation:child,operation,effect,arguments:args,
          payload:prepareHostRequest(heap,operation,args).payload,site:{start:site.start,end:site.end},...(promise?{promise}:{})});
      };
      const ret=(invocation,value)=>()=>encodeWitness('return',{invocation,value});
      rootSteps=[
        ()=>encodeWitness('bind',{name:'counterCell',value:['num',1]}),
        ()=>encodeWitness('allocate',{name:'o900',kind:'closure',code:'function:'+getter.start,properties:[],captures:[{binding:'count',cell:'counterCell'}]}),
        ()=>encodeWitness('allocate',{name:'o901',kind:'closure',code:'function:'+setter.start,properties:[],captures:[{binding:'count',cell:'counterCell'}]}),
        alloc('o902','array',[{key:'0',value:['str','count']},{key:'1',value:['ref','o900']},{key:'2',value:['ref','o901']}],3),
        call('bindnative1',binding('bindLocal'),[['str','count'],['ref','o900'],['ref','o901']]),
        p=>request(p,'bindnative1','natlang.callSync','ebind1',[['str',handle('bindLocal')],['undefined'],['ref','o902']]),
        ret('bindnative1',['undefined']),
        alloc('o903','array',[{key:'0',value:['ref','o900']},{key:'1',value:['ref','o901']}],2),
        alloc('o904','object',[{key:'count',value:['ref','o903']}]),alloc('o905','array',[],0),
        alloc('o906','array',[{key:'0',value:['num',0]},{key:'1',value:['ref','o905']},{key:'2',value:['ref','o904']}],3),
        call('inlinenative1',binding('inline'),[['num',0],['ref','o905'],['ref','o904']]),
        p=>request(p,'inlinenative1','natlang.call','einline1',[['str',handle('inline')],['undefined'],['ref','o906']],'pinline1'),
        ret('inlinenative1',['ref','pinline1']),
        p=>encodeWitness('await',{invocation:p.invocationId,promise:'pinline1',continuation:'kinline1'}),
        p=>encodeWitness('resume',{invocation:p.invocationId,continuation:'kinline1',status:'fulfilled',result:returned(p)}),
        p=>encodeWitness('bind',{name:'nativeHandle',value:new Map(heapNode(p,returned(p)[1]).properties).get('handle')}),
        p=>encodeWitness('allocate',{name:'o907',kind:'closure',code:'__selected_dynamic',
          properties:[{key:'__selected_native_callable',value:new Map(p.selectedState.locals).get('nativeHandle')}],
          captures:[{binding:'handle',cell:'nativeHandle'}]}),
        call('childnative1','__selected_dynamic',[]),
        p=>request(p,'childnative1','natlang.call','echild1',[new Map(p.selectedState.locals).get('nativeHandle'),['undefined'],['ref','o905']],'pchild1'),
        ret('childnative1',['ref','pchild1']),
        p=>encodeWitness('await',{invocation:p.invocationId,promise:'pchild1',continuation:'kchild1'}),
        p=>encodeWitness('resume',{invocation:p.invocationId,continuation:'kchild1',status:'fulfilled',result:returned(p)}),
        p=>{assert.deepEqual(new Map(p.selectedState.locals).get('counterCell'),['num',2]);return alloc('o908','object',[
          {key:'count',value:['num',2]},{key:'read',value:['ref','o907']}])();},
        alloc('o909','object',[{key:'result',value:['num',2]},{key:'returned',value:['bool',true]},{key:'bindings',value:['ref','o908']}]),
        alloc('o910','array',[{key:'0',value:['ref','o909']}],1),
        call('finishnative1',binding('finish'),[['ref','o909']]),
        p=>request(p,'finishnative1','natlang.callSync','efinish1',[['str',handle('finish')],['undefined'],['ref','o910']]),
        ret('finishnative1',['undefined']),p=>encodeWitness('return',{value:['null']}),()=>encodeWitness('end',{kind:'return'})
      ];
    }
    assert(rootSteps[count],`fixture exhausted at selected record ${count}`);return rootSteps[count](prefix);
  }});
  environment.configureNativeBindings({adaptValue:value=>selectedNativeSurface(selectedServiceSurface(value)),resourceIdentity:selectedNativeIdentity});
  const fixture=nativeSession({type:'() => number',instructions:'Return an incremented count.'},{environment,calls:false,
    agent:async child=>{
      const cell=child.lam.captures.count;const before=cell.get();nativeCalls.push(before);cell.set(before+1);child.lam.return=before+1;
    }});
  fixture.lam.let.count=1;fixture.lam.letTypes.count=parseType('number');
  try{
    const result=await fixture.session.applyAsync('eval',{code:'const read = nl<number>`Increment count and return it.`; return await read();'});
    assert.equal(result.kind,'ok',result.text);assert.equal(fixture.lam.return,2);assert.equal(fixture.lam.let.count,2);
    assert(isNatlangCallable(fixture.lam.let.read),'native callable metadata survives the selected scope output');
    assert.deepEqual(nativeCalls,[1]);
    assert.equal(environment.selectedHarness().timeline.filter(item=>item.kind==='dispatch').length,4);
    assert(selectedEntries.some(entry=>entry.startsWith('function:')),'all captured accessors execute as selected guest invocations');
  }finally{environment.close();fixture.runtime.frame.task.close();}
});

test('actual compiled sibling imports read a live selected module export through the broker',async()=>{
  const child=parseModule('/__natlang__/folder/child.ts','export const value: number = 12;',{},PATH_ONLY);
  const parent=parseModule('/__natlang__/folder/parent.ts',
    'import {value} from "./child"; export default function answer(): number {return value+1;}',{},PATH_ONLY);
  const receipts=[];
  try{
    installSelectedSemanticEnvironment(mode=>createNatlangSelectedEnvironment({mode,select:prefix=>{
      receipts.push(prefix);
      const count=prefix.committed.filter(item=>item.kind==='semantic').length;
      if(prefix.entry==='initialize'){
        const isChild=prefix.source.includes('exports.value = 12');
        const lines=isChild?[
          ()=>encodeWitness('write',{object:'o0',key:'value',value:['num',12],present:true}),
          ()=>encodeWitness('return',{value:['ref','o0']}),()=>encodeWitness('end',{kind:'return'})
        ]:[
          ()=>encodeWitness('allocate',{name:'o99',kind:'closure',code:'answer',properties:[],captures:[]}),
          ()=>encodeWitness('write',{object:'o0',key:'default',value:['ref','o99'],present:true}),
          ()=>encodeWitness('return',{value:['ref','o0']}),()=>encodeWitness('end',{kind:'return'})
        ];return lines[count]();
      }
      if(count===0){
        const getter=/function (__selected_get_\d+)\(\)/.exec(prefix.source)[1];
        return encodeWitness('call',{child:'getter1',callee:getter,receiver:['undefined'],arguments:[]});
      }
      if(count===1){
        const handle=/host.natlangReadCell\("([^"]+)"/.exec(prefix.source)[1];
        const site=analyzeSemanticProgram(prefix.source).sites.find(site=>site.operation==='natlang.readCell'&&
          prefix.source.slice(site.start,site.end).includes(JSON.stringify(handle)));
        const args=[['str',handle]],payload=prepareHostRequest({},'natlang.readCell',args).payload;
        return encodeWitness('request',{effect:'e1',operation:'natlang.readCell',arguments:args,payload,
          invocation:'getter1',site:{start:site.start,end:site.end}});
      }
      if(count===2)return encodeWitness('return',{invocation:'getter1',value:['num',12]});
      if(count===3){
        const observed=prefix.committed.findLast(item=>item.kind==='result');assert.deepEqual(observed.result,['num',12]);
        return encodeWitness('compute',{expr:{op:'binary',operator:'+',left:{op:'literal',value:observed.result},
          right:{op:'literal',value:['num',1]}},result:['num',13]});
      }
      if(count===4)return encodeWitness('return',{value:['num',13]});return encodeWitness('end',{kind:'return'});
    }}));
    const loaded=await moduleInstanceAsync(parent,{parent,child});
    assert.equal(await loaded.default(),13);
    assert.equal(await itemValue(parent,{parent,child})(),13);
    const initializations=new Set(receipts.filter(prefix=>prefix.entry==='initialize').map(prefix=>prefix.invocationId));
    assert.equal(initializations.size,2,'each dependency initialized exactly once');
    assert(receipts.some(prefix=>prefix.source.includes('__imports["./child"]')));
  }finally{restore();}
});

test('declared FileHandle surface performs selected ordered read/write/read and revokes on close',async()=>{
  const folder=Folder.fromFiles({'note.txt':'before'});
  const environment=createNatlangSelectedEnvironment({select:prefix=>{
    const count=prefix.committed.filter(item=>item.kind==='semantic').length;
    const invocation=Math.floor(count/6),phase=count%6,method=invocation===1?'writeText':'readText';
    if(count===18)return encodeWitness('return',{value:['str','before!']});
    if(count===19)return encodeWitness('end',{kind:'return'});
    const callee=new RegExp('__live\\["file"\\]\\["'+method+'"\\]\\s*=\\s*(__selected_binding_\\d+)').exec(prefix.source)[1];
    const handle=new RegExp('function '+callee+'\\([^]*?host.natlangCall\\("([^"]+)"').exec(prefix.source)[1];
    const args=invocation===1?[['str','before!']]:[],array='o'+(900+invocation),child='native'+invocation,promise='p'+invocation;
    if(phase===0)return encodeWitness('allocate',{name:array,kind:'array',length:args.length,properties:args.map((value,index)=>({key:String(index),value}))});
    if(phase===1)return encodeWitness('call',{child,callee,arguments:args,receiver:['undefined']});
    if(phase===2){
      const site=analyzeSemanticProgram(prefix.source).sites.find(site=>site.operation==='natlang.call'&&prefix.source.slice(site.start,site.end).includes(JSON.stringify(handle)));
      const heap=Object.fromEntries(prefix.selectedState.heap.map(([id,node])=>[id,{...node,properties:Object.fromEntries([...node.properties].map(([key,value])=>[key,{value}]))}]));
      const values=[['str',handle],['undefined'],['ref',array]];
      return encodeWitness('request',{invocation:child,effect:'e'+invocation,operation:'natlang.call',arguments:values,promise,
        payload:prepareHostRequest(heap,'natlang.call',values).payload,site:{start:site.start,end:site.end}});
    }
    if(phase===3)return encodeWitness('return',{invocation:child,value:['ref',promise]});
    if(phase===4)return encodeWitness('await',{invocation:prefix.invocationId,promise,continuation:'k'+invocation});
    const observed=decodeWitness(prefix.committed.findLast(item=>item.kind==='result').line).fields;
    assert.equal(observed.status,'fulfilled',JSON.stringify(observed));
    return encodeWitness('resume',{invocation:prefix.invocationId,continuation:'k'+invocation,status:observed.status,result:observed.result});
  }});
  environment.configureNativeBindings({adaptValue:selectedNativeSurface,resourceIdentity:selectedNativeIdentity});
  try{
    const result=await environment.executeAsync({path:'folder.ts',body:true,scope:{},live:{file:folder.file('note.txt')},
      code:'const text=await __live.file.readText();await __live.file.writeText(text+"!");return await __live.file.readText();'});
    assert.equal(result.result,'before!');assert.equal(await folder.readText('note.txt'),'before!');
    assert.equal(environment.selectedHarness().timeline.filter(item=>item.kind==='dispatch').length,3);
  }finally{environment.close();}
  await assert.rejects(environment.executeAsync({code:'0',body:false,path:'closed.ts',scope:{}}),/closed/);
});

test('real selected console observations and logs survive the shared EvalFailure class',async()=>{
  const environment=createNatlangSelectedEnvironment({EvalFailure,select:prefix=>{
    const count=prefix.committed.filter(item=>item.kind==='semantic').length;
    if(count===0){
      const site=analyzeSemanticProgram(prefix.source).sites.find(site=>site.operation==='platform.console.log');
      const args=[['str','before failure']];
      return encodeWitness('request',{invocation:prefix.invocationId,effect:'elog1',operation:'platform.console.log',arguments:args,
        payload:prepareHostRequest({},'platform.console.log',args).payload,site:{start:site.start,end:site.end}});
    }
    if(count===1)return encodeWitness('throw',{value:['str','broken']});return encodeWitness('end',{kind:'throw'});
  }});
  try{
    await assert.rejects(environment.executeAsync({code:'console.log("before failure");throw "broken";',body:true,path:'failure.ts',scope:{}}),error=>{
      assert(error instanceof EvalFailure);assert.deepEqual(error.debug.logs,['before failure']);
      assert(error.events.some(event=>event.kind==='result'&&event.effectId==='elog1'));return true;
    });
  }finally{environment.close();}
});

test('retained imported services resolve the current task and preserve actual effect recording',async()=>{
  const record=parseModule('/__natlang__/folder/service.ts',
    'import {store} from "natlang:services"; export default async function read(): Promise<number> {return await store.lookup(7);}',{},PATH_ONLY);
  const effects=[],calls=[],runtime=createNatlangRuntime({calls:false});let initializations=0;
  const services=base=>recordingServices({store:{base,async lookup(value){calls.push(this.base);return this.base+value;}}},event=>effects.push(event));
  try{
    installSelectedSemanticEnvironment(mode=>createNatlangSelectedEnvironment({mode,select:prefix=>{
      const count=prefix.committed.filter(item=>item.kind==='semantic').length;
      if(prefix.entry==='initialize'){
        if(count===0){initializations++;return encodeWitness('allocate',{name:'o99',kind:'closure',code:'read',properties:[],captures:[]});}
        if(count===1)return encodeWitness('write',{object:'o0',key:'default',value:['ref','o99'],present:true});
        if(count===2)return encodeWitness('return',{value:['ref','o0']});
        return encodeWitness('end',{kind:'return'});
      }
      const ordinal=Number(prefix.invocationId.replace(/\D/g,'')),array='o'+(900+ordinal);
      const promise='pservice'+ordinal,continuation='kservice'+ordinal;
      const callee=/__imports\["natlang:services"\]\["store"\]\["lookup"\]\s*=\s*(__selected_binding_\d+)/.exec(prefix.source)[1];
      const handle=new RegExp('function '+callee+'\\([^]*?host.natlangCall\\("([^"]+)"').exec(prefix.source)[1];
      if(count===0)return encodeWitness('allocate',{name:array,kind:'array',length:1,properties:[{key:'0',value:['num',7]}]});
      if(count===1)return encodeWitness('call',{child:'native1',callee,arguments:[['num',7]],receiver:['undefined']});
      if(count===2){
        const site=analyzeSemanticProgram(prefix.source).sites.find(site=>site.operation==='natlang.call'&&prefix.source.slice(site.start,site.end).includes(JSON.stringify(handle)));
        const heap=Object.fromEntries(prefix.selectedState.heap.map(([id,node])=>[id,{...node,
          properties:Object.fromEntries([...node.properties].map(([key,value])=>[key,{value}]))}]));
        const args=[['str',handle],['undefined'],['ref',array]];
        return encodeWitness('request',{invocation:'native1',effect:'eservice1',operation:'natlang.call',arguments:args,promise,
          payload:prepareHostRequest(heap,'natlang.call',args).payload,site:{start:site.start,end:site.end}});
      }
      if(count===3)return encodeWitness('return',{invocation:'native1',value:['ref',promise]});
      if(count===4)return encodeWitness('await',{invocation:prefix.invocationId,promise,continuation});
      const observed=decodeWitness(prefix.committed.findLast(item=>item.kind==='result').line).fields;
      assert.equal(observed.status,'fulfilled',JSON.stringify(observed));
      if(count===5)return encodeWitness('resume',{invocation:prefix.invocationId,continuation,status:'fulfilled',result:observed.result});
      if(count===6)return encodeWitness('return',{value:observed.result});
      return encodeWitness('end',{kind:'return'});
    }}));
    const invoke=async()=>await (await moduleInstanceAsync(record,{service:record})).default();
    assert.equal(await runtime.run(invoke,{services:services(5)}),12);
    assert.equal(await runtime.run(invoke,{services:services(20)}),27);
    assert.equal(initializations,1);assert.deepEqual(calls,[5,20]);
    assert.deepEqual(effects.map(event=>[event.phase,event.service,event.method]),[
      ['requested','store','lookup'],['completed','store','lookup'],
      ['requested','store','lookup'],['completed','store','lookup']]);
    assert.deepEqual(effects.filter(event=>event.phase==='completed').map(event=>event.exact),[
      {result:12,async:true},{result:27,async:true}]);
  }finally{runtime.close();restore();}
});

test('cyclic selected modules publish live import cells before either initializer completes',async()=>{
  const a=parseModule('/__natlang__/folder/a.ts','import "./b"; export const count: number = 5;',{},PATH_ONLY);
  const b=parseModule('/__natlang__/folder/b.ts','import {count} from "./a"; export function read(): number {return count;}',{},PATH_ONLY);
  const initialized=[];
  try{
    installSelectedSemanticEnvironment(mode=>createNatlangSelectedEnvironment({mode,select:prefix=>{
      const count=prefix.committed.filter(item=>item.kind==='semantic').length;
      if(prefix.entry==='initialize'){
        const isA=prefix.source.includes('exports.count = 5');
        if(count===0)initialized.push(isA?'a':'b');
        if(isA){
          if(count===0)return encodeWitness('write',{object:'o0',key:'count',value:['num',5],present:true});
          if(count===1)return encodeWitness('return',{value:['ref','o0']});
        }else{
          if(count===0)return encodeWitness('allocate',{name:'o99',kind:'closure',code:'read',properties:[],captures:[]});
          if(count===1)return encodeWitness('write',{object:'o0',key:'read',value:['ref','o99'],present:true});
          if(count===2)return encodeWitness('return',{value:['ref','o0']});
        }
        return encodeWitness('end',{kind:'return'});
      }
      const path='Object.defineProperty(__imports["./a"],"count"';
      const descriptor=prefix.source.slice(prefix.source.indexOf(path));
      assert(prefix.source.includes(path),'the cyclic namespace exports exist before the imported module is initialized');
      const callee=/get:(__selected_get_\d+)/.exec(descriptor)[1];
      const handle=new RegExp('function '+callee+'\\([^]*?host.natlangReadCell\\("([^"]+)"').exec(prefix.source)[1];
      if(count===0)return encodeWitness('call',{child:'getter1',callee,receiver:['undefined'],arguments:[]});
      if(count===1){
        const args=[['str',handle]],site=analyzeSemanticProgram(prefix.source).sites.find(site=>site.operation==='natlang.readCell'&&
          prefix.source.slice(site.start,site.end).includes(JSON.stringify(handle)));
        return encodeWitness('request',{invocation:'getter1',effect:'ecount1',operation:'natlang.readCell',arguments:args,
          payload:prepareHostRequest({},'natlang.readCell',args).payload,site:{start:site.start,end:site.end}});
      }
      const observed=decodeWitness(prefix.committed.findLast(item=>item.kind==='result').line).fields;
      assert.equal(observed.status,'fulfilled');assert.deepEqual(observed.result,['num',5]);
      if(count===2)return encodeWitness('return',{invocation:'getter1',value:observed.result});
      if(count===3)return encodeWitness('return',{value:observed.result});
      return encodeWitness('end',{kind:'return'});
    }}));
    const [first,second]=await Promise.all([moduleInstanceAsync(a,{a,b}),moduleInstanceAsync(b,{a,b})]);
    assert.equal(first.count,5);assert.equal(await second.read(),5);assert.deepEqual(initialized,['b','a']);
  }finally{restore();}
});
