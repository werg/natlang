// Bounded real-actor diagnostic; all model requests go through the native server.
import {mkdir,writeFile,readFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {resolve,join} from 'node:path';
import {createHash} from 'node:crypto';

const [runtime,endpoint,expectedCheckpoint,out]=process.argv.slice(2);
if(!runtime||!endpoint||!/^[a-f0-9]{64}$/.test(expectedCheckpoint||'')||!out)
  throw new Error('usage: node probe_native_writer_consumer.mjs FROZEN_RUNTIME ENDPOINT CHECKPOINT_SHA256 NEW_OUTPUT_DIRECTORY');
await mkdir(out,{recursive:false});
const load=path=>import(pathToFileURL(join(resolve(runtime),'ts-host/dist',path)).href);
const [{NodeNativeRuntime},{NativeSession},{TypeEnv},{buildPending},{FileNeuraleseStore},
 {neuraleseServerModelTurn},{NativeToolAgent},{ModelTurnRecorder,setRecorderSource},{nativeStateConsumerFromInfo}]=await Promise.all([
  load('node-runtime.js'),load('native/runtime.js'),load('native/types.js'),load('native/values.js'),
  load('neuralese/node-block-store.js'),load('model/neuralese-server.js'),load('native/agent.js'),
  load('native/model-recording.js'),load('native/neuralese.js')]);
const info=await fetch(endpoint+'/v1/neuralese/info').then(r=>{if(!r.ok)throw Error(`info HTTP ${r.status}`);return r.json()});
if(info.native_state?.checkpoint_sha256!==expectedCheckpoint)throw Error('actor checkpoint does not match requested parent');
const store=await FileNeuraleseStore.open(join(out,'state'),{residentBytes:0});
const neuralese={store,dialect:info.dialects[0],nativeStateConsumer:nativeStateConsumerFromInfo(info)};
const recorder=new ModelTurnRecorder();setRecorderSource(()=>recorder);
const server=neuraleseServerModelTurn({endpoint,model:'native-lfm3',store});
const sent=[];
const driver=Object.assign(async request=>{sent.push(structuredClone(request));return server({...request,max_tokens:768});},
  {neuralese:true,decide:server.decide});
const budget={used:0};
function open(body){const lam=buildPending({$lambda:body});const instance=new NodeNativeRuntime({neuralese,sharedEpisodeBudget:budget});
 return new NativeSession(instance,lam,new TypeEnv().child(lam.types));}
const writer=open({type:'() => Neuralese<string>',instructions:'Write brief guidance for a consumer that must return supplied IDs in ascending lexical order. Explain that it should use exact JavaScript sorting on a copied array: return [...ids].sort().'});
writer.lam.readout='template';
const agent=()=>new NativeToolAgent(driver,{maxTurns:4,maxTokens:2048,maxSeconds:180,temperature:0,neuralese});
let writerOutcome,consumerOutcome,consumer,thrown;
try{
 writerOutcome=await agent().run(writer);
 const ref=writer.lam.return;
 if(ref?.$neuralese?.id){
  consumer=open({type:'(memo: Neuralese<string>, ids: string[]) => string',instructions:'Read the memo and return the IDs in ascending lexical order as a comma-separated string.',args:{memo:ref,ids:['z','a','m']}});
  consumerOutcome=await agent().run(consumer);
 }
}catch(error){thrown=String(error?.stack||error);}
let turns=[],graph=[],recordingError;
try{turns=recorder.nativeTurns();graph=[...recorder.graphNodes.values()];}catch(error){recordingError=String(error?.stack||error);}
const id=writer.lam.return?.$neuralese?.id;
const writerEvent=writer.runtime.trace.events.find(e=>e.kind==='block_write'&&e.block===id);
const readerEvent=consumer?.runtime.trace.events.find(e=>e.kind==='block_read'&&e.block===id);
const close=info.native_state.role_marker_ids[1];
const writerTurn=turns.find(t=>t.native?.sampling?.writes?.some(w=>w.state_id===id));
const sampledClose=!!writerTurn?.native?.sampling?.samples?.some(s=>s.token_id===close&&!s.terminal);
const boundWriter=writerTurn?.native?.graph?.written_nodes?.[id];
const readerTurn=turns.find(t=>t.native?.graph?.read_bindings?.[id]===boundWriter&&boundWriter);
const graphEdge=!!writerEvent&&!!readerEvent&&readerEvent.inputs?.some(i=>i.node===writerEvent.node)&&!!readerTurn;
const failures=turns.filter(t=>t.native?.sampling?.protocol_failure);
const retryFailures=failures.map(t=>{const index=turns.indexOf(t);return {index,code:t.native.sampling.protocol_failure.code,
 next_template:sent[index+1]?.template??null, next_feedback:sent[index+1]?.messages?.at(-1)?.content??null};});
const retriesVisible=retryFailures.every(({index,next_template,next_feedback})=>{
 const request=sent[index];
 return !sent[index+1] || (JSON.stringify(next_template)===JSON.stringify(request.template)&&
  typeof next_feedback==='string'&&next_feedback.includes(turns[index].native.sampling.protocol_failure.code));
});
let statePresent=false;if(id)statePresent=!!await store.getNativeState(id);
const result=thrown||recordingError?'infrastructure_error':!id?'writer_did_not_return_state':!sampledClose?'producer_close_not_sampled':
 !statePresent?'state_missing':!retriesVisible?'retry_custody_failed':!graphEdge?'consumer_edge_missing':consumer.lam.return===undefined?'consumer_did_not_complete':
 String(consumer.lam.return).replace(/\s/g,'')!=='a,m,z'?'consumer_wrong_result':'producer_consumer_complete';
const clientFiles=['node-runtime.js','native/runtime.js','native/types.js','native/values.js','neuralese/node-block-store.js',
 'model/neuralese-server.js','native/agent.js','native/model-recording.js','native/neuralese.js'];
const clientDigests=Object.fromEntries(await Promise.all(clientFiles.map(async file=>[file,
 createHash('sha256').update(await readFile(join(resolve(runtime),'ts-host/dist',file))).digest('hex')])));
const evidence={schema:'natlang.real-native-writer-consumer-probe/1',result,checkpoint:expectedCheckpoint,info,
 writerOutcome,consumerOutcome,writerReturn:writer.lam.return,consumerReturn:consumer?.lam.return,
 writerEvent,readerEvent,sampledClose,statePresent,retriesVisible,retryFailures,thrown,recordingError,
 client_runtime:resolve(runtime),client_module_sha256:clientDigests,probe_sha256:createHash('sha256').update(await readFile(new URL(import.meta.url))).digest('hex'),
 scope:'bounded train-world synthetic mechanism diagnostic; no optimizer update or capability-lift claim'};
for(const [name,value] of Object.entries({evidence,requests:sent,turns,graph,writer_trace:writer.runtime.trace.events,consumer_trace:consumer?.runtime.trace.events??[]})){
 const data=JSON.stringify(value,null,2)+'\n';await writeFile(join(out,name+'.json'),data);evidence[name+'_sha256']=createHash('sha256').update(data).digest('hex');
}
await writeFile(join(out,'receipt.json'),JSON.stringify(evidence,null,2)+'\n');
console.log(JSON.stringify({result,checkpoint:expectedCheckpoint,state_id:id??null,protocol_failures:failures.length,output:resolve(out)}));
process.exitCode=result==='producer_consumer_complete'?0:2;
