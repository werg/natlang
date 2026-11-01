import assert from 'node:assert/strict';
import {test} from 'node:test';
import {chatCompletionModelTurn,assembleChatCompletion} from '../dist/model/chat-completion.js';
import {nativeProtocolFailure,nativeProtocolFeedback} from '../dist/native/native-protocol.js';
import {NativeToolAgent} from '../dist/native/agent.js';
import {session as open} from './support/natlang.mjs';

function failed(){
  const failure={schema:'natlang.native-marker-failure/1',code:'neuralese-native-state-close',
    message:'neuralese-native-state-close: generation emitted a produce end marker without an open span',
    trigger:'sample',sample_index:0,fed:false,own_input_count:1,generated_input_count:0,max_tokens:8,stop_token_ids:[2]};
  return {choices:[{finish_reason:'error',message:{role:'assistant',content:'<|tool_call_start|>malformed'}}],
    usage:{prompt_tokens:1,completion_tokens:1},x_natlang_native_failure:failure,
    native_state:{blocks:[],imports:[]},x_natlang_rl_capture:{schema:'natlang.native-rl-sampling/4',
      policy:{architecture:'lfm2',checkpoint_sha256:'a'.repeat(64),tokenizer_fingerprint:'b'.repeat(64),embedding_weight_sha256:'c'.repeat(64)},
      prompt:[{kind:'tokens',token_ids:[4],producing:[false]}],generated_inputs:[],writes:[],
      samples:[{index:0,token_id:65,logp:-.5,input_role:'ordinary',terminal:false,query_own_position:0}],
      finish_reason:'error',sample_range_convention:'half-open',protocol_failure:structuredClone(failure),
      sampling_law:{temperature:.6,seed:19,guidance:false,probabilities:'post-processing-normalized'}}};
}

test('typed failed reply is one captured turn; malformed-call auto retry cannot discard it',async()=>{
  const raw=failed();let calls=0;
  const driver=chatCompletionModelTurn(async()=>{calls++;return raw;});
  const turn=await driver({messages:[],tools:[],temperature:.6,seed:19,max_tokens:8});
  assert.equal(calls,1);assert.deepEqual(turn.calls,[]);assert.equal(turn.text,raw.choices[0].message.content);
  assert.deepEqual(turn.protocol_failure,raw.x_natlang_native_failure);assert.equal(turn.completion_tokens,1);
  assert.equal(turn.raw_response,raw);
});

test('runtime records failed actual turn then repairs same call without returning malformed text',async()=>{
  const requests=[];let calls=0;
  const driver=chatCompletionModelTurn(async body=>{
    requests.push(structuredClone(body));
    return calls++===0?failed():{choices:[{finish_reason:'stop',message:{role:'assistant',content:'fixed'}}],usage:{completion_tokens:1}};
  });
  const session=open({type:'() => string',instructions:'Return fixed.'}).session;
  await new NativeToolAgent(driver,{maxTurns:3}).run(session);
  assert.equal(session.lam.return,'fixed');assert.equal(calls,2);
  const feedback=requests[1].messages.at(-1);assert.equal(feedback.role,'user');
  assert.equal(feedback.content,nativeProtocolFeedback(failed().x_natlang_native_failure));
  assert.equal(requests[1].messages.at(-2).content,failed().choices[0].message.content);
  assert.equal(session.runtime.trace.events.filter(event=>event.kind==='model_turn').length,2);
  assert.equal(session.runtime.trace.events.filter(event=>event.kind==='model_protocol_failure').length,1);
});

test('streamed typed failure retains complete final sampling metadata',async()=>{
  const raw=failed();
  async function* stream(){
    yield {choices:[{delta:{role:'assistant'}}]};
    yield {choices:[{delta:{content:'partial'}}]};
    yield {...raw,choices:[{delta:{},finish_reason:'error'}],x_natlang_message:raw.choices[0].message};
  }
  const assembled=await assembleChatCompletion(stream());
  assert.deepEqual(nativeProtocolFailure(assembled),raw.x_natlang_native_failure);
  assert.equal(assembled.choices[0].message.content,raw.choices[0].message.content);
});

for(const tamper of ['missing','missing-inner','code-array','mismatch','fed','calls','finish','position'])test(`failed-turn integrity rejects ${tamper}`,()=>{
  const value=failed();
  if(tamper==='missing')delete value.x_natlang_native_failure;
  if(tamper==='missing-inner')delete value.x_natlang_rl_capture.protocol_failure;
  if(tamper==='code-array')value.x_natlang_native_failure.code=value.x_natlang_rl_capture.protocol_failure.code=['neuralese-native-state-close'];
  if(tamper==='mismatch')value.x_natlang_rl_capture.protocol_failure.code='invented';
  if(tamper==='fed')value.x_natlang_native_failure.fed=value.x_natlang_rl_capture.protocol_failure.fed=true;
  if(tamper==='calls')value.choices[0].message.tool_calls=[{function:{name:'eval',arguments:'{}'}}];
  if(tamper==='finish')value.choices[0].finish_reason='stop';
  if(tamper==='position')value.x_natlang_native_failure.own_input_count=value.x_natlang_rl_capture.protocol_failure.own_input_count=99;
  assert.throws(()=>nativeProtocolFailure(value),/neuralese-native-protocol-integrity/);
});

test('failure-looking literal content does not become protocol metadata',async()=>{
  const text=JSON.stringify(failed().x_natlang_native_failure);
  const raw={choices:[{finish_reason:'stop',message:{content:text}}]};
  assert.equal(nativeProtocolFailure(raw),undefined);
  const reply=await chatCompletionModelTurn(async()=>raw)({messages:[],tools:[]});
  assert.equal(reply.text,text);assert.equal(reply.protocol_failure,undefined);
});

test('failed native view preserves completed chunk turns and final failure before feedback',async()=>{
  const {createServer}=await import('node:http');
  const {serverViewer}=await import('../dist/neuralese/view.js');
  const {MemoryNeuraleseStore,encodeNativeState}=await import('../dist/native/neuralese-store.js');
  const {nativeStateFixture}=await import('./fixtures/native-state.mjs');
  const store=new MemoryNeuraleseStore();const artifact=await store.putNativeState(nativeStateFixture());
  const policy=Object.fromEntries(['architecture','checkpoint_sha256','tokenizer_fingerprint','embedding_weight_sha256'].map(key=>[key,artifact.metadata[key]]));
  const first={messages:[{role:'user',content:'first chunk'}],tools:[],template:{},response:{
    x_natlang_rl_capture:{schema:'natlang.native-rl-sampling/2',policy,prompt:[{kind:'tokens',token_ids:[1]}],
      generated_inputs:[60,4,5,61].map(token_id=>({token_id,input_role:'ordinary'})),
      writes:[{state_id:artifact.id,native_identity:policy,payload_own_begin:2,payload_own_end:4}]},
    native_state:{blocks:[{id:artifact.id}]}}};
  const last={messages:[{role:'user',content:'second chunk'}],tools:[],template:{},response:failed()};
  const turns=[first,last];let recorded;
  const server=createServer(async(req,res)=>{
    const chunks=[];for await(const chunk of req)chunks.push(chunk);
    res.setHeader('content-type','application/json');
    if(req.url==='/v1/neuralese/info')return res.end(JSON.stringify({dialects:['native-state/fixture'],
      capabilities:['view','native.protocol-recovery'],native_state:{enabled:true,role_marker_ids:[60,61,62,63]}}));
    if(req.url==='/v1/neuralese/view'){
      assert.equal(JSON.parse(Buffer.concat(chunks)).x_natlang_native_protocol_recovery,true);
      return res.end(JSON.stringify({source_turns:turns,x_natlang_native_failure:last.response.x_natlang_native_failure}));
    }
    if(req.url===`/v1/neuralese/native-states/${artifact.id}`){
      res.setHeader('content-type','application/octet-stream');return res.end(Buffer.from(encodeNativeState(artifact)));
    }
    res.statusCode=404;res.end('{}');
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  try{
    const viewer=serverViewer({endpoint:`http://127.0.0.1:${server.address().port}`,store,
      onSourceTurns:value=>{recorded=value;}});
    await assert.rejects(viewer({value:'two chunks'}),error=>error.code==='neuralese-native-state-close'&&error.native_source_turns.length===2);
    assert.deepEqual(recorded,turns);assert.ok(await store.getNativeState(artifact.id));
  }finally{await new Promise(resolve=>server.close(resolve));}
});

test('a malformed templated Neuralese write retries its declared typed readout and retains the failed turn',async()=>{
  const {createNatlangRuntime,loadVirtualNatlang}=await import('../dist/index.js');
  const requests=[],traces=[];let calls=0;
  const bad=failed();
  // The forced template fed a producer-open marker; the model actually
  // sampled the stop token before a producer-close marker.
  const failure={...bad.x_natlang_native_failure,code:'neuralese-native-state-open',
    message:'neuralese-native-state-open: generation stopped inside a produce span',
    own_input_count:2,generated_input_count:1};
  bad.x_natlang_native_failure=failure;
  bad.x_natlang_rl_capture.protocol_failure=structuredClone(failure);
  bad.x_natlang_rl_capture.generated_inputs=[{token_id:60,input_role:'ordinary',sample_index:null}];
  bad.x_natlang_rl_capture.samples=[{index:0,token_id:2,logp:-.5,input_role:'produce',terminal:true,query_own_position:1}];
  const block='nzs1_'+'a'.repeat(52);
  const driver=Object.assign(async request=>{
    requests.push(structuredClone(request));
    if(calls++===0)return {text:bad.choices[0].message.content,calls:[],completion_tokens:1,
      protocol_failure:bad.x_natlang_native_failure,raw_response:bad};
    return {calls:[['return_result',{status:'success',value:[{type:'native_state',id:block}]}]],completion_tokens:1};
  },{neuralese:true});
  const fn=loadVirtualNatlang({'writer.nl':'---\nargs: {}\nreturns: Neuralese<string>\nreadout: template\n---\nWrite it.\n'},'writer.nl');
  const result=await createNatlangRuntime({model:{driver},seed:{mode:'backend'},trace:t=>traces.push(t)}).run(()=>fn());
  assert.equal(result.$neuralese.id,block);assert.equal(requests.length,2);
  const expected={call:'return_result',arguments:{status:'success'},value_type:'string',value:'write'};
  assert.deepEqual(requests[0].template,expected);assert.deepEqual(requests[1].template,expected);
  assert.equal(requests[1].messages.at(-2).content,bad.choices[0].message.content);
  assert.equal(requests[1].messages.at(-1).content,nativeProtocolFeedback(bad.x_natlang_native_failure,true));
  assert.equal(traces.flatMap(t=>t.events).filter(e=>e.kind==='model_protocol_failure').length,1);
});


test('an ordinary decode-template failure keeps the existing untemplated repair',async()=>{
  const {createNatlangRuntime,loadVirtualNatlang}=await import('../dist/index.js');
  const requests=[];let calls=0;const bad=failed();
  const driver=Object.assign(async request=>{
    requests.push(structuredClone(request));
    if(calls++===0)return {text:bad.choices[0].message.content,calls:[],completion_tokens:1,
      protocol_failure:bad.x_natlang_native_failure,raw_response:bad};
    return {calls:[['return_result',{status:'success',value:'fixed'}]],completion_tokens:1};
  },{neuralese:true});
  const fn=loadVirtualNatlang({'reader.nl':'---\nargs: {}\nreturns: string\nreadout: template\n---\nRead it.\n'},'reader.nl');
  assert.equal(await createNatlangRuntime({model:{driver},seed:{mode:'backend'}}).run(()=>fn()),'fixed');
  assert.equal(requests[0].template.value,'decode');assert.equal(requests[1].template,undefined);
});

test('repeated malformed writes retain every failure and retry the typed template each time',async()=>{
  const {createNatlangRuntime,loadVirtualNatlang}=await import('../dist/index.js');
  const requests=[];let calls=0;const bad=failed();
  const driver=Object.assign(async request=>{
    requests.push(structuredClone(request));
    if(calls++<3)return {text:bad.choices[0].message.content,calls:[],completion_tokens:1,
      protocol_failure:bad.x_natlang_native_failure,raw_response:bad};
    return {calls:[['return_result',{status:'success',value:[{type:'native_state',id:'nzs1_'+'a'.repeat(52)}]}]],completion_tokens:1};
  },{neuralese:true});
  const fn=loadVirtualNatlang({'writer.nl':'---\nargs: {}\nreturns: Neuralese<string>\nreadout: template\n---\nWrite it.\n'},'writer.nl');
  await createNatlangRuntime({model:{driver,maxTurns:4},seed:{mode:'backend'}}).run(()=>fn());
  assert.equal(requests.length,4);
  assert.ok(requests.every(request=>request.template?.value==='write'));
  const feedback=nativeProtocolFeedback(bad.x_natlang_native_failure,true);
  assert.match(feedback,/single dedicated .* special token/);
  assert.match(feedback,/Ordinary text spelling those characters does not close/);
  assert.equal(requests[3].messages.filter(message=>message.role==='user'&&message.content===feedback).length,3);
  assert.equal(requests[3].messages.filter(message=>message.role==='assistant'&&message.content===bad.choices[0].message.content).length,3);
});
