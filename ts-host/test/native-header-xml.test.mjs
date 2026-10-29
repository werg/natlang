import test from 'node:test';
import assert from 'node:assert/strict';
import { chatCompletionModelTurn } from '../dist/model/chat-completion.js';
import { parseNativeXmlCalls } from '../dist/model/native-xml-calls.js';
import { trajectoryTurn } from '../dist/runtime/trajectory.js';
const tool = (name, properties, required=[]) => ({type:'function',function:{name,parameters:{type:'object',properties,required,additionalProperties:false}}});
const tools = [tool('eval',{code:{type:'string'},finish:{type:'boolean'},timeout_ms:{type:'integer',minimum:1}},['code']),
  tool('return_result',{status:{type:'string',enum:['success','blocked','failed']},value:{anyOf:[{type:'number'},{type:'null'}]},reason:{type:'string'}},['status'])];
const call=(name,params)=>`<tool_call>\n<function=${name}>\n${Object.entries(params).map(([k,v])=>`<parameter=${k}>\n${v}\n</parameter>\n`).join('')}</function>\n</tool_call>`;
const compact=call('eval',{code:'const hits = graph.links("a");\nconsole.log(JSON.stringify(hits));'});
test('complete captured-style call preserves code and handles native typed values',()=>{
  assert.deepEqual(parseNativeXmlCalls(compact,tools),[['eval',{code:'const hits = graph.links("a");\nconsole.log(JSON.stringify(hits));'}]]);
  assert.deepEqual(parseNativeXmlCalls(call('eval',{code:'if (a < b && b > 0) x.push(a);',finish:'true',timeout_ms:'10'})+'\n'+call('return_result',{status:'success',value:'null'}),tools),[['eval',{code:'if (a < b && b > 0) x.push(a);',finish:true,timeout_ms:10}],['return_result',{status:'success',value:null}]]);
  const arrayTool=tool('array',{items:{type:'array',items:{type:'integer'}}},['items']);
  assert.deepEqual(parseNativeXmlCalls(call('array',{items:'[1,2]'}),[arrayTool]),[['array',{items:[1,2]}]]);
});
test('unknown, malformed, truncated, ambiguous and mixed prose frames fail closed',()=>{
  for(const text of [compact.slice(0,-12), 'explanation '+compact, compact+' trailing',call('unknown',{code:'x'}),call('eval',{code:'x',extra:'bad'}),call('eval',{finish:'true'}),call('eval',{code:'x',finish:'maybe'}),call('return_result',{status:'invented'}),call('eval',{code:'x',timeout_ms:'0'})]) assert.throws(()=>parseNativeXmlCalls(text,tools));
});
test('header request preserves allowed schema out of band, raw response and reasoning; normal full remains strict',async()=>{
  const bodies=[];const transport=async body=>{bodies.push(body);return {choices:[{finish_reason:'stop',message:{content:compact,reasoning_content:'inspect'}}]};};
  const turn=chatCompletionModelTurn(transport);const base={messages:[{role:'user',content:'go'}],tools:[],response_tools:tools,seed:3,max_tokens:100};
  const result=await turn(base);assert.deepEqual(result.calls,[['eval',{code:'const hits = graph.links("a");\nconsole.log(JSON.stringify(hits));'}]]);assert.equal(result.reasoning,'inspect');assert.equal(result.text,'');assert.equal(result.raw_response.choices[0].message.content,compact);assert.equal(result.tool_parse.source,'assistant-content');
  const captured=trajectoryTurn(base,result);assert.deepEqual(captured.tools_offered,[]);assert.deepEqual(captured.response_tools,tools);assert.equal(captured.model_response.tool_parse.schema_custody,'response_tools');assert.deepEqual(captured.model_response.raw_calls,[]);assert.ok(!('tools'in bodies[0]));assert.ok(!('response_tools'in bodies[0]));
  const full=await turn({...base,tools,response_tools:undefined});assert.deepEqual(full.calls,[]);assert.equal(full.text,'');assert.equal(bodies.length,3);
});
test('truncated native XML never executes',async()=>{
  const turn=chatCompletionModelTurn(async()=>({choices:[{finish_reason:'length',message:{content:compact}}]}));
  const result=await turn({messages:[],tools:[],response_tools:tools,seed:3,max_tokens:1});assert.deepEqual(result.calls,[]);assert.equal(result.truncated,true);
});
