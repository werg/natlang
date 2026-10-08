/** In-process OpenAI-compatible fake provider hook for test-diagnostic-skill-evaluator-mock.mjs. */
import {pathToFileURL} from 'node:url';
import {appendFile} from 'node:fs/promises';
const runtime='/home/werg/natlang/runs/authored-root-guided-runtime-v9/frozen';
const {registerLocalEndpoint}=await import(pathToFileURL(`${runtime}/dist/model/chat-completion.js`));
const endpoint='http://127.0.0.1:65533';
const expected=JSON.parse(process.env.CRITERION_FAKE_EXPECTED??'[]');
if (expected.length!==4) throw new Error('fake provider requires four scripted typed answers');
let activeCase=-1, requestCount=0;
registerLocalEndpoint(endpoint,async(url,init)=>{
  if (!url.startsWith(endpoint+'/v1/chat/completions') || init.method!=='POST') throw new Error('unexpected fake-provider URL/method');
  const body=JSON.parse(String(init.body)); requestCount++;
  const messages=body.messages??[];
  activeCase=Math.floor((requestCount-1)/4);
  const tools=body.tools??[], names=tools.map(tool=>tool.function?.name);
  let name,args;
  if (names.length===1 && names[0]==='execution_plan') {
    name='execution_plan';args={plan:'Read the relevant bound skill, apply the supplied criteria to scoped evidence, then return the typed label.'};
  } else {
    const hasRead=messages.some(message=>message.name==='read_code'||
      (message.tool_calls??[]).some(call=>call.function?.name==='read_code'));
    if (!hasRead) { name='read_code';args={name:'skills.judge-against-criteria'}; }
    else { name='return_result';args={status:'success',value:expected[activeCase%4]}; }
  }
  if (!names.includes(name)) throw new Error(`fake provider chose unoffered ${name}; available=${names.join(',')}`);
  if (process.env.CRITERION_FAKE_LOG) await appendFile(process.env.CRITERION_FAKE_LOG,JSON.stringify({requestCount,activeCase,names,selected:name,args,hadToolCalls:messages.map(message=>(message.tool_calls??[]).length)})+'\n');
  return new Response(JSON.stringify({id:`fake-${requestCount}`,object:'chat.completion',created:1,model:'mock-luna',choices:[{
    index:0,message:{role:'assistant',content:null,tool_calls:[{id:`fake-call-${requestCount}`,type:'function',function:{name,arguments:JSON.stringify(args)}}]},finish_reason:'tool_calls'}]}),
    {status:200,headers:{'content-type':'application/json'}});
});
