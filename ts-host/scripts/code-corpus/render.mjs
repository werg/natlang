import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { readJsonl, writeJsonl, digest } from './common.mjs';
import { renderSftTurn } from '../export-native-sft.mjs';

export async function renderCodeRow(row,render) {
  if(row.kind!=='code_sft'||!row.syntax_checked||!row.completion) throw new Error('requires assembled syntax-checked code view');
  const turn={id:row.id,program_id:row.program_id,source_groups:row.source_groups,family:'code_corpus',skill:'code_generation',
    messages:[{role:'user',content:row.prompt}],tools:[],target:{role:'assistant',content:row.completion},training_admission:{approved:true,reason:'direct-code syntax checked; not execution verified'}};
  const pair=await renderSftTurn(turn,render);
  return {...pair,split:row.split,source:row.source,execution_verified:false,implementation_sha256:row.implementation_sha256};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href){
  const [input,output,server='http://127.0.0.1:8081']=process.argv.slice(2);
  if(!input||!output)throw new Error('Usage: render.mjs BUNDLE_SPLIT.jsonl NEW_OUTPUT.jsonl [TEMPLATE_SERVER]');
  const propsResponse=await fetch(`${server}/props`,{signal:AbortSignal.timeout(10000)});
  if(!propsResponse.ok)throw new Error('cannot inspect template server');
  const props=await propsResponse.json();
  const template=props.chat_template_tool_use??props.chat_template;
  if(typeof template!=='string')throw new Error('server does not expose template provenance');
  const rows=await readJsonl(input), pairs=new Array(rows.length);
  const render=async(messages,tools)=>{
    const response=await fetch(`${server}/apply-template`,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({messages,tools}),signal:AbortSignal.timeout(10000)});
    if(!response.ok)throw new Error(`template failed: ${response.status}`);
    const result=await response.json();if(typeof result.prompt!=='string')throw new Error('missing rendered prompt');return result.prompt;
  };
  let cursor=0;
  await Promise.all(Array.from({length:4},async()=>{while(cursor<rows.length){const i=cursor++;pairs[i]=await renderCodeRow(rows[i],render);}}));
  await writeJsonl(output,pairs);
  await writeJsonl(`${output}.manifest.json`,[{version:'natlang.sft.code/1',rows:pairs.length,input_sha256:digest(rows),
    renderer:{template_sha256:createHash('sha256').update(template).digest('hex'),server,end_token:'<|im_end|>'},execution_verified:false}]);
  console.log(JSON.stringify({rows:pairs.length,output}));
}
