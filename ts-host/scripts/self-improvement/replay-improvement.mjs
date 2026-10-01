/** Execute original teacher actions against the current runtime; emit current observations, never relabel old ones. */
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {migrateRevisionCode} from './migrate-revision-code.mjs';
import {Folder,improveProgram} from '../../dist/index.js';
const [casesPath,trajectoryPath,output]=process.argv.slice(2);
if(!output)throw Error('usage: replay-improvement.mjs CASES_JSONL ORIGINAL_TRAJECTORY OUTPUT_DIR');
const original=JSON.parse(await readFile(trajectoryPath,'utf8'));
const row=(await readFile(casesPath,'utf8')).trim().split('\n').map(JSON.parse).find(row=>row.id===original.id);
if(!row)throw Error('original source case missing');
await mkdir(output,{recursive:true});
const exchanges=[],traces=[];let index=0;
const driver=async request=>{
 const response=structuredClone(original.exchanges[index++]?.wireResponse?.choices?.[0]?.message);
 if(response)for(const call of response.tool_calls??[]){const args=JSON.parse(call.function.arguments);if(call.function.name==='eval'&&typeof args.code==='string')args.code=migrateRevisionCode(args.code).source;call.function.arguments=JSON.stringify(args);}
 if(!response)throw Error('original teacher action sequence exhausted');
 exchanges.push(structuredClone({request,wireRequest:{messages:request.messages,tools:request.tools},wireResponse:{choices:[{message:response}]}}));
 return {calls:(response.tool_calls??[]).map(call=>[call.function.name,JSON.parse(call.function.arguments)]),text:response.content??'',reasoning:response.reasoning_content??undefined,raw_calls:response.tool_calls??[]};
};
try{
 const result=await improveProgram({folder:Folder.fromFiles(row.files),contract:row.contract,cases:row.cases,policy:row.policy,improver:driver,
 executor:()=>{throw Error('replay requires exact target execution; inference targets need recollection');},executorId:'exact-source-replay',
 trace:trace=>traces.push(trace),directory:join(output,row.id+'.journal'),budget:{maxRollouts:40,maxModelCalls:104,maxProposals:3,maxElapsedMs:600000}});
 const source=Object.fromEntries(result.folder.filePaths().map(path=>[path,new TextDecoder().decode(result.folder.readBytesSync(path))]));
 const accepted=result.validation!==null && result.state.done && result.state.incumbent===result.folder.digest && (result.state.history.length>0 || row.control===true);
 const artifact={version:'natlang.improvement-trajectory/1',id:row.id,accepted,source,authored:result.authored,...('error'in result?{error:result.error}:{}),disposition:result.disposition,validation:result.validation,state:result.state,ledger:result.ledger,exchanges,traces,targetExchanges:[],incidents:row.incidents,caseDefinition:row,sourceGroups:row.sourceGroups??[],
 replay:{source:trajectoryPath,sha256:createHash('sha256').update(await readFile(trajectoryPath)).digest('hex'),providerCalls:0,actionsReexecuted:index,migration:'repository calls to automatic folder revisions; current authored source; fresh current observations'}};
 await writeFile(join(output,row.id+'.result.json'),JSON.stringify(artifact));console.log(JSON.stringify({accepted,actions:index}));
}catch(error){await writeFile(join(output,row.id+'.error.json'),JSON.stringify({id:row.id,error:String(error),exchanges,traces}));console.log(JSON.stringify({accepted:false,error:String(error),actions:index}));process.exitCode=1;}
