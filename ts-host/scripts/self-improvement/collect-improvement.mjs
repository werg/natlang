/** Run real authored improvement trajectories. Failed runs stay visible and never become positive SFT. */
import { readFile,writeFile,mkdir } from 'node:fs/promises';
import { join,dirname } from 'node:path';
import { AUTHORED_IMPROVER } from '../../dist/improvement/authored-source.js';
import { createHash } from 'node:crypto';
import {validateBudget} from '../../dist/evaluation/suite.js';
import { createPiModelBackend } from '../../dist/model/pi-provider.js';
import { Folder, improveProgram, openAICompatibleModelTurn, AssignmentBudget, OperationJournal } from '../../dist/index.js';
const [casesPath,output,endpoint='http://127.0.0.1:8081',model='/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf',limitText='12',workersText='1']=process.argv.slice(2);
if(!output)throw Error('usage: collect-improvement.mjs CASES_JSONL OUTPUT_DIR [ENDPOINT MODEL LIMIT]');
await mkdir(output,{recursive:true});
const rows=(await readFile(casesPath,'utf8')).trim().split('\n').filter(Boolean).map(JSON.parse).slice(0,Number(limitText));
const allocation=new AssignmentBudget({collection:120,modelCalls:12480,caseExecutions:4800,trainingJobs:6,trainingUpdates:1200,confirmation:1,cost:100},{trainingJobs:4,trainingUpdates:800,confirmation:1},3,new OperationJournal(join(dirname(output),'assignment-allocation')));
if(!Number.isSafeInteger(Number(workersText))||Number(workersText)<1||Number(workersText)>4)throw Error('workers must be 1..4');
const results=[];let publish=Promise.resolve();
async function collect(row) {
  const limits={maxModelCalls:104,maxRollouts:40,maxProposals:2*row.policy.maxExperiments,maxElapsedMs:600000,...row.budget};validateBudget(limits);
  allocation.attempt(row.incidents[0]?.cluster??row.id);allocation.allocate({collection:1,modelCalls:limits.maxModelCalls,caseExecutions:limits.maxRollouts});
  const exchanges=[],traces=[],targetExchanges=[];
  const driver=(onExchange,endpointOverride=endpoint,modelOverride=model)=> {
    const endpoint=endpointOverride,model=modelOverride;
    if(endpoint.startsWith('pi:')) {
      const backend=createPiModelBackend(endpoint.slice(3),model);
      return async (request,signal)=> {
        const turn=await backend.turn(request,signal);
        onExchange({request,wireRequest:{messages:request.messages,tools:request.tools},wireResponse:{choices:[{message:{role:'assistant',content:turn.text??null,reasoning_content:turn.reasoning??null,tool_calls:turn.raw_calls??[]}}]}});
        return turn;
      };
    }
    return openAICompatibleModelTurn({endpoint,model,apiKey:process.env.NATLANG_IMPROVEMENT_API_KEY??(endpoint.includes('api.openai.com')?process.env.OPENAI_API_KEY:undefined),request:endpoint.includes('api.openai.com')?{reasoning_effort:'none'}:{temperature:0.2},onExchange});
  };
  try {
    let improverSource;
    if(row.improverReducer){const authored={...AUTHORED_IMPROVER};const template=authored['reducers/'+row.improverReducer+'.nl'];if(!template)throw Error('unknown authored transformation reducer');authored['improveStep/rewriteProgram.nl']=template.replace('request: TransformationRequest','request: RewriteRequest').replace('returns: TransformationExplanation','returns: RewriteResult')+'\nEdit directly using await folder.file(path).writeText(text) in eval. Do not propose or evaluate from this helper. Return the actual {summary,changed,preserves}.\n';improverSource=Folder.fromFiles(authored).snapshot();}
    const result=await improveProgram({improverSource,folder:Folder.fromFiles(row.files),contract:row.contract,cases:row.cases,policy:row.policy,directory:join(output,row.id+'.journal'),
      improver:driver(exchange=>exchanges.push(structuredClone(exchange))),executor:driver(exchange=>targetExchanges.push(structuredClone(exchange)),process.env.NATLANG_TARGET_ENDPOINT??endpoint,process.env.NATLANG_TARGET_MODEL??model),executorId:process.env.NATLANG_TARGET_MODEL??model,trace:trace=>traces.push(trace),
      budget:{...limits,...(endpoint.includes('api.openai.com')?{maxCost:5,pricing:{inputPerMillion:2.5,outputPerMillion:15},requestBounds:{maxInputTokens:16000,maxOutputTokens:4096}}:{})},signal:AbortSignal.timeout(600000)});
    const accepted=result.validation!==null && result.state.done && result.state.incumbent===result.folder.digest && (result.state.history.length>0 || row.control===true);
    const source=Object.fromEntries(result.folder.filePaths().map(path=>[path,new TextDecoder().decode(result.folder.readBytesSync(path))]));
    const artifact={version:'natlang.improvement-trajectory/1',id:row.id,accepted,limits,improver:{endpoint,model},executor:{endpoint:process.env.NATLANG_TARGET_ENDPOINT??endpoint,model:process.env.NATLANG_TARGET_MODEL??model},source,authored:result.authored,validation:result.validation,baseline:result.baseline,disposition:result.disposition,sourceManifest:result.sourceManifest,sourceDiff:result.sourceDiff,...('error'in result?{error:result.error}:{}),state:result.state,ledger:result.ledger,exchanges,traces,targetExchanges,incidents:row.incidents,sourceGroups:row.sourceGroups??[],caseDefinition:row};
    await writeFile(join(output,row.id+'.result.json'),JSON.stringify(artifact));results.push({id:row.id,accepted});
  } catch(error) {await writeFile(join(output,row.id+'.error.json'),JSON.stringify({id:row.id,error:String(error),exchanges,traces,targetExchanges,incidents:row.incidents}));results.push({id:row.id,accepted:false,error:String(error)});}
  publish=publish.then(()=>writeFile(join(output,'collection.json'),JSON.stringify({results,allocation:allocation.snapshot()},null,2)));await publish;
  console.log(JSON.stringify(results.at(-1)));
}

let cursor=0;await Promise.all(Array.from({length:Math.min(Number(workersText),rows.length)},async()=>{while(cursor<rows.length){const row=rows[cursor++];try{await collect(row);}catch(error){results.push({id:row.id,accepted:false,disposition:String(error).includes('exhausted')?'assignment-allocation-exhausted':'infrastructure-failure',error:String(error)});}}}));
await writeFile(join(output,'collection.json'),JSON.stringify({results,allocation:allocation.snapshot()},null,2));
