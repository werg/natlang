/** Final source-frozen comparisons: same baseline executions, separate durable freezes for all arms. */
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {resolve,join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {enrollFollowup} from './followup-allocation.mjs';
const root=resolve(process.argv[2]??'../runs/native-next-six-20261001'),study=join(root,'clean-study'),output=join(root,'confirmation');await mkdir(output,{recursive:true});
const design=JSON.parse(await readFile(join(study,'design.json'),'utf8'));const allocation=enrollFollowup(root);
const protocol={schema:'natlang.followup-confirmation/1',programs:design.programs,arms:['native','direct','separated'],alpha:0.05/3,maxModelCalls:600,maxRollouts:100,maxElapsedMs:21600000,interpretation:'Predeclared fixed fresh cases. Three comparisons corrected by Bonferroni; pilot scope is these authored programs, not broad model learning. No held-out output is used for source edits.'};
const path=join(output,'protocol.json');try{if(JSON.stringify(JSON.parse(await readFile(path,'utf8')))!==JSON.stringify(protocol))throw Error('Confirmation protocol changed');}catch(error){if(error.code!=='ENOENT')throw error;await writeFile(path,JSON.stringify(protocol,null,2));}
const allocationHost=await import('../../dist/index.js'),journal=new allocationHost.OperationJournal(join(output,'allocation'));
const {UsageGateway}=await import('../../dist/evaluation/usage.js');const {fingerprint}=await import('../../dist/adaptation/identity.js');
const gateway=new UsageGateway({maxModelCalls:protocol.maxModelCalls,maxRollouts:protocol.maxRollouts,maxProposals:0,maxElapsedMs:protocol.maxElapsedMs},journal.read('ledger')?.value);gateway.onUpdate=ledger=>journal.record('ledger',ledger);
const signal=AbortSignal.timeout(Math.max(1,protocol.maxElapsedMs-gateway.ledger.elapsedMs));const outcomes=[];
for(const id of protocol.programs){
 const directory=join(study,id),resultPath=join(output,id+'.json');
 try{outcomes.push(JSON.parse(await readFile(resultPath,'utf8')));continue;}catch(error){if(error.code!=='ENOENT')throw error;}
 let definition,arms;
 while(!signal.aborted){
  try{definition=JSON.parse(await readFile(join(directory,'protocol.json'),'utf8'));arms=Object.fromEntries(await Promise.all(protocol.arms.map(async arm=>[arm,JSON.parse(await readFile(join(directory,arm+'.json'),'utf8'))])));break;}
  catch(error){if(error.code!=='ENOENT')throw error;try{await readFile(join(study,'result.json'));break;}catch(error){if(error.code!=='ENOENT')throw error;}await new Promise(resolve=>setTimeout(resolve,10000));}
 }
 if(!arms){outcomes.push({id,disposition:'development-incomplete-or-confirmation-allocation-exhausted'});continue;}
 const frozen={id,definition,sources:{baseline:definition.files,...Object.fromEntries(protocol.arms.map(arm=>[arm,arm==='direct'?arms[arm].files:arms[arm].sourceManifest.files]))}};
 const freezePath=join(output,id+'-freeze.json');try{if(JSON.stringify(JSON.parse(await readFile(freezePath,'utf8')))!==JSON.stringify(frozen))throw Error('Selected source changed after freeze');}catch(error){if(error.code!=='ENOENT')throw error;await writeFile(freezePath,JSON.stringify(frozen,null,2));}
 const identity=createHash('sha256').update(JSON.stringify(frozen)).digest('hex');
 const host=await import(pathToFileURL(join(directory,'runtime/index.js')).href);
 const driver=host.openAICompatibleModelTurn({...definition.executor,request:{temperature:0.2}});
 const before=host.Folder.fromFiles(frozen.sources.baseline).snapshot(),comparisons={};
 const baselineJournal=new host.OperationJournal(join(output,id+'-native'));
 for(const arm of protocol.arms){
  const armJournal=new host.OperationJournal(join(output,id+'-'+arm));
  const evaluator=new host.SourceEvaluator(definition.contract,definition.cases,driver,gateway,{executorId:definition.executor.model,timeoutMs:definition.executorTimeoutMs,signal,journal:armJournal});
  if(arm!=='native'){const rows=definition.cases.filter(row=>row.split==='test'),reference=fingerprint({source:before.digest,suite:evaluator.suiteVersion,split:'test',ids:rows.map(row=>row.id),seed:0});for(const row of rows){const key=reference+':'+row.id,cached=baselineJournal.read(key);if(cached?.status==='done')armJournal.record(key,cached.value);}}
  try{comparisons[arm]=await evaluator.confirmPair(before,host.Folder.fromFiles(frozen.sources[arm]).snapshot(),identity,protocol.alpha);}catch(error){comparisons[arm]={error:String(error)};}
 }
 const result={id,freeze:identity,comparisons};await writeFile(resultPath,JSON.stringify(result,null,2));outcomes.push(result);console.log(JSON.stringify({id,comparisons:Object.fromEntries(Object.entries(comparisons).map(([arm,r])=>[arm,{baseline:r.baseline?.quality,selected:r.selected?.quality,error:r.error}]))}));
 await writeFile(join(output,'progress.json'),JSON.stringify({protocol,outcomes,ledger:gateway.snapshot(),assignment:allocation},null,2));
}
const totals={};for(const arm of protocol.arms){const rows=outcomes.map(row=>row.comparisons?.[arm]).filter(row=>row?.selected);const wins=rows.reduce((n,r)=>n+r.wins,0),losses=rows.reduce((n,r)=>n+r.losses,0),n=wins+losses;let mass=2**(-n),p=0;for(let k=0;k<=n;k++){if(k>=wins)p+=mass;mass*=(n-k)/(k+1);}totals[arm]={cases:rows.length,baselinePassed:rows.reduce((n,r)=>n+r.baseline.passed,0),selectedPassed:rows.reduce((n,r)=>n+r.selected.passed,0),baselineRequests:rows.reduce((n,r)=>n+r.baseline.modelCalls,0),selectedRequests:rows.reduce((n,r)=>n+r.selected.modelCalls,0),wins,losses,pValue:p,supportedWithinPilot:rows.length===protocol.programs.length&&wins>losses&&p<=protocol.alpha};}
await writeFile(join(output,'result.json'),JSON.stringify({protocol,outcomes,totals,ledger:gateway.snapshot(),assignment:allocation},null,2));console.log(JSON.stringify({done:true,totals}));
