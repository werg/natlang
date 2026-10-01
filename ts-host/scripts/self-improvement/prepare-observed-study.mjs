/** Select current-student failures/cost opportunities from completed development runs, never test results. */
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {join} from 'node:path';
import {isDeepStrictEqual} from 'node:util';
import {createHash} from 'node:crypto';
import {OperationJournal} from '../../dist/index.js';
import {fingerprint} from '../../dist/adaptation/identity.js';
const [input,output]=process.argv.slice(2);
if(!output)throw Error('usage: prepare-observed-study.mjs PRIOR_STUDY OUTPUT');
const protocolText=await readFile(join(input,'protocol.json'),'utf8'),prior=JSON.parse(protocolText);
const cases=[],observations=[];
for(const original of prior.cases){
 const directory=join(input,'development',original.family);
 let result;try{result=JSON.parse(await readFile(join(directory,'result.json'),'utf8'));}catch(error){if(error.code==='ENOENT')continue;throw error;}
 if(!result.baseline)continue;
 const training=original.cases.filter(row=>row.split==='train');
 const reference=fingerprint({source:result.baseline.source,suite:result.baseline.suiteVersion,split:'train',ids:training.map(row=>row.id),seed:0});
 const journal=new OperationJournal(join(directory,'journal'));
 const measured=training.flatMap(row=>{
  const record=journal.read(reference+':'+row.id);if(record?.status!=='done'||!record.value)return [];
  const value=record.value;
  return [{caseId:row.id,passed:!value.error&&isDeepStrictEqual(value.value,row.expected),modelCalls:value.modelCalls,failureKind:value.failureKind,error:value.error}];
 });
 const fixture=measured.some(row=>row.failureKind==='fixture');
 const failure=measured.some(row=>!row.passed);
 const expensive=measured.some(row=>row.passed&&row.modelCalls>1);
 const kind=fixture?'fixture-repair':failure?'observed-target-failure':expensive?'observed-execution-cost':'no-observed-opportunity';
 observations.push({family:original.family,kind,measured});
 if(fixture||(!failure&&!expensive))continue;
 const row=structuredClone(original);
 // Keep the existing group/split assignments and gold; use a small declared development comparison.
 const chosen=measured.find(row=>!row.passed)??measured.find(row=>row.modelCalls>1);
 row.cases=row.cases.filter(c=>c.split==='test'||c.split==='train'&&c.id===chosen.caseId||c.split==='validation'&&c.id===original.cases.find(c=>c.split==='validation').id);
 row.policy={...row.policy,maxExperiments:1,objective:failure?'quality':'model-calls',goal:row.policy.goal+' Ground the edit in current Bonsai training outcomes, actual action traces and public service declarations. Preserve semantic judgments and exact arithmetic; test one concrete execution change.'};
 row.selection={kind,prior:input,measured};cases.push(row);
}
if(!cases.length)throw Error('No completed current-student development evidence supports a target edit.');
await mkdir(output,{recursive:true});
const protocol={...prior,schema:'natlang.luna-bonsai-study/3',parentProtocol:{path:join(input,'protocol.json'),sha256:createHash('sha256').update(protocolText).digest('hex')},cases,selection:observations,
 developmentBudget:{maxModelCalls:160,maxRollouts:20,maxProposals:2,maxElapsedMs:1200000},executorTimeoutMs:300000,
 design:'One grounded development experiment per selected current-student opportunity. Training-only selection; original split/groups/gold retained. All prior attempts retained. Freeze before a single final paired test; do not revise after confirmation.'};
await writeFile(join(output,'protocol.json'),JSON.stringify(protocol,null,2)+'\n');
console.log(JSON.stringify({cases:cases.map(row=>({family:row.family,selection:row.selection.kind,objective:row.policy.objective})),observations}));
