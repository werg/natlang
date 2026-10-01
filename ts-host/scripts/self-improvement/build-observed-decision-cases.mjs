/** Reconstruct native optimizer mistakes as executable repair cases; not positive SFT or reproduced student failures. */
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {isDeepStrictEqual} from 'node:util';
import {OperationJournal} from '../../dist/index.js';
import {fingerprint} from '../../dist/adaptation/identity.js';
const [study,output]=process.argv.slice(2);
if(!output)throw Error('usage: build-observed-decision-cases.mjs STUDY OUTPUT');
const definitions=[
 ['cost-opportunity','grounded-handoff','curriculum_simple_deadline','If any training case failed, report quality. If all training cases passed, report none: no wrong answer means no improvement is substantiated.'],
 ['fixture-diagnosis','refined','curriculum_simple_count_positive','If all training cases passed, report fixture: this means no target failure was evidenced. Otherwise report quality.']
];
const rows=[];
for(const [mechanism,attempt,family,instructions]of definitions){
 const path=join(study,attempt,'development',family,'result.json'),text=await readFile(path,'utf8'),result=JSON.parse(text);
 const history=result.state.history.find(row=>row.accepted===false);if(!history)throw Error('observed rejected decision missing: '+path);
 const protocol=JSON.parse(await readFile(join(study,attempt,'protocol.json'),'utf8')),original=protocol.cases.find(row=>row.family===family);
 const training=original.cases.filter(row=>row.split==='train');
 const reference=fingerprint({source:result.baseline.source,suite:result.baseline.suiteVersion,split:'train',ids:training.map(row=>row.id),seed:0});
 const journal=new OperationJournal(join(study,attempt,'development',family,'journal'));
 const observed=training.map(row=>({case:row,measurement:journal.read(reference+':'+row.id)?.value})).find(row=>row.measurement?.modelCalls>1&&!row.measurement.error&&isDeepStrictEqual(row.measurement.value,row.case.expected));
 if(!observed)throw Error('Measured training execution missing; validation counts cannot seed training cases.');
 const id='observed-native-'+mechanism;
 const goal='Classify the optimizer opportunity. A fixture failure explicitly diagnosed before target execution means fixture. Other failed executions mean quality, with timeout treated as operational failure. Correct executions taking more than one measured model request permit efficiency for a model-calls objective. A source-size objective permits source-size simplification of correct behavior. Otherwise return none. Passing answers do not finish a cost objective; never invent unknown costs or fixture failures.';
 const source=`---\nargs:\n  objective: "'quality' | 'model-calls' | 'source-size'"\n  evidence: "Evidence[]"\nreturns: "'fixture' | 'quality' | 'efficiency' | 'source-size' | 'none'"\n---\n${instructions}\n`;
 const inputs=[
  ['train','model-calls',[{passed:true,modelCalls:observed.measurement.modelCalls}], 'efficiency','observed-correct-expensive'],
  ['train','quality',[{passed:true,modelCalls:2}], 'none','independent-quality-control'],
  ['validation','model-calls',[{passed:true,modelCalls:1}], 'none','independent-minimum-cost'],
  ['validation','model-calls',[{passed:false,failureKind:'target',modelCalls:2}], 'quality','independent-failed-target'],
  ['test','model-calls',[{passed:false,failureKind:'fixture',modelCalls:0}], 'fixture','independent-fixture-boundary'],
  ['test','source-size',[{passed:true}], 'source-size','independent-size-objective'],
  ['test','model-calls',[{passed:true}], 'none','independent-unknown-cost']
 ];
 rows.push({version:'natlang.improvement-case/1',id,family:'native-optimizer-'+mechanism,files:{'diagnose.nl':source,'types.ts':'export type Evidence={passed:boolean;modelCalls?:number;failureKind?:"fixture"|"target"|"timeout"};\n'},contract:{entry:'diagnose.nl',exportName:'default',programId:id},cases:inputs.map(([split,objective,evidence,expected,tag])=>({id:id+':'+tag,group:id+':'+tag,split,args:[objective,evidence],expected})),policy:{maxExperiments:2,maxPopulation:3,mode:'instruction',strategy:'adaptive',objective:'quality',goal,allowedFiles:['diagnose.nl']},incidents:[{id:createHash('sha256').update(path+history.reason).digest('hex'),cluster:id,split:'train',route:'decision-repair',reason:history.reason,source:{file:path,hash:createHash('sha256').update(text).digest('hex'),path:'/state/history'}}],sourceGroups:original.cases.filter(row=>row.split==='train').map(row=>row.group),provenance:{kind:'reconstructed-native-optimizer-decision',historicalSource:false,source:path,startingInstructions:'Synthesized faulty instructions expressing the observed decision; not original target source.',oracle:'Independent declared decision table',disposition:'requires-target-reproduction-and-verified-collection',positiveSFT:false}});
}
await mkdir(output,{recursive:true});await writeFile(join(output,'cases.jsonl'),rows.map(JSON.stringify).join('\n')+'\n');
console.log(JSON.stringify({cases:rows.length,positiveSFT:false,disposition:'requires-target-reproduction-and-verified-collection'}));
