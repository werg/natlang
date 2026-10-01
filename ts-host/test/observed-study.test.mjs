import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,mkdirSync,writeFileSync,readFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {execFileSync} from 'node:child_process';
import {OperationJournal,Folder} from '../dist/index.js';
import {fingerprint} from '../dist/adaptation/identity.js';
import {AUTHORED_IMPROVER} from '../dist/improvement/authored-source.js';
import {compileVirtualProject} from '../dist/runtime/virtual-project.js';
import * as runtime from '../dist/runtime/node.js';

test('native bookkeeping treats correct multi-request executions as efficiency opportunities',()=>{
 const files={'main.ts':AUTHORED_IMPROVER['improveStep/context.ts'].replace("'../types'","'./types'"),'types.ts':AUTHORED_IMPROVER['types.ts']};
 const build=compileVirtualProject({files},runtime,{constrained:true,target:"node"});assert.equal(build.ok,true,JSON.stringify(build.diagnostics));
 const helper=build.require('main.ts'),policy={objective:'model-calls',goal:'Reduce requests while preserving judgment.',mode:'instruction',allowedFiles:['solve.nl']};
 const evidence=[{passed:true,modelCalls:3,modelTrace:[{calls:[{name:'return_result'}],observation:'typed answer staged'}]}];
 assert.equal(helper.opportunity(policy,evidence).kind,'efficiency');
 assert.equal(helper.opportunity({...policy,objective:'quality'},evidence).kind,'none');
 assert.equal(helper.opportunity(policy,[{passed:true}]).kind,'none');
 assert.equal(helper.opportunity(policy,[{passed:false,failureKind:'fixture',modelCalls:0}]).kind,'fixture');
 assert.equal(helper.opportunity(policy,[{passed:false,failureKind:'timeout',modelCalls:3}]).kind,'quality');
 assert.equal(helper.request(policy,'Combine computation and completion.',evidence,[]).evidence,evidence);
});

test('current-student selection uses measured train failures/costs and excludes fixture and held-out failures',()=>{
 const root=mkdtempSync(join(tmpdir(),'natlang-observed-study-')),input=join(root,'prior'),output=join(root,'next');mkdirSync(input);
 const cases=['expensive','fixture','test-only-failure','wrong-answer'].map(family=>({family,files:{'solve.nl':'---\nargs: {}\nreturns: number\n---\nReturn one.\n'},contract:{entry:'solve.nl',exportName:'default',programId:family},policy:{goal:'Return one.',maxExperiments:2},cases:['train','validation','test'].map(split=>({id:family+'-'+split,group:family+'-'+split,split,args:[],expected:1}))}));
 const protocol={optimizer:{model:'teacher'},executor:{model:'student'},cases};writeFileSync(join(input,'protocol.json'),JSON.stringify(protocol));
 for(const row of cases){
  const directory=join(input,'development',row.family);mkdirSync(directory,{recursive:true});
  writeFileSync(join(directory,'result.json'),JSON.stringify({baseline:{source:'baseline',suiteVersion:'suite'},confirmation:{quality:0}}));
  const reference=fingerprint({source:'baseline',suite:'suite',split:'train',ids:[row.family+'-train'],seed:0});
  new OperationJournal(join(directory,'journal')).record(reference+':'+row.family+'-train',row.family==='fixture'?{error:'invalid fixture',failureKind:'fixture',modelCalls:0}:{value:row.family==='wrong-answer'?2:1,modelCalls:row.family==='expensive'?3:1});
 }
 execFileSync(process.execPath,['scripts/self-improvement/prepare-observed-study.mjs',input,output]);
 const next=JSON.parse(readFileSync(join(output,'protocol.json'),'utf8'));
 assert.deepEqual(next.cases.map(row=>[row.family,row.policy.objective]),[['expensive','model-calls'],['wrong-answer','quality']]);
 assert.equal(next.selection.find(row=>row.family==='fixture').kind,'fixture-repair');
 assert.equal(next.selection.find(row=>row.family==='test-only-failure').kind,'no-observed-opportunity');
 for(const row of next.cases){assert.equal(row.policy.maxExperiments,1);assert.deepEqual(row.cases,cases.find(original=>original.family===row.family).cases);}
 assert.equal(next.developmentBudget.maxProposals,2);assert.ok(next.parentProtocol.sha256);
});

test('failure ingestion includes native result.json files and their rejected editor decisions',()=>{
 const root=mkdtempSync(join(tmpdir(),'natlang-native-incidents-')),input=join(root,'development','tickets'),output=join(root,'index');mkdirSync(input,{recursive:true});
 writeFileSync(join(input,'result.json'),JSON.stringify({sourceManifest:{schema:'natlang.program-source/1'},family:'tickets',disposition:'baseline-retained',validation:{gatesPassed:true},state:{done:true,history:[{accepted:false,reason:'Editor claimed changed paths but actual proposal was unchanged.'},{accepted:true,selected:false,reason:'Valid population child; incumbent retained.'}]}}));
 const confirmation=join(root,'development','confirmation');mkdirSync(confirmation);writeFileSync(join(confirmation,'result.json'),JSON.stringify({family:'held-out',freeze:{metric:'paired-binary-success'},selected:{quality:0}}));
 execFileSync(process.execPath,['scripts/self-improvement/index-failures.mjs',output,join(root,'development')]);
 const incidents=readFileSync(join(output,'incidents.jsonl'),'utf8').trim().split('\n').map(JSON.parse);
 assert.equal(incidents.length,2);const decision=incidents.find(row=>row.family==='tickets');assert.equal(decision.split,'train');assert.equal(decision.route,'decision-repair');assert.match(decision.reason,/unchanged/);
 assert.equal(incidents.find(row=>row.family==='held-out').route,'evaluation-only');
});

test('decision curriculum uses actual correct training costs rather than validation aggregates',()=>{
 const root=mkdtempSync(join(tmpdir(),'natlang-decision-cases-')),output=join(root,'cases');
 for(const [attempt,family]of [['grounded-handoff','curriculum_simple_deadline'],['refined','curriculum_simple_count_positive']]){
  const directory=join(root,attempt,'development',family);mkdirSync(directory,{recursive:true});
  const cases=[{id:'training',group:'training-group',split:'train',args:[],expected:1}];
  writeFileSync(join(root,attempt,'protocol.json'),JSON.stringify({cases:[{family,cases}]}));
  writeFileSync(join(directory,'result.json'),JSON.stringify({state:{history:[{accepted:false,reason:'Correct answers incorrectly ruled out cost improvement.'}]},baseline:{source:'baseline',suiteVersion:'suite',modelCalls:100,total:2}}));
  const reference=fingerprint({source:'baseline',suite:'suite',split:'train',ids:['training'],seed:0});new OperationJournal(join(directory,'journal')).record(reference+':training',{value:1,modelCalls:3});
 }
 execFileSync(process.execPath,['scripts/self-improvement/build-observed-decision-cases.mjs',root,output]);
 const rows=readFileSync(join(output,'cases.jsonl'),'utf8').trim().split('\n').map(JSON.parse);assert.equal(rows.length,2);
 for(const row of rows){assert.equal(row.cases[0].args[1][0].modelCalls,3);assert.equal(row.provenance.positiveSFT,false);assert.equal(row.provenance.historicalSource,false);assert.deepEqual(row.sourceGroups,['training-group']);}
});

test('authored experiment helper measures edits and never installs a rejected candidate',async()=>{
 const files={'main.ts':AUTHORED_IMPROVER['improveStep/experiment.ts'].replace("'../types'","'./types'").replace("'./feedback'","'./feedback'"),'capabilities.ts':AUTHORED_IMPROVER['improveStep/capabilities.ts'].replace("'../types'","'./types'"),'feedback.ts':AUTHORED_IMPROVER['improveStep/feedback.ts'].replace("'../types'","'./types'").replace("'../capabilities'","'./capabilities'"),'context.ts':AUTHORED_IMPROVER['improveStep/context.ts'].replace("'../types'","'./types'"),'types.ts':AUTHORED_IMPROVER['types.ts']};
 const build=compileVirtualProject({files},runtime,{constrained:true,target:"node"});assert.equal(build.ok,true,JSON.stringify(build.diagnostics));
 const helper=build.require('main.ts');
 const task=runtime.createNatlangRuntime();
 const execute=(...args)=>task.run(()=>helper.test(...args));
 const policy={objective:'model-calls',goal:'Batch judgments.',mode:'structural',allowedFiles:['solve.nl'],maxExperiments:3,maxPopulation:3,strategy:'adaptive'};
 let accepts=0,proposals=0,changed=true,valid=true,calls=4,fixture=false;
 const candidate={digest:'candidate',filePaths:()=>['solve.nl'],readText:async()=> 'edited candidate source'},parent={digest:'parent',branch:()=>({filePaths:()=>["solve.nl"],readText:async()=>"judge then count",propose:async()=>{proposals++;return {value:{summary:"Batch judgments."},folder:candidate,diff:{changes:changed?[{path:'solve.nl'}]:[]}};},accept:async()=>{accepts++;return candidate;}})};
 const folder={at:()=>parent};
 const evaluator={evaluate:async(source,request)=>({source:source.digest,quality:1,sourceBytes:100,modelCalls:source===candidate?calls:4,gatesPassed:true,evidence:'train',scores:request.split==='validation'?[{caseId:'v',quality:1}]:undefined}),page:()=>[{passed:!fixture,modelCalls:4,...(fixture?{failureKind:'fixture',error:'broken fixture'}:{})}],check:async()=>({valid,diagnostics:['bad contract']})};
 let result=await execute(folder,evaluator,'parent',{},policy,'Batch judgments.');assert.equal(result.accepted,false);assert.match(result.reason,/calls 4 vs 4/);assert.equal(accepts,0);assert.equal(result.feedback.sourceFiles[0].text,"edited candidate source");assert.equal(result.feedback.training[0].modelCalls,4);assert.deepEqual(result.feedback.validation,{quality:1,modelCalls:4});
 calls=2;result=await execute(folder,evaluator,'parent',{},policy,'Batch judgments.');assert.equal(result.accepted,true);assert.equal(result.source,'candidate');assert.equal(result.modelCalls,2);assert.equal(accepts,1);
 changed=false;result=await execute(folder,evaluator,'parent',{},policy,'No-op.');assert.match(result.reason,/No supported hypothesis/);assert.equal(accepts,1);
 changed=true;valid=false;result=await execute(folder,evaluator,'parent',{},policy,'Invalid edit.');assert.match(result.reason,/Compilation rejected/);assert.equal(accepts,1);
 fixture=true;const before=proposals;result=await execute(folder,evaluator,'parent',{},policy,'Invent repair.');assert.match(result.reason,/^fixture-error: broken fixture/);assert.equal(proposals,before);
});

test('retired optimizer templates migrate to the exact lifecycle and lose stale admission',async()=>{
 const {migrateLifecycle}=await import('../scripts/self-improvement/migrate-improver-lifecycle.mjs');
 const old={id:'old-turn',task:{program_ir:{semantics:{files:{'improveStep.nl':'old root','improveStep/measureBaseline.nl':'old measurement'}}}},messages:[{role:'tool',content:'historical observation'}],training_admission:{approved:true},trace_admission:{admitted:true}};
 const migrated=migrateLifecycle(old);
 assert.equal(migrated.changes,1);assert.equal(migrated.row.training_admission.approved,false);assert.equal(migrated.row.trace_admission.admitted,false);assert.equal(migrated.row.migration.disposition,'recollect-required');
 assert.equal(migrated.row.task.program_ir.semantics.files['improveStep.nl'],AUTHORED_IMPROVER['improveStep.nl']);assert.equal(migrated.row.task.program_ir.semantics.files['improveStep/measureBaseline.nl'],undefined);
 assert.equal(old.training_admission.approved,true);assert.deepEqual(migrated.row.messages,old.messages);
 assert.equal(migrateLifecycle(migrated.row).changes,0);
 assert.equal(AUTHORED_IMPROVER['improveStep/selectCandidate.nl'],undefined);
 const wrapped={id:'wrapped-plan',files:{...AUTHORED_IMPROVER,'improveStep/planExperiment.nl':'Retired plan with ExperimentPlan'},training_admission:{approved:true}};
 const current=migrateLifecycle(wrapped);assert.equal(current.changes,1);assert.equal(current.row.training_admission.approved,false);assert.equal(current.row.files['improveStep/planExperiment.nl'],undefined);assert.equal(migrateLifecycle(current.row).changes,0);
});

test('structural curriculum reconstructs measured literal writes without executing them or reusing closed tests',()=>{
 const root=mkdtempSync(join(tmpdir(),'natlang-structural-cases-')),output=join(root,'cases');
 const files={'recommending_reviews.nl':'---\nargs: {reviews: "Review[]"}\nreturns: number\n---\nJudge each review with a helper.','types.ts':'export type Review={id:string;text:string};'},replacement=files['recommending_reviews.nl'].replace('Judge each review with a helper.','Judge the reviews directly and count exactly.');
 const cases=[{id:'train',group:'train-group',split:'train',args:[[{id:'a',text:'I would buy it again.'}]],expected:1},{id:'validation',group:'validation-group',split:'validation',args:[[]],expected:0},{id:'closed-test',group:'closed-test',split:'test',args:[[]],expected:999}];
 const protocol={files,contract:{entry:'recommending_reviews.nl',exportName:'default',programId:'fixture'},cases,policy:{maxExperiments:3,maxPopulation:3,mode:'structural',strategy:'adaptive',objective:'model-calls',goal:'Reduce requests preserving semantic judgment.',allowedFiles:Object.keys(files)},budget:{maxModelCalls:100,maxRollouts:20,maxProposals:3}};
 writeFileSync(join(root,'protocol.json'),JSON.stringify(protocol));writeFileSync(join(root,'native.json'),JSON.stringify({baseline:{suiteVersion:'suite'}}));
 writeFileSync(join(root,'exchanges.ndjson'),JSON.stringify({command:'native',role:'optimizer',turn:{calls:[['eval',{code:'throw Error("saved code must not execute");await folder.file("recommending_reviews.nl").writeText('+JSON.stringify(replacement)+');'}]]}})+'\n');
 const journal=new OperationJournal(join(root,'native-journal'));
 for(const [sourceFiles,calls]of [[files,6],[{...files,'recommending_reviews.nl':replacement},2]]){const source=Folder.fromFiles(sourceFiles).snapshot().digest,reference=fingerprint({source,suite:'suite',split:'train',ids:['train'],seed:0});journal.record(reference+':train',{value:1,modelCalls:calls});}
 execFileSync(process.execPath,['scripts/self-improvement/build-structural-cases.mjs',root,output]);
 const rows=readFileSync(join(output,'cases.jsonl'),'utf8').trim().split('\n').map(JSON.parse);assert.equal(rows.length,2);
 for(const row of rows){assert.equal(row.provenance.positiveSFT,false);assert.equal(row.incidents[0].split,'train');assert.equal(row.cases.some(item=>item.id==='closed-test'||item.expected===999),false);assert.equal(row.cases.filter(item=>item.split==='test').length,2);assert.deepEqual(row.sourceGroups,['train-group']);}
 assert.equal(rows[1].files['recommending_reviews.nl'],replacement);
});
