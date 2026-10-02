import test from 'node:test';
import assert from 'node:assert/strict';
import {Folder,SourceEvaluator} from '../dist/index.js';
import {UsageGateway} from '../dist/evaluation/usage.js';
import {directoryCampaignCases} from '../scripts/self-improvement/build-directory-campaign.mjs';
import {trainingEligible} from '../scripts/self-improvement/prepare-optimizer-training.mjs';

test('optimizer weight training excludes transfer families and refuses unverified decisions',()=>{
 const row={id:'verified',split:'train',outcome:{accepted:true},training_admission:{approved:true,kind:'exact-native-runtime-oracle'},trace_admission:{admitted:true},task:{program_ir:{semantics:{evaluation_fixture:{caseDefinition:{cohort:'development',cases:[{split:'train'}]}}}}}};
 assert.equal(trainingEligible(row),true);
 assert.equal(trainingEligible({...row,task:{program_ir:{semantics:{evaluation_fixture:{caseDefinition:{cohort:'transfer'}}}}}}),false);
 assert.throws(()=>trainingEligible({...row,outcome:{accepted:false}}),/Unverified/);
 row.task.program_ir.semantics.evaluation_fixture.caseDefinition.cases.push({split:'test'});
 assert.throws(()=>trainingEligible(row),/Sealed cases/);
});

test('directory campaign starts with executing partial programs and independently exact file oracles',async()=>{
 const rows=directoryCampaignCases();assert.equal(rows.length,6);
 assert.equal(rows.filter(row=>row.cohort==='development').length,4);
 const positives=['Please reverse my payment.','Final decision: approved.','Dependable and useful. I recommend it.','Safety training is complete.'];
 for(const row of rows){
  let requests=0;
  const driver=async request=>{
   requests++;const text=request.messages.map(message=>String(message.content)).join('\n');
   if(String(request.messages[1].content).includes('Run the existing workflow with folder'))return {calls:[['eval',{code:'return await workflow.run(folder);',finish:true}]]};
   if(row.family==='confirmed-meetings')return {text:text.includes('Confirmed: Alice at Library.')?'{"owner":"Alice","place":"Library"}':'null'};
   return {text:String(positives.some(value=>text.includes(value)))};
  };
  const evaluator=new SourceEvaluator(row.contract,row.cases,driver,new UsageGateway({maxModelCalls:20,maxRollouts:2,maxProposals:0}),{executorId:'fixture-labels',timeoutMs:10000});
  const report=await evaluator.evaluate(Folder.fromFiles(row.files).snapshot(),{split:'train',caseIds:[row.cases[0].id]});
  assert.equal(report.quality,1,row.id+': '+JSON.stringify(report));assert.ok(requests>0);
  for(const example of row.cases){
   assert.equal(example.expectedFiles['README.md'],example.folder['README.md']);
   assert.equal(example.expectedFiles['archive/old.txt'],example.folder['archive/old.txt']);
   if(row.family!=='exact-edits')assert.equal(JSON.stringify(JSON.parse(example.expectedFiles['report.json'])),example.expectedFiles['report.json']);
  }
 }
});

test('source editor receives rejected helper source and causal diagnostics in its first compact card',async()=>{
 const {AUTHORED_IMPROVER}=await import('../dist/improvement/authored-source.js');
 const {compileVirtualProject}=await import('../dist/runtime/virtual-project.js');
 const runtime=await import('../dist/runtime/node.js');
 const build=compileVirtualProject({files:{'main.ts':AUTHORED_IMPROVER['improveStep/context.ts'].replace("'../types'","'./types'"),'types.ts':AUTHORED_IMPROVER['types.ts']}},runtime,{constrained:true,target:'node'});
 assert.equal(build.ok,true);
 const feedback={sourceFiles:[{path:'workflow.ts',text:'wrongJoin();'}],training:[],diagnostics:['Train second: failed; wrong invoice joined.'],reason:'Candidate rejected: file effects differ.'};
 const request=build.require('main.ts').request({goal:'Join invoices by ID.',mode:'structural',allowedFiles:['workflow.ts']},'',[{passed:false,caseId:'second'}],[{path:'workflow.ts',text:'originalJoin();'}],[],feedback);
 assert.match(request.brief,/Previous experiment: Candidate rejected/);assert.match(request.brief,/wrong invoice joined/);assert.match(request.brief,/Previous candidate workflow.ts:\nwrongJoin\(\)/);
 assert.equal(request.lastExperiment,feedback);
});
