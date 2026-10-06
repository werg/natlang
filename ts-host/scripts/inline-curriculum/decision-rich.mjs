// Bounded semantic choices become structured returns through exact code. No teacher prose is invented.
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {Random,curriculumCase,evalCall,returnCall,nlFile,literal} from './lib.mjs';
const sha=s=>createHash('sha256').update(s).digest('hex');
const SKILLS=['judge-against-criteria','calculate-from-data','move-and-verify-files'];
const LIBRARY=Object.fromEntries(SKILLS.map(id=>[id,readFileSync(new URL(`../../../training/student-skills/task-workflows-v2/skills/${id}/SKILL.md`,import.meta.url),'utf8')]));
const descriptions=Object.fromEntries(Object.entries(LIBRARY).map(([id,body])=>[id,/^description: (.+)$/m.exec(body)[1]]));
const SKILL_TASKS=[
 ['Filter incident reports by whether customers are experiencing the fault now.','The task mixes historical, hypothetical and current reports.',0,[]],
 ['Calculate the total of paid invoice rows, excluding void rows.','All numeric fields and statuses are supplied.',1,[]],
 ['Move selected documents into an archive and check that source and destination are correct.','The selection is complete; filesystem operations remain.',2,[]],
 ['Return the integer 7 supplied explicitly by the caller.','There is no outstanding computation, judgment or filesystem work.',3,[]],
 ['Continue filtering current incidents.','The criteria procedure is already in the active context; do not load it twice.',3,['judge-against-criteria']],
 ['The report title mentions file moves, but the task is to sum its expense rows.','No files should be moved; exact addition is the remaining work.',1,[]],
 ['Identify which complaints describe an unresolved problem rather than a requested feature.','Read semantic evidence before deciding membership.',0,[]],
 ['Compute tax-inclusive totals for the three supplied line items.','Use the given rate and round only the final result.',1,[]],
 ['The set of stale assets is already approved. Relocate them and verify the final tree.','Do not reconsider the classification; complete the file operations.',2,[]],
 ['Provide the already verified response.','The relevant arithmetic skill has already been supplied; there is nothing left to calculate.',3,['calculate-from-data']],
];
const EXTRACT_TASKS=[
 ['Who actually received the parcel?',['The warehouse clerk signed the incoming depot manifest.','Mira took the parcel at her front door and signed for it.','The courier plans to contact Noel tomorrow.'],1],
 ['Which statement establishes that the problem is still happening?',['The outage last week was repaired.','Users are unable to save their work today.','If the fault returns, saving could fail.'],1],
 ['Which clause permits cancellation before dispatch?',['After dispatch, cancellation is unavailable.','Customers may cancel before the goods leave our facility.','Payment is normally collected when the order is placed.'],1],
 ['Which statement commits to completing the repair?',['The requester would like the repair soon.','The engineer will deploy the fix on Friday.','The earlier attempt did not finish the repair.'],1],
 ['Which sentence confirms a completed refund?',['A refund has been requested.','The bank credited the refund to the customer yesterday.','Support will ask finance about a refund.'],1],
 ['Which sentence establishes that all accounts have access?',['One pilot account can export.','Every account is now enabled for export.','The team expects rollout to continue tomorrow.'],1],
 ['Who confirmed receipt at the final destination?',['The sorting center logged arrival.','Omar confirmed taking the package inside his apartment.','The driver scheduled a delivery visit.'],1],
 ['Which sentence describes a current failure rather than a risk?',['Yesterday the system was unavailable, but it recovered.','Customers continue to receive errors when submitting orders.','A similar outage might occur during the next upgrade.'],1],
 ['Which clause grants approval rather than merely requesting it?',['The change request is awaiting approval.','The release owner authorizes this breaking API change for this release.','The developer prefers to break the old interface.'],1],
 ['Which statement confirms completion rather than intention?',['The operator intends to archive the logs.','The logs have been archived and the source directory is empty.','Archiving the logs would save space.'],1],
];
function tag(record,domain,index){
 record.dataset_records=[`authored-bounded-decisions-v1:${domain}:fixture-${index}`];
 record.source_groups=record.dataset_records;
 record.source_revisions.push('authored-bounded-decisions/1');
 return record;
}
function metadata(record,domain,requests,captures){
 record.generation.decision_distillation={version:3,domain,capture_names:captures,requests,oracle:'authored-fixture semantics; selected fields copied exactly by code',instruction_source:'runtime objective and current scoped context'};
 return record;
}
export function decisionSkillCatalog(seed,index,split='train'){
 const fixture=split==='test'?6+index%4:index%6,[objective,history,gold,loaded]=SKILL_TASKS[fixture];
 const rng=new Random(seed,`skill-catalog:${fixture}`),order=rng.shuffle([...SKILLS,'none']);
 const candidates=Object.fromEntries(order.map((id,i)=>[`c${i}`,{id,description:id==='none'?'No additional skill is necessary.':descriptions[id],next_step:id==='none'?'Complete the remaining request directly.':`Read and apply ${id} only to the remaining work.`}]));
 const expectedKey=`c${order.indexOf(gold===3?'none':SKILLS[gold])}`,expected=candidates[expectedKey];
 const task={objective,history,loaded},policy='Choose only a skill needed for the remaining work. An already-loaded applicable procedure needs no additional retrieval. Choose none for trivial completed work. Do not choose from words in the title alone.';
 const question=`Choose the necessary next skill for: ${objective}`;
 const code="const question = 'Choose the necessary next skill for: '+task.objective; const selected = await nl<SkillPlan>`${question} Consult task, policy and candidates. Return the selected candidate record exactly.`(); console.log(JSON.stringify(selected)); return selected;";
 const criteria=Object.fromEntries(Object.entries(candidates).map(([key,value])=>[key,`${value.id}: ${value.description}`]));
 const rootCalls=[evalCall(code),...(expected.id==='none'?[]:[['read_code',{name:`skills.${expected.id}`}]]),returnCall(expected)];
 const files={'types.ts':'export type SkillPlan = { id: string; description: string; next_step: string };\n',...Object.fromEntries(Object.entries(LIBRARY).map(([id,body])=>[`select_skill/skills/${id}/SKILL.md`,body]))};
 const record=curriculumCase({family:'decision_skill_catalog',shape:`s${seed}-${index}`,variant:'catalog',split,slice:'inline_placement',domain:'other',mode:'single_call',inline:'required',
 root:{name:'select_skill',args:{task:'unknown',policy:'string',candidates:'Record<string, SkillPlan>'},returns:'SkillPlan',instructions:'Select the useful next skill using the task, history, already-loaded list and skill descriptions. Formulate an inline question from the runtime objective. This invocation selects the next skill; do not execute the task objective. Return the supplied chosen candidate record unchanged (including description and next_step), and read its bound skill with read_code before finishing unless none is selected. If none is selected, return that supplied record unchanged rather than composing a new answer to the objective.'},
 files,inputs:{task,policy,candidates},expected,reference:{root:rootCalls,children:[{match:'You are inside this call: nl@eval:',calls:[evalCall('return {task,policy,candidates,question};'),evalCall(`return candidates.${expectedKey};`),returnCall(expected)]}]}});
 metadata(record,'skill_catalog',[{id:`${record.id}:select`,state:{task,policy,candidates,question},questions:{decision:{type:'choice',instructions:question+' '+policy,criteria}},expected:expectedKey}],['task','policy','candidates','question']);
 record.generation.skill_catalog={paths:Object.keys(files).filter(p=>p.endsWith('SKILL.md')),sha256:Object.fromEntries(Object.entries(LIBRARY).map(([id,body])=>[id,sha(body)])),selected:expected.id,already_loaded:loaded};
 return [tag(record,'skills',fixture)];
}
export function decisionExtractChain(seed,index,split='train'){
 const fixture=split==='test'?6+index%4:index%6,[objective,spans,gold]=EXTRACT_TASKS[fixture],rng=new Random(seed,`extract:${index}`);
 const poolSize=split==='test'?4:6;
 const depth=1+Math.floor(index/poolSize)%4,width=1+Math.floor(index/(poolSize*4))%3;
 const directory=level=>['review',...Array.from({length:depth-1-level},(_,i)=>`merge_${depth-1-i}`)].join('/');
 const files={'types.ts':'export type Evidence = { source_id: string; quote: string; request: string; role: string };\n'},children=[],requests=[],expected=[];
 for(let i=0;i<width;i++){
  const id=`packet_${fixture}_${i}`,name=`extract_${i}`,order=rng.shuffle([0,1,2]);
  const candidates=Object.fromEntries(order.map((which,k)=>[`c${k}`,{source_id:id,quote:spans[which],request:objective,role:'selected-evidence'}]));
  const expectedKey=`c${order.indexOf(gold)}`,finding=candidates[expectedKey];expected.push(finding);
  const question=`Select the exact evidence span that answers: ${objective}`;
  const policy='Select evidence that establishes the requested fact, not a related event, wish, plan or hypothetical. Copy the selected candidate without changing its fields.';
  const packet={candidates,policy};
  files[`${directory(0)}/extract_${i}/packet.ts`]=`/** Private evidence candidates for this extractor. */\nexport function read(): unknown { return ${literal(packet)}; }\n`;
  files[`${directory(0)}/${name}.nl`]=nlFile({args:{objective:'string'},returns:'Evidence',instructions:'Read packet.read(). Compose a focused extraction question from objective. Use an inline nl<Evidence> lambda on the candidate records and policy, with those values in scope; copy the selected record exactly. Do not infer from an id.'});
  const code='const context = packet.read(); const candidates = context.candidates; const policy = context.policy; const question = "Select the exact evidence span that answers: "+objective; const finding = await nl<Evidence>`${question} Consult candidates and policy. Copy the selected candidate record exactly.`(); console.log(JSON.stringify(finding)); return finding;';
  children.push({match:`You are inside this call: ${name}(`,calls:[evalCall(code),returnCall(finding)]});
  children.push({match:['You are inside this call: nl@eval:',id],calls:[evalCall('return {candidates,policy,question};'),evalCall(`return candidates.${expectedKey};`),returnCall(finding)]});
  const criteria=Object.fromEntries(Object.entries(candidates).map(([key,value])=>[key,value.quote]));
  requests.push({id:`${id}:extract`,state:{candidates,policy,question},questions:{decision:{type:'choice',instructions:question+' '+policy,criteria}},expected:expectedKey});
 }
 let names=Array.from({length:width},(_,i)=>`extract_${i}`),code=`const findings = [${names.map(n=>`await ${n}(task.objective)`).join(',')}]; console.log(JSON.stringify(findings)); return findings;`;
 // A chain is a real nested invocation DAG, not copied prompt history. Each parent only sees child returns.
 for(let level=1;level<depth;level++){
  const name=`merge_${level}`;
  files[`${directory(level)}/${name}.nl`]=nlFile({args:{task:'unknown'},returns:'Evidence[]',instructions:`Call ${names.join(', ')} exactly once with ${level===1?'task.objective':'task'}. Combine their evidence in order, preserve every field, and return the array.`});
  children.push({match:`You are inside this call: ${name}(`,calls:[evalCall(code),returnCall(expected)]});
  names=[name];code=`const findings = (await ${name}(task)); console.log(JSON.stringify(findings)); return findings;`;
 }
 const record=curriculumCase({family:'decision_extract_chain',shape:`s${seed}-${index}`,variant:'evidence',split,slice:'nested_scoped',domain:'other',mode:'followup',inline:'required',named:'required',plausibleActions:['select evidence of the completed fact','reject a merely related intention'],decisive:[{marker:`packet_${fixture}_0`,source:'child',note:'The selected span establishes the requested fact; related plans and hypothetical events do not.'}],
 root:{name:'review',args:{task:'unknown'},returns:'Evidence[]',instructions:`Obtain evidence through ${names.join(', ')}. Give the extractors the runtime task objective. Return all evidence records in order exactly; the private packets are accessible only to their extractors.`},files,inputs:{task:{objective}},expected,reference:{root:[evalCall(code),returnCall(expected)],children}});
 metadata(record,'bounded_extraction',requests,['candidates','policy','question']);
 record.generation.recurrence={version:2,depth:depth+1,width,inline_judgments:true,controls:['correct','shuffled','zero','removed','counterfactual'],admission:'requires complete observed invocation DAG and source closure'};
 return [tag(record,'extraction',fixture)];
}
